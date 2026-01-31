# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the "License");
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/axiom/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTree utilities - pure Python implementation without JAX.

NO JAX - uses only numpy.
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union
import numpy as np


class ArrayDict:
    """Dictionary-like container for array storage - immutable."""
    
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        object.__setattr__(instance, "_fields", kwargs)
        return instance

    def get(self, key, value=None):
        return self._fields.get(key, value)

    def items(self):
        return self._fields.items()

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()

    def __getitem__(self, key):
        return self._fields[key]

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            return super().__getattribute__(item)
        try:
            return self._fields[item]
        except KeyError as exc:
            raise AttributeError(f"'{self.__class__.__name__}' object has not attribute '{item}'") from exc

    def __setattr__(self, key, value):
        raise AttributeError("Cannot modify immutable instance")

    def __delattr__(self, key):
        raise AttributeError("Cannot delete attributes of an immutable instance")

    def __repr__(self):
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._fields)
        return f"{self.__class__.__name__}({fields})"

    def __iter__(self):
        return iter(self._fields.items())

    def __contains__(self, key):
        return key in self._fields

    def __len__(self):
        return len(self._fields)


def size(array_dict) -> int:
    """Count total size of arrays in PyTree."""
    total_size = 0
    for param in tree_leaves(array_dict):
        if isinstance(param, np.ndarray):
            total_size = total_size + param.size
    return total_size


def tree_leaves(tree):
    """Flatten tree into list of leaves."""
    leaves = []
    
    def collect_leaves(node):
        if isinstance(node, (np.ndarray, int, float, str, bool, type(None))):
            leaves.append(node)
        elif isinstance(node, ArrayDict):
            for v in node.values():
                collect_leaves(v)
        elif isinstance(node, dict):
            for v in node.values():
                collect_leaves(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                collect_leaves(item)
        else:
            leaves.append(node)
    
    collect_leaves(tree)
    return leaves


def tree_map(fn, tree, is_leaf=None):
    """Map function over tree."""
    def map_node(node):
        if is_leaf is not None and is_leaf(node):
            return fn(node)
        elif isinstance(node, (np.ndarray, int, float, str, bool, type(None))):
            return fn(node)
        elif isinstance(node, ArrayDict):
            return ArrayDict(**{k: map_node(v) for k, v in node.items()})
        elif isinstance(node, dict):
            return {k: map_node(v) for k, v in node.items()}
        elif isinstance(node, (list, tuple)):
            mapped = [map_node(item) for item in node]
            return type(node)(mapped)
        else:
            return fn(node)
    
    return map_node(tree)


def sum_pytrees(*pytrees):
    """Sum multiple PyTrees element-wise."""
    def sum_fn(*args):
        result = args[0]
        for arg in args[1:]:
            if result is None:
                result = arg
            elif arg is None:
                pass
            else:
                result = result + arg
        return result
    
    return tree_map(sum_fn, pytrees[0])


def zeros_like(array_dict) -> ArrayDict:
    """Create ArrayDict with zeros matching input structure."""
    def zeros_for_value(value):
        if isinstance(value, np.ndarray):
            return np.zeros(value.shape, value.dtype)
        elif isinstance(value, ArrayDict):
            return zeros_like(value)
        elif isinstance(value, dict):
            return {k: zeros_for_value(v) for k, v in value.items()}
        else:
            return value

    return ArrayDict(**{key: zeros_for_value(val) for key, val in array_dict.items()})


def tree_copy(tree):
    """Copy a PyTree (re-references leaves)."""
    def copy(x):
        if isinstance(x, np.ndarray):
            return x.copy()
        return x
    return tree_map(copy, tree)


def apply_add(dist, updates):
    """Tree-map broadcasted addition handling None leaves."""
    def _apply_add(p, u):
        if u is None:
            return p
        else:
            return p + u

    def _is_none(x):
        return x is None

    return tree_map(lambda x, y: _apply_add(y, x), updates, is_leaf=_is_none)


def apply_scale(dist, scale=1.0):
    """Tree-map broadcasted scale handling None leaves."""
    def _apply_scale(leaf):
        if leaf is None:
            return None
        else:
            return leaf * scale

    def _is_none(x):
        return x is None

    return tree_map(_apply_scale, dist, is_leaf=_is_none)


def tree_marginalize(dist, weights: np.ndarray, dims: tuple, keepdims=False):
    """Marginalize PyTree over specified dimensions."""
    def apply_marginalization(leaf, reduce_dims=False):
        if leaf is None:
            return None
        else:
            return (leaf * weights).sum(dims, keepdims=reduce_dims)

    def _is_none(x):
        return x is None

    return tree_map(lambda x: apply_marginalization(x, reduce_dims=keepdims), dist, is_leaf=_is_none)


def map_and_multiply(a: ArrayDict, b: ArrayDict, sum_dim: int, mapping: dict = None):
    """Map and multiply with optional key mapping."""
    if mapping is not None:
        mapped_b = ArrayDict(**{a_key: b.get(mapping[a_key]) for a_key in a.keys() if mapping[a_key] in b.keys()})
    else:
        mapped_b = b

    def multiply_and_sum(x, y):
        return np.sum(x * y, axis=tuple(range(-sum_dim, 0)), keepdims=True)

    result = tree_map(multiply_and_sum, a, mapped_b)
    
    # Sum all elements
    leaves = tree_leaves(result)
    total = 0
    for leaf in leaves:
        if isinstance(leaf, np.ndarray):
            total = total + leaf
    return total


def params_to_tx(mapping):
    """Decorator to map natural parameters to sufficient statistics."""
    def decorator(cls):
        cls.params_to_tx = mapping
        return cls
    return decorator


def map_dict_names(params: ArrayDict, name_mapping: dict = None) -> ArrayDict:
    """Map ArrayDict keys to new names."""
    return ArrayDict(**{name_mapping[k]: v for k, v in params.items()})


def tree_equal(*pytrees, typematch: bool = False, rtol=0.0, atol=0.0):
    """Check PyTrees for equality."""
    if len(pytrees) < 2:
        return True
    
    def compare_trees(t1, t2):
        leaves1 = tree_leaves(t1)
        leaves2 = tree_leaves(t2)
        
        if len(leaves1) != len(leaves2):
            return False
        
        for l1, l2 in zip(leaves1, leaves2):
            if typematch and type(l1) != type(l2):
                return False
            
            if isinstance(l1, np.ndarray) and isinstance(l2, np.ndarray):
                if l1.shape != l2.shape:
                    return False
                if rtol == 0 and atol == 0:
                    if not np.all(l1 == l2):
                        return False
                else:
                    if not np.allclose(l1, l2, rtol=rtol, atol=atol):
                        return False
            else:
                if l1 != l2:
                    return False
        
        return True
    
    for t in pytrees[1:]:
        if not compare_trees(pytrees[0], t):
            return False
    
    return True


def is_array(element) -> bool:
    """Check if element is a numpy array."""
    return isinstance(element, np.ndarray)


def _array_equal(x, y, rtol, atol):
    assert x.dtype == y.dtype
    if (isinstance(rtol, (int, float)) and isinstance(atol, (int, float)) and rtol == 0 and atol == 0):
        return np.all(x == y)
    else:
        return np.allclose(x, y, rtol=rtol, atol=atol)


class _LeafWrapper:
    def __init__(self, value: Any):
        self.value = value


def _remove_leaf_wrapper(x: _LeafWrapper) -> Any:
    if not isinstance(x, _LeafWrapper):
        raise TypeError(f"Operation undefined, {x} is not a leaf of the pytree.")
    return x.value


def tree_at(
    where: Callable[[Any], Union[Any, Sequence[Any]]],
    pytree: Any,
    replace: Union[Any, Sequence[Any]] = None,
    replace_fn: Callable[[Any], Any] = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """Modify a leaf or subtree of a PyTree."""
    node_or_nodes_nowrapper = where(pytree)
    
    # Wrap pytree
    def wrap_tree(t):
        if is_leaf is not None and is_leaf(t):
            return _LeafWrapper(t)
        elif isinstance(t, ArrayDict):
            return ArrayDict(**{k: wrap_tree(v) for k, v in t.items()})
        elif isinstance(t, dict):
            return {k: wrap_tree(v) for k, v in t.items()}
        elif isinstance(t, (list, tuple)):
            return type(t)(wrap_tree(item) for item in t)
        else:
            return _LeafWrapper(t)
    
    pytree_wrapped = wrap_tree(pytree)
    node_or_nodes = where(pytree_wrapped)
    
    # Check structure matches
    leaves1 = tree_leaves(node_or_nodes_nowrapper)
    leaves2 = [x.value if isinstance(x, _LeafWrapper) else x for x in tree_leaves(node_or_nodes)]
    
    if len(leaves1) != len(leaves2) or any(l1 is not l2 for l1, l2 in zip(leaves1, leaves2)):
        raise ValueError("`where` must use just the PyTree structure of `pytree`.")
    
    # Determine if single or multiple nodes
    def is_in_tree(node, tree):
        for leaf in tree_leaves(tree):
            if leaf is node:
                return True
        return False
    
    if is_in_tree(node_or_nodes, pytree_wrapped):
        nodes = (node_or_nodes,)
        if replace is not None:
            replace = (replace,)
    else:
        nodes = node_or_nodes if isinstance(node_or_nodes, (list, tuple)) else (node_or_nodes,)
    
    if replace is None and replace_fn is None:
        raise ValueError("Precisely one of `replace` and `replace_fn` must be specified.")
    
    if replace is not None and replace_fn is not None:
        raise ValueError("Precisely one of `replace` and `replace_fn` must be specified.")
    
    # Create replacement functions
    if replace_fn is not None:
        replace_fns = [replace_fn] * len(nodes)
    else:
        if len(nodes) != len(replace):
            raise ValueError("`where` must return a sequence of leaves of the same length as `replace`.")
        replace_fns = [lambda _, r=r: r for r in replace]
    
    # Apply replacements
    def apply_replacement(t, target_nodes, node_fns):
        if isinstance(t, _LeafWrapper):
            for i, node in enumerate(target_nodes):
                if t is node:
                    return _LeafWrapper(node_fns[i](t.value))
            return t
        elif isinstance(t, ArrayDict):
            return ArrayDict(**{k: apply_replacement(v, target_nodes, node_fns) for k, v in t.items()})
        elif isinstance(t, dict):
            return {k: apply_replacement(v, target_nodes, node_fns) for k, v in t.items()}
        elif isinstance(t, (list, tuple)):
            return type(t)(apply_replacement(item, target_nodes, node_fns) for item in t)
        else:
            for i, node in enumerate(target_nodes):
                if t is node:
                    return node_fns[i](t)
            return t
    
    result = apply_replacement(pytree_wrapped, nodes, replace_fns)
    
    # Unwrap
    def unwrap_tree(t):
        if isinstance(t, _LeafWrapper):
            return t.value
        elif isinstance(t, ArrayDict):
            return ArrayDict(**{k: unwrap_tree(v) for k, v in t.items()})
        elif isinstance(t, dict):
            return {k: unwrap_tree(v) for k, v in t.items()}
        elif isinstance(t, (list, tuple)):
            return type(t)(unwrap_tree(item) for item in t)
        else:
            return t
    
    return unwrap_tree(result)
