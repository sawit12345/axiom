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

from collections.abc import Callable, Sequence
from typing import Any, Optional, TYPE_CHECKING, Union

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree, PyTreeDef
from typing import Tuple
from jax.tree_util import register_pytree_node_class
import numpy as np
import jax


@register_pytree_node_class
class ArrayDict:
    """Immutable dictionary-like PyTree node for array storage."""
    
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

    def tree_flatten(self):
        values = []
        keys = []
        for key, value in sorted(self._fields.items()):
            values.append(value)
            keys.append(key)
        return values, keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(**dict(zip(keys, values)))


def size(array_dict: PyTree) -> int:
    """Count total size of arrays in PyTree."""
    total_size = 0
    for param in jtu.tree_leaves(array_dict):
        if isinstance(param, jnp.ndarray):
            total_size = total_size + param.size
    return total_size


def sum_pytrees(*pytrees):
    """Sum multiple PyTrees element-wise."""
    return jtu.tree_map(lambda *args: sum(args), *pytrees)


def zeros_like(array_dict: ArrayDict) -> ArrayDict:
    """Create ArrayDict with zeros matching input structure."""
    def zeros_for_value(value):
        if isinstance(value, jnp.ndarray):
            return jnp.zeros(value.shape, value.dtype)
        elif isinstance(value, ArrayDict):
            return zeros_like(value)
        elif isinstance(value, dict):
            return {k: zeros_for_value(v) for k, v in value.items()}
        else:
            return value

    return ArrayDict(**{key: zeros_for_value(val) for key, val in array_dict.items()})


def tree_copy(tree: PyTree) -> PyTree:
    """Copy a PyTree (re-references leaves)."""
    def copy(x):
        return x
    return jtu.tree_map(copy, tree)


def apply_add(dist: PyTree, updates: PyTree) -> PyTree:
    """Tree-map broadcasted addition handling None leaves."""
    def _apply_add(u, p):
        if u is None:
            return p
        else:
            return p + u

    def _is_none(x):
        return x is None

    return jtu.tree_map(_apply_add, updates, dist, is_leaf=_is_none)


def apply_scale(dist: PyTree, scale=1.0) -> PyTree:
    """Tree-map broadcasted scale handling None leaves."""
    def _apply_scale(leaf):
        if leaf is None:
            return None
        else:
            return leaf * scale

    def _is_none(x):
        return x is None

    return jtu.tree_map(_apply_scale, dist, is_leaf=_is_none)


def tree_marginalize(dist: PyTree, weights: Array, dims: Tuple[int], keepdims=False) -> PyTree:
    """Marginalize PyTree over specified dimensions."""
    def apply_marginalization(leaf, reduce_dims=False):
        if leaf is None:
            return None
        else:
            return (leaf * weights).sum(dims, keepdims=reduce_dims)

    def _is_none(x):
        return x is None

    return jtu.tree_map(lambda x: apply_marginalization(x, reduce_dims=keepdims), dist, is_leaf=_is_none)


def map_and_multiply(a: ArrayDict, b: ArrayDict, sum_dim: int, mapping: dict = None):
    """Map and multiply with optional key mapping."""
    if mapping is not None:
        mapped_b = ArrayDict(**{a_key: b.get(mapping[a_key]) for a_key in a.keys() if mapping[a_key] in b.keys()})
    else:
        mapped_b = b

    def multiply_and_sum(x, y):
        return jnp.sum(x * y, axis=range(-sum_dim, 0), keepdims=True)

    result = jtu.tree_map(multiply_and_sum, a, mapped_b)
    return jtu.tree_reduce(lambda x, y: x + y, result)


def params_to_tx(mapping):
    """Decorator to map natural parameters to sufficient statistics."""
    def decorator(cls):
        cls.params_to_tx = mapping
        return cls
    return decorator


def map_dict_names(params: ArrayDict, name_mapping: dict = None) -> ArrayDict:
    """Map ArrayDict keys to new names."""
    return ArrayDict(**{name_mapping[k]: v for k, v in params.items()})


def tree_equal(*pytrees: PyTree, typematch: bool = False, rtol=0.0, atol=0.0):
    """Check PyTrees for equality."""
    flat, treedef = jtu.tree_flatten(pytrees[0])
    traced_out = True
    for pytree in pytrees[1:]:
        flat_, treedef_ = jtu.tree_flatten(pytree)
        if treedef_ != treedef:
            return False
        assert len(flat) == len(flat_)
        for elem, elem_ in zip(flat, flat_):
            if typematch:
                if not isinstance(elem, type(elem_)):
                    return False
            if isinstance(elem, (np.ndarray, np.generic)) and isinstance(elem_, (np.ndarray, np.generic)):
                if (elem.shape != elem_.shape) or (elem.dtype != elem_.dtype) or not _array_equal(elem, elem_, rtol, atol):
                    return False
            elif is_array(elem):
                if is_array(elem_):
                    if (elem.shape != elem_.shape) or (elem.dtype != elem_.dtype):
                        return False
                    traced_out = traced_out & _array_equal(elem, elem_, rtol, atol)
                else:
                    return False
            else:
                if is_array(elem_):
                    return False
                else:
                    if elem != elem_:
                        return False
    return traced_out


def is_array(element) -> bool:
    """Check if element is a JAX or NumPy array."""
    return isinstance(element, (np.ndarray, np.generic, jax.Array))


def _array_equal(x, y, rtol, atol):
    assert x.dtype == y.dtype
    npi = jnp if isinstance(x, Array) or isinstance(y, Array) else np
    if (isinstance(rtol, (int, float)) and isinstance(atol, (int, float)) and rtol == 0 and atol == 0) or not npi.issubdtype(x.dtype, npi.inexact):
        return npi.all(x == y)
    else:
        return npi.allclose(x, y, rtol=rtol, atol=atol)


class _LeafWrapper:
    def __init__(self, value: Any):
        self.value = value


def _remove_leaf_wrapper(x: _LeafWrapper) -> Any:
    if not isinstance(x, _LeafWrapper):
        raise TypeError(f"Operation undefined, {x} is not a leaf of the pytree.")
    return x.value


class _CountedIdDict:
    def __init__(self, keys, values):
        assert len(keys) == len(values)
        self._dict = {id(k): v for k, v in zip(keys, values)}
        self._count = {id(k): 0 for k in keys}

    def __contains__(self, item):
        return id(item) in self._dict

    def __getitem__(self, item):
        self._count[id(item)] += 1
        return self._dict[id(item)]

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default

    def count(self, item):
        return self._count[id(item)]


def tree_at(
    where: Callable[[PyTree], Union[Any, Sequence[Any]]],
    pytree: PyTree,
    replace: Union[Any, Sequence[Any]] = None,
    replace_fn: Callable[[Any], Any] = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """Modify a leaf or subtree of a PyTree (like .at[].set() for JAX arrays)."""
    node_or_nodes_nowrapper = where(pytree)
    pytree = jtu.tree_map(_LeafWrapper, pytree, is_leaf=is_leaf)
    node_or_nodes = where(pytree)
    leaves1, structure1 = jtu.tree_flatten(node_or_nodes_nowrapper, is_leaf=is_leaf)
    leaves2, structure2 = jtu.tree_flatten(node_or_nodes)
    leaves2 = [_remove_leaf_wrapper(x) for x in leaves2]
    if (structure1 != structure2 or len(leaves1) != len(leaves2) or any(l1 is not l2 for l1, l2 in zip(leaves1, leaves2))):
        raise ValueError("`where` must use just the PyTree structure of `pytree`.")
    del node_or_nodes_nowrapper, leaves1, structure1, leaves2, structure2

    in_pytree = False
    def _in_pytree(x):
        nonlocal in_pytree
        if x is node_or_nodes:
            in_pytree = True
        return x

    jtu.tree_map(_in_pytree, pytree, is_leaf=lambda x: x is node_or_nodes)
    if in_pytree:
        nodes = (node_or_nodes,)
        if replace is not None:
            replace = (replace,)
    else:
        nodes = node_or_nodes
    del in_pytree, node_or_nodes

    if replace is None:
        if replace_fn is None:
            raise ValueError("Precisely one of `replace` and `replace_fn` must be specified.")
        else:
            def _replace_fn(x):
                x = jtu.tree_map(_remove_leaf_wrapper, x)
                return replace_fn(x)
            replace_fns = [_replace_fn] * len(nodes)
    else:
        if replace_fn is None:
            if len(nodes) != len(replace):
                raise ValueError("`where` must return a sequence of leaves of the same length as `replace`.")
            replace_fns = [lambda _, r=r: r for r in replace]
        else:
            raise ValueError("Precisely one of `replace` and `replace_fn` must be specified.")
    node_replace_fns = _CountedIdDict(nodes, replace_fns)

    def _make_replacement(x: Any) -> Any:
        return node_replace_fns.get(x, _remove_leaf_wrapper)(x)

    out = jtu.tree_map(_make_replacement, pytree, is_leaf=lambda x: x in node_replace_fns)

    for node in nodes:
        count = node_replace_fns.count(node)
        if count == 0:
            raise ValueError("`where` does not specify an element or elements of `pytree`.")
        elif count == 1:
            pass
        else:
            raise ValueError("`where` does not uniquely identify a single element of `pytree`.")

    return out
