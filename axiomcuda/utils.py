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
Utility functions for AxiomCUDA using C++ backend.

Provides tree operations, array handling, type conversions, and batch processing utilities.
"""

import numpy as np
from typing import Any, Callable, Sequence, Tuple, List, Dict, Union
from collections.abc import Mapping
from functools import partial
import axiomcuda_backend as backend


# Tree structure representation
class PyTreeDef:
    """Definition of a pytree structure.
    
    Captures the structure of a nested container (dicts, lists, tuples)
    without the actual values.
    """
    
    def __init__(self, node_type: str, node_data: Any, children: List['PyTreeDef']):
        self.node_type = node_type
        self.node_data = node_data
        self.children = children
    
    def __eq__(self, other):
        if not isinstance(other, PyTreeDef):
            return False
        return (self.node_type == other.node_type and 
                self.node_data == other.node_data and
                len(self.children) == len(other.children) and
                all(c1 == c2 for c1, c2 in zip(self.children, other.children)))
    
    def __repr__(self):
        return f"PyTreeDef({self.node_type}, {len(self.children)} children)"


class PyLeaf:
    """Represents a leaf node in a pytree."""
    
    def __init__(self, value: Any = None):
        self.value = value
    
    def __repr__(self):
        return "PyLeaf"
    
    def __eq__(self, other):
        return isinstance(other, PyLeaf)


# Tree structure functions

def _is_leaf(x: Any) -> bool:
    """Check if x is a leaf node (not a container)."""
    return not isinstance(x, (Mapping, list, tuple)) or x == tuple()


def _get_node_type(x: Any) -> Tuple[str, Any]:
    """Get the type of a node and its data."""
    if isinstance(x, Mapping):
        return ('dict', tuple(sorted(x.keys())))
    elif isinstance(x, tuple):
        return ('tuple', len(x))
    elif isinstance(x, list):
        return ('list', len(x))
    else:
        return ('leaf', None)


def tree_flatten(tree: Any) -> Tuple[List[Any], PyTreeDef]:
    """Flatten a pytree into a list of leaves and a tree definition.
    
    Args:
        tree: A nested structure (dicts, lists, tuples, or leaves)
        
    Returns:
        Tuple of (leaves list, tree_def)
        
    Example:
        >>> tree = {'a': 1, 'b': [2, 3]}
        >>> leaves, treedef = tree_flatten(tree)
        >>> leaves
        [1, 2, 3]
    """
    leaves = []
    
    def _flatten(x):
        if _is_leaf(x):
            leaves.append(x)
            return PyLeaf()
        
        node_type, node_data = _get_node_type(x)
        
        if isinstance(x, Mapping):
            children = [_flatten(x[k]) for k in node_data]
        elif isinstance(x, tuple):
            children = [_flatten(x[i]) for i in range(len(x))]
        elif isinstance(x, list):
            children = [_flatten(x[i]) for i in range(len(x))]
        else:
            children = []
        
        return PyTreeDef(node_type, node_data, children)
    
    treedef = _flatten(tree)
    return leaves, treedef


def tree_unflatten(treedef: PyTreeDef, leaves: Sequence[Any]) -> Any:
    """Reconstruct a pytree from a tree definition and leaves.
    
    Args:
        treedef: Tree structure definition
        leaves: Sequence of leaf values
        
    Returns:
        Reconstructed pytree
        
    Example:
        >>> tree = {'a': 1, 'b': [2, 3]}
        >>> leaves, treedef = tree_flatten(tree)
        >>> tree2 = tree_unflatten(treedef, leaves)
        >>> assert tree == tree2
    """
    leaf_iter = iter(leaves)
    
    def _unflatten(td: PyTreeDef):
        if isinstance(td, PyLeaf):
            return next(leaf_iter)
        
        if td.node_type == 'dict':
            return {k: _unflatten(c) for k, c in zip(td.node_data, td.children)}
        elif td.node_type == 'tuple':
            return tuple(_unflatten(c) for c in td.children)
        elif td.node_type == 'list':
            return [_unflatten(c) for c in td.children]
        else:
            return next(leaf_iter)
    
    return _unflatten(treedef)


def tree_structure(tree: Any) -> PyTreeDef:
    """Get the structure of a pytree without the values.
    
    Args:
        tree: A pytree
        
    Returns:
        PyTreeDef representing the structure
    """
    _, treedef = tree_flatten(tree)
    return treedef


def tree_map(f: Callable, tree: Any, *rest: Any) -> Any:
    """Apply a function to all leaves of a pytree.
    
    Args:
        f: Function to apply to each leaf
        tree: The pytree to map over
        *rest: Additional pytrees with the same structure
        
    Returns:
        New pytree with f applied to leaves
        
    Example:
        >>> tree = {'a': 1, 'b': 2}
        >>> tree_map(lambda x: x * 2, tree)
        {'a': 2, 'b': 4}
    """
    if len(rest) == 0:
        if _is_leaf(tree):
            return f(tree)
        
        node_type, node_data = _get_node_type(tree)
        
        if isinstance(tree, Mapping):
            return {k: tree_map(f, tree[k]) for k in node_data}
        elif isinstance(tree, tuple):
            return tuple(tree_map(f, tree[i]) for i in range(len(tree)))
        elif isinstance(tree, list):
            return [tree_map(f, tree[i]) for i in range(len(tree))]
        else:
            return f(tree)
    else:
        # Multiple trees - need to iterate together
        leaves1, treedef = tree_flatten(tree)
        leaves_rest = [tree_flatten(r)[0] for r in rest]
        
        new_leaves = []
        for i, leaf1 in enumerate(leaves1):
            leaf_args = [leaf1] + [lr[i] for lr in leaves_rest]
            new_leaves.append(f(*leaf_args))
        
        return tree_unflatten(treedef, new_leaves)


def tree_transpose(outer_treedef: PyTreeDef, inner_treedef: PyTreeDef, tree: Any) -> Any:
    """Transpose the inner and outer levels of a nested pytree.
    
    Args:
        outer_treedef: Tree definition for the outer level
        inner_treedef: Tree definition for the inner level
        tree: Nested pytree to transpose
        
    Returns:
        Transposed pytree
    """
    # Flatten to get outer structure
    outer_leaves, _ = tree_flatten(tree)
    
    # Each outer leaf should have the inner structure
    if len(outer_leaves) == 0:
        return tree_unflatten(inner_treedef, [])
    
    # Flatten an inner leaf to get inner leaves
    inner_leaves, _ = tree_flatten(outer_leaves[0])
    n_inner = len(inner_leaves)
    n_outer = len(outer_leaves)
    
    # Transpose: create n_inner lists of n_outer elements
    transposed = []
    for j in range(n_inner):
        inner_list = []
        for i in range(n_outer):
            inner_leaves_i, _ = tree_flatten(outer_leaves[i])
            inner_list.append(inner_leaves_i[j])
        transposed.append(tree_unflatten(outer_treedef, inner_list))
    
    return tree_unflatten(inner_treedef, transposed)


def tree_reduce(f: Callable, tree: Any) -> Any:
    """Reduce a pytree to a single value.
    
    Args:
        f: Binary reduction function
        tree: The pytree to reduce
        
    Returns:
        Single reduced value
    """
    leaves, _ = tree_flatten(tree)
    
    if len(leaves) == 0:
        raise ValueError("Cannot reduce empty tree")
    
    result = leaves[0]
    for leaf in leaves[1:]:
        result = f(result, leaf)
    
    return result


def tree_all(tree: Any, predicate: Callable = bool) -> bool:
    """Check if all leaves satisfy a predicate.
    
    Args:
        tree: The pytree to check
        predicate: Function to apply to each leaf (default: bool)
        
    Returns:
        True if all leaves satisfy predicate
    """
    leaves, _ = tree_flatten(tree)
    return all(predicate(leaf) for leaf in leaves)


def tree_any(tree: Any, predicate: Callable = bool) -> bool:
    """Check if any leaf satisfies a predicate.
    
    Args:
        tree: The pytree to check
        predicate: Function to apply to each leaf (default: bool)
        
    Returns:
        True if any leaf satisfies predicate
    """
    leaves, _ = tree_flatten(tree)
    return any(predicate(leaf) for leaf in leaves)


# Array handling utilities

def to_numpy(tensor) -> np.ndarray:
    """Convert a tensor to numpy array.
    
    Args:
        tensor: Tensor or numpy array
        
    Returns:
        Numpy array (on CPU)
    """
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.array(tensor)


def from_numpy(array: np.ndarray, device=None):
    """Convert a numpy array to a tensor.
    
    Args:
        array: Numpy array
        device: Target device (None for default)
        
    Returns:
        Tensor on specified device
    """
    from .tensor import Tensor
    from .device import get_device
    
    tensor = Tensor(array)
    if device is not None:
        dev = get_device(device)
        tensor = tensor.to(dev)
    return tensor


def ensure_array(x, dtype=None) -> np.ndarray:
    """Ensure x is a numpy array.
    
    Args:
        x: Input (scalar, list, or array)
        dtype: Target dtype (optional)
        
    Returns:
        Numpy array
    """
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.array(x)
    
    if dtype is not None:
        arr = arr.astype(dtype)
    
    return arr


def broadcast_arrays(*arrays) -> List[np.ndarray]:
    """Broadcast arrays to compatible shapes.
    
    Args:
        *arrays: Arrays to broadcast
        
    Returns:
        List of broadcasted arrays
    """
    arrays = [ensure_array(a) for a in arrays]
    
    # Find the maximum shape
    max_ndim = max(a.ndim for a in arrays)
    
    # Pad shapes to match
    padded_shapes = []
    for a in arrays:
        shape = [1] * (max_ndim - a.ndim) + list(a.shape)
        padded_shapes.append(shape)
    
    # Compute broadcasted shape
    result_shape = []
    for dims in zip(*padded_shapes):
        max_dim = max(d for d in dims if d != 1)
        if any(d != 1 and d != max_dim for d in dims):
            raise ValueError(f"Cannot broadcast shapes: {[a.shape for a in arrays]}")
        result_shape.append(max_dim)
    
    # Broadcast each array
    result = []
    for a in arrays:
        # Use numpy's broadcast_to
        broadcasted = np.broadcast_to(a, result_shape)
        result.append(broadcasted)
    
    return result


# Batch handling utilities

def batch_apply(func: Callable, inputs: Any, batch_size: int = 32) -> Any:
    """Apply a function in batches over the first dimension.
    
    Args:
        func: Function to apply to each batch
        inputs: Input data (tensor or pytree of tensors)
        batch_size: Size of each batch
        
    Returns:
        Concatenated results
    """
    # Flatten to get the first tensor and determine total size
    leaves, treedef = tree_flatten(inputs)
    if len(leaves) == 0:
        return inputs
    
    total_size = leaves[0].shape[0]
    n_batches = (total_size + batch_size - 1) // batch_size
    
    results = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, total_size)
        
        # Slice each leaf
        batch_leaves = []
        for leaf in leaves:
            if hasattr(leaf, 'shape') and len(leaf.shape) > 0:
                batch_leaves.append(leaf[start:end])
            else:
                batch_leaves.append(leaf)
        
        batch_input = tree_unflatten(treedef, batch_leaves)
        batch_result = func(batch_input)
        results.append(batch_result)
    
    # Concatenate results
    result_leaves, result_treedef = tree_flatten(results[0])
    concat_leaves = []
    
    for i in range(len(result_leaves)):
        to_concat = []
        for r in results:
            rl, _ = tree_flatten(r)
            to_concat.append(rl[i])
        
        if hasattr(to_concat[0], 'shape') and len(to_concat[0].shape) > 0:
            concat_leaves.append(np.concatenate(to_concat, axis=0))
        else:
            concat_leaves.append(to_concat[0])
    
    return tree_unflatten(result_treedef, concat_leaves)


def vmap(func: Callable, in_axes=0, out_axes=0):
    """Vectorizing map (similar to jax.vmap).
    
    Creates a function that maps func over array axes.
    
    Args:
        func: Function to vectorize
        in_axes: Which axes to map over for inputs
        out_axes: Which axes to map over for outputs
        
    Returns:
        Vectorized function
    """
    def vectorized_func(*args):
        # Get the size of the mapped axis
        if isinstance(in_axes, int):
            n = args[0].shape[in_axes]
            axes = [in_axes] * len(args)
        else:
            n = args[0].shape[in_axes[0]]
            axes = in_axes
        
        results = []
        for i in range(n):
            # Slice each argument
            sliced_args = []
            for arg, axis in zip(args, axes):
                if axis is not None:
                    idx = [slice(None)] * arg.ndim
                    idx[axis] = i
                    sliced_args.append(arg[tuple(idx)])
                else:
                    sliced_args.append(arg)
            
            result = func(*sliced_args)
            results.append(result)
        
        # Stack results
        if isinstance(results[0], tuple):
            # Multiple outputs
            n_out = len(results[0])
            stacked = []
            for j in range(n_out):
                to_stack = [r[j] for r in results]
                stacked.append(np.stack(to_stack, axis=out_axes))
            return tuple(stacked)
        else:
            return np.stack(results, axis=out_axes)
    
    return vectorized_func


def pytree_vmap(func: Callable, in_axes=0, out_axes=0):
    """Vectorizing map for pytrees.
    
    Like vmap but works with nested structures.
    
    Args:
        func: Function to vectorize
        in_axes: Which axes to map over
        out_axes: Which axes to stack outputs
        
    Returns:
        Vectorized function for pytrees
    """
    def vectorized_func(*args):
        # Flatten all arguments
        flat_args = [tree_flatten(a)[0] for a in args]
        treedefs = [tree_flatten(a)[1] for a in args]
        
        # Get number of elements (from first non-empty tree)
        n = None
        for fa in flat_args:
            if len(fa) > 0:
                n = fa[0].shape[0] if len(fa[0].shape) > 0 else 1
                break
        
        if n is None:
            return func(*args)
        
        results = []
        for i in range(n):
            # Slice each leaf
            sliced_trees = []
            for fa, td in zip(flat_args, treedefs):
                sliced_leaves = []
                for leaf in fa:
                    if len(leaf.shape) > 0:
                        sliced_leaves.append(leaf[i:i+1])
                    else:
                        sliced_leaves.append(leaf)
                sliced_trees.append(tree_unflatten(td, sliced_leaves))
            
            result = func(*sliced_trees)
            results.append(result)
        
        # Concatenate results
        if len(results) == 0:
            return results
        
        result_leaves, result_treedef = tree_flatten(results[0])
        concat_leaves = []
        
        for i in range(len(result_leaves)):
            to_concat = []
            for r in results:
                rl, _ = tree_flatten(r)
                to_concat.append(rl[i])
            
            concat_leaves.append(np.concatenate(to_concat, axis=0))
        
        return tree_unflatten(result_treedef, concat_leaves)
    
    return vectorized_func


# Type checking utilities

def is_array_like(x) -> bool:
    """Check if x is array-like."""
    return isinstance(x, (np.ndarray, list, tuple)) or hasattr(x, '__array__')


def is_scalar(x) -> bool:
    """Check if x is a scalar."""
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)


def get_dtype(x):
    """Get the dtype of x."""
    if hasattr(x, 'dtype'):
        return x.dtype
    elif isinstance(x, (int, np.integer)):
        return np.int64
    elif isinstance(x, (float, np.floating)):
        return np.float64
    elif isinstance(x, (complex, np.complexfloating)):
        return np.complex128
    elif isinstance(x, bool):
        return np.bool_
    else:
        return None


def promote_types(*args):
    """Promote types to a common dtype."""
    dtypes = [get_dtype(a) for a in args if get_dtype(a) is not None]
    if len(dtypes) == 0:
        return np.float64
    return np.result_type(*dtypes)
