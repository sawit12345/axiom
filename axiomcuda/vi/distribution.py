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

import inspect
from typing import Optional, Union, Tuple
import numpy as np

from axiomcuda.vi.utils import ArrayDict

# Import the C++ backend - this is the ONLY backend we use
try:
    import axiomcuda_backend as backend
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False
    raise RuntimeError(
        "AXIOMCUDA C++ backend not found. Please build the C++ extensions:\n"
        "  pip install -e .\n"
        "The C++ backend is required - there is no JAX fallback."
    )


class Distribution:
    """Base class for probability distributions - CUDA accelerated version."""

    dim: int
    event_dim: int
    batch_dim: int
    default_event_dim: int
    event_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    pytree_data_fields = ()
    pytree_aux_fields = (
        "dim",
        "default_event_dim",
        "batch_dim",
        "event_dim",
        "batch_shape",
        "event_shape",
    )

    def __init__(self, default_event_dim: int, batch_shape: tuple, event_shape: tuple):
        self.event_dim = len(event_shape)
        self.batch_dim = len(batch_shape)
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.default_event_dim = default_event_dim
        self.dim = self.event_shape[-default_event_dim] if len(self.event_shape) > 0 else 0

    @property
    def shape(self):
        return self.batch_shape + self.event_shape

    def to_event(self, n: int) -> "Distribution":
        """Converts the distribution to an event distribution."""
        if n > 0:
            event_shape = self.batch_shape[-n:] + self.event_shape
            batch_shape = self.batch_shape[:-n]
        else:
            event_shape = self.batch_shape + self.event_shape
            batch_shape = ()

        return self.__class__(self.default_event_dim, batch_shape, event_shape)

    def get_sample_dims(self, data: np.ndarray) -> list[int]:
        """Returns the sample dimensions of the data."""
        sample_shape = self.get_sample_shape(data)
        return list(range(len(sample_shape)))

    def get_sample_shape(self, data: np.ndarray) -> tuple[int, ...]:
        """Returns the sample shape of the data."""
        return data.shape[: -self.event_dim - self.batch_dim]

    def get_batch_shape(self, data: np.ndarray) -> tuple[int, ...]:
        """Returns the batch shape of the data."""
        return data.shape[-self.event_dim - self.batch_dim : -self.event_dim]

    def get_event_dims(self) -> list[int]:
        """Return the event dimensions of the array."""
        return list(range(-self.event_dim, 0))

    def sum_events(self, x: np.ndarray, keepdims: bool = False) -> np.ndarray:
        """Sums over the event dimensions of the array."""
        return x.sum(axis=tuple(range(-self.event_dim, 0)), keepdims=keepdims)

    def sum_default_events(self, x: np.ndarray, keepdims: bool = False) -> np.ndarray:
        """Sums over the default event dimensions of the array."""
        return x.sum(axis=tuple(range(-self.default_event_dim, 0)), keepdims=keepdims)

    def expand_event_dims(self, x: np.ndarray) -> np.ndarray:
        """Adds event dimensions to the array."""
        return x.reshape(x.shape + (1,) * self.event_dim)

    def expand_default_event_dims(self, x: np.ndarray) -> np.ndarray:
        """Adds event dimensions to the array."""
        return x.reshape(x.shape + (1,) * self.default_event_dim)

    def expand_batch_dims(self, x: np.ndarray) -> np.ndarray:
        """Adds batch dimensions to the array."""
        return x.reshape(x.shape[: -self.event_dim] + (1,) * self.batch_dim + x.shape[-self.event_dim :])

    def expand_batch_shape(self, batch_relative_axes: Union[int, Tuple[int]]):
        """Returns a new Distribution object with the expanded batch shape."""
        batch_relative_axes = (batch_relative_axes,) if isinstance(batch_relative_axes, int) else batch_relative_axes

        def expand_if_possible(x, batch_dim, event_dim, batch_ax, abs_ax):
            if isinstance(x, np.ndarray):
                if x.ndim == batch_dim + event_dim:
                    return np.expand_dims(x, abs_ax)
                if x.ndim == batch_dim:
                    return np.expand_dims(x, batch_ax)
            return x

        absolute_axes = tuple([ax - self.event_dim if ax < 0 else ax for ax in batch_relative_axes])

        data_fields, aux_fields = self.tree_flatten()
        expanded_data_fields = []
        for leaf in data_fields:
            if isinstance(leaf, Distribution):
                exp_leaf = leaf.expand_batch_shape(batch_relative_axes)
            elif isinstance(leaf, ArrayDict):
                exp_leaf = ArrayDict(**{
                    k: expand_if_possible(v, self.batch_dim, self.event_dim, batch_relative_axes, absolute_axes)
                    for k, v in leaf.items()
                })
            else:
                exp_leaf = expand_if_possible(leaf, self.batch_dim, self.event_dim, batch_relative_axes, absolute_axes)

            expanded_data_fields.append(exp_leaf)

        new_batch_dim = len(batch_relative_axes) + self.batch_dim
        adjusted_axes = tuple(sorted(set(batch_relative_axes)))

        shape_it = iter(self.batch_shape)
        new_batch_shape = [1 if ax in adjusted_axes else next(shape_it) for ax in range(new_batch_dim)]

        replace = {"batch_shape": tuple(new_batch_shape)}
        unsqueezed_dist = self.tree_unflatten_and_replace(aux_fields, expanded_data_fields, replace)

        return unsqueezed_dist

    def swap_axes(self, axis1: int, axis2: int):
        """Swaps the axes of the component distributions of a Pytree."""
        def swap_axes_of_leaf(x, ax1, ax2):
            if isinstance(x, np.ndarray):
                if x.ndim >= self.batch_dim + self.event_dim:
                    return np.swapaxes(x, ax1, ax2)
                elif x.ndim > 1 and x.ndim >= self.batch_dim:
                    ax1 = ax1 + self.event_dim if ax1 < 0 else ax1
                    ax2 = ax2 + self.event_dim if ax2 < 0 else ax2
                    return np.swapaxes(x, ax1, ax2)
                else:
                    return x
            return x

        data_fields, aux_fields = self.tree_flatten()
        from axiomcuda.vi.utils import tree_map
        data_fields = tree_map(lambda x: swap_axes_of_leaf(x, axis1, axis2), data_fields)

        axis1 = axis1 + self.event_dim if axis1 < 0 else axis1
        axis2 = axis2 + self.event_dim if axis2 < 0 else axis2

        batch_shape_list = list(self.batch_shape)
        batch_shape_list[axis1], batch_shape_list[axis2] = (
            batch_shape_list[axis2],
            batch_shape_list[axis1],
        )

        replace = {"batch_shape": tuple(batch_shape_list)}
        swapped_dist = self.tree_unflatten_and_replace(aux_fields, data_fields, replace)
        return swapped_dist

    def moveaxis(self, source: int, destination: int):
        """Moves one batch axis of the component distributions to another position."""
        absolute_src = source - self.event_dim if source < 0 else source
        absolute_dest = destination - self.event_dim if destination < 0 else destination

        def moveaxis_of_leaf(x, src_batch, dest_batch, src_abs, dest_abs):
            if isinstance(x, np.ndarray):
                if x.ndim == self.batch_dim + self.event_dim:
                    return np.moveaxis(x, src_abs, dest_abs)
                elif x.ndim == self.batch_dim:
                    return np.moveaxis(x, src_batch, dest_batch)
            return x

        data_fields, aux_fields = self.tree_flatten()
        from axiomcuda.vi.utils import tree_map
        data_fields = tree_map(
            lambda x: moveaxis_of_leaf(x, source, destination, absolute_src, absolute_dest), data_fields
        )

        batch_shape_list = [dim for n, dim in enumerate(self.batch_shape) if n != absolute_src]
        batch_shape_list.insert(absolute_dest, self.batch_shape[absolute_src])

        replace = {"batch_shape": tuple(batch_shape_list)}
        moved_dist = self.tree_unflatten_and_replace(aux_fields, data_fields, replace)
        return moved_dist

    def copy(self):
        """Returns a copy of the distribution."""
        from axiomcuda.vi.utils import tree_copy
        return tree_copy(self)

    def infer_shapes(self, tensor: np.ndarray, event_dim: int) -> tuple[tuple, tuple]:
        """Infers the batch and event shapes from the tensor."""
        batch_shape = tensor.shape[:-event_dim]
        event_shape = tensor.shape[-event_dim:]
        return batch_shape, event_shape

    @classmethod
    def gather_pytree_data_fields(cls):
        """Recursively gathers all pytree_data_fields from the class hierarchy."""
        bases = inspect.getmro(cls)
        all_pytree_data_fields = ()
        for base in bases:
            if issubclass(base, Distribution):
                all_pytree_data_fields += base.__dict__.get("pytree_data_fields", ())
        return tuple(set(all_pytree_data_fields))

    @classmethod
    def gather_pytree_aux_fields(cls):
        """Gather all pytree auxiliary fields from the base classes."""
        bases = inspect.getmro(cls)
        all_pytree_aux_fields = ()
        for base in bases:
            if issubclass(base, Distribution):
                all_pytree_aux_fields += base.__dict__.get("pytree_aux_fields", ())
        return tuple(set(all_pytree_aux_fields))

    def tree_flatten(self):
        """Flattens the distribution into a tuple of data and auxiliary values."""
        data_fields = type(self).gather_pytree_data_fields()
        aux_fields = type(self).gather_pytree_aux_fields()

        data_values = tuple(getattr(self, field) for field in data_fields)
        aux_values = tuple(getattr(self, field) for field in aux_fields)

        return data_values, aux_values

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        """Reconstructs an instance from its flattened representation."""
        instance = cls.__new__(cls)
        for k, v in zip(cls.gather_pytree_data_fields(), params):
            setattr(instance, k, v)
        for k, v in zip(cls.gather_pytree_aux_fields(), aux_data):
            if k not in ["event_dim", "batch_dim"]:
                setattr(instance, k, v)

        setattr(instance, "event_dim", len(instance.event_shape))
        setattr(instance, "batch_dim", len(instance.batch_shape))
        return instance

    @classmethod
    def tree_unflatten_and_replace(cls, aux_data, params, replace):
        """Reconstructs an instance with field replacement."""
        instance = cls.__new__(cls)
        for k, v in zip(cls.gather_pytree_data_fields(), params):
            setattr(instance, k, v)
        for k, v in zip(cls.gather_pytree_aux_fields(), aux_data):
            if k not in ["event_dim", "batch_dim"]:
                if k in replace:
                    setattr(instance, k, replace[k])
                else:
                    setattr(instance, k, v)

        setattr(instance, "event_dim", len(instance.event_shape))
        setattr(instance, "batch_dim", len(instance.batch_shape))
        return instance

    def __hash__(self):
        from axiomcuda.vi.utils import tree_leaves
        return hash(tuple(tree_leaves(self)))

    def __eq__(self, other):
        from axiomcuda.vi.utils import tree_equal
        return tree_equal(self, other)


DEFAULT_EVENT_DIM = 1


class Delta(Distribution):
    """Dirac delta distribution - CUDA accelerated."""

    pytree_data_fields = ("values", "residual")

    def __init__(self, values: np.ndarray, event_dim: Optional[int] = DEFAULT_EVENT_DIM):
        batch_shape, event_shape = values.shape[:-event_dim], values.shape[-event_dim:]
        super().__init__(DEFAULT_EVENT_DIM, batch_shape, event_shape)
        self.values = values
        self.residual = np.zeros(batch_shape)

    @property
    def p(self) -> np.ndarray:
        """Returns values."""
        return self.values

    @property
    def mean(self) -> np.ndarray:
        """Returns the mean parameter (alias for `p`)"""
        return self.p

    def expected_x(self):
        return self.p

    def expected_xx(self):
        return self.p @ self.p.T

    def log_partition(self):
        return np.zeros((1,) * len(self.event_shape))

    def entropy(self):
        return np.zeros((1,) * len(self.event_shape))

    def __mul__(self, other):
        """Overloads the * operator - Delta distributions cannot be multiplied."""
        if isinstance(other, self.__class__):
            raise ValueError(f"Cannot multiply two Delta messages!")
        return self.copy()
