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
Utility functions for CUDA-accelerated variational inference.

This module provides helper functions for:
- PyTree manipulation (ArrayDict, tree operations)
- Mathematical operations (batched linear algebra, special functions)
- Input processing and validation
- Mapping utilities

All operations are designed to work efficiently with JAX and can be
JIT-compiled for optimal performance on both CPU and GPU.
"""

from .pytree import (
    ArrayDict,
    apply_add,
    apply_scale,
    params_to_tx,
    map_and_multiply,
    zeros_like,
    map_dict_names,
    size,
    tree_copy,
    sum_pytrees,
    tree_marginalize,
    tree_equal,
    tree_at,
)

from .math import (
    mvgammaln,
    mvdigamma,
    stable_logsumexp,
    stable_softmax,
    assign_unused,
    bdot,
    symmetrise,
    positive_leading_eigenvalues,
    make_posdef,
    inv_and_logdet,
)
from .input_handling import check_optim_args, get_default_args
from .maps import to_list

__all__ = [
    # PyTree utilities
    "ArrayDict",
    "apply_add",
    "apply_scale",
    "params_to_tx",
    "map_and_multiply",
    "zeros_like",
    "map_dict_names",
    "size",
    "tree_copy",
    "sum_pytrees",
    "tree_marginalize",
    "tree_equal",
    "tree_at",
    # Math utilities
    "mvgammaln",
    "mvdigamma",
    "stable_logsumexp",
    "stable_softmax",
    "assign_unused",
    "bdot",
    "symmetrise",
    "positive_leading_eigenvalues",
    "make_posdef",
    "inv_and_logdet",
    # Input handling
    "check_optim_args",
    "get_default_args",
    # Maps
    "to_list",
]
