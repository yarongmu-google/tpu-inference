# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared math helpers for Pallas TPU kernels."""

import jax
import jax.numpy as jnp

LOG2_E = 1.4426950408889634  # log2(e)


def exp(x: jax.Array) -> jax.Array:
    """exp(x) as exp2(x * log2(e)).

    The TPU's transcendental unit exposes exp2, not exp; jnp.exp2 lowers
    to it directly, so the base conversion is one explicit elementwise
    multiply instead of a backend rewrite of math.exp we cannot inspect.
    """
    return jnp.exp2(x * LOG2_E)
