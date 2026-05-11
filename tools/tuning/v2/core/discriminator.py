# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Canonical defaults for the row-level discriminator fields.

Three fields tag every kernel-tune row with plugin / hardware / row-
schema identity (architecture doc §13.4):

  kernel_variant: which kernel implementation produced this row.
  hardware:       which hardware partition the run targets.
  schema_version: the row schema version, distinct from the
                  production-envelope schema_version in
                  `core/accumulator.py` and `cli/migrate.py`. Same
                  name, different schema — see comment there.

This module is the single source of truth for the default values.
Previously the literals "rpa_v3" / "tpu_v7x" lived in three places
(enumerate_logical, service/sweep, cli/lookup) — a TPU-gen bump or
plugin rename would have to chase all three down. Importing here
makes the next rename a one-file diff.

KNOWN_KERNEL_VARIANTS is the gate both `service/sweep` (at pin-keys
resolution time) and `cli/lookup` (at deploy-env emission time) use
to fail loud when a row carries a kernel_variant we don't know how
to emit env vars for. The gate stays here so the kernel↔service
asymmetry from earlier review passes doesn't re-emerge — see
architecture doc §13.4.1 "Symmetric stamping discipline."
"""

DEFAULT_KERNEL_VARIANT = "rpa_v3"
DEFAULT_HARDWARE = "tpu_v7x"
ROW_SCHEMA_VERSION = 1

# Plugins this codebase knows how to dispatch on. Adding a new TPU
# kernel variant (e.g. "rpa_v3_hd64", "mla") or a cross-HW plugin
# ("flash_attn_cuda") extends this — and requires a matching emit
# branch in cli/lookup.lookup_env. The gate fires in two places:
#
#   1. service/sweep.resolve_kernel_pin_keys: catches a foreign
#      .kernel file at sweep startup, BEFORE the sweep records
#      thousands of FAILED rows tagged with the wrong variant.
#   2. cli/lookup.lookup_env: catches the same condition at deploy
#      time as a defense-in-depth backstop.
KNOWN_KERNEL_VARIANTS = frozenset({"rpa_v3"})
