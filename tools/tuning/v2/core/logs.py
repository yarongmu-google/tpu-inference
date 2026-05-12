# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Single-entry-point logging setup for the v2 tune stack.

Every CLI calls `configure()` at the top of its `main()`. Library
modules use `logging.getLogger(__name__)` and emit without
configuring — when called from a configured CLI they go through
the configured handler; when called from a test they go to the
default lastResort handler (WARNING+ to stderr).

Format example:

    14:23:45 INFO    tools.tuning.v2.kernel.tune: [tune    1] case=logical ...

Operator overrides:
  - `LOG_LEVEL=DEBUG` (or INFO/WARNING/ERROR) in env tightens or
    loosens the level without code changes.

Stdout vs stderr:
  - Diagnostic output (anything a human reads as "what's
    happening?") goes through this module → stderr.
  - Machine-parseable results (paths, env-var K=V lines, query
    matches) stay on `print(...)` → stdout, so shell wrappers
    can `eval $(...)` or pipe to grep without timestamp noise.
"""

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
_LOG_LEVEL_ENV = "LOG_LEVEL"

# Sentinel attribute we stamp on handlers we install, so repeated
# configure() calls don't pile up duplicates.
_V2_HANDLER_TAG = "_tuning_v2_handler"


def configure(level: int = logging.INFO) -> None:
    """Configure the root logger for v2 CLIs. Idempotent.

    Args:
      level: default log level. Overridden by `LOG_LEVEL` env var
             if present (case-insensitive: DEBUG, INFO, WARNING,
             ERROR, CRITICAL).
    """
    env_level = os.environ.get(_LOG_LEVEL_ENV)
    if env_level:
        env_level_int = logging.getLevelName(env_level.upper())
        if isinstance(env_level_int, int):
            level = env_level_int

    root = logging.getLogger()
    # Drop any prior v2 handler so the format / level stays in
    # sync across repeated calls (e.g. tests that re-configure).
    for h in list(root.handlers):
        if getattr(h, _V2_HANDLER_TAG, False):
            root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))
    setattr(handler, _V2_HANDLER_TAG, True)
    root.addHandler(handler)
    root.setLevel(level)
