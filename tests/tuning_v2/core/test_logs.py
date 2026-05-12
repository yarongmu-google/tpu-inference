# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.logs."""

import logging
import os
import unittest
from unittest import mock

from tools.tuning.v2.core.logs import (
    _LOG_LEVEL_ENV,
    _V2_HANDLER_TAG,
    configure,
)


class TestConfigure(unittest.TestCase):

    def setUp(self):
        # Snapshot root handlers + level so we can restore.
        self._saved_handlers = list(logging.getLogger().handlers)
        self._saved_level = logging.getLogger().level

    def tearDown(self):
        root = logging.getLogger()
        root.handlers = self._saved_handlers
        root.setLevel(self._saved_level)

    def test_installs_a_tagged_handler(self):
        configure()
        root = logging.getLogger()
        tagged = [
            h for h in root.handlers
            if getattr(h, _V2_HANDLER_TAG, False)
        ]
        self.assertEqual(len(tagged), 1)

    def test_idempotent_no_duplicate_handlers(self):
        """Repeated configure() calls (e.g. one CLI invoking another
        as a library) must not pile up handlers."""
        configure()
        configure()
        configure()
        root = logging.getLogger()
        tagged = [
            h for h in root.handlers
            if getattr(h, _V2_HANDLER_TAG, False)
        ]
        self.assertEqual(len(tagged), 1)

    def test_default_level_is_info(self):
        env = {k: v for k, v in os.environ.items()
               if k != _LOG_LEVEL_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            configure()
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_log_level_env_overrides(self):
        with mock.patch.dict(os.environ, {_LOG_LEVEL_ENV: "DEBUG"}):
            configure()
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_log_level_env_case_insensitive(self):
        with mock.patch.dict(os.environ, {_LOG_LEVEL_ENV: "warning"}):
            configure()
        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_configure_leaves_untagged_handlers_alone(self):
        """configure() may not delete handlers installed by other
        code paths (e.g. a parent app or pytest's CaptureHandler).
        Only handlers stamped with _V2_HANDLER_TAG go."""
        root = logging.getLogger()
        # Install a foreign untagged handler.
        other = logging.NullHandler()
        root.addHandler(other)
        try:
            configure()
            # configure() should NOT have removed the foreign handler.
            self.assertIn(other, root.handlers)
        finally:
            root.removeHandler(other)

    def test_log_level_env_invalid_falls_back_to_default(self):
        """An unrecognised level name doesn't crash; the explicit
        `level` arg passed to configure() stands."""
        with mock.patch.dict(os.environ, {_LOG_LEVEL_ENV: "NONSENSE"}):
            configure(level=logging.ERROR)
        # NONSENSE -> getLevelName returns the literal string back,
        # not an int, so we keep the explicit ERROR.
        self.assertEqual(logging.getLogger().level, logging.ERROR)


if __name__ == "__main__":
    unittest.main()
