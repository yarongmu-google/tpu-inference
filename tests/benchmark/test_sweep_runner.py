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
"""Unit tests for the runner / git / CLI parts of tools.benchmark.sweep.

All subprocess and filesystem effects are stubbed via dependency injection
or tempdirs — no vLLM, no real git pushes.

Coverage target: 100% of the runner / CLI code added in tools/benchmark/sweep.py.
"""

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.benchmark import sweep


def _make_spec(case_file="case.env", sweep_name="s",
               sweep_axes=None, coupled_axes=None, fixed=None):
    return {
        "case_file": case_file,
        "sweep_name": sweep_name,
        "sweep_axes": sweep_axes or {},
        "coupled_axes": coupled_axes or [],
        "fixed": fixed or {},
    }


class FakeProc:
    """Drop-in stand-in for subprocess.CompletedProcess in injected tests."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestRunOne(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.spec = _make_spec()
        self.combo = {"A": "1"}
        # Lock down the env so dict ordering / interactions are predictable.
        self.fake_environ = {"PATH": "/usr/bin"}

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_subprocess_writing_metrics(self, returncode=0):
        """Returns a stub for subprocess.run that writes metrics.txt into
        whatever RESULT_DIR env var the caller sets, then returns FakeProc."""

        def fake_run(cmd, env=None, check=False, **kw):
            rdir = Path(env["RESULT_DIR"])
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "metrics.txt").write_text("RequestThroughput=1.23\n")
            return FakeProc(returncode=returncode)

        return fake_run

    def test_skipped_when_already_complete(self):
        cid = sweep.combo_id(self.combo)
        rdir = sweep.result_dir(self.base, "case", "s", cid)
        rdir.mkdir(parents=True)
        # is_completed now requires a non-empty RequestThroughput.
        (rdir / "metrics.txt").write_text("RequestThroughput=8.30\n")

        # Subprocess MUST NOT be called.
        sentinel = MagicMock(side_effect=AssertionError("should not run"))
        result = sweep.run_one(self.spec, self.combo,
                               base_dir=self.base,
                               run_subprocess=sentinel,
                               environ=self.fake_environ)
        self.assertEqual(result.status, sweep.RunStatus.SKIPPED_RESUMED)
        self.assertEqual(result.combo_id, cid)
        self.assertEqual(result.result_dir, rdir)
        sentinel.assert_not_called()

    def test_completed_fresh_writes_metrics(self):
        timer = iter([100.0, 105.5])
        result = sweep.run_one(
            self.spec, self.combo,
            base_dir=self.base,
            run_subprocess=self._fake_subprocess_writing_metrics(),
            timer=lambda: next(timer),
            environ=self.fake_environ,
        )
        self.assertEqual(result.status, sweep.RunStatus.COMPLETED_FRESH)
        self.assertEqual(result.return_code, 0)
        self.assertAlmostEqual(result.duration_seconds, 5.5, places=3)
        self.assertTrue(sweep.is_completed(result.result_dir))

    def test_failed_when_returncode_nonzero(self):
        result = sweep.run_one(
            self.spec, self.combo,
            base_dir=self.base,
            run_subprocess=self._fake_subprocess_writing_metrics(returncode=2),
            environ=self.fake_environ,
        )
        self.assertEqual(result.status, sweep.RunStatus.FAILED)
        self.assertEqual(result.return_code, 2)

    def test_failed_when_subprocess_raises(self):
        def boom(cmd, env=None, check=False, **kw):
            raise OSError("ENOENT")

        result = sweep.run_one(
            self.spec, self.combo,
            base_dir=self.base,
            run_subprocess=boom,
            environ=self.fake_environ,
        )
        self.assertEqual(result.status, sweep.RunStatus.FAILED)
        self.assertIn("ENOENT", result.error)
        self.assertIsNone(result.return_code)

    def test_failed_when_subprocess_times_out(self):
        def hang(cmd, env=None, check=False, timeout=None, **kw):
            raise subprocess.TimeoutExpired(cmd, timeout)

        result = sweep.run_one(
            self.spec, self.combo,
            base_dir=self.base,
            run_subprocess=hang,
            environ=self.fake_environ,
        )
        self.assertEqual(result.status, sweep.RunStatus.FAILED)
        self.assertIn("timed out", result.error)
        # Default timeout from DEFAULT_TIMEOUT_SECONDS.
        self.assertIn(str(sweep.DEFAULT_TIMEOUT_SECONDS), result.error)

    def test_timeout_overridable_via_spec(self):
        captured = {}

        def fake_run(cmd, env=None, check=False, timeout=None, **kw):
            captured["timeout"] = timeout
            (Path(env["RESULT_DIR"]) / "metrics.txt").write_text(
                "RequestThroughput=1.0\n")
            return FakeProc(returncode=0)

        spec = _make_spec()
        spec["timeout_seconds"] = 7
        sweep.run_one(spec, self.combo, base_dir=self.base,
                      run_subprocess=fake_run, environ=self.fake_environ)
        self.assertEqual(captured["timeout"], 7)

    def test_failed_when_metrics_missing_after_zero_exit(self):
        def fake_run(cmd, env=None, check=False, **kw):
            # Don't write metrics.txt. Pretend success.
            Path(env["RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
            return FakeProc(returncode=0)

        result = sweep.run_one(
            self.spec, self.combo, base_dir=self.base,
            run_subprocess=fake_run, environ=self.fake_environ,
        )
        self.assertEqual(result.status, sweep.RunStatus.FAILED)
        # Error message changed when is_completed grew a content check
        # — covers both 'file missing' and 'file present but blank'.
        self.assertIn("missing or has no RequestThroughput", result.error)

    def test_env_overrides_passed_to_subprocess(self):
        captured = {}

        def fake_run(cmd, env=None, check=False, **kw):
            captured["cmd"] = cmd
            captured["env"] = dict(env)
            Path(env["RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
            (Path(env["RESULT_DIR"]) / "metrics.txt").write_text("RequestThroughput=8.30\n")
            return FakeProc(returncode=0)

        spec = _make_spec(case_file="cases/foo.env")
        combo = {"MAX_NUM_BATCHED_TOKENS": "4096", "BLOCK_SIZE": "128"}
        sweep.run_one(spec, combo, base_dir=self.base,
                      run_subprocess=fake_run,
                      environ={"PATH": "/usr/bin", "OTHER": "keep"})

        # Combo merged in.
        self.assertEqual(captured["env"]["MAX_NUM_BATCHED_TOKENS"], "4096")
        self.assertEqual(captured["env"]["BLOCK_SIZE"], "128")
        # Caller env preserved.
        self.assertEqual(captured["env"]["PATH"], "/usr/bin")
        self.assertEqual(captured["env"]["OTHER"], "keep")
        # RESULT_DIR points inside our base.
        self.assertTrue(captured["env"]["RESULT_DIR"].startswith(str(self.base)))
        # Cmd is [script, case_file].
        self.assertEqual(captured["cmd"][1], "cases/foo.env")

    def test_default_environ_uses_os_environ_when_none(self):
        # When environ=None, the function reads os.environ; verify by setting
        # a unique key and observing it shows up in the env dict.
        captured = {}

        def fake_run(cmd, env=None, check=False, **kw):
            captured["env"] = dict(env)
            Path(env["RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
            (Path(env["RESULT_DIR"]) / "metrics.txt").write_text("RequestThroughput=8.30\n")
            return FakeProc(returncode=0)

        with patch.dict(os.environ, {"MY_TEST_KEY": "MY_TEST_VAL"}, clear=False):
            sweep.run_one(self.spec, self.combo,
                          base_dir=self.base,
                          run_subprocess=fake_run)
        self.assertEqual(captured["env"]["MY_TEST_KEY"], "MY_TEST_VAL")


class TestRunSweep(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.spec = _make_spec(sweep_axes={"A": [1, 2]},
                               coupled_axes=[{"X": 1}, {"X": 2}])
        # Write to a temp JSON path
        self.spec_path = Path(self.tmp.name) / "spec.json"
        self.spec_path.write_text(json.dumps(self.spec))

    def tearDown(self):
        self.tmp.cleanup()

    def test_iterates_all_combos_calls_callback(self):
        seen = []

        def fake_run_one(spec, combo, **kw):
            return sweep.RunResult(
                status=sweep.RunStatus.COMPLETED_FRESH,
                combo_id=sweep.combo_id(combo),
                result_dir=Path(self.base) / "x",
                return_code=0,
                duration_seconds=0.1,
            )

        def cb(result, idx, total):
            seen.append((idx, total, result.status))

        results = sweep.run_sweep(
            str(self.spec_path), base_dir=self.base,
            run_one_fn=fake_run_one, on_result=cb)
        self.assertEqual(len(results), 4)  # 2x2 cartesian
        self.assertEqual([s[0] for s in seen], [0, 1, 2, 3])
        self.assertEqual({s[1] for s in seen}, {4})
        self.assertTrue(all(s[2] == sweep.RunStatus.COMPLETED_FRESH for s in seen))

    def test_continues_on_failure(self):
        order = []

        def fake_run_one(spec, combo, **kw):
            order.append(combo)
            status = (sweep.RunStatus.FAILED if combo["X"] == "1"
                      else sweep.RunStatus.COMPLETED_FRESH)
            return sweep.RunResult(
                status=status,
                combo_id="x", result_dir=Path("/tmp/x"))

        results = sweep.run_sweep(str(self.spec_path), base_dir=self.base,
                                  run_one_fn=fake_run_one)
        self.assertEqual(len(results), 4)
        self.assertEqual(len(order), 4)
        # Half failed, half ok
        self.assertEqual(
            sum(r.status is sweep.RunStatus.FAILED for r in results), 2)

    def test_no_callback_works(self):
        # Verifies the on_result=None branch.
        def fake_run_one(spec, combo, **kw):
            return sweep.RunResult(
                status=sweep.RunStatus.COMPLETED_FRESH,
                combo_id="x", result_dir=Path("/tmp/x"))
        results = sweep.run_sweep(str(self.spec_path), base_dir=self.base,
                                  run_one_fn=fake_run_one)
        self.assertEqual(len(results), 4)


class TestGitCommitPaths(unittest.TestCase):

    def test_returns_false_for_empty_paths(self):
        sentinel = MagicMock(side_effect=AssertionError("should not call"))
        self.assertFalse(sweep.git_commit_paths([], "msg",
                                                run_subprocess=sentinel))
        sentinel.assert_not_called()

    def test_nothing_staged_returns_false_no_commit(self):
        # add OK; diff returns 0 (nothing staged) -> no commit, no push.
        fake = MagicMock(side_effect=[
            FakeProc(returncode=0),               # git add
            FakeProc(returncode=0),               # git diff (0 = nothing staged)
        ])
        self.assertFalse(sweep.git_commit_paths(["x"], "msg",
                                                run_subprocess=fake))
        # Must have called add, diff, and stopped.
        self.assertEqual(len(fake.call_args_list), 2)

    def test_commits_and_pushes_when_changes_present(self):
        fake = MagicMock(side_effect=[
            FakeProc(returncode=0),                              # add
            FakeProc(returncode=1),                              # diff (1 = has staged)
            FakeProc(returncode=0),                              # commit
            FakeProc(returncode=0, stdout="rpa3\n"),             # rev-parse
            FakeProc(returncode=0),                              # push
        ])
        ok = sweep.git_commit_paths(["x"], "msg", run_subprocess=fake)
        self.assertTrue(ok)
        # Last call should be push to origin <branch>.
        last = fake.call_args_list[-1]
        self.assertEqual(last.args[0][:3], ["git", "push", "origin"])
        self.assertEqual(last.args[0][3], "rpa3")

    def test_explicit_branch_overrides_rev_parse(self):
        fake = MagicMock(side_effect=[
            FakeProc(returncode=0),     # add
            FakeProc(returncode=1),     # diff
            FakeProc(returncode=0),     # commit
            FakeProc(returncode=0),     # push
        ])
        ok = sweep.git_commit_paths(["x"], "msg", branch="my-branch",
                                    run_subprocess=fake)
        self.assertTrue(ok)
        # No rev-parse call.
        cmds = [call.args[0][:2] for call in fake.call_args_list]
        self.assertNotIn(["git", "rev-parse"], cmds)
        # Push targets the explicit branch.
        last = fake.call_args_list[-1]
        self.assertEqual(last.args[0][3], "my-branch")

    def test_no_push_when_disabled(self):
        fake = MagicMock(side_effect=[
            FakeProc(returncode=0),  # add
            FakeProc(returncode=1),  # diff
            FakeProc(returncode=0),  # commit
        ])
        ok = sweep.git_commit_paths(["x"], "msg", push=False,
                                    run_subprocess=fake)
        self.assertTrue(ok)
        cmds = [call.args[0][:2] for call in fake.call_args_list]
        self.assertNotIn(["git", "push"], cmds)

    def test_returns_false_on_subprocess_error(self):
        def fail_run(*a, **kw):
            raise subprocess.CalledProcessError(1, "git")
        self.assertFalse(sweep.git_commit_paths(["x"], "msg",
                                                run_subprocess=fail_run))

    def test_current_branch_helper_strips_newline(self):
        proc = FakeProc(returncode=0, stdout="some-branch\n")
        self.assertEqual(sweep._current_branch(MagicMock(return_value=proc)),
                         "some-branch")


class TestPrintProgress(unittest.TestCase):

    def test_basic(self):
        r = sweep.RunResult(
            status=sweep.RunStatus.COMPLETED_FRESH,
            combo_id="abc",
            result_dir=Path("/tmp/x"),
            duration_seconds=1.234,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            sweep._print_progress(r, idx=0, total=5)
        s = buf.getvalue()
        self.assertIn("[1/5]", s)
        self.assertIn("abc", s)
        self.assertIn("1.2s", s)
        self.assertIn("/tmp/x", s)

    def test_no_duration_renders_dash(self):
        r = sweep.RunResult(
            status=sweep.RunStatus.SKIPPED_RESUMED,
            combo_id="abc", result_dir=Path("/tmp/x"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            sweep._print_progress(r, idx=2, total=3)
        self.assertIn("—", buf.getvalue())

    def test_error_appended(self):
        r = sweep.RunResult(
            status=sweep.RunStatus.FAILED,
            combo_id="abc", result_dir=Path("/tmp/x"),
            error="boom")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sweep._print_progress(r, idx=0, total=1)
        self.assertIn("err=boom", buf.getvalue())


class TestAutoCommitCallback(unittest.TestCase):

    def _make_result(self, parent="/tmp/sw"):
        return sweep.RunResult(
            status=sweep.RunStatus.COMPLETED_FRESH,
            combo_id="abc", result_dir=Path(parent) / "abc",
            duration_seconds=0.1)

    def test_commits_every_n(self):
        calls = []

        def fake_git(paths, message, push=True, run_subprocess=None,
                     remote="origin", branch=None):
            calls.append((tuple(map(str, paths)), message, push))
            return True

        cb = sweep._make_auto_commit_callback(every=2, push=True, git_fn=fake_git)
        # idx 0,1,2,3 with total=4 -> commit at idx=1 (n=2) and idx=3 (n=4 last).
        for i in range(4):
            buf = io.StringIO()
            with redirect_stdout(buf):
                cb(self._make_result(), i, 4)
        self.assertEqual(len(calls), 2)
        # Each commit references the parent dir of the result dir.
        for paths, _, _ in calls:
            self.assertEqual(paths, ("/tmp/sw",))

    def test_commits_at_last_even_when_not_multiple(self):
        # total=3, every=5: never hits multiple but last (n=3) triggers final commit.
        calls = []

        def fake_git(paths, message, push=True, run_subprocess=None,
                     remote="origin", branch=None):
            calls.append(message)
            return True

        cb = sweep._make_auto_commit_callback(every=5, push=False, git_fn=fake_git)
        for i in range(3):
            buf = io.StringIO()
            with redirect_stdout(buf):
                cb(self._make_result(), i, 3)
        self.assertEqual(len(calls), 1)
        self.assertIn("3/3", calls[0])

    def test_push_flag_threaded_through(self):
        captured = []

        def fake_git(paths, message, push=True, run_subprocess=None,
                     remote="origin", branch=None):
            captured.append(push)
            return True

        cb = sweep._make_auto_commit_callback(every=1, push=False, git_fn=fake_git)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cb(self._make_result(), 0, 1)
        self.assertEqual(captured, [False])


class TestMainCli(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name
        spec = _make_spec(sweep_axes={"A": [1]})
        self.spec_path = Path(self.tmp.name) / "spec.json"
        self.spec_path.write_text(json.dumps(spec))

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_zero_when_all_succeed(self):
        with patch("tools.benchmark.sweep.run_sweep",
                   return_value=[sweep.RunResult(
                       status=sweep.RunStatus.COMPLETED_FRESH,
                       combo_id="x", result_dir=Path("/tmp"))]):
            rc = sweep.main([str(self.spec_path), "--base-dir", self.base])
        self.assertEqual(rc, 0)

    def test_returns_nonzero_on_failures(self):
        with patch("tools.benchmark.sweep.run_sweep",
                   return_value=[sweep.RunResult(
                       status=sweep.RunStatus.FAILED,
                       combo_id="x", result_dir=Path("/tmp"))]):
            rc = sweep.main([str(self.spec_path), "--base-dir", self.base])
        self.assertEqual(rc, 1)

    def test_auto_commit_path(self):
        # When --auto-commit-every > 0 is set, main wires the auto-commit
        # callback. We assert that run_sweep gets a callback that's NOT
        # the plain progress printer.
        with patch("tools.benchmark.sweep.run_sweep") as mock_sweep:
            mock_sweep.return_value = []
            sweep.main([str(self.spec_path), "--auto-commit-every", "3"])
            cb = mock_sweep.call_args.kwargs["on_result"]
            # The auto-commit callback is a closure created by
            # _make_auto_commit_callback; verify by name.
            self.assertEqual(cb.__qualname__,
                             "_make_auto_commit_callback.<locals>.cb")

    def test_default_callback_is_print_progress(self):
        with patch("tools.benchmark.sweep.run_sweep") as mock_sweep:
            mock_sweep.return_value = []
            sweep.main([str(self.spec_path)])
            cb = mock_sweep.call_args.kwargs["on_result"]
            self.assertIs(cb, sweep._print_progress)

    def test_no_push_flag_threaded_to_callback(self):
        # When --no-push is passed, the auto-commit callback should disable push.
        captured = {}

        def fake_make(every, push, git_fn=None):
            captured["push"] = push
            return lambda r, i, n: None

        with patch("tools.benchmark.sweep._make_auto_commit_callback",
                   side_effect=fake_make):
            with patch("tools.benchmark.sweep.run_sweep", return_value=[]):
                sweep.main([str(self.spec_path),
                            "--auto-commit-every", "1",
                            "--no-push"])
        self.assertFalse(captured["push"])

    # Note: the bottom-of-file `if __name__ == "__main__": sys.exit(main())`
    # block is not covered by these tests because runpy.run_module reloads
    # the module in a fresh namespace, bypassing any mocks on run_sweep.
    # main() itself is fully covered above through direct invocation.
    #
    # tests/benchmark/test_parse_bench_log.py::TestMainCli::test_module_main_block
    # *does* exercise the equivalent shim because parse_bench_log's main
    # reads a real file and writes stdout — no mocking required, so the
    # runpy reload is fine. If we ever refactor sweep.run_sweep to be
    # invoked through a sys.argv-driven seam (e.g. an env var pointing at
    # a script-runner stub instead of subprocess.run), the same approach
    # would let us cover sweep.py's __main__ block too.


if __name__ == "__main__":
    unittest.main()
