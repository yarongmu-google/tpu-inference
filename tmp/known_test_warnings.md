# Known third-party warnings during pytest

These warnings fire on every pytest run from third-party imports. None are
from `tpu_inference` code. Listed here so we can recognise / filter them
manually rather than blanket-suppressing them in `pyproject.toml`.

To filter them out of pytest output, pipe through grep:

```bash
python -m pytest tests/runner/ -v 2>&1 \
  | grep -vE "Failed to read commit hash|classes created inside an enum|multi-threaded.*fork|torch\.jit\.script_method"
```

## The warnings

### 1. vllm `_version` module missing — `RuntimeWarning`

```
vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
```

vLLM editable installs without a git tag don't have `vllm._version`
generated. Cosmetic, no runtime impact. Goes away if vllm is installed
from a tagged release wheel.

### 2. `tpu_info` `NamedTuple` in `Enum` — `DeprecationWarning`

```
tpu_info/device.py:33: DeprecationWarning: In 3.13 classes created inside
an enum will not become a member. Use the `member` decorator to keep the
current behavior.
```

The `tpu_info` package nests a `typing.NamedTuple` inside an `Enum`. Works
on Py 3.12 (which the repo uses), deprecated in 3.13. Fix has to happen
upstream in `tpu_info`.

### 3. multiprocessing fork in multi-threaded process — `DeprecationWarning`

```
multiprocessing/popen_fork.py:66: DeprecationWarning: This process
(pid=...) is multi-threaded, use of fork() may lead to deadlocks in
the child.
```

Triggered when JAX / TPU runtime init spawns a worker after threads have
been started. Stdlib warning since Py 3.12. Real risk only if our test
code itself forks; our tests don't.

### 4. `torch.jit.script_method` deprecated — `DeprecationWarning` (×14)

```
torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method`
is deprecated. Please switch to `torch.compile` or `torch.export`.
```

Comes from torch internals via transitive imports. `tpu_inference` does
not use `torch.jit.script_method` directly. Will go away when torch
removes the legacy API.

## When a new warning appears

Investigate before adding to this list. The whole point of *not* using
`filterwarnings` in `pyproject.toml` is so that real problems from our
own code surface as warnings during tests.
