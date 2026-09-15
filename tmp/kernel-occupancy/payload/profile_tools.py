"""Historical device-trace extraction, reused for both kernel variants."""
import gzip
import json
from pathlib import Path

def _iter_trace_events(trace_dir: str):
    for path in sorted(Path(trace_dir).rglob("*.trace.json.gz")):
        with gzip.open(path, "rt") as trace_file:
            yield json.load(trace_file).get("traceEvents", [])

def _device_pids(events) -> tuple:
    """(tc_pid_to_dev, sc_pid_to_dev) maps from the trace metadata -
    pid -> device index parsed from '/device:TPU:<n>'."""
    tc, sc = {}, {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_name = str(event.get("args", {}).get("name", ""))
            marker = "/device:TPU:"
            if marker in process_name:
                dev = process_name.split(marker, 1)[1].split(" ")[0]
                dev = int("".join(ch for ch in dev if ch.isdigit()) or -1)
                (sc if "SparseCore" in process_name else tc)[
                    event.get("pid")] = dev
    return tc, sc

def _interval_coverage_us(intervals, lo, hi) -> float:
    """Union length of [start,end) intervals clipped to [lo, hi) -
    busy WALL time, immune to parallel-lane overcounting."""
    clipped = sorted((max(a, lo), min(b, hi))
                     for a, b in intervals if b > lo and a < hi)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total

def _device_kernel_ms_per_dispatch_from_trace(
    trace_dir: str,
    *,
    jit_name_prefix: str,
) -> list:
    """Device latency of each matching top-level JIT dispatch (ported
    from bench_stacked_rpa_golden: TensorCore pids only, barrier and
    trailing-copy edges subtracted)."""
    matching_by_pid: dict = {}
    ops_by_pid: dict = {}
    sc_ops_by_dev: dict = {}
    tc_dev: dict = {}
    for events in _iter_trace_events(trace_dir):
        tc_pids, sc_pids = _device_pids(events)
        tc_dev.update(tc_pids)
        for event in events:
            pid = event.get("pid")
            if event.get("ph") != "X" or not event.get("dur"):
                continue
            name = str(event.get("name", ""))
            start_us = float(event["ts"])
            duration_us = float(event["dur"])
            end_us = start_us + duration_us
            if pid in sc_pids:
                # leaf ops only: wrapper spans (jit_*, OFFLOAD_COLLECTIVE)
                # cover the whole async region incl. WAITING for the TC
                # kernel's output - occupancy, not work (measured 898us
                # of span around 28us of actual reduce-scatter)
                if (name.startswith("jit_") or name == "OFFLOAD_COLLECTIVE"
                        or name.isdigit()):
                    continue
                sc_ops_by_dev.setdefault(sc_pids[pid], []).append(
                    (start_us, end_us))
                continue
            if pid not in tc_pids:
                continue
            if name.startswith(jit_name_prefix):
                matching_by_pid.setdefault(pid, []).append(
                    (start_us, end_us, duration_us))
            elif not name.startswith("jit_"):
                ops_by_pid.setdefault(pid, []).append(
                    (start_us, end_us, duration_us, name))
    if not matching_by_pid:
        return []
    pid, dispatches = max(matching_by_pid.items(),
                          key=lambda item: len(item[1]))
    ops = ops_by_pid.get(pid, [])
    # SC events of the SAME device only, as interval COVERAGE within the
    # dispatch window: summing durations across parallel subcore lanes
    # (or all 8 devices) overcounts absurdly - 24ms/dispatch measured.
    sc_intervals = sc_ops_by_dev.get(tc_dev.get(pid, -1), [])
    rows = []
    for start_us, end_us, duration_us in sorted(dispatches):
        children = [op for op in ops if start_us <= op[0] < end_us]
        barrier_us = sum(op[2] for op in children
                         if op[3] == "barrier-cores")
        copy_us = 0.0
        if children:
            last = max(children, key=lambda op: op[1])
            if last[3].startswith("copy"):
                copy_us = last[2]
        sc_us = _interval_coverage_us(sc_intervals, start_us, end_us)
        rows.append((max(duration_us - barrier_us - copy_us, 0.0) / 1000.0,
                     sc_us / 1000.0))
    return rows
