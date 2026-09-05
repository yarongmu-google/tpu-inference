# In-loop op census of a Mosaic dump: for every kernel in the dump dir,
# count the LLO ops INSIDE the scf.for body vs outside it. This is how a
# unit-rate probe is read: the measured cyc/1024 must be explained by
# the ops the loop actually executes per vreg (a hoisted or dead op is
# visible here as an op outside the loop, or missing).
#
# Usage: python tmp/llo_loop_census.py tmp/mosaic_units

import collections
import glob
import os
import re
import sys

MNEMONIC = re.compile(r"(llo\.[A-Za-z_0-9.]+|scf\.[a-z]+)")
SKIP = {"llo.constant", "llo.type"}


def census(path):
    inside = False
    cin, cout = collections.Counter(), collections.Counter()
    for ln in open(path):
        m = MNEMONIC.search(ln)
        if not m:
            continue
        op = m.group(1)
        if op == "scf.for":
            inside = True
        elif op == "scf.yield":
            inside = False
        elif op not in SKIP:
            (cin if inside else cout)[op] += 1
    return cin, cout


def header(orig_path):
    txt = open(orig_path).read()
    name = re.search(r"func.func @(\w+)", txt)
    pre = txt.split("scf.for", 1)[0]
    bounds = [int(v) for v in re.findall(r"arith.constant (\d+) : i32", pre)]
    return (name.group(1) if name else "?"), (max(bounds) if bounds else -1)


def main():
    for d in sys.argv[1:]:
        print(f"=== census {d}")
        for llo in sorted(glob.glob(os.path.join(d, "*-post-finalize-llo.txt"))):
            stem = os.path.basename(llo).split("-")[0]
            origs = glob.glob(os.path.join(d, f"{stem}-*-original.txt"))
            name, n = header(origs[0]) if origs else ("?", -1)
            cin, cout = census(llo)
            print(f"{name:18s} n={n:<6d} IN-LOOP  {dict(sorted(cin.items()))}")
            print(f"{'':18s} {'':9s} OUTSIDE  {dict(sorted(cout.items()))}")


if __name__ == "__main__":
    main()
