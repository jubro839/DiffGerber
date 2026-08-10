"""Every number in the paper must trace back to a script in this repository.

`verify_numbers.py` proves the manuscript agrees with the result files. That
leaves the other half open: a result file could sit in `experiments/results/`
with nothing in the release able to regenerate it, and no check would notice.
This closes that gap by walking the dependency the other way.

    paper/{verify_numbers,make_tables,make_figures}.py
        -> the result files they read
            -> the experiment script that writes each one

Anything the paper reads that no script writes is reported as an orphan and
exits non-zero: a reviewer who runs the pipeline would hit a file they cannot
reproduce. Unread result files are listed too, but they are not failures --
superseded runs are deliberately kept in the release.

    python tools/check_provenance.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
PAPER = ROOT / "paper"

# scripts assign the results directory to either name; match both
DIR_VAR = r"(?:RES|RESULTS)"
REF = re.compile(DIR_VAR + r' / f?"([^"]+)"')


def template_to_regex(name):
    """An f-string filename becomes a pattern: {TOP_N} matches any token."""
    rx = re.sub(r"\\\{[^}]*\\\}", r"[A-Za-z0-9_.]*", re.escape(name))
    return re.compile("^" + rx + "$")


def main():
    consumers = ["verify_numbers.py", "make_tables.py", "make_figures.py"]
    read = set()
    for f in consumers:
        read |= set(REF.findall((PAPER / f).read_text()))

    writers = []
    for p in sorted((ROOT / "experiments").glob("*.py")):
        for name in REF.findall(p.read_text()):
            writers.append((template_to_regex(name), p.name))

    orphans, missing, traced = [], [], {}
    for r in sorted(read):
        candidates = [r] if "{" not in r else sorted(
            x.name for x in RES.iterdir()
            if template_to_regex(r).match(x.name))
        for name in candidates:
            if not (RES / name).exists():
                missing.append(name)
                continue
            hits = sorted({w for rx, w in writers if rx.match(name)})
            traced[name] = hits
            if not hits:
                orphans.append(name)

    print(f"paper pipeline reads {len(traced)} result files\n")
    for name, hits in sorted(traced.items()):
        mark = "  " if hits else "!!"
        print(f"{mark}{name:46s} <- {', '.join(hits) or 'NO PRODUCING SCRIPT'}")

    unread = sorted(p.name for p in RES.glob("*.csv") if p.name not in traced)
    print(f"\n{len(unread)} result files are not read by the paper pipeline "
          f"(superseded runs, kept deliberately)")

    bad = 0
    if missing:
        bad += len(missing)
        print("\nMISSING from experiments/results/:")
        print("\n".join("  " + m for m in missing))
    if orphans:
        bad += len(orphans)
        print("\nORPHANED (paper reads them, no script writes them):")
        print("\n".join("  " + o for o in orphans))
    print("\n" + "=" * 70)
    print("provenance OK" if not bad else f"provenance FAILED ({bad} problems)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
