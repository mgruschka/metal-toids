"""
Collect results.json from a set of phase1 runs and interpolate to
experimental targets.

    python collect.py runs/Zn_c4_* --target-hfe -484.6 --target-iod 2.09

The interpolation is the actual fitting step: HFE and IOD are each
monotonic in the scanned parameter, so a spline through the scan points
gives the value that hits the experimental target.  Where the two
disagree you have the HFE-set / IOD-set split that motivates 12-6-4 in
the first place.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load(dirs):
    rows = []
    for d in dirs:
        path = os.path.join(d, "results.json")
        if not os.path.exists(path):
            print(f"  skip {d}: no results.json", file=sys.stderr)
            continue
        with open(path) as fh:
            r = json.load(fh)
        a = r["args"]
        row = dict(dir=d,
                   c4=a.get("c4") or 0.0,
                   sigma=a["sigma"],
                   epsilon=a["epsilon"])
        if "hfe" in r:
            raw = r["hfe"]["raw_hfe_kcal"]
            tail = r.get("corrections", {}).get("c4_tail_kcal", 0.0)
            row["hfe_raw"] = raw
            row["hfe_tail"] = tail
            row["hfe_corrected"] = raw + tail
            row["hfe_err"] = r["hfe"]["dG_err_kcal"]
        if "iod" in r:
            row["iod"] = r["iod"]["iod_A"]
            row["cn"] = r["iod"]["cn"]
        rows.append(row)
    return sorted(rows, key=lambda r: r["c4"])


def interpolate(x, y, target, label):
    """Monotonic interpolation to the target; reports if out of range."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(y)
    xs, ys = x[order], y[order]
    if not (ys.min() <= target <= ys.max()):
        print(f"  {label}: target {target} outside scanned range "
              f"[{ys.min():.3f}, {ys.max():.3f}] - extend the scan")
        return None
    val = float(np.interp(target, ys, xs))
    print(f"  {label}: target {target} reached at parameter = {val:.4f}")
    return val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--target-hfe", type=float, default=None,
                   help="kcal/mol; state your proton-anchor convention")
    p.add_argument("--target-iod", type=float, default=None, help="angstrom")
    p.add_argument("--vs", default="c4", choices=("c4", "sigma"),
                   help="which parameter was scanned")
    a = p.parse_args()

    dirs = []
    for d in a.dirs:
        dirs.extend(glob.glob(d))
    rows = load(sorted(set(dirs)))
    if not rows:
        raise SystemExit("no results found")

    hdr = f"{'C4':>8} {'sigma':>8} {'HFE_raw':>10} {'tail':>8} " \
          f"{'HFE_corr':>10} {'+/-':>7} {'IOD':>7} {'CN':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['c4']:8.1f} {r['sigma']:8.4f} "
              f"{r.get('hfe_raw', float('nan')):10.3f} "
              f"{r.get('hfe_tail', 0.0):8.3f} "
              f"{r.get('hfe_corrected', float('nan')):10.3f} "
              f"{r.get('hfe_err', float('nan')):7.3f} "
              f"{r.get('iod', float('nan')):7.3f} "
              f"{r.get('cn', float('nan')):6.2f}")

    xs = [r[a.vs] for r in rows]
    print()
    if a.target_hfe is not None:
        ys = [r.get("hfe_corrected") for r in rows]
        if all(y is not None for y in ys):
            interpolate(xs, ys, a.target_hfe, "HFE")
    if a.target_iod is not None:
        ys = [r.get("iod") for r in rows]
        if all(y is not None for y in ys):
            interpolate(xs, ys, a.target_iod, "IOD")

    print("\nReminder: HFE targets are convention-dependent.  Li/Merz used a")
    print("proton hydration free energy of -1056.98 kcal/mol (Tissandier")
    print("cluster-pair) and reported 'real' values including the surface")
    print("potential.  A different anchor shifts targets by >10 kcal/mol per")
    print("unit charge.  Pull targets from the papers, not a secondary table.")


if __name__ == "__main__":
    main()
