#!/usr/bin/env python
"""
Collect one round of the cyclic scheme:

    python -m lig1264.rounds runs/round1 [--target ddG_exp_kcal]

Reads ti_results.json from the water leg (n_ligands == 0) and every complex
leg, prints ddG = dG_complex - dG_water per alpha, and, given the
experimental target, interpolates alpha for the next round.
"""
import argparse, glob, json, os
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rounddir")
    ap.add_argument("--target", type=float, default=None, help="experimental ddG (kcal/mol)")
    ap.add_argument("--cls", default="NG2R50")
    args = ap.parse_args(argv)
    water, complexes = None, []
    for f in sorted(glob.glob(os.path.join(args.rounddir, "*", "ti_results.json"))):
        r = json.load(open(f))
        if r["n_ligands"] == 0:
            water = r
        else:
            complexes.append(r)
    if water is None:
        raise SystemExit("no water leg (n_ligands == 0) finished yet")
    print(f"water leg  {water['ion_a']} -> {water['ion_b']}: dG = {water['dG_kcal']:.3f} +/- {water['dG_sem_kcal']:.3f}")
    xs, ys = [], []
    for r in sorted(complexes, key=lambda r: r["alpha"].get(args.cls, 0)):
        a = r["alpha"].get(args.cls)
        ddg = r["dG_kcal"] - water["dG_kcal"]
        err = np.hypot(r["dG_sem_kcal"], water["dG_sem_kcal"])
        print(f"alpha({args.cls}) = {a:5.2f}: dG_complex = {r['dG_kcal']:8.3f}  ddG = {ddg:8.3f} +/- {err:.3f} kcal/mol")
        xs.append(a); ys.append(ddg)
    if args.target is not None and len(xs) >= 2:
        # linear interpolation in alpha; extrapolate if needed
        p = np.polyfit(xs, ys, 1)
        a_next = (args.target - p[1]) / p[0]
        print(f"linear fit ddG = {p[0]:.3f}*alpha + {p[1]:.3f};  target {args.target} -> next alpha = {a_next:.3f}")


if __name__ == "__main__":
    main()
