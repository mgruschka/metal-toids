#!/usr/bin/env python
"""
First-shell check for the alpha-TI windows (TI_param/README "Known gaps" 2):
does the ion's first shell (waters and ligand N) change with alpha, or drift
within a 2 ns window, in a way dU/dl cannot see?

    python water_cn.py runs/co_imid1_alpha_ti_wcn [--ref-run runs/co_imid1_alpha_ti]

For every window_NN.dcd in the run dir:
  * Co-O(water) RDF -> middle of the first-shell depletion gap; ONE common
    cutoff (from the lowest-alpha window) is used for every window so the counts are comparable
  * per-frame first-shell water count CN_w and ligand-N count CN_N
  * CN_w distribution, quarter means (drift), frames with a CN change,
    distinct waters seen in the shell (exchange)
  * <dU/dl> of the same window vs the reference run's window (a seed replicate)
-> water_cn.json in the run dir, and a table on stdout.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ligand-1264"))
import mdtraj as md                                      # noqa: E402
from lig1264.analysis import load, rdf_cn, first_min     # noqa: E402
from lig1264.build import WATER_RES                      # noqa: E402
from lig1264.config import load_config                   # noqa: E402

KCAL = 4.184


def shell_cutoff(r, g, frac=0.01):
    """
    Cutoff between the first and second shells.  A divalent ion's first shell
    is so well separated that g(r) is exactly 0 over a wide gap, and the
    "first minimum" (lig1264.analysis.first_min) is then the first empty bin,
    right at the edge of the peak -- which miscounts a Co-N/O stretched by
    0.05 A.  Take the middle of the depletion zone instead: the span after the
    first peak where g < frac * peak, up to where g recovers.  Falls back to
    first_min when there is no such gap.
    """
    ipk = int(np.argmax(g))
    low = np.where(g[ipk:] < frac * g[ipk])[0]
    if not len(low):
        return first_min(r, g)
    a = ipk + low[0]
    rec = np.where(g[a:] >= frac * g[ipk])[0]
    b = a + (rec[0] - 1 if len(rec) else len(g[a:]) - 1)
    return float(0.5 * (r[a] + r[b]))


def shell(traj, ion, idx, rc):
    """per-frame count and membership of idx within rc (nm) of ion."""
    d = md.compute_distances(traj, np.array([[ion, j] for j in idx]), periodic=True)
    inside = d < rc
    return inside.sum(axis=1), inside


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--ref-run", default=None, help="run whose dudl_wNN.csv to compare <dU/dl> with")
    ap.add_argument("--rmax", type=float, default=0.6, help="RDF range, nm")
    args = ap.parse_args(argv)

    run = args.run
    at = load_config(os.path.join(run, "config.yaml")).alpha_ti
    # from the schedule, not alpha_ti_setup.json: array tasks rewrite that
    # file concurrently, so it is not a safe source
    from lig1264 import lambda_windows as LW
    lam_all = (LW.uniform(at.n_windows) if at.schedule == "uniform"
               else LW.gauss_legendre(at.n_windows)[0]).tolist()
    dcds = sorted(glob.glob(os.path.join(run, "window_[0-9][0-9].dcd")))
    if not dcds:
        raise SystemExit(f"no window_NN.dcd in {run} (traj_ps was 0?)")
    ks = [int(re.search(r"window_(\d+)\.dcd", f).group(1)) for f in dcds]

    trajs, rows = {}, []
    for k in ks:
        traj, groups, classes = load(run, f"window_{k:02d}")
        top = traj.topology
        ion = groups["ion"][0]
        ow = [a.index for a in top.atoms if a.residue.name in WATER_RES and a.element.symbol == "O"]
        ln = [a.index for a in top.atoms if a.index in set(groups["ligand"]) and a.element.symbol == "N"]
        trajs[k] = (traj, ion, ow, ln)

    # common cutoff from the lowest-alpha window's Co-O(water) RDF
    k0 = min(ks)
    t0, ion0, ow0, _ = trajs[k0]
    r, g, cn = rdf_cn(t0, ion0, ow0, args.rmax, 300)
    rc = shell_cutoff(r, g)
    print(f"common Co-O(water) cutoff = middle of the first-shell depletion gap, window {k0}: "
          f"{rc * 10:.2f} A  (first empty bin {first_min(r, g) * 10:.2f} A)")

    for k in ks:
        traj, ion, ow, ln = trajs[k]
        rk, gk, _ = rdf_cn(traj, ion, ow, args.rmax, 300)
        cn_w, inside = shell(traj, ion, ow, rc)
        cn_n, _ = shell(traj, ion, ln, rc)
        n = len(cn_w)
        q = [float(cn_w[i * n // 4:(i + 1) * n // 4].mean()) for i in range(4)]
        vals, cnt = np.unique(cn_w, return_counts=True)
        row = dict(window=k, lam=float(lam_all[k]),
                   alpha=float(at.alpha0 + lam_all[k] * (at.alpha1 - at.alpha0)),
                   frames=int(n), rdf_first_min_A=float(first_min(rk, gk) * 10),
                   rdf_first_peak_A=float(rk[np.argmax(gk)] * 10),
                   cn_water_mean=float(cn_w.mean()), cn_water_quarters=q,
                   cn_water_dist={int(v): float(c / n) for v, c in zip(vals, cnt)},
                   cn_N_mean=float(cn_n.mean()),
                   frames_cn_change=int((np.diff(cn_w) != 0).sum()),
                   distinct_shell_waters=int(inside.any(axis=0).sum()))
        csv = os.path.join(run, f"dudl_w{k:02d}.csv")
        if os.path.exists(csv):
            a = np.loadtxt(csv, delimiter=",")[:, 1]
            row["dudl_mean_kcal"] = float(a[int(at.discard_frac * len(a)):].mean() / KCAL)
        if args.ref_run:
            ref = os.path.join(args.ref_run, f"dudl_w{k:02d}.csv")
            if os.path.exists(ref):
                a = np.loadtxt(ref, delimiter=",")[:, 1]
                row["ref_dudl_mean_kcal"] = float(a[int(at.discard_frac * len(a)):].mean() / KCAL)
        rows.append(row)

    print(f"{'w':>3} {'alpha':>6} {'CN_w':>6} {'quarters':>24} {'CN_N':>5} "
          f"{'dist (CN:frac)':>28} {'chg':>5} {'uniq':>5} {'<dU/dl>':>9} {'ref':>9}")
    for r_ in rows:
        dist = " ".join(f"{c}:{f:.2f}" for c, f in r_["cn_water_dist"].items() if f >= 0.005)
        print(f"{r_['window']:>3} {r_['alpha']:6.3f} {r_['cn_water_mean']:6.3f} "
              f"{' '.join(f'{x:.2f}' for x in r_['cn_water_quarters']):>24} {r_['cn_N_mean']:5.2f} "
              f"{dist:>28} {r_['frames_cn_change']:5d} {r_['distinct_shell_waters']:5d} "
              f"{r_.get('dudl_mean_kcal', float('nan')):9.3f} {r_.get('ref_dudl_mean_kcal', float('nan')):9.3f}")
    json.dump(dict(cutoff_A=float(rc * 10), cutoff_from_window=k0, windows=rows),
              open(os.path.join(run, "water_cn.json"), "w"), indent=2)
    print(f"wrote {run}/water_cn.json")


if __name__ == "__main__":
    main()
