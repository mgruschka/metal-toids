#!/usr/bin/env python
"""
Iterative fit of alpha(<class>) to an experimental binding free energy.

    python -m lig1264.fit_alpha runs/pmf/co_imid1_* --target -3.36 [--write-config]

Li/Merz iterate with a secant through two runs; this uses a Newton step
instead, because dU/dalpha is analytic and needs no second run:

    run a PMF at some alpha -> dG0
    |dG0 - dG0_exp| <= tol ?  record alpha, the effective C4 and dG0, stop
    otherwise               ?  d(dG0)/d(alpha) by linear response on THAT run
                               -> new alpha -> new PMF

This tool does the bookkeeping for one turn of that loop: it reads every
finished run (`config.yaml` + `pmf.json`), reports where each one sits
relative to the target, and either declares convergence or prints the next
alpha to run -- with `--write-config`, the YAML for it as well.

`--tol` defaults to 0.1 kcal/mol, the Li/Merz criterion.  That is a tight
target: it is only meaningful if the bootstrap error on dG0 is well below
it, so a run whose error bar is comparable to the tolerance is flagged
rather than declared converged.

The effective ion<->ligand-class C4 reported alongside alpha is

    C4_eff = C4_ion * alpha_class / 1.444        [kcal/mol A^4]

because the mixing rule in ions_1264.xml is
-(cfour1*arel2 + cfour2*arel1)/r^4 with arel = alpha/1.444, and ligand atoms
carry cfour = 0, so only the ion's own C4 contributes.
"""
import argparse
import csv
import json
import os

import numpy as np

from .config import load_config, REPO_ROOT

WATER_POL = 1.444
KT300 = 0.008314462618 * 300.0 / 4.184   # kcal/mol


def ion_base_c4(element, charge):
    """C4(H2O) in kcal/mol A^4 from 12-6-4_base_params.csv."""
    path = os.path.join(REPO_ROOT, "12-6-4_base_params.csv")
    want = f"{abs(charge)}{'+' if charge > 0 else '-'}"   # the CSV writes 1+, 2+, 1-
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["Element"].strip() == element and row["Charge"].strip() == want:
                return float(row[[k for k in row if k.startswith("C4")][0]])
    return None


def collect(dirs, cls):
    rows = []
    for d in sorted(dirs):
        cfg_p, pmf_p = os.path.join(d, "config.yaml"), os.path.join(d, "pmf.json")
        if not (os.path.exists(cfg_p) and os.path.exists(pmf_p)):
            continue
        cfg = load_config(cfg_p)
        if cls not in cfg.c4.alpha_by_class:
            print(f"skipping {d}: no alpha for class {cls}")
            continue
        res = json.load(open(pmf_p))
        b = res.get("binding")
        if b is None:
            print(f"skipping {d}: pmf.json predates the binding section; rerun lig1264.wham")
            continue
        rows.append(dict(dir=d, cfg=cfg, alpha=float(cfg.c4.alpha_by_class[cls]),
                         dG0=float(b["dG0_kcal"]), sem=b.get("dG0_sem_kcal"),
                         well=float(b["well_depth_w_kcal"]),
                         r_min=float(b["r_min_A"]),
                         r_bound=float(b["r_bound_A"]),
                         slope=float(b["plateau_slope_kcal_per_A"]),
                         overlap_min=res.get("overlap_min")))
    return sorted(rows, key=lambda r: r["alpha"])


def class_atom_check(run_dir, cls):
    """
    (n atoms of `cls` on the ligand, is the CV partner one of them).

    The linear-response prediction below is exact only when the fitted class
    has a single atom and it is the one in the CV -- then dU/dalpha depends
    on nothing but the collective variable.
    """
    try:
        classes = {int(k): v for k, v in
                   json.load(open(os.path.join(run_dir, "classes.json"))).items()}
        groups = json.load(open(os.path.join(run_dir, "groups.json")))
        meta = json.load(open(os.path.join(run_dir, "windows.json")))
    except (OSError, ValueError):
        return None, None
    hits = [i for i in groups["ligand"] if classes.get(i) == cls]
    return len(hits), (meta["ligand_atom"] in hits)


def linear_response(run_dir, alpha, c4_ion, kT_kcal, target, cls,
                    plateau_frac=0.15):
    """
    Predict dG0(alpha') from ONE finished run, with no new simulation.

    The C4 term is exactly linear in alpha:

        U(alpha) = U0 + alpha * V,   V = dU/dalpha = -(C4_ion/1.444) / r^4

    so dW(r)/dalpha = <V>_r.  When the fitted class has one atom and it is
    the CV partner, V is a deterministic function of the CV, so the
    conditional average is V itself and

        dW(r)/dalpha = -(C4_ion/1.444) / r^4        (closed form)

    Shifting W by that and re-running the binding integral gives dG0 at any
    alpha' to first order in (alpha' - alpha).  It is a linear-response
    estimate: it moves the PMF but not the underlying ensemble, so it
    ignores solvent reorganisation and therefore OVERSTATES how fast dG0
    falls.  Use it to choose the next alpha, not to replace running it.
    """
    from .wham import binding_free_energy
    csv_p = os.path.join(run_dir, "pmf.csv")
    if not os.path.exists(csv_p):
        return None
    d = np.loadtxt(csv_p, delimiter=",", skiprows=1)
    r_A, Wj = d[:, 0], d[:, 3]
    dWda = -(c4_ion / WATER_POL) / r_A ** 4          # kcal/mol per A^3

    def dG0_at(da):
        return binding_free_energy(r_A, Wj + da * dWda, kT_kcal,
                                   None, plateau_frac)["dG0_kcal"]

    base = dG0_at(0.0)
    eps = 0.02
    slope = (dG0_at(eps) - dG0_at(-eps)) / (2 * eps)
    # scan alpha' for the target, staying physical (alpha' > 0)
    grid = np.linspace(max(-alpha + 1e-3, -alpha), 20.0 - alpha, 800)
    vals = np.array([dG0_at(x) for x in grid])
    hit = None
    for k in range(len(grid) - 1):
        if (vals[k] - target) * (vals[k + 1] - target) <= 0:
            f = (target - vals[k]) / (vals[k + 1] - vals[k])
            hit = grid[k] + f * (grid[k + 1] - grid[k])
            break
    return dict(dG0_check=base, slope=slope, dalpha=hit,
                alpha_pred=None if hit is None else alpha + hit,
                reach=(float(vals.min()), float(vals.max())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--target", type=float, required=True, help="kcal/mol")
    ap.add_argument("--tol", type=float, default=0.1, help="kcal/mol (Li/Merz: 0.1)")
    ap.add_argument("--class", dest="cls", default="NG2R50")
    ap.add_argument("--write-config", action="store_true",
                    help="write the YAML for the next alpha next to the best run")
    args = ap.parse_args(argv)

    rows = collect(args.dirs, args.cls)
    if not rows:
        raise SystemExit("no finished runs found (need config.yaml + pmf.json)")

    ion = rows[0]["cfg"].ion
    c4_ion = ion_base_c4(ion.element, ion.charge)
    print(f"ion {ion.resname}   base C4(H2O) = "
          f"{c4_ion if c4_ion is None else f'{c4_ion:.1f}'} kcal/mol A^4   "
          f"class {args.cls}   target dG0 = {args.target:+.2f} "
          f"+/- {args.tol:.2f} kcal/mol\n")
    print(f"{'alpha':>7} {'C4_eff':>8} {'dG0':>8} {'+/-':>6} {'err':>7} "
          f"{'well w':>7} {'r*':>5} {'slope':>7} {'ovlap':>6}  run")
    for r in rows:
        c4e = "" if c4_ion is None else f"{c4_ion * r['alpha'] / WATER_POL:8.1f}"
        sem = f"{r['sem']:6.2f}" if r["sem"] else "     -"
        ov = f"{r['overlap_min']:6.3f}" if r["overlap_min"] is not None else "     -"
        flags = []
        if abs(r["slope"]) > 0.05:
            flags.append("plateau not flat")
        if r["overlap_min"] is not None and r["overlap_min"] < 0.03:
            flags.append("overlap gap")
        if r["sem"] and r["sem"] > 0.5 * args.tol:
            flags.append(f"error bar {r['sem']:.2f} vs tol {args.tol:.2f}")
        print(f"{r['alpha']:7.2f} {c4e} {r['dG0']:+8.2f} {sem} "
              f"{r['dG0'] - args.target:+7.2f} {r['well']:7.2f} {r['r_bound']:5.2f} "
              f"{r['slope']:+7.3f} {ov}  {os.path.basename(r['dir'])}"
              + ("   <- " + "; ".join(flags) if flags else ""))

    best = min(rows, key=lambda r: abs(r["dG0"] - args.target))
    off = best["dG0"] - args.target
    print()
    if abs(off) <= args.tol:
        c4e = None if c4_ion is None else c4_ion * best["alpha"] / WATER_POL
        print(f"CONVERGED: |dG0 - target| = {abs(off):.3f} <= {args.tol} kcal/mol")
        print(f"  alpha({args.cls}) = {best['alpha']:.4f}")
        if c4e is not None:
            print(f"  C4_eff({ion.element}{ion.charge:+d}<->{args.cls}) = {c4e:.1f} kcal/mol A^4")
        print(f"  dG0 = {best['dG0']:+.3f}"
              + (f" +/- {best['sem']:.3f}" if best["sem"] else "") + " kcal/mol")
        print(f"  from {best['dir']}")
        if best["sem"] and best["sem"] > 0.5 * args.tol:
            print(f"  CAUTION: the bootstrap error ({best['sem']:.2f}) is not small "
                  f"against the {args.tol} tolerance -- this may be noise, not "
                  f"convergence.  Lengthen the windows before accepting it.")
        return

    print(f"NOT CONVERGED: best run is off by {off:+.2f} kcal/mol (tol {args.tol})")

    # --- Newton step, always: the derivative comes from linear response on
    #     the best run itself, never from a secant through two runs.
    n_cls, cv_is_one = class_atom_check(best["dir"], args.cls)
    lr = None if c4_ion is None else linear_response(
        best["dir"], best["alpha"], c4_ion, KT300, args.target, args.cls)
    if lr is None:
        raise SystemExit("  no pmf.csv (or no base C4) in the best run: cannot "
                         "take a Newton step.  Run lig1264.wham first.")

    print(f"\n  Newton step from {os.path.basename(best['dir'])} "
          f"(dU/dalpha = -(C4_ion/1.444)/r^4, exact in alpha):")
    if n_cls == 1 and cv_is_one:
        print(f"    {args.cls} has 1 atom on the ligand and it is the CV partner, "
              f"so dW/dalpha is closed-form in the CV.")
    elif n_cls:
        print(f"    NOTE: {n_cls} {args.cls} atoms on the ligand; only the CV "
              f"partner is modelled, so the derivative is approximate.")
    print(f"    d(dG0)/d(alpha) = {lr['slope']:+.2f} kcal/mol per A^3 "
          f"at alpha = {best['alpha']:.3f}")
    if lr["alpha_pred"] is None:
        print(f"    -> target not reachable for alpha < 20; the shifted PMF spans "
              f"dG0 {lr['reach'][0]:+.2f} to {lr['reach'][1]:+.2f}")
        return
    nxt = lr["alpha_pred"]
    print(f"    -> next alpha({args.cls}) = {nxt:.4f}")
    if c4_ion is not None:
        print(f"       C4_eff = {c4_ion * nxt / WATER_POL:.1f} kcal/mol A^4")
    if best["sem"]:
        print(f"       a {best['sem']:.2f} kcal/mol error bar maps to "
              f"+/-{abs(best['sem'] / lr['slope']):.3f} A^3 of alpha")
    print("    Linear response moves the PMF but not the ensemble, so it "
          "overstates how fast dG0 falls: this step is a LOWER bound and the "
          "iteration approaches the target from below, monotonically, rather "
          "than overshooting.")

    # --- diagnostic only (never used to choose alpha): how much the ensemble
    #     actually damped the previous predicted response
    if len(rows) >= 2:
        prev = sorted(rows, key=lambda r: abs(r["alpha"] - best["alpha"]))[1]
        da = best["alpha"] - prev["alpha"]
        if abs(da) > 1e-9:
            realized = (best["dG0"] - prev["dG0"]) / da
            prev_lr = linear_response(prev["dir"], prev["alpha"], c4_ion,
                                      KT300, args.target, args.cls)
            if prev_lr and abs(prev_lr["slope"]) > 1e-9:
                damp = realized / prev_lr["slope"]
                print(f"\n  diagnostic (not used to pick alpha): between "
                      f"alpha={prev['alpha']:.3f} and {best['alpha']:.3f} the "
                      f"realized d(dG0)/d(alpha) was {realized:+.2f} vs "
                      f"{prev_lr['slope']:+.2f} predicted "
                      f"-> ensemble damping factor {damp:.2f}.")
                if 0 < damp < 1:
                    print(f"    At that damping the Newton step will need about "
                          f"{1 / damp:.1f} iterations' worth of travel; expect "
                          f"several turns.")

    if args.write_config:
        tag = f"a{nxt:.2f}".replace(".", "")
        src = os.path.join(best["dir"], "config.yaml")
        dst = os.path.join(os.path.dirname(best["dir"].rstrip("/")) or ".",
                           f"{os.path.basename(best['dir']).rsplit('_a', 1)[0]}_{tag}.yaml")
        out = []
        for line in open(src).read().splitlines(keepends=True):
            st = line.strip()
            if st.startswith(f"{args.cls}:"):
                line = line.split(f"{args.cls}:")[0] + \
                    f"{args.cls}: {nxt:.4f}        # Newton step from alpha={best['alpha']:.4f}\n"
            elif st.startswith("name:"):
                line = f"name: {os.path.basename(dst)[:-5]}\n"
            out.append(line)
        open(dst, "w").writelines(out)
        print(f"\n  wrote {dst}")
    print(f"\n  then:  sbatch slurm/run_pmf.sbatch <that config> <outdir> seed"
          f"\n         sbatch --array=0-{rows[0]['cfg'].pmf.n_windows - 1}%20 "
          f"slurm/run_pmf.sbatch <that config> <outdir>")


if __name__ == "__main__":
    main()
