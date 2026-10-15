#!/usr/bin/env python
"""
MBAR reweighting of ONE umbrella PMF to many values of a C4 polarizability.

    python -m lig1264.mbar runs/pmf/co_imid1_a109 --target -3.36 [--class NG2R50]
           [--alphas 0.5:6:0.05] [--bootstrap 20] [--edge-sd 2] [--force]

Replaces the Li/Merz secant loop (one new PMF per alpha) by post-processing.
LJ1264C4Force is

    U_C4 = -lambda_c4 * (cfour_i * arel_j + cfour_j * arel_i) / r_ij^4,   r < cutoff

with arel_j = alpha_j / 1.444 (x scale_by_class) on the fitted class, so the
energy is exactly linear in alpha:

    u_alpha(x) = u_alpha0(x) + beta (alpha - alpha0) V(x),   V = dU/dalpha

When every alpha-dependent pair is (ion, CV partner) -- one atom of the class,
and it is the pulled one -- V is a function of the CV alone, V(r) = c / r^4, so
the COLVARs already hold everything: at fixed r nothing else feels the
perturbation, the conditional ensemble p(x | r) is alpha-independent, and

    W_alpha(r) = W_alpha0(r) + (alpha - alpha0) V(r)       (no approximation)

Any other alpha-dependent pair (counter-ion <-> class atom, or a second class
atom) is NOT a CV function.  Counter-ion pairs are measured on smd.dcd and
reported as an unmodelled systematic; a second class atom makes this script
refuse, because then per-frame V is needed and the windows do not write it.

Estimator: the windows' production COLVARs, decorrelated per window
(statistical inefficiency of the CV), go into MBAR with u_kn = beta * bias_k.
The unbiased alpha0 weights are then shifted by -beta (alpha - alpha0) V(r_n)
for each alpha on the grid, histogrammed into W_alpha(r), and passed through
wham.binding_free_energy for r* and the plateau.  K_eq itself is summed from
the weighted samples rather than integrated over bins (see reweight_scan), so
at alpha0 it is slightly MORE negative than wham.py's binned dG0 (~0.05-0.1
kcal/mol for a well this narrow); dG0_binned_kcal is the like-for-like value.  Errors: bootstrap over
each window's decorrelated samples, re-solving MBAR every replicate.

Per alpha the script also flags
  * edge   the bound integrand has not decayed at the lowest sampled r: a
           larger alpha pulls the well inward, below the lowest window there
           is no data, and dG0 is truncated (too weak).  trunc_kcal is the
           exponential-tail estimate of the missing part; edge = trunc > 0.05.
           Add wall windows before trusting a flagged alpha.
  * n_eff  Kish effective sample size of the reweighted ensemble.

Writes mbar.json, mbar_dG0_vs_alpha.csv, and mbar_pmf_<alpha>.csv for alpha0
and the target alpha.  The same adjacent-overlap gate as wham.py applies.
"""
import argparse
import json
import os

import numpy as np
from scipy.special import logsumexp

from .wham import (KB, KCAL, MIN_OVERLAP, V_STANDARD_A3, adjacent_overlap,
                   binding_free_energy)

WATER_POL = 1.444       # A^3; arel = alpha / WATER_POL (forces.apply_c4_spec)
EDGE_BINS = 3           # lowest bins used to extrapolate the bound integrand's tail
EDGE_TRUNC_KCAL = 0.05  # flag when the extrapolated missing tail shifts dG0 by more


# ---------------------------------------------------------------- inputs

def load_samples(outdir, discard_frac=0.0):
    """Production CV samples (nm) per window, sorted by centre, with kappa."""
    meta = json.load(open(os.path.join(outdir, "windows.json")))
    rows = []
    for row in meta["windows"]:
        f = os.path.join(outdir, f"window_{row['window']:03d}_colvar.dat")
        if not os.path.exists(os.path.join(outdir, f"window_{row['window']:03d}_done")):
            continue
        x = np.atleast_2d(np.loadtxt(f))[:, 1]
        x = x[int(discard_frac * len(x)):]
        if len(x) >= 10:
            # per-window kappa: wall windows may carry their own
            rows.append((row["centre_nm"], row.get("kappa_kj_per_nm2", meta["kappa_kj_per_nm2"]), x))
    missing = len(meta["windows"]) - len(rows)
    rows.sort(key=lambda t: t[0])
    return meta, rows, missing


def alpha_pairs(outdir, cls, alpha0):
    """
    Every particle pair whose C4 energy depends on alpha(cls), from system.xml.

    Returns (lambda_c4, cutoff_nm, pairs) with pairs = [(i, j, coeff)], coeff
    in kJ/mol nm^4 per A^3 so that dU/dalpha = sum coeff / r_ij^4 (r < cutoff).
    arel is linear in alpha (alpha/1.444, times any scale_by_class), so
    d arel / d alpha = arel(alpha0) / alpha0 needs no knowledge of the scale.
    """
    import openmm as mm
    system = mm.XmlSerializer.deserialize(open(os.path.join(outdir, "system.xml")).read())
    f = next(f for f in system.getForces() if f.getName() == "LJ1264C4Force")
    if f.getUseSwitchingFunction() or f.getUseLongRangeCorrection():
        raise SystemExit("LJ1264C4Force uses switching / LRC; V(r) = c/r^4 with a "
                         "hard cutoff would be wrong -- extend alpha_pairs first")
    lam = dict((f.getGlobalParameterName(k), f.getGlobalParameterDefaultValue(k))
               for k in range(f.getNumGlobalParameters())).get("lambda_c4", 1.0)
    classes = json.load(open(os.path.join(outdir, "classes.json")))
    cls_atoms = sorted(int(i) for i, c in classes.items() if c == cls)
    if not cls_atoms:
        raise SystemExit(f"no atom of class {cls} in {outdir}/classes.json")
    par = {i: f.getParticleParameters(i) for i in range(f.getNumParticles())}
    darel = {j: par[j][1] / alpha0 for j in cls_atoms}
    sources = [i for i, p in par.items() if p[0] != 0.0]
    if f.getNumInteractionGroups():
        groups = [tuple(map(set, f.getInteractionGroupParameters(g)))
                  for g in range(f.getNumInteractionGroups())]
        interacts = lambda i, j: any((i in a and j in b) or (j in a and i in b) for a, b in groups)
    else:
        interacts = lambda i, j: True
    excl = {tuple(sorted(f.getExclusionParticles(k))) for k in range(f.getNumExclusions())}
    pairs = []
    for j in cls_atoms:
        for i in sources:
            if i != j and interacts(i, j) and tuple(sorted((i, j))) not in excl and darel[j]:
                pairs.append((i, j, -lam * par[i][0] * darel[j]))
    return lam, f.getCutoffDistance().value_in_unit(mm.unit.nanometer), pairs


def offcv_systematic(outdir, pairs, cutoff, cv_pair, r_bound_nm, r_plateau_nm):
    """
    <V_off>(bound) - <V_off>(dissociated) from smd.dcd, kJ/mol per A^3: the
    first-order dG0 shift per unit (alpha - alpha0) that the CV-only V misses.
    """
    import mdtraj
    if not pairs:
        return dict(n_pairs=0, delta=0.0, max_abs=0.0, frames=0)
    t = mdtraj.load(os.path.join(outdir, "smd.dcd"), top=os.path.join(outdir, "topology.pdb"))
    cv = mdtraj.compute_distances(t, [list(cv_pair)], periodic=True)[:, 0]
    d = mdtraj.compute_distances(t, [[i, j] for i, j, _ in pairs], periodic=True)
    c = np.array([p[2] for p in pairs])
    V = (np.where(d < cutoff, d ** -4.0, 0.0) * c).sum(axis=1)
    b, u = cv < r_bound_nm, cv > r_plateau_nm
    delta = (V[b].mean() if b.any() else 0.0) - (V[u].mean() if u.any() else 0.0)
    return dict(n_pairs=len(pairs), delta=float(delta), max_abs=float(np.abs(V).max()),
                frames=int(len(cv)), bound_frames=int(b.sum()), plateau_frames=int(u.sum()))


# ---------------------------------------------------------------- estimator

def decorrelate(samples):
    from pymbar import timeseries
    out, g = [], []
    for x in samples:
        gi = timeseries.statistical_inefficiency(x)
        out.append(x[timeseries.subsample_correlated_data(x, g=gi)])
        g.append(gi)
    return out, np.array(g)


def mbar_log_weights(centres, kappas, samples, kT, f_init=None):
    """log w_n of the unbiased alpha0 state (normalised) and the window f_k."""
    from pymbar import MBAR
    x = np.concatenate(samples)
    N_k = np.array([len(s) for s in samples])
    u_kn = 0.5 * kappas[:, None] * (x[None, :] - centres[:, None]) ** 2 / kT
    m = MBAR(u_kn, N_k, initial_f_k=f_init, solver_protocol="robust")
    lw = -logsumexp(m.f_k[:, None] - u_kn + np.log(N_k)[:, None], axis=0)
    return x, lw - logsumexp(lw), m.f_k


def reweight_scan(x, lw0, alphas, alpha0, V, kT, edges, r_bound_A=None,
                  plateau_frac=0.15):
    """dG0 etc. for every alpha from one set of unbiased alpha0 weights."""
    Vx = V(x)
    xk = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(xk) - 1)
    inside = (x >= edges[0]) & (x <= edges[-1])
    kT_kcal = kT / KCAL
    out = []
    for a in alphas:
        lw = lw0 - (a - alpha0) * Vx / kT
        lw = lw - logsumexp(lw)
        n_eff = float(np.exp(2 * logsumexp(lw) - logsumexp(2 * lw)))
        m = lw[inside].max()
        with np.errstate(divide="ignore"):
            logP = np.log(np.bincount(idx[inside], np.exp(lw[inside] - m),
                                      minlength=len(xk))) + m
        ok = np.isfinite(logP)
        r_A = xk[ok] * 10
        Wj = (-kT * logP[ok] + 2 * kT * np.log(xk[ok])) / KCAL
        b = binding_free_energy(r_A, Wj, kT_kcal, r_bound_A, plateau_frac)
        # K_eq from the samples, not the binned W: a 200-bin trapezoid over a
        # 0.08 A-wide well is biased by ~0.1 kcal/mol at alpha0 and ~0.3 at
        # alpha 3 (synthetic test), well inside the 0.1 kcal/mol fit
        # tolerance.  Bins still give r* and the plateau density.
        rs = b["r_bound_A"] / 10
        lo = r_A[-max(3, int(plateau_frac * len(r_A)))] / 10
        hi = edges[-1]
        pl = inside & (x >= lo)
        rho = np.exp(logsumexp(lw[pl])) / (4 / 3 * np.pi * (hi ** 3 - lo ** 3) * 1e3)  # per A^3
        K = float(np.exp(logsumexp(lw[inside & (x <= rs)])) / rho)
        dG0 = -kT_kcal * np.log(K / V_STANDARD_A3)
        # edge: extrapolate the bound integrand's tail below the lowest
        # sampled r as an exponential, I(r) ~ I0 exp(s (r - r0)), and turn the
        # missing area into a dG0 shift.  s <= 0 means the integrand is still
        # rising at the edge: the well continues into unsampled r.
        w = Wj - Wj[-max(3, int(plateau_frac * len(r_A))):].mean()
        lnI = 2 * np.log(r_A) - w / kT_kcal
        slope = np.polyfit(r_A[:EDGE_BINS], lnI[:EDGE_BINS], 1)[0]           # 1/A
        istar = int(np.argmin(np.abs(r_A - b["r_bound_A"])))
        area = np.trapezoid(np.exp(lnI[:istar + 1] - lnI.max()), r_A[:istar + 1])
        if slope > 0:
            trunc = float(kT_kcal * np.log1p(np.exp(lnI[0] - lnI.max()) / slope / area))
        else:
            trunc = float("inf")
        out.append(dict(alpha=float(a), dG0_kcal=dG0, dG0_binned_kcal=b["dG0_kcal"],
                        K_eq_A3=K, r_min_A=b["r_min_A"],
                        r_bound_A=b["r_bound_A"], well_depth_w_kcal=b["well_depth_w_kcal"],
                        plateau_slope=b["plateau_slope_kcal_per_A"], n_eff=n_eff,
                        trunc_kcal=trunc, edge=bool(trunc > EDGE_TRUNC_KCAL),
                        r_A=r_A, Wj=Wj))
    return out


def crossing(alphas, dG, target):
    """First alpha where dG0(alpha) crosses target, linear interpolation."""
    for k in range(len(alphas) - 1):
        a, b = dG[k] - target, dG[k + 1] - target
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0 and a != b:
            return float(alphas[k] + a / (a - b) * (alphas[k + 1] - alphas[k]))
    return None


def parse_grid(s):
    lo, hi, step = (float(v) for v in s.split(":"))
    return np.round(np.arange(lo, hi + 0.5 * step, step), 6)


# ---------------------------------------------------------------- driver

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--class", dest="cls", default="NG2R50")
    ap.add_argument("--alphas", default="0.5:6.0:0.05", help="lo:hi:step, A^3")
    ap.add_argument("--target", type=float, default=None, help="dG0, kcal/mol")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--edge-sd", type=float, default=2.0,
                    help="keep samples this many window widths below the lowest "
                         "centre (default %(default)s; 0 = wham.py's trim)")
    ap.add_argument("--bootstrap", type=int, default=20,
                    help="replicates; each re-solves MBAR (~1 min at 120 windows "
                         "x 4 ns), so run large counts on a compute node; 0 to skip")
    ap.add_argument("--r-bound", type=float, default=None, help="A (default: first barrier)")
    ap.add_argument("--plateau-frac", type=float, default=0.15)
    ap.add_argument("--discard-frac", type=float, default=None)
    ap.add_argument("--min-overlap", type=float, default=MIN_OVERLAP)
    ap.add_argument("--force", action="store_true", help="run despite a failed overlap gate")
    args = ap.parse_args(argv)
    out = args.outdir

    from .config import load_config
    cfg = load_config(os.path.join(out, "config.yaml"))
    if args.cls not in cfg.c4.alpha_by_class:
        raise SystemExit(f"config has no alpha for class {args.cls}")
    alpha0 = float(cfg.c4.alpha_by_class[args.cls])
    discard = cfg.pmf.discard_frac if args.discard_frac is None else args.discard_frac

    meta, rows, missing = load_samples(out, discard)
    if missing:
        print(f"warning: {missing} windows missing or unfinished")
    if len(rows) < 2:
        raise SystemExit("fewer than 2 finished windows")
    centres = np.array([r[0] for r in rows])
    kappas = np.array([r[1] for r in rows], float)
    kT = KB * meta["temperature_K"]

    # --- which pairs carry alpha, and is V a function of the CV?
    lam, cutoff, pairs = alpha_pairs(out, args.cls, alpha0)
    cv_pair = (meta["ion_atom"], meta["ligand_atom"])
    on_cv = [p for p in pairs if {p[0], p[1]} == set(cv_pair)]
    off_cv = [p for p in pairs if {p[0], p[1]} != set(cv_pair)]
    cls_atoms = sorted({p[1] for p in pairs})
    if len(on_cv) != 1:
        raise SystemExit(f"the CV pair {cv_pair} does not carry alpha({args.cls}); "
                         f"V is not a CV function -- per-frame V needed")
    groups = json.load(open(os.path.join(out, "groups.json")))
    counter = set(groups.get("counter_ion", []))
    foreign = [p for p in off_cv if p[0] not in counter]
    if foreign:
        raise SystemExit(f"{len(foreign)} alpha-dependent pairs other than the CV pair "
                         f"and counter-ions (class atoms {cls_atoms}); V depends on "
                         f"coordinates the COLVARs do not hold -- per-frame V needed")
    c_cv = on_cv[0][2]
    V = lambda r: np.where(r < cutoff, c_cv / r ** 4, 0.0)       # kJ/mol per A^3
    print(f"alpha0({args.cls}) = {alpha0}; V(r) = {c_cv / KCAL * 1e4:.1f}/r^4 "
          f"kcal/mol A^4 per A^3 (lambda_c4 {lam}, cutoff {cutoff * 10:.1f} A); "
          f"{len(off_cv)} counter-ion pair(s) off the CV")

    # --- decorrelate, gate, solve
    raw_n = sum(len(r[2]) for r in rows)
    samples, g = decorrelate([r[2] for r in rows])
    print(f"{len(rows)} windows, {raw_n} samples -> {sum(map(len, samples))} "
          f"decorrelated (statistical inefficiency median {np.median(g):.1f}, max {g.max():.1f})")
    ov = adjacent_overlap([r[2] for r in rows], args.bins)
    rejected = None
    if ov.min() < args.min_overlap:
        bad = np.where(ov < args.min_overlap)[0]
        rejected = (f"{len(bad)} adjacent window pair(s) overlap below {args.min_overlap}: "
                    + ", ".join(f"{centres[k] * 10:.2f}|{centres[k + 1] * 10:.2f} A ({ov[k]:.4f})"
                                for k in bad[:10]))
        print(f"REJECTED - {rejected}")
        if not args.force:
            json.dump(dict(rejected=rejected, overlap_min=float(ov.min())),
                      open(os.path.join(out, "mbar.json"), "w"), indent=2)
            raise SystemExit(2)
        print("         --force given: continuing; results are not a connected ensemble")

    x, lw0, f_k = mbar_log_weights(centres, kappas, samples, kT)
    # Upper end: trimmed at the last centre like wham.py (beyond it only one
    # window's tail, which wrecks the plateau reference).  Lower end: extended
    # args.edge_sd window widths below the first centre.  Those samples are
    # window 0's own, properly reweighted, and the bound well sits right
    # there; trimming them (as wham.py does) cost 0.2 kcal/mol at alpha0 and
    # 1.3 at alpha 2 in the synthetic test, vs 0.03 / 0.2 at 2 sd.
    sd0 = np.sqrt(kT / kappas[0])
    edges = np.linspace(centres.min() - args.edge_sd * sd0, centres.max(), args.bins + 1)
    alphas = parse_grid(args.alphas)
    if not np.any(np.isclose(alphas, alpha0)):
        alphas = np.sort(np.append(alphas, alpha0))
    scan = reweight_scan(x, lw0, alphas, alpha0, V, kT, edges, args.r_bound, args.plateau_frac)
    dG = np.array([s["dG0_kcal"] for s in scan])
    a_star = None if args.target is None else crossing(alphas, dG, args.target)
    # V < 0 everywhere, so d dG0 / d alpha = <V>_bound - <V>_plateau < 0: a
    # rise means the estimate jumped (r* landing elsewhere on a broken PMF),
    # not physics.  0.05 kcal/mol slack for noise.
    jumps = [(float(alphas[k]), float(dG[k + 1] - dG[k])) for k in range(len(dG) - 1)
             if dG[k + 1] - dG[k] > 0.05]
    if jumps:
        print(f"WARNING - dG0(alpha) rises at {len(jumps)} grid step(s) (first at alpha "
              f"{jumps[0][0]:.2f}, +{jumps[0][1]:.2f} kcal/mol); it must fall monotonically. "
              f"Check r* (dG0 vs r_bound in mbar_dG0_vs_alpha.csv) or pass --r-bound.")

    # --- bootstrap: resample each window's decorrelated samples
    sem = np.full(len(alphas), np.nan)
    a_star_sem, boots = None, []
    if args.bootstrap:
        rng = np.random.default_rng(0)
        reps, stars = [], []
        for b in range(args.bootstrap):
            res = [s[rng.integers(0, len(s), len(s))] for s in samples]
            xb, lwb, _ = mbar_log_weights(centres, kappas, res, kT, f_init=f_k)
            sb = reweight_scan(xb, lwb, alphas, alpha0, V, kT, edges,
                               args.r_bound, args.plateau_frac)
            reps.append([s["dG0_kcal"] for s in sb])
            if args.target is not None:
                stars.append(crossing(alphas, np.array(reps[-1]), args.target))
        reps = np.array(reps)
        sem = np.nanstd(reps, axis=0, ddof=1)
        stars = [s for s in stars if s is not None]
        if len(stars) > 1:
            a_star_sem = float(np.std(stars, ddof=1))
        if args.target is not None and len(stars) < args.bootstrap:
            print(f"warning: {args.bootstrap - len(stars)} bootstrap replicates "
                  f"never crossed the target inside the alpha grid")

    # --- off-CV (counter-ion) systematic, measured on the SMD trajectory
    i0 = int(np.argmin(np.abs(alphas - alpha0)))
    rb_nm = scan[i0]["r_bound_A"] / 10
    r_pl = centres.min() + (1 - args.plateau_frac) * (centres.max() - centres.min())
    sysd = offcv_systematic(out, off_cv, cutoff, cv_pair, rb_nm, r_pl)
    span = float(np.max(np.abs(alphas - alpha0)))
    sys_max = abs(sysd["delta"]) * span / KCAL
    print(f"off-CV pairs on smd.dcd ({sysd['frames']} frames): <V_off> bound - "
          f"dissociated = {sysd['delta'] / KCAL:+.4f} kcal/mol per A^3 -> up to "
          f"{sys_max:.3f} kcal/mol over this alpha grid"
          + ("  WARNING: not negligible; per-frame V needed" if sys_max > 0.05 else ""))

    # --- report
    print(f"\n{'alpha':>6} {'dG0':>7} {'+/-':>5} {'r_min':>6} {'n_eff':>8} {'trunc':>6}  flags")
    for k, s in enumerate(scan):
        if k % max(1, len(scan) // 25) and not np.isclose(s["alpha"], alpha0):
            continue
        fl = ("EDGE " if s["edge"] else "") + ("alpha0" if np.isclose(s["alpha"], alpha0) else "")
        print(f"{s['alpha']:6.2f} {s['dG0_kcal']:+7.2f} {sem[k]:5.2f} "
              f"{s['r_min_A']:6.2f} {s['n_eff']:8.0f} {s['trunc_kcal']:6.2f}  {fl}")
    # same plateau-flatness check as wham.py.  V is ~0.02 kcal/mol at 10 A, so
    # the slope barely moves with alpha; report it at alpha0 and the target.
    for lab, k in [("alpha0", i0)] + ([("target", int(np.argmin(np.abs(alphas - a_star))))]
                                      if a_star is not None else []):
        if abs(scan[k]["plateau_slope"]) > 0.05:
            print(f"WARNING - at {lab} (alpha {alphas[k]:.2f}) the dissociated end is not "
                  f"flat (slope {scan[k]['plateau_slope']:+.3f} kcal/mol/A); extend "
                  f"pmf.r_end_A or dG0 is referenced to a state that is still interacting")
    pj = os.path.join(out, "pmf.json")
    if os.path.exists(pj):
        b = json.load(open(pj)).get("binding")
        if b and b.get("dG0_kcal") is not None:
            print(f"cross-check: WHAM dG0(alpha0) = {b['dG0_kcal']:+.2f}, MBAR binned "
                  f"{scan[i0]['dG0_binned_kcal']:+.2f} (like-for-like), MBAR sample-summed "
                  f"{scan[i0]['dG0_kcal']:+.2f} kcal/mol")
    star_edge = None
    if args.target is not None:
        if a_star is None:
            print(f"\ntarget {args.target:+.2f} kcal/mol not reached for alpha in "
                  f"[{alphas[0]}, {alphas[-1]}] (dG0 spans {np.nanmin(dG):+.2f} .. {np.nanmax(dG):+.2f})")
        else:
            k = int(np.argmin(np.abs(alphas - a_star)))
            star_edge = scan[k]["edge"]
            err = f" +/- {a_star_sem:.3f}" if a_star_sem else ""
            print(f"\ntarget dG0 = {args.target:+.2f} kcal/mol at alpha({args.cls}) = "
                  f"{a_star:.3f}{err} A^3")
            if star_edge:
                print(f"WARNING - at that alpha ~{scan[k]['trunc_kcal']:.2f} kcal/mol of the bound "
                      "state lies below the lowest sampled r; "
                      "dG0 there is truncated (too weak), so alpha is biased HIGH.  "
                      "Add wall windows below the lowest centre before using it.")

    # --- write
    with open(os.path.join(out, "mbar_dG0_vs_alpha.csv"), "w") as fh:
        fh.write("alpha,dG0_kcal,dG0_sem_kcal,dG0_binned_kcal,r_min_A,r_bound_A,n_eff,"
                 "trunc_kcal,edge,plateau_slope_kcal_per_A\n")
        for k, s in enumerate(scan):
            fh.write(f"{s['alpha']:.4f},{s['dG0_kcal']:.4f},{sem[k]:.4f},{s['dG0_binned_kcal']:.4f},"
                     f"{s['r_min_A']:.4f},{s['r_bound_A']:.4f},{s['n_eff']:.1f},"
                     f"{s['trunc_kcal']:.4f},{int(s['edge'])},{s['plateau_slope']:.4f}\n")
    for a in [alpha0] + ([a_star] if a_star is not None else []):
        k = int(np.argmin(np.abs(alphas - a)))
        np.savetxt(os.path.join(out, f"mbar_pmf_{alphas[k]:.3f}.csv"),
                   np.column_stack([scan[k]["r_A"], scan[k]["Wj"] - scan[k]["Wj"].min()]),
                   delimiter=",", header="r_A,W_jacobian_kcal", comments="")
    json.dump(dict(
        cls=args.cls, alpha0=alpha0, target_kcal=args.target,
        alpha_target=a_star, alpha_target_sem=a_star_sem, alpha_target_edge=star_edge,
        dG0_alpha0_kcal=scan[i0]["dG0_kcal"], dG0_alpha0_sem_kcal=float(sem[i0]),
        plateau_slope_alpha0_kcal_per_A=scan[i0]["plateau_slope"],
        n_windows=len(rows), n_samples_raw=int(raw_n),
        n_samples_decorrelated=int(sum(map(len, samples))),
        g_median=float(np.median(g)), overlap_min=float(ov.min()), rejected=rejected,
        V_coeff_kcal_A4_per_A3=c_cv / KCAL * 1e4, cutoff_A=cutoff * 10,
        offcv=dict(sysd, delta_kcal_per_A3=sysd["delta"] / KCAL, max_over_grid_kcal=sys_max),
        nonmonotonic_steps=jumps, bootstrap=args.bootstrap, edges_A=[float(edges[0] * 10), float(edges[-1] * 10)],
    ), open(os.path.join(out, "mbar.json"), "w"), indent=2)
    print(f"wrote {out}/mbar.json, mbar_dG0_vs_alpha.csv")
    if rejected is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
