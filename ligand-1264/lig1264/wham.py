#!/usr/bin/env python
"""
WHAM for the umbrella-sampling PMF written by lig1264/pmf.py.

    python -m lig1264.wham runs/pmf/ni_imid4 [--bins 200] [--bootstrap 50]

Reads `windows.json` (centres, force constant, temperature) and the
production COLVARs `window_NNN_colvar.dat`, and writes

    pmf.csv    r_A, W_kcal, W_sem_kcal, W_jacobian_kcal
    pmf.json   well depth, barrier, per-window diagnostics, convergence
    pmf.png    PMF + per-window histograms

Binned WHAM solved by the usual self-consistent iteration, in logs:

    ln P(k) = ln S_i H_i(k)  -  logsumexp_i [ ln n_i + b f_i - b w_i(x_k) ]
    b f_i   = -logsumexp_k [ ln P(k) - b w_i(x_k) ]

with w_i(x) = 1/2 kappa (x - c_i)^2.  W(r) = -kT ln P(r).

W(r) is the PMF in the *radial* coordinate, so it still contains the 2kT ln r
volume term.  `W_jacobian` = W(r) + 2kT ln r removes it, giving the
orientation-averaged pair potential of mean force; use that one for a binding
free energy (and then add the standard-state term), and plain W for the
shape of the dissociation profile.

Errors are block bootstrap over each window's own time series (`--block-ps`),
which captures the correlation within a window but not slow degrees of
freedom orthogonal to the CV -- for a ligand swinging off a metal centre
those are real, so treat the bars as a lower bound.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.special import logsumexp

KCAL = 4.184
KB = 0.008314462618      # kJ/mol/K


def load_windows(outdir, discard_frac=0.0, with_kappa=False):
    """(meta, centres nm, samples[, kappas kJ/mol/nm^2]) for windows with data."""
    meta = json.load(open(os.path.join(outdir, "windows.json")))
    centres, samples, kappas, missing = [], [], [], []
    for row in meta["windows"]:
        f = os.path.join(outdir, f"window_{row['window']:03d}_colvar.dat")
        if not os.path.exists(f):
            missing.append(row["window"])
            continue
        d = np.loadtxt(f)
        if d.ndim == 1:
            d = d[None, :]
        x = d[:, 1]
        x = x[int(discard_frac * len(x)):]
        if len(x) < 10:
            missing.append(row["window"])
            continue
        centres.append(row["centre_nm"])
        samples.append(x)
        # rows appended by pmf.add_windows carry their own kappa
        kappas.append(row.get("kappa_kj_per_nm2", meta["kappa_kj_per_nm2"]))
    if missing:
        print(f"warning: {len(missing)} windows missing or too short: {missing}")
    if with_kappa:
        return meta, np.array(centres), samples, np.array(kappas)
    return meta, np.array(centres), samples


def wham(centres, samples, kappa, kT, bins, tol=1e-7, max_iter=100000):
    """Returns (bin centres nm, W kJ/mol, beta*f per window, iterations).
    kappa: one value, or one per window."""
    allx = np.concatenate(samples)
    edges = np.linspace(allx.min(), allx.max(), bins + 1)
    xk = 0.5 * (edges[:-1] + edges[1:])
    H = np.array([np.histogram(x, bins=edges)[0] for x in samples], float)
    n = np.array([len(x) for x in samples], float)
    # bias of every window evaluated on every bin, in kT
    kappa = np.broadcast_to(np.asarray(kappa, float), centres.shape)
    bw = 0.5 * kappa[:, None] * (xk[None, :] - centres[:, None]) ** 2 / kT
    keep = H.sum(axis=0) > 0
    xk, H, bw = xk[keep], H[:, keep], bw[:, keep]
    log_num = np.log(H.sum(axis=0))
    log_n = np.log(n)
    bf = np.zeros(len(centres))
    for it in range(max_iter):
        log_den = logsumexp(log_n[:, None] + bf[:, None] - bw, axis=0)
        logP = log_num - log_den
        bf_new = -logsumexp(logP[None, :] - bw, axis=1)
        bf_new -= bf_new[0]
        shift = np.max(np.abs(bf_new - bf))
        bf = bf_new
        if shift < tol:
            break
    W = -kT * logP
    return xk, W - W.min(), bf, it + 1


def bootstrap(centres, samples, kappa, kT, bins, xk_ref, n_boot, block_ps, colvar_ps, rng):
    """Block bootstrap: resample contiguous blocks within each window."""
    blk = max(1, int(round(block_ps / colvar_ps)))
    out = []
    for _ in range(n_boot):
        res = []
        for x in samples:
            nb = max(1, len(x) // blk)
            starts = rng.integers(0, max(1, len(x) - blk + 1), size=nb)
            res.append(np.concatenate([x[s:s + blk] for s in starts]))
        xk, W, _, _ = wham(centres, res, kappa, kT, bins)
        out.append(np.interp(xk_ref, xk, W))
    return np.array(out)


def adjacent_overlap(samples, bins, return_order=False):
    """
    Histogram overlap sum_k min(P_i(k), P_i+1(k)) of each pair of adjacent
    windows.  Estimator-independent: a pair near 0 disconnects the window
    graph, and no reweighting scheme -- WHAM or MBAR -- can then place the
    free energies on either side relative to each other.

    Windows are paired in order of their sampled mean, not their centre.  For
    one ladder that is the same order; with a stiff sub-ladder mixed in
    (pmf.add_windows) the soft windows sit F/kappa off their centres, so centre
    order would pair windows that sample far apart and report a gap that the
    data does not have.  return_order=True also returns that ordering (indices
    into samples), so callers can label the pairs.
    """
    order = np.argsort([np.mean(s) for s in samples], kind="stable")
    samples = [samples[k] for k in order]
    allx = np.concatenate(samples)
    edges = np.linspace(allx.min(), allx.max(), bins + 1)
    P = np.array([np.histogram(s, bins=edges)[0] for s in samples], float)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-300, None)
    ov = np.array([np.minimum(P[i], P[i + 1]).sum() for i in range(len(P) - 1)])
    return (ov, order) if return_order else ov


MIN_OVERLAP = 0.03


V_STANDARD_A3 = 1660.5392        # A^3 per particle at 1 M (= 1e27 / 6.02214076e23)


def binding_free_energy(r_A, Wj_kcal, kT_kcal, r_bound_A=None, plateau_frac=0.15):
    """
    Standard-state association free energy from the pair PMF.

        K_eq = int_0^{r*} 4 pi r^2 exp(-w(r)/kT) dr        [A^3]
        dG0  = -kT ln(K_eq / V0),   V0 = 1660.54 A^3

    `Wj_kcal` is W(r) + 2kT ln r -- the orientation-averaged pair PMF w(r) --
    which this function re-zeroes on the dissociated plateau (the last
    `plateau_frac` of the sampled range) whatever shift the caller applied.
    The CV is a bare distance with the ligand free to rotate, so w(r) is
    already orientation-averaged and no restraint correction is needed.

    r* is the edge of the bound state: the first maximum of w(r) past the
    minimum, else the first point where w climbs back to within 0.05
    kcal/mol of the plateau.  The choice matters, so the caller is handed
    dG0 over a range of r* as well.
    """
    npl = max(3, int(plateau_frac * len(r_A)))
    w = Wj_kcal - Wj_kcal[-npl:].mean()
    imin = int(np.argmin(w))
    if r_bound_A is None:
        tail = w[imin:]
        istar = None
        for k in range(1, len(tail) - 1):          # first local maximum
            if tail[k] >= tail[k - 1] and tail[k] > tail[k + 1] and tail[k] > 0:
                istar = imin + k
                break
        if istar is None:                          # else first return to ~0
            above = np.where(tail > -0.05)[0]
            istar = imin + (above[0] if len(above) else len(tail) - 1)
    else:
        istar = int(np.argmin(np.abs(r_A - r_bound_A)))

    integrand = 4 * np.pi * r_A ** 2 * np.exp(-w / kT_kcal)
    K = float(np.trapezoid(integrand[:istar + 1], r_A[:istar + 1]))
    dG = -kT_kcal * np.log(K / V_STANDARD_A3) if K > 0 else float("nan")

    scan = []
    for rb in np.linspace(r_A[imin] + 0.2, min(r_A[-1], r_A[imin] + 4.0), 9):
        k = int(np.argmin(np.abs(r_A - rb)))
        Kk = float(np.trapezoid(integrand[:k + 1], r_A[:k + 1]))
        if Kk > 0:
            scan.append((float(r_A[k]), float(-kT_kcal * np.log(Kk / V_STANDARD_A3))))
    # how flat is the "plateau" really?  a residual slope means r_end is short
    tail_slope = float(np.polyfit(r_A[-npl:], w[-npl:], 1)[0])
    return dict(dG0_kcal=dG, K_eq_A3=K, r_bound_A=float(r_A[istar]),
                r_min_A=float(r_A[imin]), well_depth_w_kcal=float(-w[imin]),
                plateau_slope_kcal_per_A=tail_slope, dG0_vs_rbound=scan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--discard-frac", type=float, default=None,
                    help="of each window COLVAR (default: config pmf.discard_frac)")
    ap.add_argument("--bootstrap", type=int, default=50, help="0 to skip")
    ap.add_argument("--block-ps", type=float, default=100.0)
    ap.add_argument("--zero", choices=["min", "last"], default="min",
                    help="where W is set to 0 ('last' = dissociated plateau)")
    ap.add_argument("--r-bound", type=float, default=None,
                    help="A; bound-state cutoff for the binding integral "
                         "(default: first barrier past the minimum)")
    ap.add_argument("--plateau-frac", type=float, default=0.15,
                    help="fraction of the r range treated as dissociated")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep bins outside the span of the umbrella centres "
                         "(they are unanchored; only for diagnostics)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--min-overlap", type=float, default=MIN_OVERLAP,
                    help="refuse to report dG0 if any adjacent-window overlap "
                         "is below this (default %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="report dG0 even when the overlap gate fails")
    args = ap.parse_args(argv)

    out = args.outdir
    discard = args.discard_frac
    if discard is None:
        try:
            from .config import load_config
            discard = load_config(os.path.join(out, "config.yaml")).pmf.discard_frac
        except Exception:
            discard = 0.0
    meta, centres, samples, kappa = load_windows(out, discard, with_kappa=True)
    kT = KB * meta["temperature_K"]
    colvar_ps = json.load(open(os.path.join(out, "pmf_provenance.json"))
                          )["protocol"].get("colvar_ps", 1.0) \
        if os.path.exists(os.path.join(out, "pmf_provenance.json")) else 1.0

    xk, W, bf, iters = wham(centres, samples, kappa, kT, args.bins)
    print(f"WHAM: {len(centres)} windows, {sum(len(s) for s in samples)} samples, "
          f"{len(xk)} occupied bins, converged in {iters} iterations")

    # Bins outside the span of the umbrella centres are reached only by the
    # sparse tail of a single window: nothing overlaps them, so WHAM cannot
    # anchor them and they come out as noise of order kT.  Left in, they wreck
    # the dissociated-plateau reference that the binding integral needs.
    if not args.no_trim:
        keep = (xk >= centres.min()) & (xk <= centres.max())
        dropped = len(xk) - int(keep.sum())
        xk, W = xk[keep], W[keep] - W[keep].min()
        if dropped:
            print(f"trimmed {dropped} bins outside the umbrella centre span "
                  f"[{centres.min() * 10:.2f}, {centres.max() * 10:.2f}] A")

    # --- diagnostics: neighbouring-window histogram overlap
    ov, order = adjacent_overlap(samples, args.bins, return_order=True)
    print(f"adjacent-window overlap: min {ov.min():.3f}, median {np.median(ov):.3f} "
          f"({(ov < args.min_overlap).sum()} pairs below {args.min_overlap})")
    # Gate: co_imid1_a109 printed a confident dG0 = -0.10 from a window graph
    # split in two (overlap_min 0.0002).  A disconnected graph has no defined
    # free-energy offset across the gap, so there is no dG0 to report.
    rejected = None
    if ov.min() < args.min_overlap:
        bad = np.where(ov < args.min_overlap)[0]
        cs = centres[order] * 10      # centres of the windows, in sampled-mean order
        pairs = ", ".join(f"{cs[k]:.2f}|{cs[k + 1]:.2f} A ({ov[k]:.4f})" for k in bad[:10])
        rejected = (f"{len(bad)} adjacent window pair(s) overlap below "
                    f"{args.min_overlap}: {pairs}{' ...' if len(bad) > 10 else ''}")
        print(f"REJECTED - {rejected}")
        if args.force:
            print("         --force given: reporting dG0 anyway; it is not a "
                  "free energy of a connected ensemble")
            rejected = None

    sem = np.zeros_like(W)
    dG0_sem = None
    if args.bootstrap and rejected is None:
        rng = np.random.default_rng(0)
        B = bootstrap(centres, samples, kappa, kT, args.bins, xk,
                      args.bootstrap, args.block_ps, colvar_ps, rng)
        B = B - B.min(axis=1, keepdims=True)
        sem = B.std(axis=0, ddof=1)

    r_A = xk * 10.0
    W_k = W / KCAL
    Wj = (W + 2 * kT * np.log(xk)) / KCAL
    if args.zero == "last":
        W_k = W_k - W_k[-1]
        Wj = Wj - Wj[-1]
    else:
        W_k, Wj = W_k - W_k.min(), Wj - Wj.min()

    np.savetxt(os.path.join(out, "pmf.csv"),
               np.column_stack([r_A, W_k, sem / KCAL, Wj]), delimiter=",",
               header="r_A,W_kcal,W_sem_kcal,W_jacobian_kcal", comments="")

    if rejected is not None:
        bind = None
        print("binding: dG0 WITHHELD (overlap gate failed; --force to override)")
    else:
        bind = binding_free_energy(r_A, Wj, kT / KCAL, args.r_bound, args.plateau_frac)
    if bind is not None and dG0_sem is None and args.bootstrap:
        # propagate the same replicas through the binding integral, holding r*
        # fixed at the main estimate so the bar reflects sampling noise and
        # not jitter in the cutoff search
        reps = [binding_free_energy(r_A, (B[b] + 2 * kT * np.log(xk)) / KCAL,
                                    kT / KCAL, bind["r_bound_A"],
                                    args.plateau_frac)["dG0_kcal"]
                for b in range(len(B))]
        reps = np.array([v for v in reps if np.isfinite(v)])
        dG0_sem = float(reps.std(ddof=1)) if len(reps) > 1 else None
        bind["dG0_sem_kcal"] = dG0_sem
    if bind is not None:
        err = f" +/- {bind['dG0_sem_kcal']:.2f}" if bind.get("dG0_sem_kcal") else ""
        print(f"binding: dG0 = {bind['dG0_kcal']:+.2f}{err} kcal/mol "
              f"(K_eq = {bind['K_eq_A3']:.1f} A^3, bound state r < "
              f"{bind['r_bound_A']:.2f} A, standard state 1 M = 1660.5 A^3)")
        print(f"         pair-PMF well depth {bind['well_depth_w_kcal']:.2f} kcal/mol "
              f"at {bind['r_min_A']:.2f} A; plateau slope "
              f"{bind['plateau_slope_kcal_per_A']:+.3f} kcal/mol/A")
        if abs(bind["plateau_slope_kcal_per_A"]) > 0.05:
            print("         WARNING - the dissociated end is not flat; extend "
                  "pmf.r_end_A or dG0 is referenced to a state that is still "
                  "interacting")

    imin = int(np.argmin(W_k))
    # same plateau window as binding_free_energy (--plateau-frac), so the
    # well depth and dG0 in pmf.json share one zero (was the last 5% of bins)
    npl = max(3, int(args.plateau_frac * len(W_k)))
    plateau = float(np.mean(W_k[-npl:]))
    res = dict(n_windows=len(centres), n_samples=int(sum(len(s) for s in samples)),
               discard_frac=discard, bins=args.bins, iterations=iters,
               kT_kcal=kT / KCAL,
               overlap_min=float(ov.min()), overlap_median=float(np.median(ov)),
               min_overlap=args.min_overlap,
               overlap_below_min=int((ov < args.min_overlap).sum()),
               r_min_A=float(r_A[imin]),
               well_depth_kcal=float(plateau - W_k[imin]),
               well_depth_sem_kcal=float(np.hypot(sem[imin], sem[-npl:].mean()) / KCAL),
               plateau_kcal=plateau,
               binding=bind,
               rejected=rejected,
               free_energies_kT=[float(v) for v in bf])
    json.dump(res, open(os.path.join(out, "pmf.json"), "w"), indent=2)
    print(f"minimum at {res['r_min_A']:.2f} A; well depth "
          f"{res['well_depth_kcal']:.2f} +/- {res['well_depth_sem_kcal']:.2f} kcal/mol")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1]))
        ax[0].plot(r_A, W_k, lw=1.8, label="W(r)")
        if args.bootstrap:
            ax[0].fill_between(r_A, W_k - sem / KCAL, W_k + sem / KCAL, alpha=0.3)
        ax[0].plot(r_A, Wj, lw=1.0, ls="--", label="W(r) + 2kT ln r")
        ax[0].set_ylabel("PMF (kcal/mol)"); ax[0].legend(); ax[0].grid(alpha=0.3)
        for s in samples:
            ax[1].hist(s * 10, bins=60, histtype="step", lw=0.6)
        ax[1].set_xlabel("ion-ligand distance (A)"); ax[1].set_ylabel("counts")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "pmf.png"), dpi=150)
        print(f"wrote {os.path.join(out, 'pmf.png')}")
    if rejected is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
