#!/usr/bin/env python
"""
Bound-state thermodynamic integration of the 12-6-4 polarizability:
dG_bound(alpha0 -> alpha) for one metal-ligand pair, with alpha on ligand
class `alpha_ti.cls` switched linearly by lambda (forces.make_alpha_alchemical).

    python TI_param/run_TI.py TI_param/configs/co_imid1_alpha_ti.yaml --outdir TI_param/runs/co_imid1 [--resume]

Protocol (config `alpha_ti:`, system blocks as for lig1264.pmf):
    build     same build as pmf.py / ti.py (lig1264.build.build_system_files);
              system.xml is the plain alpha0 system, the lambda force is
              added when each Context is created
    prep      at lambda=0, CV d(ion, cv_atom) held harmonically at the bound
              distance: 3-stage minimize -> NVT heat -> NPT -> npt.xml
    windows   NVT, NO restraint, one fresh Context per window (seed + 1 + i):
              window_equil_ps unrecorded, then ns_per_window with dU/dl and
              the CV every dudl_every_ps -> dudl_wNN.csv, end state window_NN.xml.
              Sequential (window i starts from window i-1); --window N starts
              from npt.xml (SLURM array).
    analyze   alpha_k = alpha0 + lambda_k (alpha1 - alpha0) at every window;
              uniform schedule: cumulative trapezoid TI + MBAR over lambda
              (U is linear in lambda, so dU/dl alone feeds MBAR);
              gauss_legendre: endpoint only.  Per-window fraction of frames
              with CV > bound_r_max_A is reported -- the windows are not
              restrained, so a dissociation is visible instead of silently
              biasing <dU/dl>.
              -> alpha_ti_results.json, alpha_ti_dG_vs_alpha.csv

Unbound-leg check (alpha_ti.cv_walls_A = [lo, hi] A): the windows hold the
CV in that shell (the PMF's plateau) with a flat-bottom restraint, analyze
keeps only frames inside it, and the TI is dG_unbound itself.  It is
compared with plateau_shift (analytic c/r^4 over an r^2-weighted shell) and
with mbar's dG_plateau column, and split into the ion-CV-pair part (c/r^4
on the sampled r) and the rest (counter-ion pairs) -> unbound_check.csv.

Binding cycle: dG0(alpha) = dG0(alpha0) + dG_bound(alpha0->alpha) - dG_unbound,
and dG_unbound ~ 0 (the class atom's arel couples only to the ion, within
the 1 nm cutoff).  Compare with `python -m lig1264.mbar <pmf run>` at the
same alpha values.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ligand-1264"))

import openmm as mm                                          # noqa: E402
from openmm import app, unit                                 # noqa: E402

from lig1264.config import load_config                       # noqa: E402
from lig1264 import build, forces, stages, lambda_windows    # noqa: E402
from lig1264.build import log                                # noqa: E402
from lig1264.run import get_platform                         # noqa: E402
from lig1264.pmf import (pull_pair, kappa_kj_nm2, save_state,  # noqa: E402
                         load_state, sim_cv as _sim_cv, CV)

KCAL = 4.184
KB = 0.008314462618      # kJ/mol/K
# what --seed-from copies: everything load_system_files and class_atoms read
SYSTEM_FILES = ("system.xml", "topology.pdb", "groups.json", "classes.json", "resnames.json")


def sim_cv(sim, i, j):
    """d(i, j) in nm (pmf.sim_cv takes a CV object since centroid CVs)."""
    return _sim_cv(sim, CV(i, [j]))


def schedule(at):
    if at.schedule == "uniform":
        return lambda_windows.uniform(at.n_windows), None
    if at.schedule == "gauss_legendre":
        return lambda_windows.gauss_legendre(at.n_windows)
    raise SystemExit(f"alpha_ti.schedule must be uniform | gauss_legendre, not {at.schedule!r}")


def class_atoms(out, groups, cls):
    classes = json.load(open(os.path.join(out, "classes.json")))
    lig = set(groups["ligand"])
    idx = sorted(int(i) for i, c in classes.items() if c == cls and int(i) in lig)
    if not idx:
        raise SystemExit(f"no ligand atom of class {cls} in {out}/classes.json")
    return idx


def open_sim(cfg, out, platform, props, cls_idx, seed_offset=0, cv_restraint=None,
             walls=None):
    """
    Fresh Context from system.xml + the lambda_alpha C4 force (+ the prep CV
    restraint when given; + the alpha_ti.cv_walls_A flat-bottom shell on
    d(walls) when `walls=(i, j)` and the config sets it).  Barostat added at
    frequency 0; NVT unless the caller switches it on.
    """
    at = cfg.alpha_ti
    topology, positions, system, groups = build.load_system_files(out)
    forces.make_alpha_alchemical(system, cls_idx, at.alpha0, at.alpha1)
    if cv_restraint is not None:
        forces.add_cv_restraint_force(system, *cv_restraint)
    if walls is not None and at.cv_walls_A:
        lo, hi = at.cv_walls_A
        forces.add_cv_walls_force(system, *walls, kappa_kj_nm2(at.cv_walls_kappa),
                                  lo * 0.1, hi * 0.1)
    integ = mm.LangevinMiddleIntegrator(cfg.md.temperature_K * unit.kelvin,
                                        cfg.md.friction_per_ps / unit.picosecond,
                                        at.dt_fs * unit.femtosecond)
    integ.setRandomNumberSeed(cfg.seed + seed_offset)
    stages.add_barostat(system, cfg)
    baro = next(f for f in system.getForces() if isinstance(f, mm.MonteCarloBarostat))
    baro.setRandomNumberSeed(cfg.seed + seed_offset)
    baro.setFrequency(0)
    sim = app.Simulation(topology, system, integ, platform, props)
    sim.context.setPositions(positions)
    sim.context.setParameter("lambda_c4", cfg.c4.lambda_c4)
    sim.context.setParameter(forces.LAMBDA_ALPHA, 0.0)
    return sim, baro, groups


def stage_prep(cfg, out, platform, props, args, cls_idx, i, j):
    at, md = cfg.alpha_ti, cfg.md
    P = lambda f: os.path.join(out, f)
    done = lambda f: args.resume and os.path.exists(P(f))
    if done("npt.xml"):
        log("prep: done, npt.xml present")
        return
    cv_res = None
    if at.prep_restraint_kappa:
        r0 = at.prep_restraint_A or cfg.ion.distance_A
        cv_res = (i, j, kappa_kj_nm2(at.prep_restraint_kappa), r0 * 0.1)
        log(f"prep: CV restrained at {r0:.2f} A ({at.prep_restraint_kappa} kcal/mol/A^2); "
            f"the windows do not see it")
    sim, baro, groups = open_sim(cfg, out, platform, props, cls_idx, cv_restraint=cv_res)
    restraint = forces.find_force(sim.system, forces.RESTRAINT_FORCE_NAME)
    # the shared stage helpers read lengths off cfg.md
    md.min_max_iter, md.dt_fs, md.heat_ns = at.min_max_iter, at.dt_fs, at.heat_ns

    if done("min_3_all_free.pdb"):
        log("minimize: done")
        sim.context.setPositions(app.PDBFile(P("min_3_all_free.pdb")).positions)
    else:
        stages.minimize_3stage(sim, restraint, groups, cfg, out)
    if args.stop_after == "minimize":
        return
    if done("heat.xml"):
        log("heat: done")
        load_state(sim, P("heat.xml"))
    else:
        stages.heat_nvt(sim, cfg, out)
        save_state(sim, P("heat.xml"))
    log(f"prep: CV after heat = {sim_cv(sim, i, j) * 10:.2f} A")
    if args.stop_after == "heat":
        return
    baro.setFrequency(25)
    sim.context.setParameter(baro.Pressure(), md.pressure_bar * unit.bar)
    sim.context.reinitialize(preserveState=True)
    stages.run_npt(sim, cfg, out, "equil", at.npt_ns)
    save_state(sim, P("npt.xml"))
    log(f"prep: CV after NPT = {sim_cv(sim, i, j) * 10:.2f} A")


def stage_relax(cfg, out, platform, props, args, cls_idx, i, j):
    """
    Long NVT run at lambda = --relax-lambda from --start-state (default
    npt.xml) -> relax.xml, a starting state for a hysteresis ladder.  dU/dl
    and the CV are recorded throughout (relax_dudl.csv), so slow drift at
    that alpha shows up directly in the quarter means logged at the end.
    """
    at = cfg.alpha_ti
    P = lambda f: os.path.join(out, f)
    if args.resume and os.path.exists(P("relax.xml")):
        log("relax: done, relax.xml present")
        return
    start = args.start_state or P("npt.xml")
    lam = float(args.relax_lambda)
    sim, _, _ = open_sim(cfg, out, platform, props, cls_idx, seed_offset=100, walls=(i, j))
    load_state(sim, start)
    n_steps = int(round(args.relax_ns * 1e6 / at.dt_fs))
    every = int(round(at.dudl_every_ps * 1e3 / at.dt_fs))
    log(f"relax: lambda={lam:.5f} (alpha={at.alpha0 + lam * (at.alpha1 - at.alpha0):.4f}) "
        f"from {start}, {args.relax_ns} ns NVT")
    d = lambda_windows.run_window(
        sim, P("relax_dudl.csv"), forces.LAMBDA_ALPHA, lam, n_steps, every, at.dt_fs,
        lambda ctx: forces.dudl(ctx, forces.LAMBDA_ALPHA),
        extra=[("cv_A", lambda s: sim_cv(s, i, j) * 10)])
    save_state(sim, P("relax.xml"))
    q = [d[m * len(d) // 4:(m + 1) * len(d) // 4].mean() / KCAL for m in range(4)]
    log("relax: <dU/dl> by quarter = " + ", ".join(f"{x:.3f}" for x in q) + " kcal/mol")


def stage_window(cfg, out, platform, props, args, cls_idx, i, j, k, lam):
    at = cfg.alpha_ti
    P = lambda f: os.path.join(out, f)
    csvf, endf = P(f"dudl_w{k:02d}.csv"), P(f"window_{k:02d}.xml")
    if args.resume and os.path.exists(endf) and os.path.exists(csvf):
        log(f"window {k}: done")
        return
    if args.window is not None or k == 0:
        start = args.start_state or P("npt.xml")
    else:
        start = P(f"window_{k - 1:02d}.xml")
        if not os.path.exists(start):
            raise SystemExit(f"window {k}: {start} missing (sequential mode); "
                             f"run the earlier windows or use --window")
    sim, _, _ = open_sim(cfg, out, platform, props, cls_idx, seed_offset=1 + k, walls=(i, j))
    load_state(sim, start)
    alpha_k = at.alpha0 + lam * (at.alpha1 - at.alpha0)
    sim.context.setParameter(forces.LAMBDA_ALPHA, float(lam))
    n_eq = int(round(at.window_equil_ps * 1e3 / at.dt_fs))
    n_steps = int(round(at.ns_per_window * 1e6 / at.dt_fs))
    every = int(round(at.dudl_every_ps * 1e3 / at.dt_fs))
    traj_every = int(round(at.traj_ps * 1e3 / at.dt_fs)) if at.traj_ps else 0
    log(f"window {k}: lambda={lam:.5f} (alpha={alpha_k:.4f}) from {os.path.basename(start)}, "
        f"{at.window_equil_ps} ps equil + {at.ns_per_window} ns")
    sim.step(n_eq)
    d = lambda_windows.run_window(
        sim, csvf, forces.LAMBDA_ALPHA, lam, n_steps, every, at.dt_fs,
        lambda ctx: forces.dudl(ctx, forces.LAMBDA_ALPHA),
        extra=[("cv_A", lambda s: sim_cv(s, i, j) * 10)],
        traj_path=P(f"window_{k:02d}.dcd"), traj_every=traj_every)
    save_state(sim, endf)
    _, cv = lambda_windows.load_window(csvf, at.discard_frac)
    d = d[int(at.discard_frac * len(d)):]
    if at.cv_walls_A:
        lo, hi = at.cv_walls_A
        occ = f"outside shell {((cv[:, 0] < lo) | (cv[:, 0] > hi)).mean():.3f}"
    else:
        occ = f"unbound frac {(cv[:, 0] > at.bound_r_max_A).mean():.3f}"
    log(f"window {k}: <dU/dl> = {d.mean() / KCAL:.4f} kcal/mol, "
        f"CV mean {cv[:, 0].mean():.2f} A, {occ}")


def plateau_shell(mb, bins, plateau_frac):
    """(lo, hi) A of the plateau shell, exactly as lig1264.mbar.reweight_scan bins it."""
    edges = np.linspace(mb["edges_A"][0], mb["edges_A"][1], bins + 1)
    xk = 0.5 * (edges[:-1] + edges[1:])
    return float(xk[-max(3, int(plateau_frac * len(xk)))]), float(edges[-1])


def pmf_mbar_json(at):
    run = Path(at.pmf_run)
    if not run.is_absolute():
        run = Path(__file__).resolve().parents[1] / run
    return run, json.load(open(run / "mbar.json"))


def plateau_shift(at, alphas, kT_kcal):
    """
    dG_unbound(alpha_k) in kcal/mol: the shift of the dissociated plateau that
    lig1264.mbar applies when it reweights the PMF to alpha.  The plateau shell
    is the last `plateau_frac` of mbar's bins up to the outermost edge
    (reweight_scan's lo / hi); W is flat there, so the shell is r^2-weighted:

        dG_unbound = -kT ln < exp(-beta (alpha - alpha0) c / r^4) >_shell

    c and the edges come from the PMF run's own mbar.json, so the subtraction
    is the same term the reweighting curve contains.  Exact only when V is a
    function of the CV alone (one class atom, the CV partner) -- mbar.py
    refuses otherwise, so a written mbar.json implies it.
    """
    if not at.pmf_run:
        return None
    run = Path(at.pmf_run)
    if not run.is_absolute():
        run = Path(__file__).resolve().parents[1] / run
    mb = json.load(open(run / "mbar.json"))
    if mb["cls"] != at.cls or abs(mb["alpha0"] - at.alpha0) > 1e-9:
        raise SystemExit(f"{run}/mbar.json is for {mb['cls']} at alpha0 {mb['alpha0']}, "
                         f"not {at.cls} at {at.alpha0}")
    # mbar.py with per-frame V (window_NNN_V.dat) also accepts runs where
    # c / r^4 is not V (a second class atom); this analytic shift would be wrong
    if not mb.get("V_cv_function", True) or mb.get("V_coeff_kcal_A4_per_A3") is None:
        raise SystemExit(f"{run}/mbar.json: V is not a function of the CV (v_source "
                         f"{mb.get('v_source')}); take dG_unbound from the "
                         f"dG_plateau_kcal column of mbar_dG0_vs_alpha.csv instead")
    c = mb["V_coeff_kcal_A4_per_A3"]
    # mbar.json keeps only the binned range; mbar.py bins it as
    # linspace(lo, hi, bins + 1) with --bins (default 200, not stored)
    bins = mb.get("bins", 200)
    if "bins" not in mb:
        log(f"plateau: {run.name}/mbar.json has no 'bins'; assuming mbar.py's default {bins}")
    lo, hi = plateau_shell(mb, bins, at.plateau_frac)
    r = np.linspace(lo, hi, 2001)
    w = r ** 2
    shift = []
    for a in alphas:
        u = (a - at.alpha0) * c / r ** 4 / kT_kcal
        shift.append(-kT_kcal * np.log(np.trapezoid(w * np.exp(-u), r) / np.trapezoid(w, r)))
    log(f"plateau correction from {run.name}/mbar.json: V = {c:.1f}/r^4 kcal/mol A^4 per A^3, "
        f"shell {lo:.2f}-{hi:.2f} A; dG_unbound at alpha {alphas[-1]:.2f} = {shift[-1]:+.4f} kcal/mol")
    return np.array(shift), dict(pmf_run=str(run), c=c, shell_A=[float(lo), float(hi)])


def unbound_check(cfg, out, lam, rows, samples, cvs, kT):
    """
    Unbound-leg TI vs the implemented plateau corrections, per alpha_k:
      ti / mbar_lambda   this run: dG_unbound(alpha0 -> alpha_k), all pairs
      ti_cv_pair         TI of the ion-CV-pair term alone, c/r^4 on the
                         sampled r (c from pmf_run's mbar.json)
      ti_rest            ti - ti_cv_pair: the counter-ion pairs
      analytic           plateau_shift: c/r^4, r^2-weighted shell (run_TI)
      pmf_mbar           dG_plateau from pmf_run's mbar_dG0_vs_alpha.csv,
                         interpolated to alpha_k (absent if mbar.py predates it)
    -> unbound_check.csv
    """
    at = cfg.alpha_ti
    if not at.pmf_run:
        raise SystemExit("cv_walls_A needs alpha_ti.pmf_run (the PMF whose plateau is checked)")
    run, mb = pmf_mbar_json(at)
    bins = mb.get("bins", 200)
    lo, hi = plateau_shell(mb, bins, at.plateau_frac)
    if abs(lo - at.cv_walls_A[0]) > 0.01 or abs(hi - at.cv_walls_A[1]) > 0.01:
        log(f"WARNING: cv_walls_A {at.cv_walls_A} differs from {run.name}'s plateau "
            f"shell [{lo:.3f}, {hi:.3f}] A -- not the same ensemble")
    alphas = [r["alpha"] for r in rows]
    dalpha = at.alpha1 - at.alpha0
    c = mb["V_coeff_kcal_A4_per_A3"]                           # kcal/mol A^4 per A^3
    rc = mb["cutoff_A"]
    # dU/dlambda of the ion-CV pair alone, per sample, kcal/mol
    cv_means = np.array([np.mean(dalpha * np.where(r < rc, c / r ** 4, 0.0)) for r in cvs])
    cv_sems = np.array([lambda_windows.block_sem(dalpha * np.where(r < rc, c / r ** 4, 0.0))
                        for r in cvs])
    ti_cv, _ = lambda_windows.cumulative_ti(lam, cv_means, cv_sems)
    analytic, _ = plateau_shift(at, alphas, kT / KCAL)
    pmf_mb = None
    csv = run / "mbar_dG0_vs_alpha.csv"
    if csv.exists():
        t = np.genfromtxt(csv, delimiter=",", names=True)
        if "dG_plateau_kcal" in t.dtype.names:
            pmf_mb = np.interp(alphas, t["alpha"], t["dG_plateau_kcal"])
    if pmf_mb is None:
        log(f"{run.name}/mbar_dG0_vs_alpha.csv has no dG_plateau_kcal column "
            f"(rerun lig1264.mbar there for it); comparing with the analytic shift only")
    with open(os.path.join(out, "unbound_check.csv"), "w") as fh:
        fh.write("alpha,ti_kcal,ti_sem_kcal,mbar_lambda_kcal,ti_cv_pair_kcal,ti_rest_kcal,"
                 "analytic_kcal,pmf_mbar_kcal,ti_minus_analytic_kcal,outside_shell_frac\n")
        for k, r in enumerate(rows):
            pm = float(pmf_mb[k]) if pmf_mb is not None else float("nan")
            fh.write(f"{r['alpha']:.4f},{r['dG_ti_kcal']:.5f},{r['dG_ti_sem_kcal']:.5f},"
                     f"{r['dG_mbar_kcal']:.5f},{ti_cv[k]:.5f},{r['dG_ti_kcal'] - ti_cv[k]:.5f},"
                     f"{analytic[k]:.5f},{pm:.5f},{r['dG_ti_kcal'] - analytic[k]:.5f},"
                     f"{r['outside_shell_frac']:.3f}\n")
            log(f"alpha {r['alpha']:.3f}: dG_unbound TI {r['dG_ti_kcal']:+.4f} +/- "
                f"{r['dG_ti_sem_kcal']:.4f} (MBAR-lambda {r['dG_mbar_kcal']:+.4f}; ion-CV pair "
                f"{ti_cv[k]:+.4f}, rest {r['dG_ti_kcal'] - ti_cv[k]:+.4f})  vs  analytic "
                f"{analytic[k]:+.4f}" + (f", PMF-MBAR {pm:+.4f}" if pmf_mb is not None else "")
                + " kcal/mol")
    return dict(pmf_run=str(run), shell_A=[lo, hi], walls_A=list(at.cv_walls_A),
                ti_kcal=[r["dG_ti_kcal"] for r in rows], ti_cv_pair_kcal=ti_cv.tolist(),
                analytic_kcal=list(map(float, analytic)),
                pmf_mbar_kcal=None if pmf_mb is None else pmf_mb.tolist())


def analyze(cfg, out, lam, wts):
    at = cfg.alpha_ti
    kT = KB * cfg.md.temperature_K
    rows, samples, cvs = [], [], []
    for k in range(len(lam)):
        f = os.path.join(out, f"dudl_w{k:02d}.csv")
        if not os.path.exists(f):
            log(f"missing {f}; dG not computed")
            return None
        d, extra = lambda_windows.load_window(f, at.discard_frac)
        cv = extra[:, 0]
        n_all = len(d)
        if at.cv_walls_A:
            # shell-conditional ensemble: drop frames that leaked past a wall
            inside = (cv >= at.cv_walls_A[0]) & (cv <= at.cv_walls_A[1])
            d, cv = d[inside], cv[inside]
        samples.append(d)
        cvs.append(cv)
        rows.append(dict(window=k, lam=float(lam[k]),
                         alpha=float(at.alpha0 + lam[k] * (at.alpha1 - at.alpha0)),
                         dudl_mean_kcal=float(d.mean() / KCAL),
                         dudl_sem_kcal=lambda_windows.block_sem(d) / KCAL,
                         n=int(len(d)), cv_mean_A=float(cv.mean()), cv_max_A=float(cv.max()),
                         unbound_frac=float((cv > at.bound_r_max_A).mean()),
                         outside_shell_frac=float(1 - len(d) / n_all)))
    bad = [] if at.cv_walls_A else [r["window"] for r in rows
                                    if r["unbound_frac"] > at.max_unbound_frac]
    if bad:
        log(f"WARNING: windows {bad} spent > {at.max_unbound_frac:.0%} of frames with "
            f"CV > {at.bound_r_max_A} A -- the ligand left; their <dU/dl> is not bound-state")
    means = np.array([r["dudl_mean_kcal"] for r in rows])
    sems = np.array([r["dudl_sem_kcal"] for r in rows])
    res = dict(name=cfg.name, ion=f"{cfg.ion.element}{cfg.ion.charge:+d}", cls=at.cls,
               alpha0=at.alpha0, alpha1=at.alpha1, schedule=at.schedule,
               bound_r_max_A=at.bound_r_max_A, windows=rows, unbound_windows=bad)
    if wts is None:
        ti, ti_err = lambda_windows.cumulative_ti(lam, means, sems)
        mb, mb_err, g, overlap = lambda_windows.mbar_over_lambda(lam, samples, kT)
        mb, mb_err = mb / KCAL, mb_err / KCAL
        for r, a, b, c, e, gi in zip(rows, ti, ti_err, mb, mb_err, g):
            r.update(dG_ti_kcal=float(a), dG_ti_sem_kcal=float(b),
                     dG_mbar_kcal=float(c), dG_mbar_sem_kcal=float(e), stat_ineff=float(gi))
        res["adjacent_overlap"] = overlap.tolist()
        res["dG_ti_kcal"], res["dG_mbar_kcal"] = float(ti[-1]), float(mb[-1])
        if at.cv_walls_A:
            res["unbound_check"] = unbound_check(cfg, out, lam, rows, samples, cvs, kT)
            json.dump(res, open(os.path.join(out, "alpha_ti_results.json"), "w"), indent=2)
            return res
        # binding-cycle closure: ddG0(alpha0 -> alpha) = dG_bound - dG_unbound
        pl = plateau_shift(at, [r["alpha"] for r in rows], kT / KCAL)
        if pl is not None:
            shift, res["plateau"] = pl
            for r, s in zip(rows, shift):
                r.update(dG_unbound_kcal=float(s), ddG0_ti_kcal=r["dG_ti_kcal"] - float(s),
                         ddG0_mbar_kcal=r["dG_mbar_kcal"] - float(s))
        with open(os.path.join(out, "alpha_ti_dG_vs_alpha.csv"), "w") as fh:
            # columns after unbound_frac exist only with the plateau correction
            fh.write("alpha,lambda,dG_ti_kcal,dG_ti_sem_kcal,dG_mbar_kcal,dG_mbar_sem_kcal,unbound_frac"
                     + (",dG_unbound_kcal,ddG0_ti_kcal,ddG0_mbar_kcal" if pl is not None else "") + "\n")
            for r in rows:
                fh.write(f"{r['alpha']:.4f},{r['lam']:.5f},{r['dG_ti_kcal']:.4f},"
                         f"{r['dG_ti_sem_kcal']:.4f},{r['dG_mbar_kcal']:.4f},"
                         f"{r['dG_mbar_sem_kcal']:.4f},{r['unbound_frac']:.3f}"
                         + (f",{r['dG_unbound_kcal']:.4f},{r['ddG0_ti_kcal']:.4f},"
                            f"{r['ddG0_mbar_kcal']:.4f}" if pl is not None else "") + "\n")
                log(f"alpha {r['alpha']:.3f}: dG_bound TI {r['dG_ti_kcal']:+.3f} +/- "
                    f"{r['dG_ti_sem_kcal']:.3f}, MBAR {r['dG_mbar_kcal']:+.3f} +/- "
                    f"{r['dG_mbar_sem_kcal']:.3f} kcal/mol"
                    + (f";  - dG_unbound {r['dG_unbound_kcal']:+.4f} -> ddG0 TI "
                       f"{r['ddG0_ti_kcal']:+.3f}" if pl is not None else ""))
        step = (at.alpha1 - at.alpha0) / (len(lam) - 1)
        lo, hi = sorted((at.alpha0, at.alpha1))
        # mbar.parse_grid already makes `hi` inclusive
        log(f"compare: python -m lig1264.mbar <pmf run> --alphas {lo:g}:{hi:g}:{abs(step):g}"
            f"  -> dG0(alpha) - dG0({at.alpha0:g})")
    else:
        dG = float(np.sum(wts * means))
        res["dG_ti_kcal"] = dG
        res["dG_ti_sem_kcal"] = float(np.sqrt(np.sum((wts * sems) ** 2)))
        log(f"dG_bound(alpha {at.alpha0:g} -> {at.alpha1:g}) = {dG:+.3f} +/- "
            f"{res['dG_ti_sem_kcal']:.3f} kcal/mol (Gauss-Legendre)")
    json.dump(res, open(os.path.join(out, "alpha_ti_results.json"), "w"), indent=2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--window", type=int, default=None,
                    help="run only this window (0-based) from npt.xml (SLURM array)")
    ap.add_argument("--analyze", action="store_true", help="only analyze existing dudl files")
    ap.add_argument("--stop-after", choices=["build", "minimize", "heat", "npt", "relax", "windows"],
                    default="windows")
    # hysteresis ladders: same system as another run, different starting state
    ap.add_argument("--seed-from", default=None,
                    help="run dir whose built system (system.xml, topology, groups, "
                         "classes) is copied in, so the System is identical; skips prep")
    ap.add_argument("--start-state", default=None,
                    help="State XML the windows (array mode / window 0) start from "
                         "instead of npt.xml")
    ap.add_argument("--relax-ns", type=float, default=0.0,
                    help="run this long at --relax-lambda first -> relax.xml")
    ap.add_argument("--relax-lambda", type=float, default=1.0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.alpha_ti is None:
        raise SystemExit("config has no `alpha_ti:` section")
    at = cfg.alpha_ti
    out = args.outdir
    os.makedirs(out, exist_ok=True)
    if args.analyze:
        # the run's own copy, so lambdas match what was simulated
        cfg_saved = os.path.join(out, "config.yaml")
        if os.path.exists(cfg_saved):
            cfg = load_config(cfg_saved)
        analyze(cfg, out, *schedule(cfg.alpha_ti))
        return
    lam, wts = schedule(at)

    if not (args.window is not None and os.path.exists(os.path.join(out, "config.yaml"))):
        shutil.copy(args.config, os.path.join(out, "config.yaml"))
    platform, props = get_platform(args.platform or cfg.platform, args.allow_cpu)
    log(f"{cfg.name}: alpha({at.cls}) {at.alpha0:g} -> {at.alpha1:g}, {at.n_windows} "
        f"{at.schedule} windows: " + " ".join(f"{l:.4f}" for l in lam))

    if args.seed_from:
        for f in SYSTEM_FILES:
            dst = os.path.join(out, f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(args.seed_from, f), dst)
        log(f"system copied from {args.seed_from}")
    if args.start_state and not os.path.exists(args.start_state):
        raise SystemExit(f"--start-state {args.start_state} does not exist")

    if os.path.exists(os.path.join(out, "system.xml")) and (
            args.resume or args.window is not None or args.seed_from):
        topology, positions, _, groups = build.load_system_files(out)
    else:
        topology, positions, _, groups = build.build_system_files(cfg, out)
    if args.stop_after == "build":
        return
    cls_idx = class_atoms(out, groups, at.cls)
    xyz = np.array(positions.value_in_unit(unit.nanometer))
    i, j = pull_pair(topology, xyz, groups,
                     SimpleNamespace(pull_atom=at.cv_atom, pull_resid=at.cv_resid))
    log(f"alpha-TI atoms ({at.cls}): {cls_idx}")
    json.dump(dict(cv_pair=[i, j], class_atoms=cls_idx, lambdas=lam.tolist()),
              open(os.path.join(out, "alpha_ti_setup.json"), "w"), indent=2)

    if args.window is not None:
        if not args.start_state and not os.path.exists(os.path.join(out, "npt.xml")):
            raise SystemExit(f"{out}/npt.xml missing: run --stop-after npt first")
        stage_window(cfg, out, platform, props, args, cls_idx, i, j, args.window, lam[args.window])
        return

    if args.seed_from:
        log("prep: skipped (--seed-from)")
    else:
        stage_prep(cfg, out, platform, props, args, cls_idx, i, j)
    if args.stop_after in ("minimize", "heat", "npt"):
        return
    if args.relax_ns:
        stage_relax(cfg, out, platform, props, args, cls_idx, i, j)
    if args.stop_after == "relax":
        return
    for k, l in enumerate(lam):
        stage_window(cfg, out, platform, props, args, cls_idx, i, j, k, l)
    analyze(cfg, out, lam, wts)


if __name__ == "__main__":
    main()
