#!/usr/bin/env python
"""
MIRIAM-style thermodynamic integration: ion A -> ion B in the presence of
`ligand.count` ligands (or bare, for the water leg).

    python -m lig1264.ti configs/ni_co_imidazole4.yaml --outdir runs/ni_co_imid4 [--resume]

Protocol (config `ti:`):
    build     ion A system + alchemical forces (forces.make_ion_alchemical)
    minimize  same 3 stages as run.py, at lambda = first quadrature node
    heat      NVT 0 -> T over ti.heat_ns
    npt       ti.npt_ns with barostat (MIRIAM: 2 ns)
    windows   n_windows Gauss-Legendre nodes, NVT, ti.ns_per_window each,
              sequential (window i starts from the end of window i-1);
              dU/dlambda sampled every ti.dudl_every_ps -> dudl_wNN.csv,
              and, if ti.traj_ps > 0, coordinates every ti.traj_ps -> window_NN.dcd
    analysis  dG = sum_i w_i <dU/dlambda>_i  (kcal/mol), block-averaged error
              -> ti_results.json

Every stage leaves a checkpoint; --resume continues from the last complete
window.  --window N runs only window N from the NPT state (for array jobs);
then run --analyze once all dudl_*.csv exist.
"""
import argparse
import json
import os
import shutil

import numpy as np
from openmm import app, unit
import openmm as mm

from .config import load_config, REPO_ROOT
from . import build, forces, stages
from .build import log
from .run import get_platform

KCAL = 4.184


def gauss_legendre(n):
    """nodes and weights on [0, 1]; n=12 reproduces Amber's clambda list."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--window", type=int, default=None,
                    help="run only this window (0-based) from the NPT state")
    ap.add_argument("--analyze", action="store_true", help="only integrate existing dudl files")
    ap.add_argument("--stop-after", choices=["build", "minimize", "heat", "npt", "windows"],
                    default="windows")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.ti is None:
        raise SystemExit("config has no `ti:` section")
    ti, md = cfg.ti, cfg.md
    out = args.outdir
    os.makedirs(out, exist_ok=True)
    P = lambda f: os.path.join(out, f)
    lam, wts = gauss_legendre(ti.n_windows)

    if args.analyze:
        return analyze(out, cfg, lam, wts)

    shutil.copy(args.config, P("config.yaml"))
    platform, props = get_platform(args.platform or cfg.platform, args.allow_cpu)
    done = lambda f: args.resume and os.path.exists(P(f))
    csv = str(REPO_ROOT / "12-6-4_base_params.csv")
    pA = forces.ion_params(cfg.ion.element, cfg.ion.charge, csv)
    pB = forces.ion_params(ti.element_b, ti.charge_b, csv)
    log(f"TI {cfg.ion.element}{cfg.ion.charge:+d} -> {ti.element_b}{ti.charge_b:+d}: "
        f"sigma {pA['sigma']:.4f}->{pB['sigma']:.4f} nm, eps {pA['epsilon']:.4f}->{pB['epsilon']:.4f} kJ/mol, "
        f"C4 {pA['c4']/4.184e-4:.0f}->{pB['c4']/4.184e-4:.0f} kcal A^4/mol")
    log(f"{ti.n_windows} windows: " + " ".join(f"{l:.5f}" for l in lam))

    # ---------------- build ----------------
    if done("system.xml"):
        topology, positions, system, groups = build.load_system_files(out)
    else:
        def add_alchemical(system, groups):
            (ion_idx,) = groups["ion"]
            forces.make_ion_alchemical(system, ion_idx, pA, pB)
        topology, positions, system, groups = build.build_system_files(cfg, out, add_alchemical)
    if args.stop_after == "build":
        return

    restraint = forces.find_force(system, forces.RESTRAINT_FORCE_NAME)
    integ = mm.LangevinMiddleIntegrator(md.temperature_K * unit.kelvin,
                                        md.friction_per_ps / unit.picosecond,
                                        ti.dt_fs * unit.femtosecond)
    integ.setRandomNumberSeed(cfg.seed)
    stages.add_barostat(system, cfg)
    baro = next(f for f in system.getForces() if isinstance(f, mm.MonteCarloBarostat))
    baro.setFrequency(0)
    sim = app.Simulation(topology, system, integ, platform, props)
    sim.context.setPositions(positions)
    sim.context.setParameter("lambda_c4", cfg.c4.lambda_c4)
    sim.context.setParameter(forces.LAMBDA_ION, float(lam[0]))

    # md lengths for the shared stages come from the ti section
    cfg.md.min_max_iter = ti.min_max_iter
    cfg.md.heat_ns = ti.heat_ns
    cfg.md.dt_fs = ti.dt_fs

    if done("min_3_all_free.pdb"):
        log("minimize: done, loading min_3_all_free.pdb")
        sim.context.setPositions(app.PDBFile(P("min_3_all_free.pdb")).positions)
    else:
        stages.minimize_3stage(sim, restraint, groups, cfg, out)
    if args.stop_after == "minimize":
        return

    if done("heat.chk"):
        log("heat: done, loading heat.chk")
        sim.loadCheckpoint(P("heat.chk"))
    else:
        stages.heat_nvt(sim, cfg, out)
        sim.saveCheckpoint(P("heat.chk"))
    if args.stop_after == "heat":
        return

    baro.setFrequency(25)
    sim.context.reinitialize(preserveState=True)
    if done("npt.chk"):
        log("npt: done, loading npt.chk")
        sim.loadCheckpoint(P("npt.chk"))
    else:
        stages.run_npt(sim, cfg, out, "equil", ti.npt_ns)
        sim.saveCheckpoint(P("npt.chk"))
    if args.stop_after == "npt":
        return

    # ---------------- windows (NVT) ----------------
    baro.setFrequency(0)
    sim.context.reinitialize(preserveState=True)
    windows = [args.window] if args.window is not None else range(ti.n_windows)
    n_steps = int(round(ti.ns_per_window * 1e6 / ti.dt_fs))
    every = int(round(ti.dudl_every_ps * 1e3 / ti.dt_fs))
    traj_every = int(round(ti.traj_ps * 1e3 / ti.dt_fs)) if ti.traj_ps else 0
    for i in windows:
        chk, csvf = P(f"window_{i:02d}.chk"), P(f"dudl_w{i:02d}.csv")
        if done(chk) and os.path.exists(csvf):
            log(f"window {i}: done")
            if args.window is None:
                sim.loadCheckpoint(chk)
            continue
        if args.window is not None or (args.resume and i > 0 and not os.path.exists(P(f"window_{i-1:02d}.chk"))):
            sim.loadCheckpoint(P("npt.chk"))   # array mode: every window starts from NPT
        sim.context.setParameter(forces.LAMBDA_ION, float(lam[i]))
        log(f"window {i}: lambda={lam[i]:.5f}, {ti.ns_per_window} ns, {n_steps} steps")
        # unwrapped, as for the NPT DCDs: the complex stays whole, and anything
        # involving the bulk (ion-water CN) needs minimum image in analysis anyway
        if traj_every:
            sim.reporters.append(app.DCDReporter(P(f"window_{i:02d}.dcd"), traj_every,
                                                 enforcePeriodicBox=False))
        try:
            with open(csvf, "w") as fh:
                fh.write("# time_ps,dudl_kJmol\n")
                t = 0.0
                for _ in range(n_steps // every):
                    sim.step(every)
                    t += every * ti.dt_fs * 1e-3
                    fh.write(f"{t:.3f},{forces.dudl(sim.context):.6f}\n")
        finally:
            sim.reporters.clear()   # closes the DCD before the next window opens one
        sim.saveCheckpoint(chk)
        # keep a quick running estimate in the log
        d = np.loadtxt(csvf, delimiter=",")[:, 1]
        log(f"window {i}: <dU/dl> = {d[int(ti.discard_frac*len(d)):].mean()/KCAL:.3f} kcal/mol")
    if args.window is None:
        analyze(out, cfg, lam, wts)


def block_sem(x, nblocks=5):
    x = np.asarray(x)
    n = len(x) // nblocks * nblocks
    if n < nblocks:
        return float("nan")
    means = x[:n].reshape(nblocks, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(nblocks))


def analyze(out, cfg, lam, wts):
    ti = cfg.ti
    rows, dG, var = [], 0.0, 0.0
    for i in range(ti.n_windows):
        f = os.path.join(out, f"dudl_w{i:02d}.csv")
        if not os.path.exists(f):
            log(f"missing {f}; dG not computed")
            return
        d = np.loadtxt(f, delimiter=",")[:, 1] / KCAL
        d = d[int(ti.discard_frac * len(d)):]
        m, se = d.mean(), block_sem(d)
        rows.append(dict(window=i, lam=float(lam[i]), weight=float(wts[i]),
                         mean=float(m), sem=se, n=int(len(d))))
        dG += wts[i] * m
        var += (wts[i] * se) ** 2
    res = dict(ion_a=f"{cfg.ion.element}{cfg.ion.charge:+d}",
               ion_b=f"{ti.element_b}{ti.charge_b:+d}",
               n_ligands=cfg.ligand.count, ligand=cfg.ligand.resnames,
               c4_scale=cfg.c4.scale_by_class, alpha=cfg.c4.alpha_by_class,
               dG_kcal=float(dG), dG_sem_kcal=float(np.sqrt(var)), windows=rows)
    json.dump(res, open(os.path.join(out, "ti_results.json"), "w"), indent=2)
    log(f"dG({res['ion_a']} -> {res['ion_b']}, {cfg.ligand.count} ligands) = "
        f"{dG:.3f} +/- {np.sqrt(var):.3f} kcal/mol")
    return res


if __name__ == "__main__":
    main()
