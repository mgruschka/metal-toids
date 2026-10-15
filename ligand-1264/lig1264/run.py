#!/usr/bin/env python
"""
Run one ion/ligand system end to end.

    python -m lig1264.run configs/zn_peptoid.yaml --outdir runs/zn_peptoid [--resume]

Stages and the files that mark them done (used by --resume):
    build     system.xml, topology.pdb, groups.json
    minimize  min_3_all_free.pdb
    heat      heat_final.pdb, heat.chk
    equil     npt_equil_final.pdb, npt_equil.chk
    prod      npt_prod_final.pdb
"""
import argparse
import json
import os
import shutil
import sys

from openmm import app, unit
import openmm as mm

from .config import load_config
from . import build, forces, stages
from .build import log


def get_platform(name, allow_cpu=False):
    plat = mm.Platform.getPlatformByName(name)
    props = {}
    if name == "CUDA":
        props = {"Precision": "mixed"}
    elif name == "CPU" and not allow_cpu:
        raise SystemExit("refusing CPU platform without --allow-cpu")
    return plat, props


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--platform", default=None, help="override config.platform")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", choices=["build", "minimize", "heat", "equil", "prod"],
                    default="prod")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = args.outdir
    os.makedirs(out, exist_ok=True)
    shutil.copy(args.config, os.path.join(out, "config.yaml"))
    platform, props = get_platform(args.platform or cfg.platform, args.allow_cpu)
    log(f"{cfg.name}: platform {platform.getName()} {props}")

    done = lambda f: args.resume and os.path.exists(os.path.join(out, f))
    P = lambda f: os.path.join(out, f)

    # ---------------- build ----------------
    if done("system.xml"):
        topology, positions, system, groups = build.load_system_files(out)
    else:
        topology, positions, system, groups = build.build_system_files(cfg, out)
    if args.stop_after == "build":
        return

    restraint = forces.find_force(system, forces.RESTRAINT_FORCE_NAME)
    md = cfg.md
    integ = mm.LangevinMiddleIntegrator(md.temperature_K * unit.kelvin,
                                        md.friction_per_ps / unit.picosecond,
                                        md.dt_fs * unit.femtosecond)
    integ.setRandomNumberSeed(cfg.seed)

    # barostat is added before Simulation() so one Context serves every
    # stage; it is switched off (pressure coupling frequency 0) until NPT
    stages.add_barostat(system, cfg)
    baro = next(f for f in system.getForces() if isinstance(f, mm.MonteCarloBarostat))
    baro.setFrequency(0)

    sim = app.Simulation(topology, system, integ, platform, props)
    sim.context.setPositions(positions)
    sim.context.setParameter("lambda_c4", cfg.c4.lambda_c4)

    # ---------------- minimize ----------------
    if done("min_3_all_free.pdb"):
        log("minimize: done, loading min_3_all_free.pdb")
        sim.context.setPositions(app.PDBFile(P("min_3_all_free.pdb")).positions)
    else:
        stages.minimize_3stage(sim, restraint, groups, cfg, out)
    if args.stop_after == "minimize":
        return

    # ---------------- heat (NVT) ----------------
    if done("heat.chk"):
        log("heat: done, loading heat.chk")
        sim.loadCheckpoint(P("heat.chk"))
    else:
        stages.heat_nvt(sim, cfg, out)
        sim.saveCheckpoint(P("heat.chk"))
    if args.stop_after == "heat":
        return

    # ---------------- NPT ----------------
    baro.setFrequency(25)
    sim.context.setParameter(baro.Pressure(), md.pressure_bar * unit.bar)
    sim.context.reinitialize(preserveState=True)

    if done("npt_equil_final.pdb"):
        log("equil: done, loading npt_equil.chk")
        sim.loadCheckpoint(P("npt_equil.chk"))
    else:
        stages.run_npt(sim, cfg, out, "equil", md.npt_equil_ns)
    if args.stop_after == "equil":
        return

    if done("npt_prod_final.pdb"):
        log("prod: already done")
    else:
        stages.run_npt(sim, cfg, out, "prod", md.npt_prod_ns)
    log("finished")


if __name__ == "__main__":
    main()
