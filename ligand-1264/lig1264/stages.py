"""
The three MD stages.  Each takes a live Simulation and leaves the Context
in the state the next stage expects; run.py handles checkpoints / resume.

  minimize_3stage : (1) solute (ligand + all ions) restrained, water relaxes
                    (2) everything except ligand hydrogens restrained
                    (3) all free
  heat_nvt        : T ramped 0 -> T_target in `heat_steps` increments
  run_npt         : MonteCarloBarostat added, equil then production
"""
import os
import numpy as np
from openmm import app, unit
import openmm as mm

from . import forces
from .build import log


def _steps(ns, dt_fs):
    return int(round(ns * 1e6 / dt_fs))


def _energy(sim):
    e = sim.context.getState(getEnergy=True).getPotentialEnergy()
    return e.value_in_unit(unit.kilojoule_per_mole)


def minimize_3stage(sim, restraint, groups, cfg, outdir):
    md = cfg.md
    all_idx = set(range(sim.topology.getNumAtoms()))
    solute = groups["ligand"] + groups["ion"] + groups["counter_ion"]
    stages = [
        ("1_solute_restrained", solute, md.min_max_iter[0]),
        ("2_ligandH_only",      sorted(all_idx - set(groups["ligand_H"])), md.min_max_iter[1]),
        ("3_all_free",          [], md.min_max_iter[2]),
    ]
    for tag, restrained, max_iter in stages:
        pos = sim.context.getState(getPositions=True).getPositions()
        forces.set_restrained(sim.context, restraint, restrained, md.min_restraint_k, pos)
        e0 = _energy(sim)
        sim.minimizeEnergy(maxIterations=max_iter)
        e1 = _energy(sim)
        log(f"min {tag}: {len(restrained)} restrained, {e0:.1f} -> {e1:.1f} kJ/mol")
        _write_pdb(sim, os.path.join(outdir, f"min_{tag}.pdb"))
    forces.set_restrained(sim.context, restraint, [], 0.0)


def heat_nvt(sim, cfg, outdir):
    md = cfg.md
    n_total = _steps(md.heat_ns, md.dt_fs)
    per = n_total // md.heat_steps
    temps = np.linspace(0.0, md.temperature_K, md.heat_steps + 1)[1:]
    sim.context.setVelocitiesToTemperature(1.0 * unit.kelvin, cfg.seed)
    sim.reporters.append(app.StateDataReporter(
        os.path.join(outdir, "heat.csv"), _steps(md.report_ps * 1e-3, md.dt_fs),
        step=True, time=True, temperature=True, potentialEnergy=True))
    log(f"NVT heating 0 -> {md.temperature_K} K over {md.heat_ns} ns "
        f"({md.heat_steps} x {per} steps)")
    for T in temps:
        sim.integrator.setTemperature(T * unit.kelvin)
        sim.step(per)
    sim.reporters.clear()
    _write_pdb(sim, os.path.join(outdir, "heat_final.pdb"))


def run_npt(sim, cfg, outdir, phase, ns):
    """phase = 'equil' | 'prod'; reporters are phase-specific files."""
    md = cfg.md
    n = _steps(ns, md.dt_fs)
    every = _steps(md.report_ps * 1e-3, md.dt_fs)
    sim.reporters.append(app.StateDataReporter(
        os.path.join(outdir, f"npt_{phase}.csv"), every, step=True, time=True,
        temperature=True, density=True, volume=True, potentialEnergy=True, speed=True))
    sim.reporters.append(app.DCDReporter(
        os.path.join(outdir, f"npt_{phase}.dcd"), every, enforcePeriodicBox=False))
    sim.reporters.append(app.CheckpointReporter(
        os.path.join(outdir, f"npt_{phase}.chk"), _steps(md.checkpoint_ps * 1e-3, md.dt_fs)))
    log(f"NPT {phase}: {ns} ns ({n} steps)")
    sim.step(n)
    sim.reporters.clear()
    _write_pdb(sim, os.path.join(outdir, f"npt_{phase}_final.pdb"))


def add_barostat(system, cfg):
    md = cfg.md
    system.addForce(mm.MonteCarloBarostat(md.pressure_bar * unit.bar,
                                          md.temperature_K * unit.kelvin, 25))


def _write_pdb(sim, path):
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(path, "w") as fh:
        app.PDBFile.writeFile(sim.topology, st.getPositions(), fh)
