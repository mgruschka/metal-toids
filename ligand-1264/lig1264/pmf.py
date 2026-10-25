#!/usr/bin/env python
"""
PLUMED/OpenMM potential of mean force for ion <-> ligand dissociation.

    python -m lig1264.pmf configs/ni_imid4_pmf.yaml --outdir runs/pmf/ni_imid4 [--resume]
    python -m lig1264.pmf <cfg> --outdir <dir> --window 57        # one umbrella window
    python -m lig1264.wham <dir>                                  # PMF from the COLVARs
    python -m lig1264.pmf <cfg> --outdir <dir> --add-windows 2.2 2.95 16 300
                                      # append a stiff sub-ladder (lo hi n kappa)

Same system setup as run.py / ti.py (build.build_system_files), then:

    minimize  three stages (water only / non-ligand-H / everything free)
    heat      NVT 0 -> T over pmf.heat_ns
    equil     NPT, pmf.npt_equil_steps
    prod      NPT, pmf.npt_prod_steps
              minimize/heat/equil/prod all run under a harmonic restraint on
              the CV (pmf.prep_restraint_kappa) so a weakly bound ion cannot
              drift off before the pull; it is dropped for everything below,
              so no free energy is ever computed under it
    smd       steered MD: PLUMED MOVINGRESTRAINT pulls d(ion, NE2) from the
              bound distance out to pmf.r_end_A over pmf.smd_ns (Jarzynski
              work recorded in smd_colvar.dat); -> smd.dcd
    seed      pmf.n_windows frames picked from smd.dcd, one per umbrella
              centre, evenly spaced in the CV -> windows.json
    windows   per window: pmf.window_equil_ns NPT equilibration then
              pmf.window_prod_ns NPT production, both under a PLUMED
              harmonic RESTRAINT at that centre -> window_NNN_colvar.dat
    wham      lig1264/wham.py

Deviations from the AMBER protocol this reproduces, all forced by OpenMM and
all recorded in pmf_provenance.json:

  * Minimizer.  OpenMM has only L-BFGS, no steepest-descent/conjugate-gradient
    pair, so "10,000 SD + 10,000 CG" per stage maps onto maxIterations=20000.
  * Barostat.  OpenMM ships no Berendsen barostat (it exposes no virial, so
    one cannot be written either).  MonteCarloBarostat is used instead; it
    samples a rigorous NPT ensemble, unlike Berendsen, but its relaxation is
    set by the MC move frequency rather than a 5 ps time constant.
SHAKE is *not* one of them.  OpenMM does implement SHAKE: on the CUDA
platform IntegrationUtilities partitions the constraints into SETTLE (rigid
water), SHAKE (a heavy atom with its attached hydrogens) and CCMA (anything
left over).  With constraints=HBonds + rigidWater=True every constraint in
this system is one of the first two, so the GPU runs use SHAKE and SETTLE,
as AMBER does.  The Reference and CPU platforms have no SHAKE path and fall
back to CCMA, so a CPU test run differs numerically from a CUDA production
run -- same constrained degrees of freedom, different solver, both to the
integrator's 1e-5 constraint tolerance.

Everything else -- 2 fs timestep, 10 A cutoff, Langevin thermostat at
1 ps^-1, 300 K -- is as specified.
"""
import argparse
import json
import os
import shutil

import numpy as np
from openmm import app, unit
from openmm.app.internal import unitcell
import openmm as mm

from .config import load_config
from . import build, forces, stages
from .build import log
from .run import get_platform

KCAL = 4.184
KB_KCAL = 0.008314462618 / KCAL     # kcal/mol/K

# Switching the barostat on calls Context.reinitialize(), which rebuilds the
# PlumedForce; PLUMED then re-opens its PRINT file and renames the previous
# one to bck.0.<name>.  Two stray (empty) backups per window x 120 windows is
# just clutter -- the live file holds the post-reinitialize data either way.
os.environ.setdefault("PLUMED_MAXBACKUP", "0")


def kappa_kj_nm2(kcal_per_A2):
    """kcal/mol/A^2 -> kJ/mol/nm^2 (PLUMED's units for a distance RESTRAINT)."""
    return kcal_per_A2 * KCAL * 100.0


# ---------------------------------------------------------------- CV selection

def pull_pair(topology, xyz_nm, groups, pmf):
    """
    (ion index, ligand atom index) for the CV, 0-based.

    The ligand copy is `pmf.pull_resid` if given, otherwise the one whose
    `pmf.pull_atom` sits closest to the ion -- i.e. a bound one.
    """
    ion = groups["ion"][0]
    lig = set(groups["ligand"])
    cands = [a for a in topology.atoms()
             if a.index in lig and a.name == pmf.pull_atom]
    if pmf.pull_resid is not None:
        cands = [a for a in cands if str(a.residue.id) == str(pmf.pull_resid)]
    if not cands:
        raise SystemExit(f"pmf.pull_atom={pmf.pull_atom!r} "
                         f"(resid {pmf.pull_resid}) matched no ligand atom")
    d = [np.linalg.norm(xyz_nm[a.index] - xyz_nm[ion]) for a in cands]
    pick = cands[int(np.argmin(d))]
    log(f"CV: d(ion atom {ion}, {pick.name} of {pick.residue.name}{pick.residue.id} "
        f"atom {pick.index}) = {min(d) * 10:.2f} A"
        + ("" if len(cands) == 1 else f"  [closest of {len(cands)} copies]"))
    return ion, pick.index


# ---------------------------------------------------------------- PLUMED input

PLUMED_HEADER = "UNITS LENGTH=nm ENERGY=kj/mol TIME=ps\n"


def fresh_colvar(path):
    """
    Remove a stale COLVAR and PLUMED's bck.N.* copies of it.

    PLUMED renames an existing output file to bck.0.<name> rather than
    overwriting, so a requeued window would otherwise leave one backup per
    attempt -- 120 windows makes that a mess, and the backups are never read.
    """
    d, base = os.path.dirname(path), os.path.basename(path)
    for f in [path] + [os.path.join(d, n) for n in os.listdir(d or ".")
                       if n.startswith("bck.") and n.endswith(base)]:
        if os.path.exists(f):
            os.remove(f)


def smd_script(i, j, r0_nm, r1_nm, n_steps, kappa, colvar, stride):
    """MOVINGRESTRAINT pull; PLUMED atom numbering is 1-based."""
    return (PLUMED_HEADER +
            f"d: DISTANCE ATOMS={i + 1},{j + 1}\n"
            f"smd: MOVINGRESTRAINT ARG=d "
            f"STEP0=0 AT0={r0_nm:.5f} KAPPA0={kappa:.3f} "
            f"STEP1={n_steps} AT1={r1_nm:.5f} KAPPA1={kappa:.3f}\n"
            f"PRINT ARG=d,smd.work,smd.bias STRIDE={stride} FILE={colvar}\n"
            f"FLUSH STRIDE={stride}\n")


def umbrella_script(i, j, centre_nm, kappa, colvar, stride):
    return (PLUMED_HEADER +
            f"d: DISTANCE ATOMS={i + 1},{j + 1}\n"
            f"res: RESTRAINT ARG=d AT={centre_nm:.5f} KAPPA={kappa:.3f}\n"
            f"PRINT ARG=d,res.bias STRIDE={stride} FILE={colvar}\n"
            f"FLUSH STRIDE={stride}\n")


# ---------------------------------------------------------------- state I/O

def save_state(sim, path):
    """
    Positions + velocities + box as a State XML.  Checkpoints are tied to the
    exact Force layout, and the umbrella/SMD Contexts carry an extra
    PlumedForce, so states -- not checkpoints -- move between stages.
    """
    st = sim.context.getState(getPositions=True, getVelocities=True)
    with open(path, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(st))


def load_state(sim, path):
    st = mm.XmlSerializer.deserialize(open(path).read())
    sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    sim.context.setPositions(st.getPositions())
    sim.context.setVelocities(st.getVelocities())


def _steps(ns, dt_fs):
    return int(round(ns * 1e6 / dt_fs))


# ---------------------------------------------------------------- simulation

def alpha_deriv_spec(cfg, out):
    """(class atoms, alpha0) for open_sim's alpha_deriv, or None if v_class is off."""
    cls = cfg.pmf.v_class
    if not cls:
        return None
    if cls not in cfg.c4.alpha_by_class:
        raise SystemExit(f"pmf.v_class = {cls} needs c4.alpha_by_class[{cls}] (alpha0)")
    classes = json.load(open(os.path.join(out, "classes.json")))
    atoms = sorted(int(i) for i, c in classes.items() if c == cls)
    if not atoms:
        raise SystemExit(f"pmf.v_class: no atom of class {cls} in {out}/classes.json")
    return atoms, cfg.c4.alpha_by_class[cls]


class DUDAlphaReporter:
    """
    time_ps and V = dU/dalpha (kJ/mol per A^3) every `interval` steps.  The C4
    force is rebuilt with alpha1 = alpha0 + 1 A^3, so dU/dlambda_alpha is V
    itself.  Rows are keyed by time so they join with the PLUMED COLVAR.
    """
    def __init__(self, path, interval, dt_fs):
        self._fh = open(path, "w")
        self._fh.write("# time_ps  V_kJ_per_mol_per_A3  (dU/dalpha, lambda_alpha = 0)\n")
        self._interval, self._dt_ps = interval, dt_fs * 1e-3

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return dict(steps=steps, periodic=None, include=["parameterDerivatives"])

    def report(self, simulation, state):
        v = state.getEnergyParameterDerivatives()[forces.LAMBDA_ALPHA]
        self._fh.write(f"{simulation.currentStep * self._dt_ps:.3f} {v:.8g}\n")

    def close(self):
        self._fh.close()


def open_sim(cfg, out, platform, props, plumed=None, seed_offset=0,
             cv_restraint=None, alpha_deriv=None):
    """
    Fresh Context from system.xml (+ a PlumedForce when `plumed` is given,
    + a harmonic CV restraint when `cv_restraint=(i, j, k_kj_nm2, r0_nm)`,
    + a dU/dalpha derivative on LJ1264C4Force when `alpha_deriv=(atoms,
    alpha0)`, at lambda_alpha = 0 so energies and forces are unchanged).
    The barostat is added with frequency 0; callers switch it on for NPT.
    """
    topology, positions, system, groups = build.load_system_files(out)
    pmf = cfg.pmf
    if plumed is not None:
        from openmmplumed import PlumedForce
        system.addForce(PlumedForce(plumed))
    if cv_restraint is not None:
        forces.add_cv_restraint_force(system, *cv_restraint)
    if alpha_deriv is not None:
        atoms, alpha0 = alpha_deriv
        forces.make_alpha_alchemical(system, atoms, alpha0, alpha0 + 1.0)
    integ = mm.LangevinMiddleIntegrator(cfg.md.temperature_K * unit.kelvin,
                                        cfg.md.friction_per_ps / unit.picosecond,
                                        pmf.dt_fs * unit.femtosecond)
    integ.setRandomNumberSeed(cfg.seed + seed_offset)
    stages.add_barostat(system, cfg)
    baro = next(f for f in system.getForces() if isinstance(f, mm.MonteCarloBarostat))
    baro.setRandomNumberSeed(cfg.seed + seed_offset)
    baro.setFrequency(0)
    sim = app.Simulation(topology, system, integ, platform, props)
    sim.context.setPositions(positions)
    sim.context.setParameter("lambda_c4", cfg.c4.lambda_c4)
    if alpha_deriv is not None:
        sim.context.setParameter(forces.LAMBDA_ALPHA, 0.0)
    return sim, baro, topology, groups


def set_npt(sim, baro, cfg, on=True):
    baro.setFrequency(25 if on else 0)
    sim.context.setParameter(baro.Pressure(), cfg.md.pressure_bar * unit.bar)
    sim.context.reinitialize(preserveState=True)


def run_md(sim, out, tag, n_steps, dt_fs, report_ps, traj_ps=0.0, hook=None,
           extra_reporters=()):
    """
    NPT segment with StateData (+ optional DCD, + `extra_reporters`)
    reporters, cleanly closed.
    `hook=(n, fn)` calls fn() after the first n steps, mid-segment, so an
    early check sees the live COLVAR without splitting the reporter files.
    """
    sim.reporters.extend(extra_reporters)
    every = max(1, int(round(report_ps * 1e3 / dt_fs)))
    sim.reporters.append(app.StateDataReporter(
        os.path.join(out, f"{tag}.csv"), every, step=True, time=True,
        temperature=True, density=True, volume=True, potentialEnergy=True, speed=True))
    if traj_ps:
        sim.reporters.append(app.DCDReporter(
            os.path.join(out, f"{tag}.dcd"), max(1, int(round(traj_ps * 1e3 / dt_fs))),
            enforcePeriodicBox=False))
    log(f"{tag}: {n_steps} steps ({n_steps * dt_fs * 1e-6:.3f} ns)")
    try:
        if hook is not None and 0 < hook[0] < n_steps:
            sim.step(hook[0])
            hook[1]()
            sim.step(n_steps - hook[0])
        else:
            sim.step(n_steps)
    finally:
        sim.reporters.clear()   # closes the DCD before the next stage opens one
        for r in extra_reporters:
            r.close()


# ---------------------------------------------------------------- stages

def prep_restraint(cfg, i, j):
    """(i, j, k, r0) for open_sim's cv_restraint, or None if disabled."""
    pmf = cfg.pmf
    k = pmf.prep_restraint_kappa
    if not k:
        return None
    r0_A = pmf.prep_restraint_A or pmf.r_start_A or cfg.ion.distance_A
    return i, j, kappa_kj_nm2(k), r0_A * 0.1


def stage_prep(cfg, out, platform, props, args, i, j):
    """
    minimize -> heat -> NPT equil -> NPT prod.  Leaves prod.xml.

    Unbiased except for the harmonic CV restraint (pmf.prep_restraint_kappa),
    which keeps the ion on the ligand so the SMD starts from the bound state.
    """
    pmf, md = cfg.pmf, cfg.md
    P = lambda f: os.path.join(out, f)
    done = lambda f: args.resume and os.path.exists(P(f))
    if done("prod.xml"):
        log("prep: done, prod.xml present")
        return

    cv_res = prep_restraint(cfg, i, j)
    if cv_res is None:
        log("prep: WARNING - no CV restraint (pmf.prep_restraint_kappa = 0); "
            "a weakly bound ion may dissociate before the pull starts")
    else:
        log(f"prep: CV restrained at {cv_res[3] * 10:.2f} A "
            f"({pmf.prep_restraint_kappa} kcal/mol/A^2) through minimize/heat/"
            f"equil/prod; the SMD and the windows do not see it")
    sim, baro, topology, groups = open_sim(cfg, out, platform, props,
                                           cv_restraint=cv_res)
    restraint = forces.find_force(sim.system, forces.RESTRAINT_FORCE_NAME)

    # the shared stage helpers read lengths off cfg.md
    md.min_max_iter, md.dt_fs, md.heat_ns = pmf.min_max_iter, pmf.dt_fs, pmf.heat_ns

    if done("min_3_all_free.pdb"):
        log("minimize: done")
        sim.context.setPositions(app.PDBFile(P("min_3_all_free.pdb")).positions)
    else:
        stages.minimize_3stage(sim, restraint, groups, cfg, out)

    if done("heat.xml"):
        log("heat: done")
        load_state(sim, P("heat.xml"))
    else:
        stages.heat_nvt(sim, cfg, out)
        save_state(sim, P("heat.xml"))
    log(f"prep: CV after heat = {sim_cv(sim, i, j) * 10:.2f} A")

    set_npt(sim, baro, cfg, True)
    if done("equil.xml"):
        log("equil: done")
        load_state(sim, P("equil.xml"))
    else:
        run_md(sim, out, "equil", pmf.npt_equil_steps, pmf.dt_fs,
               md.report_ps, pmf.traj_ps)
        save_state(sim, P("equil.xml"))
    log(f"prep: CV after equil = {sim_cv(sim, i, j) * 10:.2f} A")
    run_md(sim, out, "prod", pmf.npt_prod_steps, pmf.dt_fs, md.report_ps, pmf.traj_ps)
    save_state(sim, P("prod.xml"))
    log(f"prep: CV after prod = {sim_cv(sim, i, j) * 10:.2f} A")


# ---------------------------------------------------------------- SMD

def min_image(v, box):
    """Shortest v under triclinic PBC (box rows are the lattice vectors)."""
    best = v
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                w = v + a * box[0] + b * box[1] + c * box[2]
                if np.dot(w, w) < np.dot(best, best):
                    best = w
    return float(np.linalg.norm(best))


def state_cv(path, i, j):
    st = mm.XmlSerializer.deserialize(open(path).read())
    xyz = np.array(st.getPositions().value_in_unit(unit.nanometer))
    box = np.array(st.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    return min_image(xyz[i] - xyz[j], box)


def sim_cv(sim, i, j):
    """Minimum-image d(i, j) in nm from a live Context."""
    st = sim.context.getState(getPositions=True)
    xyz = np.array(st.getPositions().value_in_unit(unit.nanometer))
    box = np.array(st.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    return min_image(xyz[i] - xyz[j], box)


def stage_smd(cfg, out, platform, props, args, i, j):
    pmf = cfg.pmf
    P = lambda f: os.path.join(out, f)
    if args.resume and os.path.exists(P("smd.xml")):
        log("smd: done")
        return
    r0 = pmf.r_start_A * 0.1 if pmf.r_start_A else state_cv(P("prod.xml"), i, j)
    r1 = pmf.r_end_A * 0.1
    if r0 >= r1:
        raise SystemExit(
            f"smd: the pull would run inwards ({r0 * 10:.2f} -> {r1 * 10:.2f} A).  "
            f"The ion is not bound at the end of prep, so there is no bound "
            f"state to pull from and the umbrella centres would all collapse "
            f"onto r_end_A.  Raise pmf.prep_restraint_kappa (now "
            f"{pmf.prep_restraint_kappa} kcal/mol/A^2), or set pmf.r_start_A "
            f"explicitly, and rerun prep.")
    n = _steps(pmf.smd_ns, pmf.dt_fs)
    stride = max(1, int(round(pmf.colvar_ps * 1e3 / pmf.dt_fs)))
    script = smd_script(i, j, r0, r1, n, kappa_kj_nm2(pmf.smd_kappa),
                        P("smd_colvar.dat"), stride)
    fresh_colvar(P("smd_colvar.dat"))
    open(P("smd.plumed"), "w").write(script)
    log(f"smd: pulling d {r0 * 10:.2f} -> {r1 * 10:.2f} A over {pmf.smd_ns} ns "
        f"({pmf.smd_kappa} kcal/mol/A^2)")
    sim, baro, _, _ = open_sim(cfg, out, platform, props, plumed=script)
    load_state(sim, P("prod.xml"))
    set_npt(sim, baro, cfg, True)
    run_md(sim, out, "smd", n, pmf.dt_fs, cfg.md.report_ps,
           traj_ps=pmf.smd_traj_ps)
    save_state(sim, P("smd.xml"))
    w = np.atleast_2d(np.loadtxt(P("smd_colvar.dat")))
    t, cv = w[:, 0], w[:, 1]
    centre = r0 + (r1 - r0) * t / (t[-1] if t[-1] > 0 else 1.0)
    lag = (centre - cv) * 10
    log(f"smd: Jarzynski work over the pull = {w[-1, 2] / KCAL:.2f} kcal/mol "
        f"(one realisation; the pull is for seeding, not for a free energy)")
    log(f"smd: CV reached {cv.max() * 10:.2f} A; max lag behind the moving "
        f"centre {lag.max():.2f} A")
    if cv.max() < 0.9 * r1 or lag.max() > 1.0:
        log("smd: WARNING - the ligand is not following the restraint.  Raise "
            "pmf.smd_kappa or lengthen pmf.smd_ns, or the windows will be "
            "seeded from a CV range with holes in it.")


# ---------------------------------------------------------------- seeding

def stage_seed(cfg, out, i, j, args):
    """Pick one SMD frame per umbrella centre, evenly spaced in the CV."""
    import mdtraj
    pmf = cfg.pmf
    P = lambda f: os.path.join(out, f)
    if args.resume and os.path.exists(P("windows.json")):
        meta = json.load(open(P("windows.json")))
        stored = meta.get("kappa_kcal_per_A2")
        if stored is not None and abs(stored - pmf.window_kappa) > 1e-9:
            raise SystemExit(
                f"seed: windows.json was written with window_kappa={stored} "
                f"kcal/mol/A^2 but the config now says {pmf.window_kappa}.  Any "
                f"window already in this directory was sampled under the old "
                f"value, so the two cannot be mixed in one WHAM.  Delete "
                f"windows.json and window_* to reseed at the new kappa (prep "
                f"and smd are unaffected and will be reused).")
        log("seed: done")
        return meta
    traj = mdtraj.load(P("smd.dcd"), top=P("topology.pdb"))
    d = mdtraj.compute_distances(traj, [[i, j]], periodic=True)[:, 0]
    lo = pmf.r_start_A * 0.1 if pmf.r_start_A else float(d.min())
    hi = pmf.r_end_A * 0.1
    centres = np.linspace(lo, hi, pmf.n_windows)
    rows = []
    for w, c in enumerate(centres):
        f = int(np.argmin(np.abs(d - c)))
        rows.append(dict(window=w, centre_nm=float(c), frame=f,
                         frame_cv_nm=float(d[f])))
    gap = np.abs(np.array([r["frame_cv_nm"] for r in rows]) - centres)
    spacing = (hi - lo) / (pmf.n_windows - 1)
    log(f"seed: {pmf.n_windows} windows over {lo * 10:.2f}-{hi * 10:.2f} A "
        f"(spacing {spacing * 10:.3f} A); "
        f"worst frame-to-centre gap {gap.max() * 10:.3f} A")

    # Width/spacing budget for this ladder.  Informational, not a gate: it
    # catches windows too sparse to overlap, which is a different failure from
    # a restraint too soft to hold a barrier (that one only shows up once the
    # windows have run -- see the overlap check in wham.py).
    kT_kcal = KB_KCAL * cfg.md.temperature_K
    width = (kT_kcal / pmf.window_kappa) ** 0.5           # A
    ratio = (spacing * 10) / width
    ceiling = kT_kcal / (spacing * 10) ** 2
    log(f"seed: kappa {pmf.window_kappa} kcal/mol/A^2 -> window width "
        f"{width:.3f} A, spacing/width {ratio:.2f} "
        f"(kappa ceiling for this ladder {ceiling:.0f})")
    if ratio > 2.0:
        log("seed: WARNING - spacing exceeds 2x the window width; adjacent "
            "windows will barely overlap.  Lower window_kappa or raise "
            "n_windows.")
    elif ratio < 0.5:
        log(f"seed: note - spacing is {1 / ratio:.1f}x finer than the window "
            f"width, so the ladder is redundant; n_windows could drop to "
            f"~{int((hi - lo) * 10 / width) + 1} at this kappa.")
    if gap.max() > 2 * spacing:
        bad = int((gap > 2 * spacing).sum())
        # Fatal, not a warning: the seed set is the only thing the 120-window
        # GPU pool consumes, and a bad one costs 4 jobs x 4 GPUs x 36 h before
        # anyone notices.  Exiting non-zero also leaves the pool's afterok
        # dependency unsatisfied, so the windows never start.
        raise SystemExit(
            f"seed: {bad} of {pmf.n_windows} windows have no SMD frame within "
            f"2 spacings ({2 * spacing * 10:.3f} A) of their centre; worst gap "
            f"{gap.max() * 10:.3f} A.  The pull skipped part of the CV range, "
            f"so those windows would start far from their restraint and WHAM "
            f"would have no overlap there.  Raise pmf.smd_kappa, lengthen "
            f"pmf.smd_ns, or shorten pmf.smd_traj_ps for more frames, then "
            f"delete smd.* and windows.json and rerun from the SMD.")
    meta = dict(ion_atom=i, ligand_atom=j, kappa_kcal_per_A2=pmf.window_kappa,
                kappa_kj_per_nm2=kappa_kj_nm2(pmf.window_kappa),
                temperature_K=cfg.md.temperature_K, windows=rows)
    json.dump(meta, open(P("windows.json"), "w"), indent=2)
    return meta


def add_windows(cfg, out, lo_A, hi_A, n, kappa_kcal, max_widths=None):
    """
    Append a sub-ladder of n windows over [lo_A, hi_A] at its own kappa to an
    already seeded windows.json, seeded from smd.dcd like stage_seed.

    For a steep wall the main kappa cannot hold: a window sits F/kappa off
    its centre, and where W'' approaches -kappa the windows go bistable.
    Windows already run stay valid -- WHAM and MBAR take a kappa per window
    -- so a stiff patch over that range is cheaper than reseeding everything.
    New rows get indices after the existing ones and carry kappa, spacing,
    a ladder tag and (optionally) their own drift_max_widths.
    """
    import mdtraj
    P = lambda f: os.path.join(out, f)
    meta = json.load(open(P("windows.json")))
    tag = f"k{kappa_kcal:g}_{lo_A:g}-{hi_A:g}"
    if any(r.get("ladder") == tag for r in meta["windows"]):
        raise SystemExit(f"add-windows: ladder {tag} is already in windows.json")
    traj = mdtraj.load(P("smd.dcd"), top=P("topology.pdb"))
    d = mdtraj.compute_distances(traj, [[meta["ion_atom"], meta["ligand_atom"]]],
                                 periodic=True)[:, 0]
    centres = np.linspace(lo_A * 0.1, hi_A * 0.1, n)
    spacing = float(centres[1] - centres[0])
    w0 = len(meta["windows"])
    rows = []
    for k, c in enumerate(centres):
        f = int(np.argmin(np.abs(d - c)))
        if abs(d[f] - c) > 2 * spacing:
            raise SystemExit(f"add-windows: no SMD frame within 2 spacings of "
                             f"{c * 10:.3f} A (nearest {d[f] * 10:.3f} A)")
        row = dict(window=w0 + k, centre_nm=float(c), frame=f, frame_cv_nm=float(d[f]),
                   ladder=tag, kappa_kcal_per_A2=kappa_kcal,
                   kappa_kj_per_nm2=kappa_kj_nm2(kappa_kcal), spacing_nm=spacing)
        if max_widths is not None:
            row["drift_max_widths"] = max_widths
        rows.append(row)
    meta["windows"] += rows
    _atomic_write(P("windows.json"), json.dumps(meta, indent=2))
    width = (KB_KCAL * cfg.md.temperature_K / kappa_kcal) ** 0.5
    log(f"add-windows: {tag}: windows {w0}-{w0 + n - 1}, spacing {spacing * 10:.3f} A, "
        f"width {width:.3f} A (spacing/width {spacing * 10 / width:.2f})")


# ---------------------------------------------------------------- windows

ABORT = "ABORT"


def abort_run(out, msg):
    """
    Stop the whole umbrella pool, not just this window.  A window that cannot
    hold its centre means the ladder (kappa, spacing) is wrong for this PMF,
    and every other window would be spent on a run WHAM has to reject.
    run_pmf_pool.sbatch checks for this file before each window and does not
    resubmit; delete it to override.
    """
    with open(os.path.join(out, ABORT), "a") as fh:
        fh.write(msg + "\n")
    raise SystemExit(msg)


def _colvar_mean(path):
    return float(np.atleast_2d(np.loadtxt(path))[:, 1].mean())


def stage_window(cfg, out, platform, props, args, w):
    import mdtraj
    pmf = cfg.pmf
    P = lambda f: os.path.join(out, f)
    meta = json.load(open(P("windows.json")))
    row = meta["windows"][w]
    i, j = meta["ion_atom"], meta["ligand_atom"]
    centre = row["centre_nm"]
    # windows.json is the single source of truth for kappa, not the config.
    # wham.py unbiases with meta["kappa_kj_per_nm2"]; if this read the config
    # instead, editing window_kappa and rerunning with --resume would simulate
    # at the new value and unbias at the old one -- a silently wrong PMF with
    # no error anywhere.  stage_seed refuses to let the two disagree.
    # Rows appended by add_windows carry their own kappa / spacing / gate
    # limit; the top-level values are the main ladder's.
    kappa = row.get("kappa_kj_per_nm2", meta["kappa_kj_per_nm2"])
    stride = max(1, int(round(pmf.colvar_ps * 1e3 / pmf.dt_fs)))
    prod_colvar = P(f"window_{w:03d}_colvar.dat")
    equil_colvar = P(f"window_{w:03d}_equil_colvar.dat")
    spacing = row.get("spacing_nm",
                      abs(meta["windows"][1]["centre_nm"] - meta["windows"][0]["centre_nm"]))
    max_widths = row.get("drift_max_widths", pmf.drift_max_widths)
    if args.resume and os.path.exists(P(f"window_{w:03d}_done")):
        log(f"window {w}: done")
        return

    # --- equilibration, its own Context so its COLVAR is a separate file and
    #     WHAM never has to guess which samples are production
    if args.resume and os.path.exists(P(f"window_{w:03d}_equil.xml")):
        log(f"window {w}: equil done")
    else:
        fresh_colvar(equil_colvar)
        script = umbrella_script(i, j, centre, kappa, equil_colvar, stride)
        sim, baro, _, _ = open_sim(cfg, out, platform, props, plumed=script,
                                   seed_offset=w + 1)
        frame = mdtraj.load_frame(P("smd.dcd"), row["frame"], top=P("topology.pdb"))
        # A DCD stores the cell as lengths+angles in float32, and mdtraj
        # rebuilds vectors that are an equivalent lattice but not necessarily
        # in OpenMM's reduced form -- for this truncated octahedron |c_y| lands
        # a rounding error past the b_y/2 bound and setPeriodicBoxVectors
        # refuses it.  Rebuilding from lengths/angles gives the reduced
        # representation of the same lattice (c' = c + a - b here), so
        # unwrapped positions and minimum-image distances are unchanged.
        a, b, c = frame.unitcell_lengths[0]
        alpha, beta, gamma = np.radians(frame.unitcell_angles[0])
        sim.context.setPeriodicBoxVectors(
            *unitcell.computePeriodicBoxVectors(a, b, c, alpha, beta, gamma))
        sim.context.setPositions(frame.xyz[0] * unit.nanometer)
        sim.context.setVelocitiesToTemperature(cfg.md.temperature_K * unit.kelvin,
                                               cfg.seed + w + 1)
        set_npt(sim, baro, cfg, True)
        log(f"window {w}: centre {centre * 10:.3f} A, seeded from smd frame {row['frame']} "
            f"(CV {row['frame_cv_nm'] * 10:.3f} A)")
        def drift_gate():
            # skip the first 5 ps: the seed frame may start up to 2 spacings
            # off-centre (stage_seed's limit) and needs a moment to relax
            d = np.atleast_2d(np.loadtxt(equil_colvar))[:, 1]
            d = d[int(round(5.0 / pmf.colvar_ps)):]
            off = float(d.mean()) - centre
            width = (KB_KCAL * KCAL * cfg.md.temperature_K / kappa) ** 0.5   # nm
            msg = (f"window {w}: drift check at {pmf.drift_check_ps:g} ps: "
                   f"<d> {d.mean() * 10:.3f} A vs centre {centre * 10:.3f} A, "
                   f"offset {off / spacing:+.1f} spacings / {off / width:+.1f} widths "
                   f"(limits {pmf.drift_max_spacings:g} / {max_widths:g})")
            if (abs(off) > pmf.drift_max_spacings * spacing
                    and abs(off) > max_widths * width):
                abort_run(out, msg + ".  The restraint is not holding this "
                          "window; raise window_kappa (and n_windows to keep "
                          "overlap) and restart the ladder from seed.")
            log(msg)
        hook = None
        if pmf.drift_check_ps and pmf.drift_max_spacings and max_widths:
            hook = (_steps(pmf.drift_check_ps * 1e-3, pmf.dt_fs), drift_gate)
        run_md(sim, out, f"window_{w:03d}_equil", _steps(pmf.window_equil_ns, pmf.dt_fs),
               pmf.dt_fs, cfg.md.report_ps, hook=hook)
        save_state(sim, P(f"window_{w:03d}_equil.xml"))
        del sim

    # --- production
    fresh_colvar(prod_colvar)
    script = umbrella_script(i, j, centre, kappa, prod_colvar, stride)
    deriv = alpha_deriv_spec(cfg, out)
    sim, baro, _, _ = open_sim(cfg, out, platform, props, plumed=script,
                               seed_offset=w + 1, alpha_deriv=deriv)
    load_state(sim, P(f"window_{w:03d}_equil.xml"))
    set_npt(sim, baro, cfg, True)
    extra = ([DUDAlphaReporter(P(f"window_{w:03d}_V.dat"), stride, pmf.dt_fs)]
             if deriv is not None else [])
    run_md(sim, out, f"window_{w:03d}", _steps(pmf.window_prod_ns, pmf.dt_fs),
           pmf.dt_fs, cfg.md.report_ps, traj_ps=pmf.traj_ps, extra_reporters=extra)
    save_state(sim, P(f"window_{w:03d}.xml"))
    open(P(f"window_{w:03d}_done"), "w").close()
    d = np.atleast_2d(np.loadtxt(prod_colvar))[:, 1]
    log(f"window {w}: <d> = {d.mean() * 10:.3f} A (centre {centre * 10:.3f}), "
        f"sd {d.std() * 10:.3f} A, {len(d)} samples")

    # Neighbour degeneracy: at a working kappa the sampled means follow the
    # centres in order.  Two neighbours sampling the same place (or swapped)
    # is the other signature of a restraint too soft for a barrier.  Only a
    # warning -- the pool runs neighbours concurrently, so this sees each pair
    # once, from whichever window finishes second -- and wham.py's overlap
    # gate is what actually refuses the PMF.  Neighbours are index +/- 1 within
    # the same ladder (an appended sub-ladder starts right after the main one).
    for nb in (w - 1, w + 1):
        if (0 <= nb < len(meta["windows"])
                and meta["windows"][nb].get("ladder") == row.get("ladder")
                and os.path.exists(P(f"window_{nb:03d}_done"))):
            sep = (d.mean() - _colvar_mean(P(f"window_{nb:03d}_colvar.dat"))) * np.sign(w - nb)
            if sep < 0.25 * spacing:
                msg = (f"window {w}: WARNING - <d> is {sep / spacing:+.2f} spacings "
                       f"from window {nb}'s (expected ~+1); the pair is degenerate")
                log(msg)
                with open(P(f"window_{w:03d}_neighbour_warn.txt"), "a") as fh:
                    fh.write(msg + "\n")


# ---------------------------------------------------------------- driver

def resolve_cv(cfg, out, topology, positions):
    """CV atom pair, taken from windows.json if a run is already seeded."""
    P = os.path.join(out, "windows.json")
    if os.path.exists(P):
        m = json.load(open(P))
        return m["ion_atom"], m["ligand_atom"]
    groups = json.load(open(os.path.join(out, "groups.json")))
    prod = os.path.join(out, "prod.xml")
    if os.path.exists(prod):   # pick the bound copy from the equilibrated state
        st = mm.XmlSerializer.deserialize(open(prod).read())
        xyz = np.array(st.getPositions().value_in_unit(unit.nanometer))
    else:
        xyz = np.array(positions.value_in_unit(unit.nanometer))
    return pull_pair(topology, xyz, groups, cfg.pmf)


def _atomic_write(path, text):
    """
    Write via a unique temp file + os.replace.

    Several window processes share one output directory (the GPU pool runs
    4 per node), and they all rewrite config.yaml and pmf_provenance.json
    with identical content.  Writing in place would let one process read a
    half-written file; os.replace is atomic on POSIX.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_provenance(cfg, out, i, j):
    pmf = cfg.pmf
    _atomic_write(os.path.join(out, "pmf_provenance.json"), json.dumps(dict(
        cv=dict(ion_atom=i, ligand_atom=j, pull_atom=pmf.pull_atom,
                pull_resid=pmf.pull_resid),
        protocol=dict(
            dt_fs=pmf.dt_fs, cutoff_nm=cfg.md.cutoff_nm,
            constraints=cfg.md.constraints, temperature_K=cfg.md.temperature_K,
            friction_per_ps=cfg.md.friction_per_ps,
            min_max_iter=pmf.min_max_iter, heat_ns=pmf.heat_ns,
            npt_equil_steps=pmf.npt_equil_steps, npt_prod_steps=pmf.npt_prod_steps,
            prep_restraint_kcal_per_A2=pmf.prep_restraint_kappa,
            prep_restraint_A=pmf.prep_restraint_A,
            smd_ns=pmf.smd_ns, smd_kappa_kcal_per_A2=pmf.smd_kappa,
            r_end_A=pmf.r_end_A, n_windows=pmf.n_windows,
            window_kappa_kcal_per_A2=pmf.window_kappa,
            window_equil_ns=pmf.window_equil_ns,
            window_prod_ns=pmf.window_prod_ns,
            colvar_ps=pmf.colvar_ps, v_class=pmf.v_class),
        c4=dict(
            alpha_by_class=dict(cfg.c4.alpha_by_class),
            scale_by_class=dict(cfg.c4.scale_by_class),
            zero_other_classes=cfg.c4.zero_other_classes,
            lambda_c4=cfg.c4.lambda_c4,
            convention=("MIRIAM: C4 on ion<->water and the named ligand "
                        "class(es) only" if cfg.c4.zero_other_classes else
                        "ParmEd default: Miller polarizability on every class")),
        substitutions=dict(
            minimizer="OpenMM L-BFGS, maxIterations=%s per stage; the AMBER "
                      "protocol's 10000 steepest-descent + 10000 conjugate-"
                      "gradient cycles have no OpenMM equivalent" % pmf.min_max_iter,
            barostat="MonteCarloBarostat (freq 25) instead of Berendsen with a "
                     "5 ps relaxation time: OpenMM ships no Berendsen barostat "
                     "and exposes no virial, so one cannot be added"),
        constraints=dict(
            requested="SHAKE on bonds to hydrogen + rigid water",
            actual="constraints=HBonds, rigidWater=True.  On CUDA, OpenMM "
                   "solves these with SETTLE (water) and SHAKE (heavy atom + "
                   "its hydrogens), i.e. as requested; CCMA handles anything "
                   "left over, of which this system has none.  The Reference "
                   "and CPU platforms have no SHAKE path and use CCMA, so CPU "
                   "runs differ numerically from CUDA ones."),
    ), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--window", type=int, default=None,
                    help="run only this umbrella window (0-based; SLURM array)")
    ap.add_argument("--stop-after",
                    choices=["build", "prep", "smd", "seed", "windows"],
                    default="windows")
    ap.add_argument("--add-windows", nargs=4, metavar=("LO_A", "HI_A", "N", "KAPPA"),
                    help="append N windows over [LO_A, HI_A] at KAPPA kcal/mol/A^2 "
                         "to a seeded windows.json, then exit")
    ap.add_argument("--drift-max-widths", type=float, default=None,
                    help="with --add-windows: gate limit stored on the new rows "
                         "(default: use the config's)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.pmf is None:
        raise SystemExit("config has no `pmf:` section")
    out = args.outdir
    if args.add_windows:
        lo, hi, n, k = args.add_windows
        add_windows(cfg, out, float(lo), float(hi), int(n), float(k), args.drift_max_widths)
        return
    os.makedirs(out, exist_ok=True)
    _atomic_write(os.path.join(out, "config.yaml"), open(args.config).read())
    platform, props = get_platform(args.platform or cfg.platform, args.allow_cpu)
    log(f"{cfg.name}: platform {platform.getName()} {props}")

    if args.resume and os.path.exists(os.path.join(out, "system.xml")):
        topology, positions, _, _ = build.load_system_files(out)
    else:
        topology, positions, _, _ = build.build_system_files(cfg, out)
    if args.stop_after == "build":
        return

    i, j = resolve_cv(cfg, out, topology, positions)
    write_provenance(cfg, out, i, j)

    # array mode: the prep/SMD/seed stages must already exist
    if args.window is not None:
        if os.path.exists(os.path.join(out, ABORT)):
            raise SystemExit(f"{out}/{ABORT} exists, refusing to start window "
                             f"{args.window}:\n" + open(os.path.join(out, ABORT)).read())
        if not os.path.exists(os.path.join(out, "windows.json")):
            raise SystemExit(
                f"{out}/windows.json is missing: run --stop-after seed once "
                f"before submitting the window array")
        stage_window(cfg, out, platform, props, args, args.window)
        return

    stage_prep(cfg, out, platform, props, args, i, j)
    if args.stop_after == "prep":
        return
    i, j = resolve_cv(cfg, out, topology, positions)   # re-pick from prod.xml
    write_provenance(cfg, out, i, j)                  # prep may have changed it
    stage_smd(cfg, out, platform, props, args, i, j)
    if args.stop_after == "smd":
        return
    meta = stage_seed(cfg, out, i, j, args)
    if args.stop_after == "seed":
        return
    for w in range(len(meta["windows"])):
        stage_window(cfg, out, platform, props, args, w)
    log(f"all windows done; now: python -m lig1264.wham {out}")


if __name__ == "__main__":
    main()
