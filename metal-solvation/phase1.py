#!/usr/bin/env python
"""
Phase 1: hydration free energy and ion-oxygen distance for a 12-6 or
12-6-4 ion in SPC/E water.

Two modes:
    --mode hfe    alchemical decoupling via replica exchange -> dG
    --mode iod    plain NPT/NVT -> RDF, IOD, coordination number
    --mode both   iod first (cheap), then hfe

Three-stage decoupling for 12-6-4, in this order:
    A. lambda_c4       1 -> 0   (charges and sterics fully on)
    B. lambda_electrostatics
    C. lambda_sterics  (softcore)

Stage A must be first: -C4/r^4 diverges at short range, so softening the
repulsive wall while C4 is on lets water collapse onto the ion.

Deliberately does NOT load the CHARMM peptoid XML - a bare ion in bulk
water never sees it, and that file carries an embedded <Script> that
chokes on runtime-generated templates.

Examples
--------
    # 12-6 validation run against a published Joung-Cheatham value
    python phase1.py --mode both --name ION --element Na --charge 1 \
        --sigma 0.2439 --epsilon 0.3658 --outdir runs/na_126

    # one point of a C4 scan
    python phase1.py --mode hfe --name ION --element Zn --charge 2 \
        --mass 65.38 --sigma 0.1950 --epsilon 0.0125 --c4 231.0 \
        --outdir runs/zn_c4_231

    # resume an interrupted run (same outdir, same arguments)
    python phase1.py --mode hfe ... --outdir runs/zn_c4_231 --resume
"""

import argparse
import copy
import json
import os
import sys
import time
import tempfile

import numpy as np

from openmm import app, unit, Vec3
import openmm as mm


# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
C4_CONV = 4.184 * 1e-4          # kcal*A^4/mol -> kJ*nm^4/mol
XI_EW = 2.837297
ONE_4PI_EPS0 = 138.935456       # kJ*nm/(mol*e^2)
KCAL = 4.184

TEMP = 298.15 * unit.kelvin
PRES = 1.0 * unit.bar
CUTOFF = 1.0 * unit.nanometer
PME_ALPHA = 3.12 / unit.nanometer
DT = 2.0 * unit.femtosecond
FOURIER_SPACING = 0.08          # nm


def log(msg):
    """Timestamped, flushed - SLURM buffers stdout aggressively."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# platform
# ----------------------------------------------------------------------
def get_platform(name, allow_cpu):
    """
    Select a platform and refuse to silently run on CPU.  An alchemical
    ladder that falls back to CPU turns a 6-hour job into a 6-week one
    and there is no warning unless you ask for it.
    """
    if name == "auto":
        for candidate in ("CUDA", "OpenCL", "CPU"):
            try:
                p = mm.Platform.getPlatformByName(candidate)
                break
            except Exception:
                continue
    else:
        p = mm.Platform.getPlatformByName(name)

    props = {}
    if p.getName() in ("CUDA", "OpenCL"):
        props["Precision"] = "mixed"
    elif not allow_cpu:
        raise SystemExit(
            f"Refusing to run on {p.getName()}. Pass --allow-cpu to override."
        )
    log(f"platform: {p.getName()} {props}")
    return p, props


# ----------------------------------------------------------------------
# system construction
# ----------------------------------------------------------------------
ION_XML = """<ForceField>
 <AtomTypes>
  <Type name="ION-{name}" class="{name}" element="{element}" mass="{mass}"/>
 </AtomTypes>
 <Residues>
  <Residue name="{name}">
   <Atom name="{name}" type="ION-{name}"/>
  </Residue>
 </Residues>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="ION-{name}" charge="{charge}" sigma="{sigma}" epsilon="{epsilon}"/>
 </NonbondedForce>
</ForceField>
"""


def build(args):
    """Solvated single ion.  Ion is always particle index 0."""
    fd, ion_xml = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as fh:
        fh.write(ION_XML.format(
            name=args.name, element=args.element, mass=args.mass,
            charge=args.charge, sigma=args.sigma, epsilon=args.epsilon))

    ff = app.ForceField(args.water_xml, ion_xml)
    if getattr(ff, "_scripts", []):
        log(f"WARNING: {len(ff._scripts)} embedded <Script> block(s) loaded; "
            "Phase 1 expects none")

    top = app.Topology()
    res = top.addResidue(args.name, top.addChain())
    top.addAtom(args.name, app.Element.getBySymbol(args.element), res)

    modeller = app.Modeller(top, [Vec3(0, 0, 0)] * unit.nanometer)
    modeller.addSolvent(ff, model="spce",
                        boxSize=Vec3(1, 1, 1) * args.box * unit.nanometer,
                        neutralize=False)
    n_wat = modeller.topology.getNumResidues() - 1
    log(f"{n_wat} waters, box {args.box} nm, net charge {args.charge:+g}")
    os.unlink(ion_xml)
    return ff, modeller


def fft_friendly(n):
    while True:
        m, x = n, n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return x
        n += 1


def make_system(ff, topology, args):
    system = ff.createSystem(topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=CUTOFF, constraints=None,
                             rigidWater=True)
    nb = next(f for f in system.getForces()
              if isinstance(f, mm.NonbondedForce))
    g = fft_friendly(int(np.ceil(args.box / FOURIER_SPACING)))
    nb.setPMEParameters(PME_ALPHA, g, g, g)
    nb.setUseDispersionCorrection(False)     # matches DispCorr = No
    nb.setUseSwitchingFunction(False)        # plain cutoff

    # Confirm the ion got the parameters we asked for and not some
    # template that won a graph match.  Cheap, and the one place a wrong
    # template would produce entirely plausible numbers.
    q, sig, eps = nb.getParticleParameters(0)
    q = q.value_in_unit(unit.elementary_charge)
    sig = sig.value_in_unit(unit.nanometer)
    eps = eps.value_in_unit(unit.kilojoule_per_mole)
    log(f"ion params: q={q:+.4f} sigma={sig:.6f} eps={eps:.6f}")
    for got, want, label in ((q, args.charge, "charge"),
                             (sig, args.sigma, "sigma"),
                             (eps, args.epsilon, "epsilon")):
        if abs(got - want) > 1e-6:
            raise SystemExit(f"ion {label} is {got}, expected {want}")
    log(f"PME grid {g}^3")
    return system


def add_c4(system, topology, c4_kcal, switch_frac=0.9):
    """
    -lambda_c4 * C4_ij / r^4 for ion <-> solvent only.

    C4_ij = C4_iw * (alpha_j / alpha_water); water oxygen has arel = 1 by
    definition and hydrogens get 0, per the Li/Merz convention.
    """
    nb = next(f for f in system.getForces()
              if isinstance(f, mm.NonbondedForce))
    cnb = mm.CustomNonbondedForce(
        "-lambda_c4*(cfour1*arel2 + cfour2*arel1)/(r^4)")
    cnb.addGlobalParameter("lambda_c4", 1.0)
    cnb.addPerParticleParameter("cfour")
    cnb.addPerParticleParameter("arel")

    ions, others = [], []
    for atom in topology.atoms():
        i = atom.index
        if i == 0:
            cnb.addParticle([c4_kcal * C4_CONV, 0.0])
            ions.append(i)
        elif atom.element.symbol == "O":
            cnb.addParticle([0.0, 1.0])
            others.append(i)
        else:
            cnb.addParticle([0.0, 0.0])
            others.append(i)

    cnb.addInteractionGroup(ions, others)
    cnb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(CUTOFF)
    cnb.setUseSwitchingFunction(True)
    cnb.setSwitchingDistance(CUTOFF * switch_frac)
    # OpenMM's built-in long-range correction ignores interaction groups,
    # so it would be wrong here.  Corrected analytically instead.
    cnb.setUseLongRangeCorrection(False)
    system.addForce(cnb)
    return cnb


# ----------------------------------------------------------------------
# equilibration
# ----------------------------------------------------------------------
def equilibrate(ff, modeller, args, platform, props):
    system = make_system(ff, modeller.topology, args)
    if args.c4:
        add_c4(system, modeller.topology, args.c4)
    system.addForce(mm.MonteCarloBarostat(PRES, TEMP, 25))

    integ = mm.LangevinMiddleIntegrator(TEMP, 1.0 / unit.picosecond, DT)
    sim = app.Simulation(modeller.topology, system, integ, platform, props)
    sim.context.setPositions(modeller.positions)
    log("minimizing")
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(TEMP)

    n = int(args.equil_ns * unit.nanoseconds / DT)
    sim.reporters.append(app.StateDataReporter(
        os.path.join(args.outdir, "equil.csv"), 5000, step=True,
        temperature=True, density=True, volume=True, potentialEnergy=True))
    log(f"NPT equilibration, {args.equil_ns} ns")
    sim.step(n)

    st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    box = st.getPeriodicBoxVectors()
    edge = box[0][0].value_in_unit(unit.nanometer)
    log(f"equilibrated box edge {edge:.4f} nm")
    app.PDBFile.writeFile(modeller.topology, st.getPositions(),
                          open(os.path.join(args.outdir, "equil.pdb"), "w"))
    return st.getPositions(), box


# ----------------------------------------------------------------------
# mode: iod
# ----------------------------------------------------------------------
def run_iod(ff, modeller, args, platform, props, positions, box):
    """NVT production at the equilibrated box, then RDF."""
    system = make_system(ff, modeller.topology, args)
    if args.c4:
        add_c4(system, modeller.topology, args.c4)
    system.setDefaultPeriodicBoxVectors(*box)

    integ = mm.LangevinMiddleIntegrator(TEMP, 1.0 / unit.picosecond, DT)
    sim = app.Simulation(modeller.topology, system, integ, platform, props)
    sim.context.setPositions(positions)
    sim.context.setPeriodicBoxVectors(*box)
    sim.context.setVelocitiesToTemperature(TEMP)

    dcd = os.path.join(args.outdir, "iod.dcd")
    sim.reporters.append(app.DCDReporter(dcd, 500))
    sim.reporters.append(app.StateDataReporter(
        os.path.join(args.outdir, "iod.csv"), 5000, step=True,
        temperature=True, potentialEnergy=True))
    log(f"NVT production for RDF, {args.iod_ns} ns")
    sim.step(int(args.iod_ns * unit.nanoseconds / DT))

    top_pdb = os.path.join(args.outdir, "equil.pdb")
    return analyze_rdf(dcd, top_pdb, args)


def analyze_rdf(dcd, top_pdb, args):
    import mdtraj as md
    traj = md.load(dcd, top=top_pdb)
    ion = traj.topology.select(f"name {args.name}")
    ow = traj.topology.select("name O or name OW or name OH2")
    if len(ion) != 1:
        raise SystemExit(f"expected 1 ion, found {len(ion)}")

    pairs = np.array([[ion[0], o] for o in ow])
    r, g = md.compute_rdf(traj, pairs, r_range=(0.10, 0.80), bin_width=0.002)

    i_max = int(np.argmax(g))
    tail = g[i_max:]
    i_min = i_max + int(np.argmin(tail[:max(1, len(tail) // 2)]))
    rho_o = len(ow) / float(np.mean(traj.unitcell_volumes))
    dr = r[1] - r[0]
    cn = float(np.sum(4 * np.pi * r[:i_min] ** 2 * g[:i_min] * rho_o) * dr)

    np.savetxt(os.path.join(args.outdir, "rdf.dat"),
               np.column_stack([r, g]), header="r_nm  g_r")

    res = dict(iod_A=float(r[i_max] * 10), g_max=float(g[i_max]),
               first_min_A=float(r[i_min] * 10), cn=cn,
               rho_o_per_nm3=float(rho_o))
    log(f"IOD {res['iod_A']:.3f} A   g_max {res['g_max']:.2f}   CN {cn:.2f}")
    return res


# ----------------------------------------------------------------------
# mode: hfe
# ----------------------------------------------------------------------
def lambda_schedule(with_c4, n_c4, n_ele, n_vdw):
    """Sequential three-stage path; no stage overlaps another."""
    s = []
    if with_c4:
        for l in np.linspace(1.0, 0.0, n_c4):
            s.append(dict(c4=float(l), ele=1.0, vdw=1.0))
    for l in np.linspace(1.0, 0.0, n_ele)[1 if with_c4 else 0:]:
        s.append(dict(c4=0.0, ele=float(l), vdw=1.0))
    for l in np.linspace(1.0, 0.0, n_vdw)[1:]:
        s.append(dict(c4=0.0, ele=0.0, vdw=float(l)))
    return s


def run_hfe(ff, modeller, args, platform, props, positions, box):
    from openmmtools import alchemy, states, mcmc, multistate
    from openmmtools.cache import ContextCache

    storage = os.path.join(args.outdir, "hfe.nc")

    # ---- resume path
    if args.resume and os.path.exists(storage):
        log(f"resuming from {storage}")
        sampler = multistate.ReplicaExchangeSampler.from_storage(storage)
        done = sampler.iteration
        want = int(args.ns_per_state * 1000)
        if done < want:
            log(f"extending {done} -> {want} iterations")
            sampler.extend(want - done)
        else:
            log(f"already at {done} iterations")
        return analyze_hfe(storage, args)

    system = make_system(ff, modeller.topology, args)
    if args.c4:
        add_c4(system, modeller.topology, args.c4)
    system.setDefaultPeriodicBoxVectors(*box)

    factory = alchemy.AbsoluteAlchemicalFactory(
        consistent_exceptions=False, alchemical_pme_treatment="exact")
    region = alchemy.AlchemicalRegion(
        alchemical_atoms=[0],
        annihilate_electrostatics=True, annihilate_sterics=True)
    alch_system = factory.create_alchemical_system(system, region)

    # The factory only transforms NonbondedForce, so the C4
    # CustomNonbondedForce should pass through untouched - but verify,
    # because a silently dropped C4 term produces a clean-looking wrong
    # answer.  If this ever fails, add C4 *after* the factory call.
    if args.c4:
        found = False
        for f in alch_system.getForces():
            if isinstance(f, mm.CustomNonbondedForce):
                names = [f.getGlobalParameterName(k)
                         for k in range(f.getNumGlobalParameters())]
                if "lambda_c4" in names:
                    found = True
        log(f"lambda_c4 survived alchemical factory: {found}")
        if not found:
            raise SystemExit("C4 force lost during alchemical setup")

    composable = [alchemy.AlchemicalState.from_system(alch_system)]
    if args.c4:
        class C4State(states.GlobalParameterState):
            lambda_c4 = states.GlobalParameterState.GlobalParameter(
                "lambda_c4", standard_value=1.0)
        composable.append(C4State.from_system(alch_system))

    compound = states.CompoundThermodynamicState(
        states.ThermodynamicState(alch_system, temperature=TEMP),
        composable_states=composable)

    sched = lambda_schedule(bool(args.c4), args.n_c4, args.n_ele, args.n_vdw)
    log(f"{len(sched)} lambda states, {args.ns_per_state} ns each "
        f"({len(sched) * args.ns_per_state:.0f} ns total)")

    thermo_states, sampler_states = [], []
    for p in sched:
        s = copy.deepcopy(compound)
        s.lambda_electrostatics = p["ele"]
        s.lambda_sterics = p["vdw"]
        if args.c4:
            s.lambda_c4 = p["c4"]
        thermo_states.append(s)
        sampler_states.append(states.SamplerState(positions=positions,
                                                  box_vectors=box))

    move = mcmc.LangevinDynamicsMove(
        timestep=DT, collision_rate=1.0 / unit.picosecond,
        n_steps=args.steps_per_iter, reassign_velocities=False)

    sampler = multistate.ReplicaExchangeSampler(
        mcmc_moves=move,
        number_of_iterations=int(args.ns_per_state * 1000),
        online_analysis_interval=None)

    # Route contexts onto the chosen platform.  The attribute names here
    # have moved between openmmtools versions; fall back to the global
    # cache if the per-sampler attributes are absent.
    try:
        cache = ContextCache(platform=platform)
        sampler.energy_context_cache = cache
        sampler.sampler_context_cache = cache
    except AttributeError:
        from openmmtools import cache as _cache
        _cache.global_context_cache.platform = platform
        log("using global_context_cache for platform selection")

    reporter = multistate.MultiStateReporter(
        storage, checkpoint_interval=args.checkpoint_interval)
    sampler.create(thermodynamic_states=thermo_states,
                   sampler_states=sampler_states, storage=reporter)
    log("running replica exchange")
    sampler.run()
    return analyze_hfe(storage, args)


def analyze_hfe(storage, args):
    from openmmtools import multistate
    reporter = multistate.MultiStateReporter(storage, open_mode="r")
    an = multistate.MultiStateSamplerAnalyzer(reporter)
    dF, ddF = an.get_free_energy()
    kt = (unit.MOLAR_GAS_CONSTANT_R * TEMP).value_in_unit(
        unit.kilocalorie_per_mole)

    dG = float(dF[0, -1] * kt)
    err = float(ddF[0, -1] * kt)
    log(f"dG(decoupling) = {dG:8.3f} +/- {err:.3f} kcal/mol")
    log(f"raw HFE        = {-dG:8.3f} kcal/mol   (before corrections)")

    # Replica mixing is the diagnostic that tells you whether the ladder
    # was adequate.  Poor mixing in the sterics leg means add windows,
    # not run longer.
    try:
        mix = an.mixing_statistics
        log(f"mixing statistics available: {type(mix)}")
    except Exception as exc:
        log(f"(mixing statistics unavailable: {exc})")

    return dict(dG_decoupling_kcal=dG, dG_err_kcal=err, raw_hfe_kcal=-dG)


# ----------------------------------------------------------------------
# corrections
# ----------------------------------------------------------------------
def c4_tail(c4_kcal, cutoff_nm, rho_o_per_nm3):
    """
    Neglected -C4/r^4 energy beyond the cutoff, per ion, kcal/mol:

        E = -4*pi*rho*C4 / r_c

    At rho = 33.4 /nm^3, C4 = 100 kcal*A^4/mol, rc = 1.0 nm this is about
    -4.2 kcal/mol - larger than the whole accuracy target of the fit.
    """
    return -4.0 * np.pi * rho_o_per_nm3 * (c4_kcal * 1e-4) / cutoff_nm


def wigner_scale(charge, box_nm):
    """
    UNSCREENED leading periodicity term, an order-of-magnitude indicator
    only.  This is NOT the correction to apply: the physical value is far
    smaller because of solvent screening, and the full treatment has
    additional terms.  Use Rocklin et al. JCP 139:184103 (2013), or run
    3-4 box sizes and extrapolate linearly in 1/L.
    """
    return XI_EW * charge ** 2 * ONE_4PI_EPS0 / (2.0 * box_nm) / KCAL


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    # ion
    p.add_argument("--name", default="ION",
                   help="residue/atom name; avoid CHARMM names like SOD "
                        "which collide by graph match")
    p.add_argument("--element", default="Na")
    p.add_argument("--mass", type=float, default=22.99)
    p.add_argument("--charge", type=float, default=1.0)
    p.add_argument("--sigma", type=float, required=True, help="nm")
    p.add_argument("--epsilon", type=float, required=True, help="kJ/mol")
    p.add_argument("--c4", type=float, default=None,
                   help="kcal*A^4/mol; omit for plain 12-6")
    # system
    p.add_argument("--box", type=float, default=3.0, help="nm")
    p.add_argument("--water-xml", default="spce.xml")
    # protocol
    p.add_argument("--mode", choices=("hfe", "iod", "both"), default="both")
    p.add_argument("--equil-ns", type=float, default=0.5)
    p.add_argument("--iod-ns", type=float, default=20.0)
    p.add_argument("--ns-per-state", type=float, default=5.0)
    p.add_argument("--steps-per-iter", type=int, default=500)
    p.add_argument("--n-c4", type=int, default=6)
    p.add_argument("--n-ele", type=int, default=9)
    p.add_argument("--n-vdw", type=int, default=11)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    # runtime
    p.add_argument("--outdir", required=True)
    p.add_argument("--platform", default="auto")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    if args.seed is not None:
        np.random.seed(args.seed)

    log(f"openmm {mm.version.version}")
    log(f"outdir {args.outdir}")
    platform, props = get_platform(args.platform, args.allow_cpu)

    ff, modeller = build(args)
    positions, box = equilibrate(ff, modeller, args, platform, props)

    results = dict(args=vars(args))
    if args.mode in ("iod", "both"):
        results["iod"] = run_iod(ff, modeller, args, platform, props,
                                 positions, box)
    if args.mode in ("hfe", "both"):
        results["hfe"] = run_hfe(ff, modeller, args, platform, props,
                                 positions, box)

    # corrections, reported separately from the raw number
    edge = box[0][0].value_in_unit(unit.nanometer)
    rho_o = results.get("iod", {}).get("rho_o_per_nm3", 33.4)
    corr = dict(box_edge_nm=float(edge))
    if args.c4:
        corr["c4_tail_kcal"] = float(c4_tail(args.c4, 1.0, rho_o))
    corr["periodicity_scale_kcal"] = float(wigner_scale(args.charge, edge))
    corr["note"] = ("periodicity_scale is an unscreened upper bound, not the "
                    "correction; use Rocklin 2013 or extrapolate in 1/L")
    results["corrections"] = corr

    out = os.path.join(args.outdir, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    log(f"wrote {out}")

    if "hfe" in results:
        raw = results["hfe"]["raw_hfe_kcal"]
        tail = corr.get("c4_tail_kcal", 0.0)
        log(f"raw HFE {raw:+.3f}, C4 tail {tail:+.3f}, "
            f"sum {raw + tail:+.3f} kcal/mol "
            "(finite-size correction still to apply)")


if __name__ == "__main__":
    main()
