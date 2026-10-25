"""
Force-level helpers:

* positional restraints used by the 3-stage minimization
* post-hoc editing of the C4 force that ions_1264.xml's <Script> installs
  (named "LJ1264C4Force", per-particle params cfour / arel), so that the
  ion<->ligand-class C4 can be scaled without rebuilding the System.
"""
import numpy as np
import openmm as mm
from openmm import unit

C4_FORCE_NAME = "LJ1264C4Force"
RESTRAINT_FORCE_NAME = "PositionalRestraint"
CV_RESTRAINT_FORCE_NAME = "CVRestraint"


# ----------------------------------------------------------------------
# positional restraints
# ----------------------------------------------------------------------
def add_restraint_force(system, positions):
    """
    Harmonic restraint on every particle with a per-particle k (kJ/mol/nm^2)
    that starts at 0.  Stages switch groups on/off with set_restrained().
    periodicdistance() so wrapped waters are not yanked across the box.
    """
    f = mm.CustomExternalForce("0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2")
    f.setName(RESTRAINT_FORCE_NAME)
    f.addPerParticleParameter("k")
    for p in ("x0", "y0", "z0"):
        f.addPerParticleParameter(p)
    for i, pos in enumerate(positions):
        x, y, z = pos.value_in_unit(unit.nanometer)
        f.addParticle(i, [0.0, x, y, z])
    system.addForce(f)
    return f


def set_restrained(context, force, indices, k, positions=None):
    """
    Restrain `indices` with stiffness k, release everything else.  If
    `positions` is given the anchors are moved there first (so each stage
    restrains to the *current* structure, not the initial one).
    """
    restrained = set(indices)
    for i in range(force.getNumParticles()):
        idx, (k_old, x0, y0, z0) = force.getParticleParameters(i)
        if positions is not None:
            x0, y0, z0 = positions[idx].value_in_unit(unit.nanometer)
        force.setParticleParameters(i, idx, [k if idx in restrained else 0.0, x0, y0, z0])
    force.updateParametersInContext(context)


# ----------------------------------------------------------------------
# CV restraint (ion held on the ligand through the unbiased prep stages)
# ----------------------------------------------------------------------
def add_cv_restraint_force(system, i, j, k, r0):
    """
    Harmonic restraint on d(i, j); k in kJ/mol/nm^2, r0 in nm.  `j` may be a
    list of atoms: then d is to their unweighted centroid (the pmf.pull_atoms
    CV), via CustomCentroidBondForce.

    pmf.stage_prep runs minimize/heat/equil/prod with no bias and then starts
    the SMD from whatever separation prod.xml happens to end at.  A complex
    worth only a few kcal/mol -- which is the whole point of the Li/Merz fit --
    has a sub-nanosecond residence time, so it dissociates during those 2 ns
    and the pull starts from a random bulk distance.  This force holds the CV
    at the bound value so the pull always starts bound.  It is absent from the
    SMD and umbrella Contexts, so no free energy is computed under it.

    Periodic, so a ligand that wraps mid-run is not yanked across the box.
    """
    js = [int(x) for x in np.atleast_1d(j)]
    if len(js) == 1:
        f = mm.CustomBondForce("0.5*k_cv*(r - r0_cv)^2")
        f.addPerBondParameter("k_cv")
        f.addPerBondParameter("r0_cv")
        f.addBond(i, js[0], [k, r0])
    else:
        f = mm.CustomCentroidBondForce(2, "0.5*k_cv*(distance(g1, g2) - r0_cv)^2")
        f.addPerBondParameter("k_cv")
        f.addPerBondParameter("r0_cv")
        g1 = f.addGroup([i])
        g2 = f.addGroup(js, [1.0] * len(js))      # unweighted, as PLUMED CENTER
        f.addBond([g1, g2], [k, r0])
    f.setName(CV_RESTRAINT_FORCE_NAME)
    f.setUsesPeriodicBoundaryConditions(True)
    system.addForce(f)
    return f


# ----------------------------------------------------------------------
# C4 tuning
# ----------------------------------------------------------------------
def find_force(system, name):
    for f in system.getForces():
        if f.getName() == name:
            return f
    raise KeyError(f"no force named {name!r} in system; forces: "
                   f"{[f.getName() for f in system.getForces()]}")


def atom_classes(forcefield, topology, residue_templates):
    """atom index -> CHARMM atom class, via the same template matching
    createSystem used (so ligand classes are exactly what the FF assigned)."""
    data = forcefield._SystemData(topology)
    # ForceField keeps the matching logic private; call it the way
    # createSystem does.  With recordParameters=True it fills data.atomType.
    forcefield._matchAllResiduesToTemplates(
        data, topology, residue_templates, ignoreExternalBonds=False)
    classes = {}
    for atom in topology.atoms():
        classes[atom.index] = forcefield._atomTypes[data.atomType[atom]].atomClass
    return classes


def apply_c4_spec(system, topology, classes, ligand_indices, c4spec, water_pol=1.444):
    """
    Edit LJ1264C4Force per-particle `arel` for ligand atoms according to
    c4spec.alpha_by_class (absolute, A^3) then c4spec.scale_by_class
    (multiplicative).  With c4spec.zero_other_classes, every ligand class not
    named in either dict is set to arel = 0 instead (MIRIAM convention: C4 on
    the ligating atom only).  Water and the ion are untouched either way, so
    the ion<->water C4 keeps its fitted value.  Returns
    {class: [(index, arel_old, arel_new)]} so the run log shows exactly what
    changed.
    """
    f = find_force(system, C4_FORCE_NAME)
    named = set(c4spec.alpha_by_class) | set(c4spec.scale_by_class)
    changed = {}
    for i in ligand_indices:
        cls = classes[i]
        cfour, arel = f.getParticleParameters(i)
        new = arel
        if cls in c4spec.alpha_by_class:
            new = c4spec.alpha_by_class[cls] / water_pol
        if cls in c4spec.scale_by_class:
            new *= c4spec.scale_by_class[cls]
        if c4spec.zero_other_classes and cls not in named:
            new = 0.0
        if new != arel:
            f.setParticleParameters(i, [cfour, new])
            changed.setdefault(cls, []).append((i, arel, new))
    return changed


def set_lambda_c4(context, value):
    context.setParameter("lambda_c4", value)


def set_class_scale(context, force, indices_by_class, base_arel, cls, scale):
    """
    Scan helper: rescale arel for one ligand class in a live Context.
    `base_arel[i]` is the arel before any scaling (store it at build time).
    """
    for i in indices_by_class[cls]:
        cfour, _ = force.getParticleParameters(i)
        force.setParticleParameters(i, [cfour, base_arel[i] * scale])
    force.updateParametersInContext(context)


# ----------------------------------------------------------------------
# alchemical ion swap (TI): ion A (lambda_ion=0) -> ion B (lambda_ion=1)
# ----------------------------------------------------------------------
LAMBDA_ION = "lambda_ion"
ION_LJ_FORCE_NAME = "IonLJAlchemical"


def ion_params(element, charge, csv_path):
    """sigma (nm), epsilon (kJ/mol), C4 (kJ nm^4/mol) from 12-6-4_base_params.csv"""
    import sys
    from .config import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT))
    import make_1264_xml as M
    _, ions = M.read_csv(csv_path)
    for ion in ions:
        if ion["element"] == element and ion["charge"] == charge:
            return dict(sigma=ion["rmin_half"] * M.RMIN_TO_SIGMA,
                        epsilon=ion["eps_kcal"] * M.KCAL,
                        c4=ion["c4_kcal"] * 4.184e-4, charge=float(charge))
    raise KeyError(f"{element}{charge:+d} not in {csv_path}")


def _lj_from_table(lj):
    """Per-type (sigma, epsilon) from the diagonal of acoef/bcoef."""
    n_x, n_y, a = lj.getTabulatedFunction(0).getFunctionParameters()
    _, _, b = lj.getTabulatedFunction(1).getFunctionParameters()
    out = {}
    for t in range(n_x):
        A, B = a[t * n_y + t], b[t * n_y + t]
        if A > 0 and B > 0:
            out[t] = ((A / B) ** (1.0 / 6.0), B * B / (4.0 * A))
        else:
            out[t] = (1.0, 0.0)
    return out, (n_x, n_y, list(a), list(b))


def make_ion_alchemical(system, ion_index, pA, pB):
    """
    Linear energy mixing U = (1-l) U_A + l U_B for everything that involves
    the ion, with l = global parameter `lambda_ion`:

    * charge   NonbondedForce parameter offset  q = qA + l (qB - qA)
    * LJ       the ion's row/column in the LennardJones table is zeroed and
               replaced by IonLJAlchemical (ion <-> all, Lorentz-Berthelot)
    * C4       LJ1264C4Force gets an extra per-particle `dcfour` so the ion's
               C4 becomes cfour + l*dcfour

    dU/dl is then available from getEnergyParameterDerivatives().
    """
    # --- charge ---------------------------------------------------------
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    q, sig, eps = nb.getParticleParameters(ion_index)
    q0 = q.value_in_unit(unit.elementary_charge)
    if abs(q0 - pA["charge"]) > 1e-6:
        raise SystemExit(f"ion charge in system {q0} != csv {pA['charge']}")
    nb.addGlobalParameter(LAMBDA_ION, 0.0)
    nb.addParticleParameterOffset(LAMBDA_ION, ion_index, pB["charge"] - pA["charge"], 0.0, 0.0)

    # --- LJ -------------------------------------------------------------
    lj = find_force(system, "LennardJones")
    per_type, (n_x, n_y, a, b) = _lj_from_table(lj)
    ion_type = int(lj.getParticleParameters(ion_index)[0])
    sigA_tab, epsA_tab = per_type[ion_type]
    if abs(sigA_tab - pA["sigma"]) > 1e-5 or abs(epsA_tab - pA["epsilon"]) > 1e-5:
        raise SystemExit(f"ion LJ in table ({sigA_tab:.5f}, {epsA_tab:.5f}) != csv "
                         f"({pA['sigma']:.5f}, {pA['epsilon']:.5f})")
    for t in range(n_x):
        a[ion_type * n_y + t] = a[t * n_y + ion_type] = 0.0
        b[ion_type * n_y + t] = b[t * n_y + ion_type] = 0.0
    lj.getTabulatedFunction(0).setFunctionParameters(n_x, n_y, a)
    lj.getTabulatedFunction(1).setFunctionParameters(n_x, n_y, b)

    ilj = mm.CustomNonbondedForce(
        "(1-lambda_ion)*4*epsA*((sA/r)^12-(sA/r)^6) + lambda_ion*4*epsB*((sB/r)^12-(sB/r)^6);"
        "epsA=sqrt(epsAion*epsO); sA=0.5*(sAion+sigO);"
        "epsB=sqrt(epsBion*epsO); sB=0.5*(sBion+sigO);"
        # the ion's own entry is (0, 0), so the sum picks out the partner's
        # parameters whichever side of the pair it is on
        "epsO=eps1+eps2; sigO=sig1+sig2")
    ilj.setName(ION_LJ_FORCE_NAME)
    ilj.addGlobalParameter(LAMBDA_ION, 0.0)
    for name, val in (("sAion", pA["sigma"]), ("epsAion", pA["epsilon"]),
                      ("sBion", pB["sigma"]), ("epsBion", pB["epsilon"])):
        ilj.addGlobalParameter(name, val)
    ilj.addPerParticleParameter("sig")
    ilj.addPerParticleParameter("eps")
    others = []
    for i in range(system.getNumParticles()):
        t = int(lj.getParticleParameters(i)[0])
        ilj.addParticle([0.0, 0.0] if i == ion_index else list(per_type[t]))
        if i != ion_index:
            others.append(i)
    ilj.addInteractionGroup([ion_index], others)
    ilj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    ilj.setCutoffDistance(lj.getCutoffDistance())
    ilj.setUseSwitchingFunction(lj.getUseSwitchingFunction())
    if lj.getUseSwitchingFunction():
        ilj.setSwitchingDistance(lj.getSwitchingDistance())
    ilj.setUseLongRangeCorrection(False)     # ignores interaction groups anyway
    # platforms require every nonbonded force to carry the same exclusions
    for i in range(lj.getNumExclusions()):
        ilj.addExclusion(*lj.getExclusionParticles(i))
    ilj.addEnergyParameterDerivative(LAMBDA_ION)
    system.addForce(ilj)

    # --- C4 -------------------------------------------------------------
    old = find_force(system, C4_FORCE_NAME)
    c4 = mm.CustomNonbondedForce(
        "-lambda_c4*((cfour1+lambda_ion*dcfour1)*arel2 + (cfour2+lambda_ion*dcfour2)*arel1)/(r^4)")
    c4.setName(C4_FORCE_NAME)
    c4.addGlobalParameter("lambda_c4", old.getGlobalParameterDefaultValue(0))
    c4.addGlobalParameter(LAMBDA_ION, 0.0)
    for p in ("cfour", "arel", "dcfour"):
        c4.addPerParticleParameter(p)
    for i in range(old.getNumParticles()):
        cfour, arel = old.getParticleParameters(i)
        if i == ion_index:
            if abs(cfour - pA["c4"]) > 1e-9:
                raise SystemExit(f"ion C4 in system {cfour} != csv {pA['c4']}")
            c4.addParticle([cfour, arel, pB["c4"] - pA["c4"]])
        else:
            c4.addParticle([cfour, arel, 0.0])
    for g in range(old.getNumInteractionGroups()):
        c4.addInteractionGroup(*old.getInteractionGroupParameters(g))
    for i in range(old.getNumExclusions()):
        c4.addExclusion(*old.getExclusionParticles(i))
    c4.setNonbondedMethod(old.getNonbondedMethod())
    c4.setCutoffDistance(old.getCutoffDistance())
    c4.setUseSwitchingFunction(old.getUseSwitchingFunction())
    if old.getUseSwitchingFunction():
        c4.setSwitchingDistance(old.getSwitchingDistance())
    c4.setUseLongRangeCorrection(old.getUseLongRangeCorrection())
    c4.addEnergyParameterDerivative(LAMBDA_ION)
    for i, f in enumerate(system.getForces()):
        if f is old or f.getName() == C4_FORCE_NAME and f.getNumPerParticleParameters() == 2:
            system.removeForce(i)
            break
    system.addForce(c4)
    return ilj, c4


def dudl(context, name=LAMBDA_ION):
    """dU/d lambda (kJ/mol) from analytic parameter derivatives."""
    st = context.getState(getParameterDerivatives=True)
    return st.getEnergyParameterDerivatives()[name]


# ----------------------------------------------------------------------
# alchemical polarizability (alpha-TI, TI_param/run_TI.py):
# alpha0 (lambda_alpha=0) -> alpha1 (lambda_alpha=1) on one ligand class
# ----------------------------------------------------------------------
LAMBDA_ALPHA = "lambda_alpha"


def make_alpha_alchemical(system, atom_indices, alpha0, alpha1):
    """
    Rebuild LJ1264C4Force so that arel on `atom_indices` (the fitted class)
    is arel + lambda_alpha*darel, i.e. alpha goes linearly alpha0 -> alpha1.

    darel = arel(alpha0) * (alpha1/alpha0 - 1): arel is linear in alpha
    (alpha/1.444 times any scale_by_class), so this needs no knowledge of the
    scale -- the same argument as mbar.alpha_pairs.  The energy is exactly
    linear in lambda_alpha, and dU/dlambda = (alpha1-alpha0) * V, V being the
    per-frame dU/dalpha that mbar.py reweights with.  Ion and water keep
    their parameters, so the ion<->water C4 is untouched.

    The C4 clone mirrors the one in make_ion_alchemical (kept separate so the
    ion-swap TI is not touched).  Returns (force, {index: darel}).
    """
    if alpha0 <= 0:
        raise ValueError("alpha0 must be > 0 (darel is scaled off arel(alpha0))")
    old = find_force(system, C4_FORCE_NAME)
    if old.getNumPerParticleParameters() != 2:
        raise SystemExit(f"{C4_FORCE_NAME} already modified "
                         f"({old.getNumPerParticleParameters()} per-particle params)")
    targets = set(atom_indices)
    c4 = mm.CustomNonbondedForce(
        "-lambda_c4*(cfour1*(arel2+lambda_alpha*darel2) + cfour2*(arel1+lambda_alpha*darel1))/(r^4)")
    c4.setName(C4_FORCE_NAME)
    c4.addGlobalParameter("lambda_c4", old.getGlobalParameterDefaultValue(0))
    c4.addGlobalParameter(LAMBDA_ALPHA, 0.0)
    for p in ("cfour", "arel", "darel"):
        c4.addPerParticleParameter(p)
    darel = {}
    for i in range(old.getNumParticles()):
        cfour, arel = old.getParticleParameters(i)
        d = arel * (alpha1 / alpha0 - 1.0) if i in targets else 0.0
        if i in targets:
            if arel == 0.0:
                raise SystemExit(f"atom {i} has arel = 0; alpha cannot be scaled off it")
            darel[i] = d
        c4.addParticle([cfour, arel, d])
    for g in range(old.getNumInteractionGroups()):
        c4.addInteractionGroup(*old.getInteractionGroupParameters(g))
    for i in range(old.getNumExclusions()):
        c4.addExclusion(*old.getExclusionParticles(i))
    c4.setNonbondedMethod(old.getNonbondedMethod())
    c4.setCutoffDistance(old.getCutoffDistance())
    c4.setUseSwitchingFunction(old.getUseSwitchingFunction())
    if old.getUseSwitchingFunction():
        c4.setSwitchingDistance(old.getSwitchingDistance())
    c4.setUseLongRangeCorrection(old.getUseLongRangeCorrection())
    c4.setForceGroup(old.getForceGroup())
    c4.addEnergyParameterDerivative(LAMBDA_ALPHA)
    # by name: SWIG may hand back a fresh wrapper, so `f is old` is unreliable
    # (the new force is not in the System yet, so the first match is the old one)
    for i, f in enumerate(system.getForces()):
        if f.getName() == C4_FORCE_NAME:
            system.removeForce(i)
            break
    system.addForce(c4)
    return c4, darel
