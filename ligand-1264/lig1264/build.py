"""
System construction: ligand PDB (+ ion) -> solvated truncated octahedron ->
OpenMM System with the C4 force edited per config.
"""
import json
import os

import numpy as np
from openmm import app, unit, Vec3
import openmm as mm

from .config import Config
from . import forces

WATER_RES = {"HOH", "SOL", "WAT", "TIP3", "SPCE"}


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_forcefield(cfg: Config):
    paths = cfg.ff_paths()
    ff = app.ForceField(*paths)
    register_template_bonds(ff)
    log(f"force field: {[p.split('/')[-1] for p in paths]}")
    return ff


def register_template_bonds(ff):
    """
    Make PDBFile.createStandardBonds() know the force field's residues.
    Topology.loadBondDefinitions() only reads residues.xml-style files, not
    force-field XML, so we feed the template bonds into the same table.
    Only intra-residue bonds; inter-residue links still need CONECT records
    (structure_maker writes them).  Built-in definitions are left alone.
    """
    n = 0
    for name, tmpl in ff._templates.items():
        if name in app.Topology._standardBonds:
            continue
        atoms = list(tmpl.atoms)
        app.Topology._standardBonds[name] = [
            (atoms[i].name, atoms[j].name) for i, j in tmpl.bonds]
        n += 1
    log(f"registered bond definitions for {n} force-field residues")


def residue_template_map(topology, cfg: Config):
    """
    Pin every non-water residue to the template with its own name.  The
    ions need this because Zn2+ (ions_1264.xml) and ZN2 (CHARMM) are
    graph-identical; the peptoid residues need it for the same reason.
    Modeller adds counter-ions as residues NA / CL, which we map onto the
    12-6-4 Na+ / Cl- templates.
    """
    counter = {"NA": "Na+", "CL": "Cl-", "K": "K+", "MG": "Mg2+", "CA": "Ca2+"}
    m = {}
    for res in topology.residues():
        if res.name in WATER_RES:
            continue
        m[res] = counter.get(res.name, res.name)
    return m


def load_solute(cfg: Config):
    ion = cfg.ion
    if cfg.ligand.count == 0:
        # bare ion at the origin (TI water leg)
        top = app.Topology()
        res = top.addResidue(ion.resname, top.addChain())
        top.addAtom(ion.element.upper(), app.Element.getBySymbol(ion.element), res)
        log(f"bare {ion.resname}, no ligand")
        return app.Modeller(top, [Vec3(0, 0, 0)] * unit.nanometer)
    pdb = app.PDBFile(cfg.ligand.pdb)
    if cfg.ligand.pdb_rename:
        n = 0
        for res in pdb.topology.residues():
            if res.name in cfg.ligand.pdb_rename:
                res.name = cfg.ligand.pdb_rename[res.name]
                n += 1
        log(f"renamed {n} residue(s) per ligand.pdb_rename {cfg.ligand.pdb_rename}")
    if cfg.ligand.contains_ion:
        return app.Modeller(pdb.topology, pdb.positions)
    if cfg.ligand.count == 1:
        top, pos = place_ion(pdb.topology, pdb.positions, cfg)
        return app.Modeller(top, pos)
    return place_ligands(pdb, cfg)


def _anchor(top, pos, cfg: Config):
    ion = cfg.ion
    lig = [a for a in top.atoms() if a.residue.name in cfg.ligand.resnames]
    if not lig:
        raise SystemExit(f"no atoms with residue names {cfg.ligand.resnames} in {cfg.ligand.pdb}")
    if ion.near_atom is None:
        raise SystemExit("ion.near_atom is required unless ligand.contains_ion")
    cands = [a for a in lig if a.name == ion.near_atom and
             (ion.near_resid is None or a.residue.id == str(ion.near_resid))]
    if len(cands) != 1:
        raise SystemExit(f"ion.near_atom={ion.near_atom!r} (resid {ion.near_resid}) "
                         f"matched {len(cands)} ligand atoms; set ion.near_resid")
    xyz = np.array([p.value_in_unit(unit.nanometer) for p in pos])
    com = xyz[[a.index for a in lig]].mean(axis=0)
    return xyz, com, cands[0]


def place_ion(top, pos, cfg: Config):
    """Add the ion `distance_A` from `near_atom`, away from ligand COM."""
    ion = cfg.ion
    xyz, com, anchor_atom = _anchor(top, pos, cfg)
    anchor = xyz[anchor_atom.index]
    v = anchor - com
    n = np.linalg.norm(v)
    v = v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    ion_xyz = anchor + v * ion.distance_A * 0.1

    new = app.Modeller(top, pos)
    itop = app.Topology()
    res = itop.addResidue(ion.resname, itop.addChain())
    itop.addAtom(ion.element.upper(), app.Element.getBySymbol(ion.element), res)
    new.add(itop, [Vec3(*ion_xyz)] * unit.nanometer)
    log(f"placed {ion.resname} {ion.distance_A} A from {ion.near_atom} "
        f"(residue {anchor_atom.residue.name}{anchor_atom.residue.id})")
    return new.topology, new.positions


OCTAHEDRAL = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def _rotation_onto(a, b):
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    v = np.cross(a, b); c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _rotation_about(axis, theta):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def place_ligands(pdb, cfg: Config):
    """
    Ion at the origin, `count` copies of the ligand on octahedral directions
    with `near_atom` at `distance_A` and the COM->near_atom axis pointing at
    the ion (so e.g. the imidazole lone pair faces the metal).
    """
    ion = cfg.ion
    n = cfg.ligand.count
    if n > len(OCTAHEDRAL):
        raise SystemExit(f"ligand.count={n} > {len(OCTAHEDRAL)} supported positions")
    xyz, com, anchor_atom = _anchor(pdb.topology, pdb.positions, cfg)
    axis = xyz[anchor_atom.index] - com
    axis /= np.linalg.norm(axis)
    local = xyz - xyz[anchor_atom.index]          # near_atom at origin

    itop = app.Topology()
    res = itop.addResidue(ion.resname, itop.addChain())
    itop.addAtom(ion.element.upper(), app.Element.getBySymbol(ion.element), res)
    modeller = app.Modeller(itop, [Vec3(0, 0, 0)] * unit.nanometer)
    placed_all = []
    for k in range(n):
        d = OCTAHEDRAL[k]
        R = _rotation_onto(axis, -d)              # ligand axis points at ion
        base = local @ R.T
        # spin about the ion-ligand axis to maximise the closest approach to
        # the copies already placed (coplanar rings 90 deg apart overlap)
        best, best_sep = None, -1.0
        for ang in np.arange(0, 180, 5.0):
            Rs = _rotation_about(-d, np.radians(ang))
            cand = base @ Rs.T + d * ion.distance_A * 0.1
            sep = min((np.linalg.norm(cand[:, None, :] - q[None, :, :], axis=2).min()
                       for q in placed_all), default=1.0)
            if sep > best_sep:
                best, best_sep = cand, sep
        placed_all.append(best)
        modeller.add(pdb.topology, [Vec3(*p) for p in best] * unit.nanometer)
    if n > 1:
        sep = min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min()
                  for i, a in enumerate(placed_all) for b in placed_all[i + 1:])
        log(f"closest inter-ligand atom distance {sep * 10:.2f} A")
    log(f"{ion.resname} at origin with {n} x {cfg.ligand.resnames} at "
        f"{ion.distance_A} A ({ion.near_atom} facing the ion)")
    return modeller


def inscribed_ratio(modeller, shape):
    """(min distance between opposite faces) / (box-vector length)."""
    a, b, c = (np.array(v) for v in modeller._computeBoxVectors(1.0, shape))
    vol = abs(np.dot(a, np.cross(b, c)))
    return min(vol / np.linalg.norm(np.cross(b, c)),
               vol / np.linalg.norm(np.cross(c, a)),
               vol / np.linalg.norm(np.cross(a, b)))


def solvate(modeller, ff, cfg: Config):
    b = cfg.box
    tmap = residue_template_map(modeller.topology, cfg)
    xyz = np.array([p.value_in_unit(unit.nanometer) for p in modeller.positions])
    centre = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
    radius = np.linalg.norm(xyz - centre, axis=1).max() * 10      # A
    kw = dict(model=b.water, boxShape=b.shape, neutralize=b.neutralize,
              positiveIon=b.positive_ion, negativeIon=b.negative_ion,
              residueTemplates=tmap)
    # Face-to-face (minimum-image) width.  Modeller's own `padding` sets the
    # box-*vector* length to 2*padding, which for the octahedron is only
    # 0.82x the face-to-face width, so it does not reproduce tLEaP; build
    # the vectors ourselves in both modes.  Modeller centres the water box
    # on the solute, so no translation is needed.
    if b.buffer_A is not None:
        diameter = 2 * radius + 2 * b.buffer_A
        how = f"buffer {b.buffer_A} A around solute radius {radius:.1f} A"
    else:
        diameter = b.diameter_A
        how = f"fixed diameter {diameter} A"
    width = diameter * 0.1 / inscribed_ratio(modeller, b.shape)
    kw["boxVectors"] = modeller._computeBoxVectors(width, b.shape)
    if 2 * radius + cfg.md.cutoff_nm * 10 > diameter:
        log(f"WARNING: solute diameter {2 * radius:.1f} A + cutoff exceeds the "
            f"{diameter:.1f} A face-to-face width; solute sees its own images")
    modeller.addSolvent(ff, **kw)
    n_wat = sum(1 for r in modeller.topology.residues() if r.name in WATER_RES)
    a, bb, c = (np.array(v.value_in_unit(unit.angstrom))
                for v in modeller.topology.getPeriodicBoxVectors())
    vol = abs(np.dot(a, np.cross(bb, c)))
    n_ion = sum(1 for r in modeller.topology.residues() if r.name == cfg.ion.resname)
    conc_mM = n_ion / (vol * 1e-27) / 6.02214e23 * 1e3
    log(f"{b.shape} box, {how}: face-to-face {diameter:.1f} A, |a| = "
        f"{np.linalg.norm(a):.1f} A, {vol:.0f} A^3, {n_wat} waters, "
        f"{cfg.ion.resname} {conc_mM:.1f} mM")
    return modeller


def classify_atoms(topology, cfg: Config):
    """index lists for the restraint stages + analysis."""
    groups = {"ligand": [], "ligand_H": [], "ion": [], "counter_ion": [], "water": []}
    for a in topology.atoms():
        rn = a.residue.name
        if rn in WATER_RES:
            groups["water"].append(a.index)
        elif rn in cfg.ligand.resnames:
            groups["ligand"].append(a.index)
            if a.element is not None and a.element.symbol == "H":
                groups["ligand_H"].append(a.index)
        elif rn == cfg.ion.resname or (cfg.ligand.contains_ion and
                                       a.element is not None and
                                       a.element.symbol == cfg.ion.element):
            groups["ion"].append(a.index)
        else:
            groups["counter_ion"].append(a.index)
    return groups


def make_system(modeller, ff, cfg: Config):
    md = cfg.md
    tmap = residue_template_map(modeller.topology, cfg)
    constraints = {"None": None, "HBonds": app.HBonds}[str(md.constraints)]
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=md.cutoff_nm * unit.nanometer,
                             constraints=constraints, rigidWater=True,
                             residueTemplates=tmap)
    groups = classify_atoms(modeller.topology, cfg)
    classes = forces.atom_classes(ff, modeller.topology, tmap)

    # C4 edits for the ligand classes under study
    changed = forces.apply_c4_spec(system, modeller.topology, classes,
                                   groups["ligand"], cfg.c4)
    if cfg.c4.zero_other_classes and groups["counter_ion"]:
        # MIRIAM's polfile is zero for Cl- too, so the metal<->counter-ion C4
        # cross term vanishes there.  Zeroing arel here does the same without
        # touching the counter-ion's own fitted C4 to water (that term is
        # cfour_Cl * arel_O and does not involve arel_Cl).
        c4f = forces.find_force(system, forces.C4_FORCE_NAME)
        for i in groups["counter_ion"]:
            cfour, arel = c4f.getParticleParameters(i)
            if arel != 0.0:
                c4f.setParticleParameters(i, [cfour, 0.0])
                changed.setdefault(classes[i] + " (counter-ion)", []).append((i, arel, 0.0))
    for cls, rows in changed.items():
        log(f"C4 arel for class {cls}: {rows[0][1]:.4f} -> {rows[0][2]:.4f} "
            f"({len(rows)} atoms)")
    ligand_classes = sorted({classes[i] for i in groups["ligand"]})
    log(f"ligand classes present: {ligand_classes}")
    if cfg.c4.zero_other_classes:
        kept = sorted(set(cfg.c4.alpha_by_class) | set(cfg.c4.scale_by_class))
        log(f"MIRIAM C4 convention: arel zeroed on every ligand class except "
            f"{kept}; ion<->water C4 unaffected")

    # sanity: ion got the 12-6-4 template, not CHARMM's
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    c4f = forces.find_force(system, forces.C4_FORCE_NAME)
    for i in groups["ion"]:
        q, sig, eps = nb.getParticleParameters(i)
        cfour, arel = c4f.getParticleParameters(i)
        # sigma/eps in NonbondedForce are the 1/0 placeholders of the
        # LennardJonesForce convention (LJ lives in the CustomNonbondedForce
        # named "LennardJones"), so only charge and C4 are informative here.
        log(f"ion atom {i}: q={q.value_in_unit(unit.elementary_charge):+.1f} "
            f"C4={cfour / 4.184e-4:.1f} kcal A^4/mol  arel={arel:.3f}")
        if cfour == 0.0:
            raise SystemExit("ion has C4 = 0: matched a CHARMM ion template "
                             "instead of ions_1264.xml (check residue_template_map)")
    return system, groups, classes


def build_system_files(cfg: Config, out, post_system=None):
    """
    Full build -> topology.pdb / system.xml / groups.json / classes.json in
    `out`.  `post_system(system, groups)` can modify the System before it is
    serialized (the TI driver adds the alchemical forces there).  The
    positional-restraint force is always included.
    """
    ff = load_forcefield(cfg)
    modeller = load_solute(cfg)
    modeller = solvate(modeller, ff, cfg)
    system, groups, classes = make_system(modeller, ff, cfg)
    if post_system is not None:
        post_system(system, groups)
    forces.add_restraint_force(system, modeller.positions)
    with open(os.path.join(out, "topology.pdb"), "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)
    json.dump(groups, open(os.path.join(out, "groups.json"), "w"))
    json.dump({str(k): v for k, v in classes.items()}, open(os.path.join(out, "classes.json"), "w"))
    # PDBFile truncates residue names to 3 characters (IMIA -> IMI, Zn2+ -> Zn2)
    json.dump([r.name for r in modeller.topology.residues()],
              open(os.path.join(out, "resnames.json"), "w"))
    with open(os.path.join(out, "system.xml"), "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    return modeller.topology, modeller.positions, system, groups


def load_system_files(out):
    log("build: reusing system.xml / topology.pdb")
    pdb = app.PDBFile(os.path.join(out, "topology.pdb"))
    for res, name in zip(pdb.topology.residues(),
                         json.load(open(os.path.join(out, "resnames.json")))):
        res.name = name
    with open(os.path.join(out, "system.xml")) as fh:
        system = mm.XmlSerializer.deserialize(fh.read())
    groups = json.load(open(os.path.join(out, "groups.json")))
    return pdb.topology, pdb.positions, system, groups
