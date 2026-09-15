#!/usr/bin/env python
"""
Translate the Li/Merz 12-6-4 ion parameters (12-6-4_base_params.csv) into an
OpenMM force-field XML that can be loaded together with the MoSiC/CHARMM
peptoid XMLs produced by translate_ff.ipynb:

    ForceField('mosic.xml', 'spce_ions.xml', 'ions_1264.xml')

What goes into the XML
----------------------
* <AtomTypes>/<Residues>   one single-atom residue per ion.  Residue and class
  names follow the Amber ion library (Zn2+, Na+, Cl-, Ce3+, Ce4+, Cu+, ...),
  which are unambiguous for the multi-oxidation-state metals in the CSV and do
  not collide with any CHARMM residue (ZN2, MG, SOD, CLA, CO2, NH4, ...).
  The atom name is the upper-case element symbol, matching Amber PDB output.
* 12-6 part            R_min/2 (A) and epsilon (kcal/mol) from the CSV, converted
  to sigma (nm) / epsilon (kJ/mol).  With --lj-style lennardjones (default)
  these go into <LennardJonesForce> and the <NonbondedForce> entry carries
  sigma=1/epsilon=0, which is the convention ParmEd used for mosic.xml /
  spce_mosic.xml (and the stock charmm36 XMLs).  With --lj-style nonbonded
  they go straight into <NonbondedForce>, for Amber-style water XMLs that keep
  LJ there.  Mixing conventions silently drops the ion-water LJ, so match the
  style to whichever water/peptoid XML you load.
* C4 part              a <Script> block that adds a CustomNonbondedForce,
      E = -lambda_c4 * (C4_i*arel_j + C4_j*arel_i) / r^4
  restricted (interaction group) to ion <-> everything-else pairs.  arel_j is
  the atomic polarizability of atom j relative to water oxygen (1.444 A^3),
  exactly the ParmEd `add12_6_4` mixing rule, so water O sees the base C4 from
  the CSV, water H sees zero, peptoid atoms see C4 scaled by their element /
  hybridisation polarizability (Miller 1990 values from lj_1264_pol_custom.dat)
  and ion-ion pairs get the summed term.  `lambda_c4` is a global parameter
  (default 1) so the C4 term can be scaled or switched off from a Context.

Units in the XML are OpenMM's: nm, kJ/mol, and kJ/mol*nm^4 for C4.

Using it
--------
A lone Zn atom matches both CHARMM's ZN2 template and our Zn2+ by graph, so
createSystem raises "Multiple non-identical matching templates" unless the ion
residues are pinned by name, exactly as the peptoid residues already are in
translate_ff.ipynb / run_sim.py:

    ff = ForceField('mosic.xml', 'spce_ions.xml', 'ions_1264.xml')
    resmap = {res: res.name for res in top.residues() if res.name != 'HOH'}
    system = ff.createSystem(top, nonbondedMethod=PME, residueTemplates=resmap)

(mosic.xml + spce_ions.xml come from build_mosic_xml.py; the older monolithic
spce_mosic.xml also works on its own.)  Ion residues in the PDB must be named Zn2+, Cl-, ...
with atom name ZN, CL, ... .  The C4 force is named "LJ1264C4Force" and can be
scaled with context.setParameter('lambda_c4', x).

Usage
-----
    python make_1264_xml.py                                   # -> ions_1264.xml
    python make_1264_xml.py --lj-style nonbonded --out ions_1264_amberlj.xml \
        --coulomb14scale 0.8333333333 --lj14scale 0.5
"""

import argparse
import csv
import re
import sys
from xml.sax.saxutils import quoteattr

from openmm.app import Element

KCAL = 4.184
C4_CONV = KCAL * 1e-4                 # kcal/mol*A^4 -> kJ/mol*nm^4
RMIN_TO_SIGMA = 2.0 / 2.0 ** (1.0 / 6.0) * 0.1   # (R_min/2 in A) -> sigma in nm
WATER_POL = 1.444                     # A^3, water oxygen (Eisenberg & Kauzmann)

# Atom masses for the united-atom ammonium bead
NH4_MASS = Element.getBySymbol("N").mass._value + 4 * Element.getBySymbol("H").mass._value


def parse_charge(s):
    """'3+' -> 3, '1-' -> -1"""
    m = re.fullmatch(r"(\d+)([+-])", s.strip())
    if m is None:
        raise ValueError(f"cannot parse charge {s!r}")
    return int(m.group(1)) * (1 if m.group(2) == "+" else -1)


def ion_name(element, charge):
    """Amber-library style residue/class name: Zn2+, Cu+, Cl-, NH4+"""
    mag = abs(charge)
    return f"{element}{mag if mag > 1 else ''}{'+' if charge > 0 else '-'}"


def read_csv(path):
    ions = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for lineno, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            element, charge, rmin_half, eps, c4 = (cell.strip() for cell in row[:5])
            if not element:
                print(f"WARNING: line {lineno} has no element symbol, skipped: {row}",
                      file=sys.stderr)
                continue
            ions.append(dict(
                element=element,
                charge=parse_charge(charge),
                rmin_half=float(rmin_half),
                eps_kcal=float(eps),
                c4_kcal=float(c4),
            ))
    return header, ions


def read_polfile(path):
    """Amber polarizability file: <type> <alpha A^3> [comment]"""
    pol = {}
    if path is None:
        return pol
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pol[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return pol


# --------------------------------------------------------------------------
# The embedded <Script>.  Runs inside ForceField.createSystem with the locals
# of that function available (sys, data, topology, nonbondedMethod,
# nonbondedCutoff, args, self).  Everything it needs is baked in below.
# --------------------------------------------------------------------------
SCRIPT_TEMPLATE = '''
# ---- 12-6-4 C4 term (generated by make_1264_xml.py) -----------------------
import openmm as _mm
from openmm.app.forcefield import NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, LJPME

# C4 (kJ/mol nm^4) and polarizability (A^3) per ion atom type
_c4_by_type = {c4_by_type}
_pol_by_type = {pol_by_type}
_WATER_POL = {water_pol}
_WATER_RES = {{"HOH", "SOL", "WAT", "TIP3", "TP3M", "SPCE", "SPC", "TIP3P", "H2O"}}

# Polarizabilities (A^3) for everything that is not one of our ions.
# Miller 1990 values as used in Amber's lj_1264_pol_custom.dat, mapped from
# CHARMM atom classes by element / hybridisation.
_pol_element = {pol_element}
_ion_pol_by_element = {ion_pol_by_element}
_SP3_CARBON = ("CT", "CG3", "CC3", "CP1", "CP2", "CP3", "CS", "CN7", "CN8", "CN9", "C3")
_SINGLE_O = ("OH", "OG3", "OS", "OC3", "ON5", "ON6")


def _alpha(atom, atype, aclass):
    """Atomic polarizability in A^3 for a non-ion atom."""
    res = atom.residue.name
    el = atom.element
    if el is None:
        return 0.0
    sym = el.symbol
    if res in _WATER_RES:
        return _WATER_POL if sym == "O" else 0.0
    if len(list(atom.residue.atoms())) == 1 and sym in _ion_pol_by_element:
        return _ion_pol_by_element[sym]          # CHARMM monatomic ions
    if sym == "C":
        return 1.061 if aclass.startswith(_SP3_CARBON) else 1.352
    if sym == "O":
        return 0.637 if aclass.startswith(_SINGLE_O) else 0.569
    if sym == "H":
        return 0.0 if self._atomTypes[atype].mass == 0.0 else 0.387   # CHARMM lone pairs are massless "H"
    return _pol_element.get(sym, 0.0)


_ions, _others = [], []
_params = []
for _atom in data.atoms:
    _atype = data.atomType[_atom]
    _aclass = self._atomTypes[_atype].atomClass
    if _atype in _c4_by_type:
        _ions.append(_atom.index)
        _params.append((_c4_by_type[_atype], _pol_by_type[_atype] / _WATER_POL))
    else:
        _others.append(_atom.index)
        _params.append((0.0, _alpha(_atom, _atype, _aclass) / _WATER_POL))

if _ions:
    _nb = [f for f in sys.getForces() if isinstance(f, _mm.NonbondedForce)]
    _cnb = _mm.CustomNonbondedForce("-lambda_c4*(cfour1*arel2 + cfour2*arel1)/(r^4)")
    _cnb.setName("LJ1264C4Force")
    _cnb.addGlobalParameter("lambda_c4", 1.0)
    _cnb.addPerParticleParameter("cfour")
    _cnb.addPerParticleParameter("arel")
    for _p in _params:
        _cnb.addParticle(list(_p))
    # ion <-> everything (including other ions); ion-ion pairs are counted once
    _cnb.addInteractionGroup(set(_ions), set(_ions) | set(_others))
    if _nb:
        for _k in range(_nb[0].getNumExceptions()):
            _i, _j = _nb[0].getExceptionParameters(_k)[:2]
            _cnb.addExclusion(_i, _j)
    if nonbondedMethod is NoCutoff:
        _cnb.setNonbondedMethod(_mm.CustomNonbondedForce.NoCutoff)
    elif nonbondedMethod in (CutoffPeriodic, Ewald, PME, LJPME):
        _cnb.setNonbondedMethod(_mm.CustomNonbondedForce.CutoffPeriodic)
        _cnb.setCutoffDistance(nonbondedCutoff)
    else:
        _cnb.setNonbondedMethod(_mm.CustomNonbondedForce.CutoffNonPeriodic)
        _cnb.setCutoffDistance(nonbondedCutoff)
    if args.get("switchDistance", None) is not None and nonbondedMethod is not NoCutoff:
        _cnb.setUseSwitchingFunction(True)
        _cnb.setSwitchingDistance(args["switchDistance"])
    # The r^-4 tail beyond a ~1 nm cutoff is small for a handful of ions and
    # the correction would be recomputed every time lambda_c4 changes.
    _cnb.setUseLongRangeCorrection(False)
    sys.addForce(_cnb)
# ---- end 12-6-4 C4 term -----------------------------------------------------
'''


def build_xml(ions, pol, lj_style, prefix, coulomb14, lj14):
    lines = ["<ForceField>",
             " <!-- Li/Merz 12-6-4 ion parameters, generated by make_1264_xml.py -->",
             " <AtomTypes>"]
    entries = []
    for ion in ions:
        name = ion_name(ion["element"], ion["charge"])
        if ion["element"] == "NH4":
            mass, el_sym, atom_name = NH4_MASS, "N", "NH4"
        else:
            el = Element.getBySymbol(ion["element"])
            mass, el_sym, atom_name = el.mass._value, el.symbol, el.symbol.upper()
        tname = f"{prefix}-{name}"
        sigma = ion["rmin_half"] * RMIN_TO_SIGMA
        eps = ion["eps_kcal"] * KCAL
        c4 = ion["c4_kcal"] * C4_CONV
        # ion polarizability: Amber pol file keys are the Amber lib names
        alpha = pol.get(name, pol.get(ion["element"], 0.0))
        entries.append(dict(name=name, tname=tname, atom=atom_name, mass=mass, el=el_sym,
                            q=ion["charge"], sigma=sigma, eps=eps, c4=c4, alpha=alpha,
                            src=ion))
        lines.append(f'  <Type name={quoteattr(tname)} class={quoteattr(name)} '
                     f'element="{el_sym}" mass="{mass:.4f}"/>')
    lines.append(" </AtomTypes>")

    lines.append(" <Residues>")
    for e in entries:
        # override="1" lets these win over same-named templates in stock XMLs (amber14 has U4+, V2+)
        lines.append(f'  <Residue name={quoteattr(e["name"])} override="1">')
        lines.append(f'   <Atom name="{e["atom"]}" type={quoteattr(e["tname"])} charge="{e["q"]:.1f}"/>')
        lines.append("  </Residue>")
    lines.append(" </Residues>")

    lines.append(f' <NonbondedForce coulomb14scale="{coulomb14}" lj14scale="{lj14}">')
    for e in entries:
        if lj_style == "lennardjones":
            lines.append(f'  <Atom type={quoteattr(e["tname"])} charge="{e["q"]:.1f}" sigma="1.0" epsilon="0.0"/>')
        else:
            lines.append(f'  <Atom type={quoteattr(e["tname"])} charge="{e["q"]:.1f}" '
                         f'sigma="{e["sigma"]:.8f}" epsilon="{e["eps"]:.8f}"/>')
    lines.append(" </NonbondedForce>")

    if lj_style == "lennardjones":
        lines.append(f' <LennardJonesForce lj14scale="{lj14}">')
        for e in entries:
            s = e["src"]
            lines.append(f'  <!-- {e["name"]}: R_min/2={s["rmin_half"]} A, eps={s["eps_kcal"]} kcal/mol, '
                         f'C4={s["c4_kcal"]} kcal/mol A^4 -->')
            lines.append(f'  <Atom class={quoteattr(e["name"])} sigma="{e["sigma"]:.8f}" epsilon="{e["eps"]:.8f}"/>')
        lines.append(" </LennardJonesForce>")

    c4_by_type = {e["tname"]: round(e["c4"], 10) for e in entries}
    pol_by_type = {e["tname"]: e["alpha"] for e in entries}
    pol_element = {"H": 0.387, "C": 1.352, "N": 1.090, "O": 0.569, "S": 3.000, "P": 1.538,
                   "F": 0.32, "Cl": 1.91, "Br": 2.88, "I": 4.69}
    # for CHARMM's own monatomic ion residues (SOD, POT, CLA, MG, CAL, ZN2, ...)
    ion_pol_by_element = {}
    for key, val in pol.items():
        m = re.fullmatch(r"([A-Z][a-z]?)\d?[+-]", key)
        if m:
            ion_pol_by_element.setdefault(m.group(1), val)
    script = SCRIPT_TEMPLATE.format(
        c4_by_type=repr(c4_by_type), pol_by_type=repr(pol_by_type), water_pol=WATER_POL,
        pol_element=repr(pol_element), ion_pol_by_element=repr(ion_pol_by_element))
    lines.append(" <Script>")
    lines.append(script.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    lines.append(" </Script>")
    lines.append("</ForceField>")
    return "\n".join(lines) + "\n", entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="12-6-4_base_params.csv")
    ap.add_argument("--out", default="ions_1264.xml")
    ap.add_argument("--polfile", default="sri_docs/lj_1264_pol_custom.dat",
                    help="Amber polarizability file used for ion-ion C4 mixing (optional)")
    ap.add_argument("--lj-style", choices=["lennardjones", "nonbonded"], default="lennardjones",
                    help="where the 12-6 parameters live; must match the water/peptoid XML")
    ap.add_argument("--prefix", default="M1264", help="atom-type name prefix")
    ap.add_argument("--coulomb14scale", default="1.0")
    ap.add_argument("--lj14scale", default="1.0")
    a = ap.parse_args()

    header, ions = read_csv(a.csv)
    try:
        pol = read_polfile(a.polfile)
    except FileNotFoundError:
        print(f"WARNING: polarizability file {a.polfile} not found; ion-ion C4 mixing "
              "will use alpha=0 for the ions", file=sys.stderr)
        pol = {}
    xml, entries = build_xml(ions, pol, a.lj_style, a.prefix, a.coulomb14scale, a.lj14scale)
    with open(a.out, "w") as fh:
        fh.write(xml)

    print(f"wrote {a.out}: {len(entries)} ions, lj-style={a.lj_style}")
    print(f"{'res':6s} {'q':>3s} {'sigma(nm)':>10s} {'eps(kJ/mol)':>12s} {'C4(kJ/mol nm^4)':>16s} {'alpha(A^3)':>10s}")
    for e in entries:
        print(f"{e['name']:6s} {e['q']:+3d} {e['sigma']:10.6f} {e['eps']:12.6f} {e['c4']:16.6f} {e['alpha']:10.4f}")


if __name__ == "__main__":
    main()
