#!/usr/bin/env python
"""
Rebuild the OpenMM XML files from the MoSiC/CHARMM36 GROMACS force field using
GromacsParameterSet from gromacs_parameterset.py (a local copy of the custom
ParmEd module; needs only stock ParmEd).  Replaces the hand-edited workflow
from translate_ff.ipynb.  Output goes next to this script.

    mosic.xml       peptoid/CHARMM residues, all classes + NBFIX; the water
                    and monatomic-ion residues that merged.rtp carries are
                    left out so they cannot collide with a real water XML
    spce_ions.xml   SPC/E water (SOL, from spce.itp) + CHARMM ions (NA, CL,
                    K, MG, ZN, ... from ions.itp); classes='used' so nothing
                    is defined twice across the two files

Load with  ForceField('mosic.xml', 'spce_ions.xml'[, 'ions_1264.xml']).
Pass defines={'HEAVY_H': 1} etc. to GromacsParameterSet for other branches.

After writing, the templates are validated (`--check-only` runs just this on
the existing files):
  * only polymer residues -- those with a +/- bond of their own in the rtp --
    may carry <ExternalBond>.  A phantom external bond makes a template
    unmatchable for the free molecule (the converter once put one on every
    residue with an atom named C2, N, C, P or O3': 953 of 1280 templates,
    22BPY among them).
  * free small molecules (ligand-1264/ligands: IMIA, 22BPY) and a capped
    peptide chain (openmm_viable_AAAAA.pdb, ACE-ALA5-NME) must template-match.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gromacs_parameterset import GromacsParameterSet as G

FF = '/scratch/midway3/gruschka/toid_explore/forcefields/charmm36-feb2021.ff/'

# (pdb, {pdb resname: template name}) -- the renames are for template names
# longer than the 4 characters a PDB resName holds
MATCH_TESTS = [
    ('ligand-1264/ligands/imidazole/imidazole.pdb', {}),
    ('ligand-1264/ligands/bipyridine/bipyridine.pdb', {'BPY': '22BPY'}),
    ('openmm_viable_AAAAA.pdb', {}),
]


def polymer_residues(p):
    """Residues with a cross-residue (+/-) bond of their own in the rtp."""
    return {name for name, r in p.residues.items() if getattr(r, '_raw_ext_bonds', [])}


def check_external_bonds(xml_path, polymers):
    bad = []
    for r in ET.parse(xml_path).getroot().iter('Residue'):
        ext = [e.get('atomName') for e in r.findall('ExternalBond')]
        if ext and r.get('name') not in polymers:
            bad.append((r.get('name'), ext))
    if bad:
        raise SystemExit(f'{xml_path}: {len(bad)} non-polymer residue(s) carry '
                         f'external bonds, e.g. {bad[:5]}')
    print(f'external bonds: only on polymer residues ({len(polymers)} polymer templates)')


def check_template_matching(xml_paths):
    from openmm import app
    ff = app.ForceField(*xml_paths)
    for name, tmpl in ff._templates.items():          # as lig1264.build does
        if name not in app.Topology._standardBonds:
            atoms = list(tmpl.atoms)
            app.Topology._standardBonds[name] = [(atoms[i].name, atoms[j].name)
                                                 for i, j in tmpl.bonds]
    for rel, rename in MATCH_TESTS:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            print(f'template match: {rel} missing, skipped')
            continue
        pdb = app.PDBFile(path)
        for res in pdb.topology.residues():
            res.name = rename.get(res.name, res.name)
        if rename:          # bonds of renamed residues were unknown at read time
            pdb.topology.createStandardBonds()
        tmap = {res: res.name for res in pdb.topology.residues()}
        try:
            ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                            residueTemplates=tmap)
        except Exception as e:
            raise SystemExit(f'template match FAILED for {rel}: {e}')
        print(f'template match: {rel} ok ({", ".join(r.name for r in pdb.topology.residues())})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check-only', action='store_true',
                    help='validate the existing mosic.xml / spce_ions.xml, do not rebuild')
    args = ap.parse_args()
    mosic = os.path.join(HERE, 'mosic.xml')
    spce = os.path.join(HERE, 'spce_ions.xml')

    # forcefield.itp #includes ffnonbonded/ffbonded/gb/cmap/nbfix, as for grompp
    p = G(FF + 'forcefield.itp', FF + 'atomtypes.atp', FF + 'merged.rtp')
    if not args.check_only:
        p.write_xml(mosic, exclude_residues=G.WATER_RESIDUES + G.ION_RESIDUES)
        w = G(FF + 'forcefield.itp', FF + 'spce.itp', FF + 'ions.itp')
        w.write_xml(spce, include_residues=list(w.residues), classes='used')

    check_external_bonds(mosic, polymer_residues(p))
    check_template_matching([mosic, spce])


if __name__ == '__main__':
    main()
