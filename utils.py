from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import os

def add_bonds_from_forcefield(topology, forcefield):
    """Add missing bonds to a topology using force field templates."""
    
    # Track existing bonds to avoid duplicates
    existing = set()
    for bond in topology.bonds():
        existing.add((bond[0].index, bond[1].index))
        existing.add((bond[1].index, bond[0].index))
    
    def safe_add(a1, a2):
        if (a1.index, a2.index) not in existing:
            topology.addBond(a1, a2)
            existing.add((a1.index, a2.index))
            existing.add((a2.index, a1.index))
    
    # Intra-residue bonds from templates
    for res in topology.residues():
        if res.name not in forcefield._templates:
            continue
        template = forcefield._templates[res.name]
        atom_map = {a.name: a for a in res.atoms()}
        template_atoms = list(template.atoms)
        for bond in template.bonds:
            a1_name = template_atoms[bond[0]].name
            a2_name = template_atoms[bond[1]].name
            if a1_name in atom_map and a2_name in atom_map:
                safe_add(atom_map[a1_name], atom_map[a2_name])
