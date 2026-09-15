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
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gromacs_parameterset import GromacsParameterSet as G

FF = '/scratch/midway3/gruschka/toid_explore/forcefields/charmm36-feb2021.ff/'

# forcefield.itp #includes ffnonbonded/ffbonded/gb/cmap/nbfix, as for grompp
p = G(FF + 'forcefield.itp', FF + 'atomtypes.atp', FF + 'merged.rtp')
p.write_xml(os.path.join(HERE, 'mosic.xml'), exclude_residues=G.WATER_RESIDUES + G.ION_RESIDUES)

w = G(FF + 'forcefield.itp', FF + 'spce.itp', FF + 'ions.itp')
w.write_xml(os.path.join(HERE, 'spce_ions.xml'), include_residues=list(w.residues), classes='used')
