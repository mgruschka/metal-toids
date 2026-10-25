# ligands/

One sub-directory per non-water ligand:

    ligands/<name>/<name>.pdb    coordinates, residue names matching the templates
    ligands/<name>/<name>.xml    OpenMM force-field XML for it (only if the residue
                                 is not already in mosic.xml)

MoSiC/CHARMM peptoid and peptide residues, and the CGenFF 4.4 residues merged
into the GROMACS CHARMM36 port (e.g. `IMIA` imidazole -- see
`imidazole/make_imidazole.py`), need no XML: point the config at the PDB and
list `resnames`.  Template names longer than 4 characters (a PDB resName
holds 4) are written short and renamed on load, e.g. `bipyridine/` writes
`BPY` and the config sets `ligand.pdb_rename: {BPY: 22BPY}`, `resnames: [22BPY]`
(CONECT records carry the bonds, so the short name needs no definition).  Atom names must match the template; check with
`grep -A12 'Residue name="IMIA"' ../mosic.xml`.  For other molecules, produce the XML the same way
the repo builds mosic.xml: `gromacs_parameterset.GromacsParameterSet` on
CGenFF-style .itp/.rtp -> `write_xml`.  The XML must use the LennardJonesForce
convention (LJ in `<LennardJonesForce>`, `sigma=1 epsilon=0` in
`<NonbondedForce>`), otherwise the ion/ligand LJ is silently dropped.
