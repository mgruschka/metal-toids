#!/usr/bin/env python
"""
Build imidazole.pdb with CGenFF's IMIA residue/atom names (mosic.xml carries
the IMIA template, so no extra XML is needed).

    obabel SMILES -> 3D  ->  name atoms by ring topology  ->  PDB

Naming (top_all36_cgenff.rtf, RESI IMIA):
    ND1  ring N carrying H          HD1
    NE2  ring N without H (the metal-binding site, class NG2R50)
    CE1  C between the two N        HE1
    CD2  C bonded to NE2            HD2
    CG   C bonded to ND1            HG
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "imidazole.pdb"


def obabel_xyz(smiles):
    r = subprocess.run(["obabel", f"-:{smiles}", "-oxyz", "--gen3d", "-h"],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    n = int(lines[0])
    atoms = [(l.split()[0], np.array(list(map(float, l.split()[1:4]))))
             for l in lines[2:2 + n]]
    return atoms


def bonds_from_distance(atoms, cut=1.75):
    b = {i: set() for i in range(len(atoms))}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ei, ej = atoms[i][0], atoms[j][0]
            c = 1.3 if "H" in (ei, ej) else cut
            if np.linalg.norm(atoms[i][1] - atoms[j][1]) < c:
                b[i].add(j); b[j].add(i)
    return b


def name_atoms(atoms):
    el = [a[0] for a in atoms]
    bonds = bonds_from_distance(atoms)
    heavy_nb = lambda i: [j for j in bonds[i] if el[j] != "H"]
    h_of = lambda i: [j for j in bonds[i] if el[j] == "H"]
    Ns = [i for i in range(len(el)) if el[i] == "N"]
    ND1 = next(i for i in Ns if h_of(i))
    NE2 = next(i for i in Ns if not h_of(i))
    CE1 = next(i for i in heavy_nb(ND1) if i in heavy_nb(NE2))
    CG = next(i for i in heavy_nb(ND1) if i != CE1)
    CD2 = next(i for i in heavy_nb(NE2) if i != CE1)
    names = {ND1: "ND1", NE2: "NE2", CE1: "CE1", CG: "CG", CD2: "CD2"}
    for heavy, hn in ((ND1, "HD1"), (CE1, "HE1"), (CG, "HG"), (CD2, "HD2")):
        (h,) = h_of(heavy)
        names[h] = hn
    assert len(names) == 9, names
    return names


def main():
    atoms = obabel_xyz("c1c[nH]cn1")
    names = name_atoms(atoms)
    # CGenFF template order, ring centred at the origin
    order = ["CG", "HG", "CD2", "HD2", "ND1", "HD1", "CE1", "HE1", "NE2"]
    by_name = {names[i]: atoms[i] for i in names}
    centre = np.mean([by_name[n][1] for n in order if not n.startswith("H")], axis=0)
    with open(OUT, "w") as fh:
        fh.write("REMARK   imidazole, CGenFF IMIA names, built by make_imidazole.py\n")
        for k, n in enumerate(order, start=1):
            el, xyz = by_name[n]
            x, y, z = xyz - centre
            # PDB columns: name 13-16 (from col 14 for short names), the
            # 4-letter CHARMM resName fills 18-21 so chain is at 22, resSeq
            # 23-26, x from 31
            fh.write(f"HETATM{k:5d}  {n:<3s} IMIAA   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00          {el:>2s}\n")
        fh.write("END\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
