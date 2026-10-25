#!/usr/bin/env python
"""
Build 2,2'-bipyridine PDBs with CGenFF's 22BPY atom names (mosic.xml carries
the 22BPY template, including CGenFF's 4-term N1-C2-C8-N7 torsion, so no extra
XML is needed).

    obabel SMILES -> 3D  ->  name atoms by ring topology  ->  set N1-C2-C8-N7
    torsion by rotating ring 2 about C2-C8  ->  PDB (+ CONECT)

    bipyridine.pdb        cis  (N1-C2-C8-N7 =   0 deg), the chelating conformer
    bipyridine_trans.pdb  trans (            = 180 deg), the free-ligand minimum

Naming (top_all36_cgenff.rtf, RESI 22BPY; both N are class NG2R60):
    ring 1:  N1, C2 (inter-ring C, bonded to N1), C3, C4, C5, C6 (bonded to N1)
    ring 2:  N7, C8 (inter-ring C, bonded to N7), C9, C10, C11, C12 (bonded to N7)
    H3..H6, H9..H12 on the carbon of the same number

The residue name is written as BPY: a PDB resName holds at most 4 characters
and "22BPY" has 5.  Configs map it back with `ligand.pdb_rename: {BPY: 22BPY}`
(build.load_solute).  CONECT records carry the bonds, so loading does not
depend on the residue name being known.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PDB_RESNAME = "BPY"
# CGenFF template order
ORDER = ["N1", "C2", "C3", "H3", "C4", "H4", "C5", "H5", "C6", "H6",
         "N7", "C8", "C9", "H9", "C10", "H10", "C11", "H11", "C12", "H12"]
BONDS = [("N1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "C6"),
         ("C6", "N1"), ("C3", "H3"), ("C4", "H4"), ("C5", "H5"), ("C6", "H6"),
         ("C2", "C8"),
         ("N7", "C8"), ("C8", "C9"), ("C9", "C10"), ("C10", "C11"), ("C11", "C12"),
         ("C12", "N7"), ("C9", "H9"), ("C10", "H10"), ("C11", "H11"), ("C12", "H12")]


def obabel_xyz(smiles):
    r = subprocess.run(["obabel", f"-:{smiles}", "-oxyz", "--gen3d", "-h"],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    n = int(lines[0])
    return [(l.split()[0], np.array(list(map(float, l.split()[1:4]))))
            for l in lines[2:2 + n]]


def bonds_from_distance(atoms, cut=1.75):
    b = {i: set() for i in range(len(atoms))}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            c = 1.3 if "H" in (atoms[i][0], atoms[j][0]) else cut
            if np.linalg.norm(atoms[i][1] - atoms[j][1]) < c:
                b[i].add(j); b[j].add(i)
    return b


def name_atoms(atoms):
    el = [a[0] for a in atoms]
    bonds = bonds_from_distance(atoms)
    heavy_nb = lambda i: [j for j in bonds[i] if el[j] != "H"]
    h_of = lambda i: [j for j in bonds[i] if el[j] == "H"]
    Ns = [i for i in range(len(el)) if el[i] == "N"]
    assert len(Ns) == 2, Ns
    names = {}
    for n_idx, (nN, nCa, ring) in zip(Ns, (("N1", "C2", (3, 4, 5, 6)),
                                          ("N7", "C8", (9, 10, 11, 12)))):
        ca = next(j for j in heavy_nb(n_idx) if not h_of(j))    # inter-ring C
        cb = next(j for j in heavy_nb(n_idx) if j != ca)        # C6 / C12
        names[n_idx], names[ca] = nN, nCa
        # walk away from N over H-bearing carbons: C3, C4, C5, then C6 = cb.
        # Ca's neighbours are N, the other ring's Ca (no H) and C3 (has H).
        prev, cur = ca, next(j for j in heavy_nb(ca) if h_of(j))
        path = [cur]
        while len(path) < 3:
            prev, cur = cur, next(j for j in heavy_nb(cur) if j != prev)
            path.append(cur)
        assert cb in heavy_nb(path[-1]), "ring walk did not close on the N-bonded C"
        for k, idx in zip(ring, path + [cb]):
            names[idx] = f"C{k}"
            (h,) = h_of(idx)
            names[h] = f"H{k}"
    assert sorted(names.values()) == sorted(ORDER), sorted(names.values())
    return names


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def rotate_about(points, origin, axis, angle_deg):
    k = axis / np.linalg.norm(axis)
    t = np.radians(angle_deg)
    p = points - origin
    rot = (p * np.cos(t) + np.cross(k, p) * np.sin(t)
           + np.outer(p @ k, k) * (1 - np.cos(t)))
    return rot + origin


def set_torsion(xyz, target_deg):
    """Rotate ring 2 (N7..H12) about C2-C8 so N1-C2-C8-N7 = target."""
    cur = dihedral(xyz["N1"], xyz["C2"], xyz["C8"], xyz["N7"])
    ring2 = ORDER[ORDER.index("N7"):]
    pts = np.array([xyz[n] for n in ring2])
    pts = rotate_about(pts, xyz["C8"], xyz["C8"] - xyz["C2"], target_deg - cur)
    out = dict(xyz)
    out.update(zip(ring2, pts))
    return out


def write_pdb(path, xyz, el, label):
    centre = np.mean([xyz[n] for n in ORDER if not n.startswith("H")], axis=0)
    serial = {n: k for k, n in enumerate(ORDER, start=1)}
    with open(path, "w") as fh:
        fh.write(f"REMARK   2,2'-bipyridine {label}, CGenFF 22BPY names "
                 f"(resName {PDB_RESNAME}), built by make_bipyridine.py\n")
        for n in ORDER:
            x, y, z = xyz[n] - centre
            name = f" {n:<3s}" if len(n) < 4 else n
            fh.write(f"HETATM{serial[n]:5d} {name} {PDB_RESNAME:>3s} A   1    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el[n]:>2s}\n")
        nbrs = {n: [] for n in ORDER}
        for a, b in BONDS:
            nbrs[a].append(serial[b]); nbrs[b].append(serial[a])
        for n in ORDER:
            fh.write("CONECT" + f"{serial[n]:5d}"
                     + "".join(f"{s:5d}" for s in sorted(nbrs[n])) + "\n")
        fh.write("END\n")
    print(f"wrote {path}  (N1-C2-C8-N7 = "
          f"{dihedral(xyz['N1'], xyz['C2'], xyz['C8'], xyz['N7']):+.1f} deg, "
          f"N1-N7 = {np.linalg.norm(xyz['N1'] - xyz['N7']):.2f} A)")


def main():
    atoms = obabel_xyz("c1ccnc(c1)-c1ccccn1")
    names = name_atoms(atoms)
    xyz = {names[i]: atoms[i][1] for i in names}
    el = {names[i]: atoms[i][0] for i in names}
    write_pdb(HERE / "bipyridine.pdb", set_torsion(xyz, 0.0), el, "cis (chelating)")
    write_pdb(HERE / "bipyridine_trans.pdb", set_torsion(xyz, 180.0), el, "trans (free)")


if __name__ == "__main__":
    sys.exit(main())
