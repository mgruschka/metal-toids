"""
Cyclic (head-to-tail) peptoid builder, companion to make_structure.py.

Run from inside structure_maker/ (sidechain templates are resolved relative to
the working directory, exactly like make_structure.py):

    # every residue at a named Ramachandran minimum, ring closed by least squares
    python make_cyclic.py --seq ZZZZZZ --mini A+C --file cyc6.pdb

    # explicit backbone angles for all residues, then vacuum-minimise with MoSiC
    python make_cyclic.py --seq 555555 --phi -60 --psi -50 --omega 180 --relax

    # per-residue cis/trans pattern (c = 0 deg, t = 180 deg) on top of phi/psi
    python make_cyclic.py --seq ZZZZZZ --phi -60 --psi -50 --omega-pattern ctctct

    # one (phi, psi, omega) row per residue (degrees or radians), row i = residue i
    python make_cyclic.py --seq ZZZZZZ --anglefile angles.txt

    # also write the ring-closure path as a multi-frame PDB (vmd cyc6_closure.pdb)
    python make_cyclic.py --seq ZZZZZZ --mini A+C --file cyc6.pdb --trajectory

    # ... and render it to a movie (matplotlib + bundled ffmpeg; --movie gif for a GIF)
    python make_cyclic.py --seq ZZZZZZ --mini A+C --file cyc6.pdb --movie

How it works
------------
1. A glycine backbone (N, CA, C, O) is built from internal coordinates with the
   same bond lengths/angles PeptideBuilder uses for the linear builder.
2. The requested angles almost never close a ring, so phi/psi (and omega too if
   --flex-omega) are refined by least squares: the virtual residue n+1 must land
   on residue 1, with a weak pull back toward the requested angles.  Several
   random restarts are tried; the best closure is reported (max atom mismatch
   and the RMS angle change).  --closure-tol sets what counts as closed.
3. From there the pipeline is the linear one: obabel adds hydrogens, N-H
   hydrogens become sidechain attachment points, make_structure.load_sequence
   places the sidechains and pdb_to_peptoid_forcefield applies MoSiC atom
   names.  No end caps; the closing CLP(n)-NL(1) bond is written as CONECT.

Proline ('P'): the pyrrolidine ring is placed with PeptideBuilder's ring geometry
and phi is held at -65 deg during closure (the ring locks it); like the linear
builder the residue keeps no N-H, so no sidechain is attached.
Limitations (v1): glycine backbone only (no --nsa), n >= 4.
The MoSiC residue templates already carry ExternalBond entries on NL and CLP,
so a cyclic molecule matches them as-is in OpenMM.
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import mdtraj as md
from scipy.optimize import least_squares

import make_structure as ms

# PeptideBuilder glycine geometry (Angstrom / degrees)
B_N_CA, B_CA_C, B_C_N, B_C_O = 1.46, 1.52, 1.33, 1.23
A_N_CA_C, A_CA_C_N, A_C_N_CA, A_CA_C_O = 110.8, 116.6, 121.7, 120.5
# PeptideBuilder ProGeo: wider N-CA-C angle, phi locked by the ring, and the
# pyrrolidine ring internal coordinates (bond, angle, torsion off the backbone)
PRO_N_CA_C = 112.7499
PRO_PHI = -65.0
PRO_RING = dict(CB=(1.52, 109.5, 115.2975),     # from (N, C, CA)
                CG=(1.49, 104.21, 29.6),        # from (N, CA, CB)
                CD=(1.50, 105.03, -34.8))       # from (CA, CB, CG)



# ---------------------------------------------------------------------------
# backbone geometry
# ---------------------------------------------------------------------------
def place(a, b, c, bond, angle, torsion):
    """NeRF: position of D given A-B-C, |CD|, angle BCD and torsion ABCD (deg)."""
    angle, torsion = np.radians(angle), np.radians(torsion)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(angle),
                   bond * np.sin(angle) * np.cos(torsion),
                   bond * np.sin(angle) * np.sin(torsion)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def build_backbone(angles, sequence=None):
    """
    angles: (n, 3) array of (phi_i, psi_i, omega_i) in degrees, where
    omega_i = CA(i-1)-C(i-1)-N(i)-CA(i) is the amide bond *preceding* residue i.
    sequence (optional) selects the proline N-CA-C angle at 'P' positions.
    Returns N, CA, C of residues 1..n+1 (residue n+1 is the virtual copy of
    residue 1 used for ring closure) and O of residues 1..n.
    """
    n = len(angles)
    tau = np.full(n, A_N_CA_C)
    if sequence is not None:
        tau[[i for i, c in enumerate(sequence) if c == "P"]] = PRO_N_CA_C
    N = np.zeros((n + 1, 3)); CA = np.zeros((n + 1, 3)); C = np.zeros((n + 1, 3))
    O = np.zeros((n, 3))
    N[0] = [0.0, 0.0, 0.0]
    CA[0] = [B_N_CA, 0.0, 0.0]
    ang = np.radians(tau[0])
    C[0] = CA[0] + B_CA_C * np.array([-np.cos(ang), np.sin(ang), 0.0])
    for i in range(n):
        j = (i + 1) % n
        phi_next, omega_next = angles[j, 0], angles[j, 2]
        N[i + 1] = place(N[i], CA[i], C[i], B_C_N, A_CA_C_N, angles[i, 1])       # psi_i
        CA[i + 1] = place(CA[i], C[i], N[i + 1], B_N_CA, A_C_N_CA, omega_next)   # omega_{i+1}
        C[i + 1] = place(C[i], N[i + 1], CA[i + 1], B_CA_C, tau[j], phi_next)    # phi_{i+1}
        O[i] = place(N[i + 1], CA[i], C[i], B_C_O, A_CA_C_O, 180.0)              # O trans to N(i+1)
    return N, CA, C, O


def proline_ring(N, CA, C, i):
    """CB, CG, CD of a proline at residue i, from PeptideBuilder's ring geometry."""
    cb = place(N[i], C[i], CA[i], *PRO_RING["CB"])
    cg = place(N[i], CA[i], cb, *PRO_RING["CG"])
    cd = place(CA[i], cb, cg, *PRO_RING["CD"])
    return cb, cg, cd


_CLASH_CACHE = {}


def _clash_pairs(n):
    """Backbone atom pairs (flat index into the (n,4,3) N/CA/C/O array) that are
    >= 1-4 apart along the ring, with their minimum allowed distance."""
    if n in _CLASH_CACHE:
        return _CLASH_CACHE[n]
    idx = lambda i, k: 4 * (i % n) + k          # k: 0=N 1=CA 2=C 3=O
    adj = {a: set() for a in range(4 * n)}
    for i in range(n):
        for a, b in ((idx(i, 0), idx(i, 1)), (idx(i, 1), idx(i, 2)), (idx(i, 2), idx(i, 3)),
                     (idx(i, 2), idx(i + 1, 0))):
            adj[a].add(b); adj[b].add(a)
    # BFS topological distances (capped at 4)
    pairs, dmin = [], []
    for a in range(4 * n):
        dist = {a: 0}; frontier = [a]
        for step in range(1, 4):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = step; nxt.append(v)
            frontier = nxt
        for b in range(a + 1, 4 * n):
            td = dist.get(b, 4)
            if td == 3:
                pairs.append((a, b)); dmin.append(2.6)
            elif td >= 4:
                pairs.append((a, b)); dmin.append(3.0)
    _CLASH_CACHE[n] = (np.array(pairs), np.array(dmin))
    return _CLASH_CACHE[n]


def _clash(N, CA, C, O, n):
    """Penalties for backbone heavy atoms closer than allowed (see _clash_pairs)."""
    atoms = np.stack([N[:n], CA[:n], C[:n], O], axis=1).reshape(-1, 3)
    pairs, dmin = _clash_pairs(n)
    d = np.linalg.norm(atoms[pairs[:, 0]] - atoms[pairs[:, 1]], axis=1)
    return np.maximum(0.0, dmin - d)


def close_ring(target, flex_omega=False, weight=0.1, restarts=20, seed=0, tol=0.05,
               clash_weight=1.0, sequence=None, record=None):
    """
    Adjust phi/psi (and omega if flex_omega) so that residue n+1 coincides with
    residue 1, without letting the backbone fold through itself.  Proline phi
    is held fixed (the pyrrolidine ring locks it).
    record: optional list; the angle sets along the winning restart's descent
    path are appended to it (one entry per accepted improvement).
    Returns (angles, max_atom_mismatch_A, rms_angle_change_deg).
    """
    n = len(target)
    cols = [0, 1, 2] if flex_omega else [0, 1]
    # boolean mask of free (residue, angle) entries
    free = np.zeros((n, 3), bool)
    free[:, cols] = True
    if sequence is not None:
        for i, c in enumerate(sequence):
            if c == "P":
                free[i, 0] = False
    rng = np.random.default_rng(seed)

    def unpack(x):
        a = target.copy()
        a[free] = x
        return a

    path = []            # frames of the current restart: (cost, angles)

    def residual(x):
        a = unpack(x)
        N, CA, C, O = build_backbone(a, sequence)
        gap = np.concatenate([N[n] - N[0], CA[n] - CA[0], C[n] - C[0]])
        dev = np.radians(((a[free] - target[free]) + 180) % 360 - 180)
        r = np.concatenate([gap, weight * dev, clash_weight * _clash(N, CA, C, O, n)])
        if record is not None:
            # least_squares calls this for finite-difference probes too; keeping
            # only strict improvements leaves the clean descent path
            cost = float(r @ r)
            if not path or cost < path[-1][0] * (1 - 1e-4):
                path.append((cost, a.copy()))
        return r

    best = None
    for k in range(restarts):
        x0 = target[free].copy()
        if k:
            x0 = x0 + rng.normal(0, 15 * min(k, 4), size=x0.shape)
        path = []
        sol = least_squares(residual, x0, method="trf", max_nfev=5000)
        a = unpack(sol.x)
        N, CA, C, O = build_backbone(a, sequence)
        gap = max(np.linalg.norm(N[n] - N[0]), np.linalg.norm(CA[n] - CA[0]),
                  np.linalg.norm(C[n] - C[0]))
        clash = _clash(N, CA, C, O, n).max()
        dev = ((a[free] - target[free]) + 180) % 360 - 180
        ok = gap <= tol and clash < 0.5
        score = (not ok, gap + clash if not ok else np.sqrt(np.mean(dev ** 2)))
        if best is None or score < best[0]:
            best = (score, a, gap, np.sqrt(np.mean(dev ** 2)), [f for _, f in path])
        if ok and k >= 3:
            break
    _, a, gap, rms, frames = best
    if record is not None:
        record.extend(frames)
    a = ((a + 180) % 360) - 180
    return a, gap, rms


# ---------------------------------------------------------------------------
# templates for the sidechain-attachment step (mirrors make_polyglycine_angles)
# ---------------------------------------------------------------------------
def write_backbone_pdb(fname, N, CA, C, O, sequence):
    n = len(O)
    lines = []
    serial = 1
    idx = {}
    for i in range(n):
        atoms = [("N", N[i]), ("CA", CA[i]), ("C", C[i]), ("O", O[i])]
        resname = "GLY"
        if sequence[i] == "P":
            resname = "PRO"
            atoms += list(zip(("CB", "CG", "CD"), proline_ring(N, CA, C, i)))
        for name, xyz in atoms:
            lines.append(f"ATOM  {serial:5d}  {name:<3s} {resname} A{i + 1:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {name[0]}")
            idx[(i, name)] = serial
            serial += 1
    conect = []
    for i in range(n):
        j = (i + 1) % n
        pairs = [(idx[(i, "N")], idx[(i, "CA")]), (idx[(i, "CA")], idx[(i, "C")]),
                 (idx[(i, "C")], idx[(i, "O")]), (idx[(i, "C")], idx[(j, "N")])]
        if sequence[i] == "P":
            pairs += [(idx[(i, "CA")], idx[(i, "CB")]), (idx[(i, "CB")], idx[(i, "CG")]),
                      (idx[(i, "CG")], idx[(i, "CD")]), (idx[(i, "CD")], idx[(i, "N")])]
        for a, b in pairs:
            conect.append(f"CONECT{a:5d}{b:5d}")
            conect.append(f"CONECT{b:5d}{a:5d}")
    with open(fname, "w") as f:
        f.write("\n".join(lines + conect + ["END"]) + "\n")


def interpolate_angles(frames, n_out):
    """Resample a list of (n,3) angle sets to n_out frames, interpolating each
    optimizer step linearly in torsion space along the shortest arc."""
    frames = [np.asarray(f, float) for f in frames]
    if len(frames) < 2 or n_out <= len(frames):
        return frames
    # cumulative "distance" = RMS angle change per step, so big steps get more frames
    steps = [np.sqrt(np.mean((((b - a) + 180) % 360 - 180) ** 2)) for a, b in zip(frames[:-1], frames[1:])]
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    out = []
    for t in np.linspace(0, cum[-1], n_out):
        k = min(np.searchsorted(cum, t, side="right") - 1, len(frames) - 2)
        f = 0.0 if steps[k] == 0 else (t - cum[k]) / steps[k]
        a, b = frames[k], frames[k + 1]
        delta = ((b - a) + 180) % 360 - 180
        out.append(((a + f * delta) + 180) % 360 - 180)
    return out


def write_closure_trajectory(fname, frames, sequence, n_frames=200):
    """Multi-MODEL PDB of the backbone (plus proline rings) along the closure
    path, ending with the closed ring.  Load with  vmd <fname>  (or mdtraj)."""
    n = len(sequence)
    frames = interpolate_angles(list(frames), n_frames)
    with open(fname, "w") as f:
        f.write(f"REMARK  ring-closure path, {len(frames)} frames; last frame is the closed ring\n")
        for m, a in enumerate(frames, start=1):
            N, CA, C, O = build_backbone(a, sequence)
            f.write(f"MODEL     {m:4d}\n")
            serial = 1
            for i in range(n):
                atoms = [("N", N[i]), ("CA", CA[i]), ("C", C[i]), ("O", O[i])]
                resname = "GLY"
                if sequence[i] == "P":
                    resname = "PRO"
                    atoms += list(zip(("CB", "CG", "CD"), proline_ring(N, CA, C, i)))
                for name, xyz in atoms:
                    f.write(f"ATOM  {serial:5d}  {name:<3s} {resname} A{i + 1:4d}    "
                            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {name[0]}\n")
                    serial += 1
            # the virtual residue n+1 shows the gap that is being closed
            f.write(f"HETATM{serial:5d}  N   GAP A{n + 1:4d}    "
                    f"{N[n][0]:8.3f}{N[n][1]:8.3f}{N[n][2]:8.3f}  1.00  0.00           N\n")
            f.write("ENDMDL\n")
        f.write("END\n")


def make_templates(sequence, angles):
    """Write template_{seq}.pdb (backbone + HA1/HA2 [+HN for G]) and
    template_{seq}_nh.pdb (backbone + N-H) like make_polyglycine_angles does."""
    n = len(sequence)
    N, CA, C, O = build_backbone(angles, sequence)
    bare = f"template_{sequence}.pdb"
    write_backbone_pdb(bare, N[:n], CA[:n], C[:n], O, sequence)

    obabel = os.path.join(os.path.dirname(sys.executable), "obabel")
    if not os.path.exists(obabel):
        obabel = shutil.which("obabel") or obabel
    if not os.path.exists(obabel):
        raise RuntimeError("obabel not found; run with the metal-toid python (conda-forge openbabel)")
    subprocess.run([obabel, bare, "-O", f"template_{sequence}_h.pdb", "-h"],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    th = md.load(f"template_{sequence}_h.pdb", standard_names=False)

    # ring closure bond may not have been perceived by mdtraj; add it for the
    # bonded-hydrogen bookkeeping below
    keep_c = np.zeros(th.n_atoms, bool)      # carbon template (HA's, HN on Gly)
    keep_n = np.zeros(th.n_atoms, bool)      # nitrogen template (N-H attachment points)
    heavy = th.top.select("not element H")
    keep_c[heavy] = keep_n[heavy] = True
    ha_names = ["HA1", "HA2"]
    ha_count = np.zeros(n, int)
    hn_count = np.zeros(n, int)
    ring_count = [dict() for _ in range(n)]       # proline CB/CG/CD hydrogen counters
    ctop = th.top.copy()
    ntop = th.top.copy()
    for x, y in th.top.bonds:
        if x.element.symbol == "H" and y.element.symbol != "H":
            a, b = x, y                      # a = hydrogen, b = heavy atom
        elif y.element.symbol == "H" and x.element.symbol != "H":
            a, b = y, x
        else:
            continue
        r = b.residue.index
        if b.name == "N":
            hn_count[r] += 1
            keep_n[a.index] = True
            ntop.atom(a.index).name = "H"
            if sequence[r] == "G":
                keep_c[a.index] = True
                ctop.atom(a.index).name = "HN"
        elif b.name == "CA":
            if ha_count[r] >= 2:
                raise ValueError(f"residue {r + 1}: obabel placed more than 2 HA")
            name = "HA" if sequence[r] == "P" else ha_names[ha_count[r]]
            ctop.atom(a.index).name = name
            ntop.atom(a.index).name = name
            ha_count[r] += 1
            keep_c[a.index] = keep_n[a.index] = True
        elif sequence[r] == "P" and b.name in ("CB", "CG", "CD"):
            k = ring_count[r].get(b.name, 0) + 1
            ring_count[r][b.name] = k
            name = f"H{b.name[1]}{k}"                 # HB1, HB2, HG1, ...
            ctop.atom(a.index).name = name
            ntop.atom(a.index).name = name
            keep_c[a.index] = keep_n[a.index] = True
    bad = []
    for i in range(n):
        if sequence[i] == "P":
            if ha_count[i] != 1 or hn_count[i] != 0 or \
                    any(ring_count[i].get(c, 0) != 2 for c in ("CB", "CG", "CD")):
                bad.append(i + 1)
        elif ha_count[i] != 2 or hn_count[i] != 1:
            bad.append(i + 1)
    if bad:
        raise ValueError(f"obabel did not place the expected hydrogens on residues {bad} "
                         f"(Gly: 2 HA + 1 HN, Pro: 1 HA + 2 H per ring carbon); "
                         "the backbone is probably too strained - try other angles "
                         "or --flex-omega")

    for keep, top, suffix in ((keep_c, ctop, ""), (keep_n, ntop, "_nh")):
        t = md.Trajectory(th.xyz[0, keep][None], top.subset(np.where(keep)[0]))
        # obabel appends hydrogens after all heavy atoms; regroup per residue
        t = ms.regroup_by_residue(t)
        t.save_pdb(f"template_{sequence}{suffix}.pdb")
    os.remove(f"template_{sequence}_h.pdb")


def add_closure_conect(filename, sequence):
    """Append the CLP(n)-NL(1) bond that pdb_to_peptoid_forcefield cannot know about."""
    with open(filename) as f:
        lines = f.readlines()
    n = len(sequence)
    nl1 = clpn = None
    for l in lines:
        if l.startswith("ATOM"):
            name, resseq = l[12:16].strip(), int(l[22:26])
            if name == "NL" and resseq == 1:
                nl1 = int(l[6:11])
            if name == "CLP" and resseq == n:
                clpn = int(l[6:11])
    if nl1 is None or clpn is None:
        raise RuntimeError("could not find NL(1)/CLP(n) for the closing bond")
    out = []
    for l in lines:
        if l.startswith("ENDMDL") or l.strip() == "END":
            out.append(f"CONECT{clpn:5d}{nl1:5d}\n")
            out.append(f"CONECT{nl1:5d}{clpn:5d}\n")
            out.append(l)
            # only once
            lines = [x for x in lines if x is not l]
            break
        out.append(l)
    rest = lines[len(out) - 2:]  # anything after ENDMDL (END)
    with open(filename, "w") as f:
        f.writelines(out + [x for x in rest if not (x.startswith("ENDMDL"))])


# ---------------------------------------------------------------------------
def relax(filename, max_iterations=2000):
    """Vacuum minimisation with the MoSiC force field (mosic.xml + spce_ions.xml
    from the repo root); overwrites filename, keeping the original as *_raw.pdb."""
    from openmm import app, unit
    import openmm as mm
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from utils import add_bonds_from_forcefield
    ff = app.ForceField(os.path.join(root, "mosic.xml"), os.path.join(root, "spce_ions.xml"))
    pdb = app.PDBFile(filename)
    add_bonds_from_forcefield(pdb.topology, ff)
    resmap = {r: r.name for r in pdb.topology.residues()}
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, residueTemplates=resmap)
    sim = app.Simulation(pdb.topology, system, mm.VerletIntegrator(1 * unit.femtosecond),
                         mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(pdb.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    sim.minimizeEnergy(maxIterations=max_iterations)
    st = sim.context.getState(getEnergy=True, getPositions=True)
    print(f"relax: {e0.value_in_unit(unit.kilojoule_per_mole):.3g} -> "
          f"{st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
    raw = filename.replace(".pdb", "_raw.pdb")
    os.replace(filename, raw)
    # rewrite with the original text (names/CONECT) and new coordinates
    pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    out, k = [], 0
    for l in open(raw):
        if l.startswith("ATOM"):
            l = f"{l[:30]}{pos[k][0]:8.3f}{pos[k][1]:8.3f}{pos[k][2]:8.3f}{l[54:]}"
            k += 1
        out.append(l)
    with open(filename, "w") as f:
        f.writelines(out)


def create_cyclic_peptoid(sequence, angles, filename="cyclic.pdb", flex_omega=False,
                          closure_tol=0.05, restarts=20, seed=0, angle_weight=0.1,
                          do_relax=False, trajectory=False, trajectory_frames=200, movie=None):
    if not filename.endswith(".pdb"):
        filename += ".pdb"
    n = len(sequence)
    if n < 4:
        raise ValueError("cyclic peptoids need at least 4 residues")
    for pos, letter in enumerate(sequence, start=1):
        if letter not in ms.FILENAMES:
            raise ValueError(f"{letter} is not a valid residue code")
        if letter == "P":
            print(f"proline at position {pos}: phi held at {PRO_PHI} deg")

    angles = np.asarray(angles, float)
    if angles.ndim == 1:
        angles = np.tile(angles, (n, 1))
    assert angles.shape == (n, 3), f"need one (phi, psi, omega) row per residue, got {angles.shape}"
    angles[[i for i, c in enumerate(sequence) if c == "P"], 0] = PRO_PHI

    trajectory = trajectory or movie is not None
    record = [] if trajectory else None
    closed, gap, rms = close_ring(angles, flex_omega=flex_omega, restarts=restarts,
                                  seed=seed, tol=closure_tol, weight=angle_weight,
                                  sequence=sequence, record=record)
    print(f"ring closure: max backbone mismatch {gap:.3f} A, RMS angle change {rms:.1f} deg")
    if trajectory:
        traj_name = filename.replace(".pdb", "_closure.pdb")
        write_closure_trajectory(traj_name, record + [closed], sequence, trajectory_frames)
        print(f"closure path: {len(record) + 1} optimizer steps -> {max(trajectory_frames, len(record) + 1)} "
              f"interpolated frames in {traj_name}  (vmd {traj_name})")
        if movie is not None:
            from render_closure import render
            render(traj_name, filename.replace(".pdb", f"_closure.{movie}"))
    if gap > closure_tol:
        raise ValueError(f"could not close the ring within {closure_tol} A (best {gap:.2f} A); "
                         "try --flex-omega, other angles, or more --restarts")
    print("final backbone angles (phi, psi, omega):")
    for i, (p, s, o) in enumerate(closed):
        print(f"  {i + 1:3d} {sequence[i]}  {p:8.1f} {s:8.1f} {o:8.1f}")
    np.savetxt(filename.replace(".pdb", "_angles.txt"), closed, fmt="%9.3f",
               header="phi psi omega (deg); row i = residue i, omega_i precedes residue i")

    tmp = os.path.join(os.path.dirname(filename), "temp_" + os.path.basename(filename))
    make_templates(sequence, closed)
    ms.load_sequence(sequence, tmp)
    ms.pdb_to_peptoid_forcefield(sequence, tmp, "NME")
    add_closure_conect(tmp, sequence)
    os.rename(tmp, filename)
    for suffix in ("", "_nh"):
        ms._osremove(f"template_{sequence}{suffix}.pdb")
    if do_relax:
        relax(filename)
    return md.load(filename, standard_names=False)


def parse_angles(args, n):
    if args.anglefile is not None:
        a = np.loadtxt(args.anglefile)
        if np.all(np.abs(a) <= 2 * np.pi):
            a = np.degrees(a)
        if a.ndim == 1:
            a = np.tile(a, (n, 1))
    elif args.mini is not None:
        if args.mini not in ms.MINIMA_DICT:
            raise ValueError(f"--mini must be one of {ms.MINIMA_FILENAMES}")
        a = np.tile(np.array(ms.MINIMA_DICT[args.mini], float), (n, 1))
    elif args.phi is not None and args.psi is not None:
        omega = 180.0 if args.omega is None else args.omega
        a = np.tile([args.phi, args.psi, omega], (n, 1))
    else:
        raise ValueError("give --mini, --phi/--psi[/--omega], or --anglefile")
    if args.omega_pattern:
        if len(args.omega_pattern) != n:
            raise ValueError("--omega-pattern must have one c/t per residue")
        a[:, 2] = [0.0 if ch.lower() == "c" else 180.0 for ch in args.omega_pattern]
    return a


def run():
    p = argparse.ArgumentParser(description="Cyclic peptoid builder (see module docstring)")
    p.add_argument("--seq", required=True, help="residue codes, as in make_structure.py")
    p.add_argument("--mini", help="named Ramachandran minimum for every residue")
    p.add_argument("--phi", type=float); p.add_argument("--psi", type=float)
    p.add_argument("--omega", type=float, help="default 180 (trans)")
    p.add_argument("--omega-pattern", help="per-residue amide pattern, e.g. ctctct")
    p.add_argument("--anglefile", help="text file, one phi psi omega row per residue")
    p.add_argument("--file", default="cyclic.pdb")
    p.add_argument("--flex-omega", action="store_true", help="let omega relax during ring closure")
    p.add_argument("--closure-tol", type=float, default=0.05, help="max backbone mismatch (A)")
    p.add_argument("--restarts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--angle-weight", type=float, default=0.1,
                   help="pull toward requested angles during closure (larger = more faithful, harder to close)")
    p.add_argument("--trajectory", action="store_true",
                   help="write <file>_closure.pdb: multi-frame backbone PDB of the ring-closure path")
    p.add_argument("--trajectory-frames", type=int, default=200,
                   help="frames in the closure movie (optimizer steps are interpolated in torsion space)")
    p.add_argument("--movie", nargs="?", const="mp4", choices=["mp4", "gif"],
                   help="also render the closure path to <file>_closure.mp4 (or gif) with matplotlib; implies --trajectory")
    p.add_argument("--relax", action="store_true",
                   help="minimise the result in vacuum with mosic.xml (needs OpenMM); raw build kept as *_raw.pdb")
    args = p.parse_args()
    angles = parse_angles(args, len(args.seq))
    create_cyclic_peptoid(args.seq, angles, args.file, flex_omega=args.flex_omega,
                          closure_tol=args.closure_tol, restarts=args.restarts, seed=args.seed,
                          angle_weight=args.angle_weight, do_relax=args.relax,
                          trajectory=args.trajectory, trajectory_frames=args.trajectory_frames,
                          movie=args.movie)


if __name__ == "__main__":
    run()
