# 2026-09-15 — 12-6-4 ions, force-field rebuild, CUDA fix, cyclic peptoid builder

## Summary

One session covering four threads: (1) an OpenMM XML for the Li/Merz 12-6-4 ion
parameters, (2) finding and fixing why the GROMACS→OpenMM force-field translation
carried "hidden" water/ion definitions, (3) getting the `metal-toid` env to run
OpenMM on the Midway3 GPUs with PLUMED, and (4) a cyclic (head-to-tail) peptoid
builder with ring-closure visualisation.

## 1. 12-6-4 ion parameters → `ions_1264.xml`

- `12-6-4_base_params.csv` is byte-for-byte ParmEd's `DEFAULT_C4_PARAMS['SPCE']`
  (Li/Merz SPC/E set). The row with a blank element was Fe³⁺ (C4 = 442) and was
  filled in.
- `make_1264_xml.py` writes `ions_1264.xml`: Amber-library residue/class names
  (`Zn2+`, `Cl-`, `Ce3+`/`Ce4+`, `Cu+`/`Cu2+`, …) to avoid CHARMM collisions
  (`ZN2`, `CO2`, `NH4`, …); 12-6 terms in `<LennardJonesForce>` to match the
  MoSiC convention (sigma=1/epsilon=0 in `<NonbondedForce>`), with
  `--lj-style nonbonded` for Amber-style XMLs; and an embedded `<Script>` adding
  a `CustomNonbondedForce` (`LJ1264C4Force`) with the ParmEd `add12_6_4` mixing
  rule `C4_ij = C4_i·α_j/1.444` (water O → base C4, water H → 0, peptoid atoms by
  element/hybridisation polarizability, ion–ion summed). Global parameter
  `lambda_c4` for scanning/decoupling.
- Verified analytically: ion–water C4 at 2.5 Å, ion–ion mixed term counted once,
  LB-mixed LJ with water; PME run of solvated peptoid + Zn²⁺/Cl⁻ stable.
- Ion residues must be pinned via `residueTemplates={res: res.name}` because a
  lone Zn graph-matches both `ZN2` and `Zn2+`.

## 2. Force-field translation: root causes of the "hidden water models"

`GromacsParameterSet` (custom code, formerly inside the env's ParmEd at
`parmed/gromacs/parameters.py`) had three leaks:

1. `read_itp` dropped every `#` line, so **both branches of every `#ifdef`** were
   parsed and the last definition won — `OT` (CHARMM water O) got the `HEAVY_H`
   mass 9.9514; `#define _FF_CHARMM` was lost.
2. `merged.rtp` itself carries water and monatomic-ion residues (`TIP3`, `TP3M`,
   `HOH`, `SOD`, `CLA`, `ZN2`, …) with no way to filter them; two water templates
   carry CHARMM's H–H bond and never match PDB water, the rest collide with any
   real water XML ("Multiple non-identical matching templates").
3. `spce.itp`/`ions.itp` produced nothing because `[ moleculetype ]` blocks were
   skipped (`spce_ions.xml` was an empty stub).

Fixes (now `gromacs_parameterset.py` in the repo, stock ParmEd only):
- ITPs run through ParmEd's `CPreProcessor`; `#include` followed (each file once,
  shared defines); `defines={...}` selects branches.
- `[ moleculetype ]` → residue templates (atoms, bonds, `[ settles ]` → O–H bonds
  plus zero-k bond/angle types carrying the rigid geometry for OpenMM's
  `rigidWater`).
- `write_xml(exclude_residues=, include_residues=, drop_water_hh_bond=True,
  classes='all'|'used')`; `'used'` writes only referenced classes/bonded terms and
  no `NBFixPair` (duplicates across files are an OpenMM error).
- Improper `<Script>` now resolves ±1 residue neighbours by **actual bonds**
  rather than index, so ring-spanning impropers in cyclic molecules are included
  (12 for a hexamer; linear molecules unchanged at 15).
- `build_mosic_xml.py` reproducibly generates `mosic.xml` (peptoid residues,
  water/ions excluded) + `spce_ions.xml` (SPC/E `SOL` + CHARMM ions). Regenerated
  `mosic.xml` matched the hand-edited one residue-for-residue except the now
  correct `OT` ε. Load: `ForceField('mosic.xml','spce_ions.xml','ions_1264.xml')`.
- `translate_ff.ipynb`: XML-building cells commented out with a superseded note.
- Env audit (SHA-256 vs conda/pip records, plus diff against upstream ParmEd
  4.3.1): the only hand edits in `metal-toid` were `parmed/gromacs/parameters.py`
  and a 2-line `__init__.py` import; OpenMM, pdbfixer, mdtraj pristine.
  `alchemist` env untouched.

## 3. GPU / PLUMED in `metal-toid`

- PLUMED: `openmm-plumed 2.1` + PLUMED 2.9.2 work on CUDA/OpenCL/CPU (tested
  with a DISTANCE + RESTRAINT on Zn–Cl in SPC/E).
- CUDA was broken: the conda OpenMM 8.4 was a **CUDA-13** build but Midway3 GPU
  nodes run **driver 535 (CUDA 12.2)** → `cuCtxCreate_v4` missing. The CUDA-12.6
  build loads but fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (nvrtc 12.6 PTX
  too new for the driver's JIT). Fix: OpenMM 8.4 CUDA-12.6 build + **nvrtc pinned
  by hand to 12.2.140** (symlinks; note in `targets/x86_64-linux/lib/README_nvrtc_pin.txt`).
  Any conda transaction touching `cuda-nvrtc` undoes the pin.
- Result on A30: CUDA 160 ns/day, OpenCL 177, CPU 6.6 (2650-atom box), identical
  with or without `module load cuda/12.0|12.2|13.2` — the plugins resolve nvrtc
  from the env via RPATH, so **no CUDA module is needed**.
- System `conda`/`mamba` on login nodes are broken (libstdc++); standalone
  micromamba works. `openbabel` 3.2.1 and `imageio-ffmpeg` added to the env.

## 4. Cyclic peptoid builder (`structure_maker/make_cyclic.py`)

- Glycine backbone from per-residue (φ, ψ, ω) via NeRF (PeptideBuilder geometry);
  ring closure by `scipy.optimize.least_squares` over φ/ψ (ω too with
  `--flex-omega`) with a topological-distance backbone clash term, weak pull
  toward requested angles (`--angle-weight`), random restarts. Then the linear
  pipeline (obabel → `load_sequence` → `pdb_to_peptoid_forcefield`), no end caps,
  closing `CLP(n)–NL(1)` CONECT. `--omega-pattern ctctct`, `--anglefile`,
  `--mini`, `--relax` (MoSiC vacuum minimisation), `--trajectory` (multi-MODEL PDB
  of the closure path, optimizer steps interpolated in torsion space; residue
  `GAP` marks the virtual N of residue n+1), `--movie [mp4|gif]` via
  `render_closure.py` (matplotlib 3D, off-screen). Sample:
  `structure_maker/example_cyc6_closure.mp4`.
- Proline: pyrrolidine ring from PeptideBuilder `ProGeo` internal coordinates
  (`place()` verified identical to `calculateCoordinates`), φ frozen at −65°
  during closure; `PROd` naming as in the linear builder.
- Verified: hexamers/octamers incl. `ZPZPZP`, `ZGZGZG`; every residue matches its
  MoSiC template (2 external bonds), ring survives minimisation
  (CLP–NL 1.34–1.37 Å, proline rings intact).
- Findings on cis/trans fidelity: without `--flex-omega` the built ω pattern is
  exact by construction; after `--relax`, hexamer/octamer patterns survive within
  ~10–30°. Pentamers are strain-limited: `ccctt` and `ttttt` flip an amide on
  relaxation; cis-rich patterns (`ccccc`, `cctct`) survive. `--flex-omega` treats
  ω as soft as φ/ψ and can park amides at ~−100°; a stiffer ω weight would be the
  next improvement. Template idea (closed torsion sets per ring size/pattern,
  symmetry-constrained closure) discussed but not implemented.
- **Linear builder fix**: current mdtraj no longer regroups stacked sidechains per
  residue in `Topology.from_dataframe` (old mdtraj in `peptoid_env` did), so
  `make_structure.py` under `metal-toid` produced split residues OpenMM could not
  match. Added `regroup_by_residue()` (used by both builders) and an
  `if __name__ == "__main__"` guard so the module is importable.

## Files

Added: `make_1264_xml.py`, `ions_1264.xml`, `12-6-4_base_params.csv`,
`gromacs_parameterset.py`, `build_mosic_xml.py`, `structure_maker/make_cyclic.py`,
`structure_maker/render_closure.py`, `structure_maker/example_cyc6_closure.mp4`,
`CLAUDE.md`, `sri_docs/lj_1264_pol_custom.dat`, `sri_docs/parmed_1264.in`.
Modified: `mosic.xml`, `spce_ions.xml`, `structure_maker/make_structure.py`,
`translate_ff.ipynb`, `.gitignore` (stale merge markers removed; large data ignored).
