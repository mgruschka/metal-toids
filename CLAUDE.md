# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project runs on Midway3 (UChicago RCC). The full stack is activated via a shell alias:

```bash
toid_stack
# expands to:
module load python/anaconda-2022.05
source activate peptoid_env
module load rcc cmake openmpi/4.1.2+gcc-10.2.0 lapack/3.10.0 gsl/2.7 plumed/2.8.0+annfunc+opes vmd fftw3/3.3.9 gromacs/2021.4+plumed2-2.8.0
export GMXLIB="/scratch/midway3/gruschka/toid_explore/forcefields/"
```

All Python scripts and the notebook require this environment. External dependencies include `mdtraj`, `parmed`, `openmm`, `biopython`, `PeptideBuilder`, `numpy`, and `openbabel` (CLI tool `obabel`). The `metal-toid` env (below) now carries all of these including `obabel`, so `structure_maker/*.py` run there too; `make_structure.regroup_by_residue` was added because current mdtraj no longer regroups stacked sidechains per residue (older mdtraj in `peptoid_env` did), which otherwise produces PDBs whose residues OpenMM cannot match.

## OpenMM / GPU Environment (`metal-toid`)

`/scratch/midway3/gruschka/metal-toid` (Python 3.10) is the OpenMM env: OpenMM 8.4.0 (conda-forge CUDA-12.6 build), `openmm-plumed` 2.1 + PLUMED 2.9.2 (`from openmmplumed import PlumedForce` works on CUDA/OpenCL/CPU), ParmEd 4.3.1, pdbfixer. Midway3 GPU nodes (A30/L40S) run **driver 535 = CUDA 12.2**, so the env's nvrtc is hand-pinned to 12.2.140 (symlinks in `lib/` and `targets/x86_64-linux/lib/`, see `targets/x86_64-linux/lib/README_nvrtc_pin.txt`); newer nvrtc gives `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, and the CUDA-13 openmm build fails to load at all (`cuCtxCreate_v4`). Any conda/micromamba transaction touching `cuda-nvrtc` undoes the pin. **No `module load cuda/...` is needed or used** — the plugins resolve nvrtc/cufft from the env via RPATH, and loading cuda/12.0, 12.2 or 13.2 modules changes nothing (verified). The system `conda`/`mamba` on the login nodes are broken (libstdc++/entry-point errors); a standalone micromamba works (`curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba`). The CUDA platform is only testable on a GPU node (`sbatch` to `andrewferguson-gpu`); on login nodes only Reference/CPU load.

## Running the Structure Maker

`structure_maker/make_structure.py` must be run from inside the `structure_maker/` directory — it resolves `.nobkb_residue_pdb/` relative to the working directory.

```bash
cd structure_maker

# Build a peptoid at a named Ramachandran minimum
python make_structure.py --seq AAAAA --mini A+C --file molecule.pdb

# Build at explicit backbone angles
python make_structure.py --seq AAAAA --phi -61 --psi -50 --omega -11 --file molecule.pdb

# Load angles from a text file (numpy-loadtxt format, degrees or radians)
python make_structure.py --seq AAAAA --anglefile angle_files/somefile.txt

# Build a peptide instead of a peptoid
python make_structure.py --seq AAAAA --mini A+C --tide

# N-substituted alanine backbone (default is glycine)
python make_structure.py --seq AAAAA --mini A+C --nsa

# C-terminus options: nme (default), NH2, NDM
python make_structure.py --seq AAAAA --mini A+C --cter NH2
```

Intermediate files (`template_*.pdb`, `template_*_h.pdb`, `template_*_nh.pdb`, `temp_*.pdb`) are cleaned up automatically on success.

## Cyclic Peptoids (`structure_maker/make_cyclic.py`)

Head-to-tail macrocycles, same residue codes and working-directory rule as `make_structure.py`; run with the `metal-toid` python. A glycine backbone is built from per-residue (φ, ψ, ω) by NeRF, then φ/ψ (and ω with `--flex-omega`) are refined by least squares to close the ring without backbone self-overlap; the RMS deviation from the requested angles is printed and the final angles saved to `<file>_angles.txt`. Then the linear pipeline is reused (obabel → `load_sequence` → `pdb_to_peptoid_forcefield`) with no end caps and a `CLP(n)–NL(1)` CONECT. `--omega-pattern ctctct` sets cis/trans per residue (ω_i is the amide *preceding* residue i); `--relax` vacuum-minimises with `mosic.xml`; `--trajectory` writes `<file>_closure.pdb`, a multi-MODEL backbone PDB of the ring-closure path (optimizer steps interpolated in torsion space to `--trajectory-frames`, default 200; residue `GAP` marks the virtual N of residue n+1 that has to meet residue 1) — load with `vmd <file>_closure.pdb`, or `--movie [mp4|gif]` renders it in pure Python via `structure_maker/render_closure.py` (matplotlib 3D; MP4 through the ffmpeg bundled in `imageio-ffmpeg`, GIF through Pillow; `render_closure.py <closure.pdb>` also works standalone). `example_cyc6_closure.mp4` is a sample. Proline (`P`) gets its pyrrolidine ring from PeptideBuilder's `ProGeo` internal coordinates with φ frozen at −65° during closure (no N–H, so no sidechain step, as in the linear builder). Not supported: `--nsa`, n < 4. Extended patterns (e.g. `AD-T`) cannot close and are bent heavily — check the printed deviation. The improper `<Script>` in `mosic.xml` resolves ±1 neighbours by actual bonds (not index), so ring-spanning impropers are included (12 for a hexamer).

```bash
cd structure_maker
python make_cyclic.py --seq ZZZZZZ --phi -90 --psi 180 --omega-pattern ctctct --file cyc6.pdb --relax
```

## Residue Codes

Single-character sequence codes (passed to `--seq`) map to sidechain PDB templates in `structure_maker/.nobkb_residue_pdb/`. Standard amino acids use their one-letter codes; peptoid-specific residues use digits/special characters:

| Code | Residue | Code | Residue |
|------|---------|------|---------|
| `1` | NTBp | `2` | NPHp |
| `3`–`8` | BTM/BTE/CTM/CTE/FTM/FTE | `9`/`0` | ITM/ITE |
| `Z` | NSPE | `X` | NAEk | `J` | EME |
| `O` | NOE | `U` | DECp | `z` | NRPE |

## Named Ramachandran Minima

`MINIMA_FILENAMES` in `make_structure.py` defines the named configurations: `AD-C`, `AD+C`, `A-C`, `A+C`, `C7B-C`, `C7B+C`, and the same with `-T`/`+T` suffixes, plus `HELIX`. Each maps to a `[phi, psi, omega]` triplet.

## Force Field Translation (`translate_ff.ipynb`)

The notebook translates the MoSiC GROMACS force field (CHARMM36-based, located at `$GMXLIB`) into OpenMM XML format. Key workflow:

1. **ParmED** loads GROMACS `.top`/`.gro` files and converts to AMBER prmtop/inpcrd.
2. **`GromacsParameterSet`** (parmed) parses MoSiC `.itp` files and writes `mosic.xml` / `spce_mosic.xml`.
3. Validation compares CMAP torsion maps, nonbonded terms, and impropers against the reference OpenMM CHARMM36 XML. Known convention difference: GROMACS stores CMAP on a 0–2π grid vs OpenMM's −π to π; the matrix also requires transposition (phi-psi vs psi-phi ordering).

The `sri_docs/` directory contains files for adding 12-6-4 metal ion LJ parameters (for Cl⁻ with TIP3P) via ParmED's `add12_6_4` command.

`GromacsParameterSet` is custom code kept in the repo as `gromacs_parameterset.py` (needs only stock ParmEd; the original copy inside the `metal-toid` env at `parmed/gromacs/parameters.py` is now redundant). It runs ITP files through ParmEd's `CPreProcessor` (so `#ifdef HEAVY_H`-style branches and `#include` are honoured; pass `defines={...}` to pick other branches), turns `[ moleculetype ]` blocks (water models, ions) into residue templates, and `write_xml(..., exclude_residues=, include_residues=, classes='all'|'used')` filters what is written. `build_mosic_xml.py` is the reproducible replacement for the hand-edited XMLs: `mosic.xml` (peptoid residues, water/ion residues from `merged.rtp` excluded) + `spce_ions.xml` (SPC/E `SOL` + CHARMM ions, used classes only, no NBFIX). Load them together: `ForceField('mosic.xml', 'spce_ions.xml', 'ions_1264.xml')`. `spce_mosic.xml` is the older monolithic file with TIP3P-charged `HOH` and hand-deleted entries; prefer the pair above.

## 12-6-4 Ion Parameters (`make_1264_xml.py`)

`12-6-4_base_params.csv` holds the Li/Merz 12-6-4 SPC/E ion set (R_min/2, ε, base C4; identical to ParmEd's `DEFAULT_C4_PARAMS['SPCE']`). `make_1264_xml.py` turns it into `ions_1264.xml`: Amber-style single-atom residues (`Zn2+`, `Cl-`, `Ce3+`, ... atom name `ZN`, `CL`, ...), 12-6 terms in `<LennardJonesForce>` to match the mosic XMLs, and an embedded `<Script>` that adds a `CustomNonbondedForce` named `LJ1264C4Force` with the ParmEd C4 mixing rule (`C4_ij = C4_i * α_j/1.444`, global parameter `lambda_c4`). Load with `ForceField('mosic.xml', 'spce_ions.xml', 'ions_1264.xml')` and pin ion residues via `residueTemplates={res: res.name ...}`, otherwise they also graph-match CHARMM's `ZN2`/`CLA`. `--lj-style nonbonded` produces the Amber convention (LJ inside `<NonbondedForce>`) for use with `amber14/spce.xml`. Requires an OpenMM-enabled Python (e.g. `/scratch/midway3/gruschka/metal-toid/bin/python`; `peptoid_env` has no OpenMM).

## GROMACS Utilities

`structure_maker/_gromacs_utils.py` provides Python wrappers for common GROMACS commands: `grompp`, `mdrun`, `pdb2gmx`, `editconf`, `insert_molecules`, `make_ndx`, and `energy`. Use `mpirun=False` for single-node (`gmx`) and `mpirun=True` for MPI (`gmx_mpi`).

## Key File Locations

- Force fields: `/scratch/midway3/gruschka/toid_explore/forcefields/` (set via `$GMXLIB`)
- Simulation outputs referenced in notebook: `/scratch/midway3/gruschka/toid_explore/simulations/`
- CHARMM36 toppar: `toppar_c36_jul21/` (standard CHARMM36 July 2021 release)

## Ligand 12-6-4 Parametrization (`ligand-1264/`)

OpenMM pipeline for tuning ion↔ligand-class C4 mixing (`arel` scale per CHARMM class) with the ion's base C4 fixed. YAML config → build (ion placed near a ligand atom, truncated-octahedron SPC/E box with 48 Å face-to-face width (= Li/Merz's tLEaP buffer 24 Å; ~5100 waters, ~10 mM), `mosic.xml`+`spce_ions.xml`+`ions_1264.xml`; CGenFF residues such as `IMIA` imidazole are already in `mosic.xml`, and template bonds are registered so ligand PDBs need no CONECT) → 3-stage restrained minimization → 1 ns NVT heating → NPT equil/prod → `analysis.py`. `lig1264/ti.py` is the MIRIAM-style TI driver (ion A → B alchemical swap with linear energy mixing, 12 Gauss–Legendre windows, `--window N` for SLURM arrays, `--analyze`); `ligand.count` places several ligands around the ion (`count: 0` = water leg). Run with the `metal-toid` python: `python -m lig1264.run configs/x.yaml --outdir runs/x [--resume] [--stop-after build]`; `slurm/run_one.sbatch` for GPU jobs. See `ligand-1264/README.md` for the layout and design notes.
