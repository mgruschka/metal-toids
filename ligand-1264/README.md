# ligand-1264 — 12-6-4 parameters for non-water ligand atom types

Second half of the Li/Merz 12-6-4 scheme.  `metal-solvation/` fixes each ion's
base C4 against water (HFE + IOD).  Here that base C4 is held fixed and what is
tuned is how it is mixed with **ligand** atom classes — the `arel` factor in

    E_C4 = -lambda_c4 * (C4_ion * arel_j + C4_j * arel_ion) / r^4

which `ions_1264.xml` sets from Miller polarizabilities (α_j / 1.444).  A
config's `c4.scale_by_class` multiplies `arel` for chosen CHARMM classes on the
ligand (carbonyl `O`, amide `NH1`, carboxylate `OC`, ...) — that scale factor is
the fitted parameter.  Everything is OpenMM + the repo's CHARMM/MoSiC XMLs.

## Protocol (per system)

| stage    | what                                                                                  | outputs |
|----------|---------------------------------------------------------------------------------------|---------|
| build    | ligand PDB (+ ion placed `distance_A` from `near_atom`; `ligand.count` > 1 arranges copies octahedrally around the ion, `count: 0` is a bare ion), truncated-octahedron SPC/E box, `diameter_A` = face-to-face (minimum-image) width (44.5 Å = tLEaP `solvateoct 24.0` as in MIRIAM — box *size* only, MIRIAM's solvent is OPC: 124,600 Å³, ~4040 waters; 48 Å gives the ~5100 waters / 10 mM quoted in the paper) or `buffer_A` = solute-to-face distance, neutralising Na⁺/Cl⁻ from `ions_1264.xml` | `topology.pdb`, `system.xml`, `groups.json`, `classes.json` |
| minimize | 1. ligand + all ions restrained (water relaxes)  2. everything restrained except ligand H  3. all free | `min_1_solute_restrained.pdb`, `min_2_ligandH_only.pdb`, `min_3_all_free.pdb` |
| heat     | NVT, T ramped 0 → 300 K over `heat_ns` (1 ns) in `heat_steps` increments             | `heat.csv`, `heat_final.pdb`, `heat.chk` |
| equil    | NPT (MC barostat), `npt_equil_ns`                                                     | `npt_equil.{csv,dcd,chk}`, `npt_equil_final.pdb` |
| prod     | NPT, `npt_prod_ns`                                                                    | `npt_prod.{csv,dcd,chk}`, `npt_prod_final.pdb` |
| analysis | ion↔ligand-class min distance / RDF / CN, ion–water IOD & CN                          | `analysis_prod.{json,npz}` |

Restraints are a `CustomExternalForce` with per-particle `k`, so all three
minimizations, heating and NPT share one Context; the barostat is present from
the start with frequency 0 and switched on for NPT.  Every stage leaves a marker
file, and `--resume` skips completed stages (requeue-safe).

## Thermodynamic integration (`lig1264/ti.py`) — consistency check

Mirrors MIRIAM (`github.com/lizhen62017/MIRIAM`, `TI/`): the ion is
alchemically transformed **A → B** (e.g. Ni²⁺ → Co²⁺) once in the presence of
`ligand.count` ligands and once in plain water (`count: 0`); the difference of
the two ΔG's is the *relative* binding free energy of the two metals.

α(NG2R50) is no longer fitted here — that is the PMF's job (next section).
TI keeps a different and useful role: it measures a metal-vs-metal difference
by a completely different route (alchemical, no reaction coordinate, no
WHAM), so agreement between a PMF-fitted α and the TI ΔΔG at that α is a real
check, whereas fitting both to the same number would not be.

* Linear energy mixing `U = (1−λ)U_A + λU_B` for everything involving the ion
  (`forces.make_ion_alchemical`): charge via a `NonbondedForce` parameter
  offset, LJ by zeroing the ion's row of the `LennardJonesForce` table and
  adding an ion↔all `CustomNonbondedForce` with Lorentz–Berthelot mixing, C4
  via an extra `dcfour` per-particle term.  Validated: energies at λ=0/1 match
  plain A/B systems to <0.1 kJ/mol (the residual is the ion's share of the LJ
  dispersion correction, λ-independent), and analytic dU/dλ from
  `getEnergyParameterDerivatives` equals `U_B − U_A`.
* Protocol (`ti:` section): build → 3-stage minimization → NVT heat 0.36 ns →
  NPT 2 ns, all at the first λ node → 12 Gauss–Legendre windows (Amber's
  `clambda` list) × 5 ns NVT at 1 fs, sequential; dU/dλ every 0.5 ps.
  ΔG = Σ wᵢ⟨dU/dλ⟩ᵢ with block-averaged errors → `ti_results.json`.
* `ti.traj_ps` > 0 also writes `window_NN.dcd` every that many ps (default 0 =
  off, as in round 1).  Needed to check *post hoc* that the coordination shell
  survived production — without it the windows leave only checkpoints.  At
  10 ps that is 500 frames / ~30 MB per window, ~350 MB per transformation.
  Written unwrapped (`enforcePeriodicBox=False`), like the NPT DCDs, so the
  complex stays whole; `python -m lig1264.analysis <run> --phase window_05`
  reads them (the `npt_<phase>.dcd` name falls back to `<phase>.dcd`).
* `--window N` runs one window from the NPT checkpoint (SLURM array),
  `--analyze` integrates whatever `dudl_w*.csv` exist.

Cost per transformation ≈ 62 ns at 1 fs.  Per metal in the cyclic scheme
(water leg + ~4 complex-leg iterations) ≈ 3–4 GPU-days serial, ≲1 day wall
with array parallelism.

## Potential of mean force (`lig1264/pmf.py` + `lig1264/wham.py`) — the fitting engine

Umbrella sampling of **one ligand being pulled off the ion**, biased through
PLUMED (`openmm-plumed` 2.1 / PLUMED 2.9.2, which works on CUDA here).  The
collective variable is the single distance d(ion, `pmf.pull_atom`) — for
imidazole the Nε — of the ligand copy nearest the ion; the other copies stay
bound.  The PMF well depth is what `c4.alpha_by_class` is now fitted against.

| stage    | what                                                                                   | outputs |
|----------|----------------------------------------------------------------------------------------|---------|
| build    | identical to `run.py` / `ti.py` (`build.build_system_files`)                             | `topology.pdb`, `system.xml`, `groups.json`, `classes.json` |
| minimize | (a) water only, imidazoles + ions restrained  (b) ligand H only free  (c) everything free | `min_*.pdb` |
| heat     | NVT 0 → 300 K over `pmf.heat_ns` (1 ns)                                                  | `heat.csv`, `heat.xml` |
| equil    | NPT, `pmf.npt_equil_steps` (550,000 × 2 fs = 1.1 ns)                                     | `equil.csv`, `equil.xml` |
| prod     | NPT, `pmf.npt_prod_steps` (500,000 × 2 fs = 1.0 ns)                                      | `prod.csv`, `prod.xml` |
| smd      | PLUMED `MOVINGRESTRAINT` pulls the CV from the bound distance to `pmf.r_end_A` over `pmf.smd_ns`; Jarzynski work recorded | `smd.dcd`, `smd_colvar.dat`, `smd.plumed` |
| seed     | `pmf.n_windows` (120) SMD frames, one per umbrella centre, evenly spaced in the CV        | `windows.json` |
| windows  | per window: `pmf.window_equil_ns` (2 ns) then `pmf.window_prod_ns` (**4 ns**, see below), both NPT under a PLUMED `RESTRAINT` at that centre | `window_NNN_colvar.dat`, `window_NNN.xml` |
| wham     | binned WHAM + block bootstrap                                                             | `pmf.csv`, `pmf.json`, `pmf.png` |

Equilibration and production of each window run in **separate Contexts** so
their COLVARs are separate files and WHAM never has to guess which samples
are production.  Stages hand over through serialized `State` XMLs, not
checkpoints, because the SMD and umbrella Contexts carry an extra
`PlumedForce` and a checkpoint is tied to the exact Force layout.

**Window length: 4 ns, deliberately short.**  MIRIAM ran both 4 and 40 ns per
window and found the π-stacked intermediate on the dissociation path only
resolves at 40 ns; missing it overestimates the binding of a single imidazole,
which here biases the fitted α **low**.  4 ns is the cheap setting for turning
the secant loop (120 windows × 4 ns = 480 ns vs 4.8 μs per iteration).  Treat
a 4 ns α as provisional and re-run the accepted value at
`window_prod_ns: 40.0` before quoting it.  `pmf_provenance.json` records the
length used.

`W(r)` from WHAM is the PMF in the radial coordinate, so it still contains the
2kT ln r volume term; `pmf.csv` also carries `W_jacobian = W(r) + 2kT ln r`,
which is the one to use for a binding free energy (plus the usual
standard-state term).

### Where this departs from the AMBER protocol it reproduces

Both are forced by OpenMM, and both are recorded in each run's
`pmf_provenance.json`:

* **Minimizer.**  OpenMM has only L-BFGS; there is no steepest-descent /
  conjugate-gradient pair.  "10,000 SD + 10,000 CG cycles" per stage maps onto
  `maxIterations: 20000`.  The stage *structure* (water → ligand H → all) is
  exactly as specified.
* **Barostat.**  OpenMM ships no Berendsen barostat, and exposes no virial, so
  one cannot be written either.  `MonteCarloBarostat` (frequency 25) is used.
  It samples a rigorous NPT ensemble — which Berendsen does not — but its
  relaxation is set by the MC acceptance rate, not by a 5 ps time constant.

**SHAKE is not a deviation** — OpenMM does implement it.  On the CUDA
platform `IntegrationUtilities` splits the constraints three ways: SETTLE for
rigid water, **SHAKE** for a heavy atom with its attached hydrogens, and CCMA
for anything left over.  With `constraints=HBonds` + `rigidWater=True` every
constraint in this system falls in the first two, so the production runs use
SHAKE and SETTLE exactly as AMBER does.  One caveat: the Reference and CPU
platforms have no SHAKE path and fall back to CCMA, so a CPU test run is not
numerically identical to the CUDA production run — same constrained degrees
of freedom, different solver, both converged to the integrator's 1e-5
constraint tolerance.

Everything else is as specified: 2 fs timestep, 10 Å nonbonded cutoff,
Langevin thermostat at 1 ps⁻¹, 300 K.

### From a PMF to a binding free energy

The PMF **well depth is not a binding free energy** and must not be compared
to an experimental ΔG directly — it leaves out the translational entropy of
localising the ligand into the bound volume, and it has no standard state.
`lig1264.wham` does the conversion:

    w(r)  = W(r) + 2kT ln r          # orientation-averaged pair PMF, 0 at the plateau
    K_eq  = int_0^{r*} 4 pi r^2 exp(-w(r)/kT) dr        [A^3]
    dG0   = -kT ln(K_eq / V0),        V0 = 1660.54 A^3 (1 M)

and prints `dG0`, `K_eq`, the bound-state cutoff `r*` it chose (the first
barrier past the minimum, overridable with `--r-bound`) and `dG0` over a
range of `r*` in `pmf.json`, because that choice is a real source of
ambiguity.  No orientational restraint correction is needed: the CV is a bare
distance with the ligand free to rotate, so w(r) is already the
orientation-averaged pair PMF.

Two things the analysis guards against, both found by testing the estimator
against an analytic Morse pair PMF (recovered ΔG⁰ = −3.18 vs −3.227
kcal/mol exact):

* Bins outside the span of the umbrella centres are reached only by one
  window's sparse tail, so WHAM cannot anchor them and they come out as
  noise of order kT.  They are trimmed by default (`--no-trim` to keep them);
  left in, they wrecked the plateau reference and shifted ΔG⁰ by 0.03 kcal/mol
  while firing a spurious warning.
* If the dissociated end is not flat, `r_end_A` is too short and ΔG⁰ is
  referenced to a state that is still interacting.  `wham` reports the
  plateau slope and warns above 0.05 kcal/mol/Å.

### Fitting alpha: the Li/Merz iteration, Co(II) + imidazole

Li/Merz do **not** scan and fit — they run a secant iteration to a tolerance:

    run a PMF at some alpha -> dG0
    |dG0 - dG0_exp| <= 0.1 kcal/mol ?  record alpha, C4 and dG0, stop
    otherwise                       ?  linear extrapolation through the two
                                       nearest points -> new alpha -> new PMF

`lig1264.fit_alpha` does one turn of that loop.  It reads every finished run
(`config.yaml` + `pmf.json`), tabulates α / C4_eff / ΔG⁰ / error / well depth
/ r* / plateau slope / overlap, and either declares convergence or prints the
next α — with `--write-config`, the YAML for it too.

Target ΔG⁰ = −3.36 kcal/mol is −RT ln K with log K = 2.46 at 298 K, i.e. the
**stepwise first** association constant, so the system is one Co²⁺ and **one**
imidazole (`ligand.count: 1`), not the four-ligand TI complex.  Co's 12-6 and
base C4 stay fixed (R_min/2 1.457 Å, ε 0.0271 kcal/mol, C4 209.2 kcal Å⁴/mol);
only α(NG2R50) moves.  The effective pair term reported alongside it is

    C4_eff = C4_ion * alpha_class / 1.444        [kcal/mol A^4]

because `ions_1264.xml` mixes as `-(cfour1*arel2 + cfour2*arel1)/r^4` with
`arel = alpha/1.444`, and ligand atoms carry `cfour = 0`.

**Iteration 1 starts at α = 1.090 Å³**, which is not a chosen scan point: it
is what `_alpha()` in `ions_1264.xml` assigns to any aromatic nitrogen by
default (Miller N in `_pol_element`), i.e. the unmodified 12-6-4 model — the
same starting point Li/Merz use.  One α at a time, as the paper does:

    cd ligand-1264
    PY=/scratch/midway3/gruschka/metal-toid/bin/python
    R=runs/pmf/co_imid1_a109
    sbatch slurm/run_pmf.sbatch configs/pmf/co_imid1_a109.yaml $R seed
    # when windows.json exists:
    sbatch --array=0-119%20 slurm/run_pmf.sbatch configs/pmf/co_imid1_a109.yaml $R
    $PY -m lig1264.wham $R --bootstrap 50
    $PY -m lig1264.fit_alpha runs/pmf/co_imid1_* --target -3.36 --write-config
    # -> next alpha + its config; run it, wham it, call fit_alpha again

**After one run there is still a slope**, because the C4 term's origin is the
charge–induced-dipole energy

    U_ind = -(1/2) alpha_0 [q / (4 pi eps0 eps_r r^2)]^2  ==  -C4_ij / r^4
    =>  C4_ij = (1/2) * 332.0637 * q^2 * alpha_0 / eps_r^2   [kcal/mol A^4]

which is **exactly linear in alpha_0**.  (Checked against the fitted table:
Co²⁺ / Ni²⁺ / Zn²⁺ all come out at eps_r ≈ 2.1, so the empirical C4 is ~22%
of the gas-phase value; Na⁺ needs eps_r ≈ 6.3, i.e. for weakly polarising
ions the fit absorbs more than polarisation.)  Linearity means

    U(alpha) = U0 + alpha * V,     V = dU/dalpha = -(C4_ion/1.444) / r^4

so `dW(r)/dalpha = <V>_r` by linear response.  And because **imidazole has
exactly one NG2R50 atom — NE2, which is the CV partner** (ND1 is NG2R51, a
different class), V is a deterministic function of the collective variable,
the conditional average is V itself, and

    dW(r)/dalpha = -(C4_ion/1.444) / r^4        closed form, no averaging

`fit_alpha` shifts the measured PMF by that and re-runs the binding integral,
predicting ΔG⁰ at any alpha' from iteration 1 alone — a Newton step instead of
a second expensive PMF spent only to get a slope.  Verified against an
independent Boltzmann average of dU/dalpha over the bound state: both give
−6.64 kcal/mol per Å³ on the same test PMF.

This is a *linear-response* estimate: it moves the PMF but not the underlying
ensemble, so it ignores solvent reorganisation and **overstates** how fast
ΔG⁰ falls.  Treat the predicted alpha as a lower bound on the step, run it,
then let the secant take over with two real points.  `fit_alpha` says so, and
warns when the fitted class has more than one atom on the ligand (the
4-imidazole system has four; only the CV partner is modelled, so there the
prediction is approximate).  As a cruder cross-check it also reports the bare
pair-term sensitivity C4_ion/(1.444 r_min⁴) at the PMF minimum.

`configs/pmf/co_imid1_a{275,450}.yaml` are left over from an earlier
scan-and-fit plan and are marked as not part of the iteration.  The
"α = 2.75 = MIRIAM MUTE" attribution they inherited from
`configs/ni_co_imidazole4.yaml` has since been traced upstream — see
[Provenance of α = 2.75](#provenance-of-α--275-and-what-miriam-actually-did)
below.  It is a real number from the literature, but it is **not** ours to
reuse: it is an OPC/ff19SB value, it is the superseded single-imidazole
parameter, and MIRIAM applied it under a different C4 convention.  Keep
treating it as a scan point.

**The ±0.1 kcal/mol criterion is tight, and only means something if the
statistical error is well under it.**  `wham --bootstrap N` now propagates its
replicas through the binding integral and reports `dG0_sem_kcal`;
`fit_alpha` flags any run whose error exceeds half the tolerance and refuses
to call that convergence.  Two reasons to watch this:

* The estimator itself carries ~0.05 kcal/mol of binning error even on
  perfectly independent samples (measured against an analytic Morse pair
  PMF).  That is already half the budget before any MD noise.
* r* is a choice.  `pmf.json` records ΔG⁰ over a range of r*; if ΔG⁰ moves by
  more than ~0.1 kcal/mol across sensible cutoffs, the tolerance is finer
  than the definition of the bound state and chasing it is not meaningful.

Parameters are fitted **per metal–ligand pair**, as Li/Merz do: the α(NG2R50)
fitted on Co²⁺ is α(Co–NG2R50) only, and Ni²⁺ gets its own value from its own
single-imidazole PMF against its own log K₁.  In a single-metal system a
per-class parameter and a per-pair parameter are the same thing, so the PMF
code needs nothing extra.

The Ni→Co TI is therefore a **consistency check between two fits**, not an
independent test.  It can only run once both pair parameters exist, and it
needs the imidazole N's `arel` to differ between the end states (α(Ni–N) at
λ = 0, α(Co–N) at λ = 1).  `forces.make_ion_alchemical` does not do that yet:
it morphs only the ion's own C4 and keeps one ligand `arel` for both end
states.  The pair α cannot be folded into the ion's C4 instead, because that
also sets the ion–water term, which must stay at its HFE/IOD-fitted value.

### Cost

120 windows × 42 ns = **5.04 μs per α value**, and the Li/Merz iteration is
strictly sequential — budget one turn per day of wall time, typically 3–5
turns from the α = 1.09 start.  At the ~114 ns/day the TI legs
managed at 1 fs on a V100, 2 fs gives roughly 230 ns/day, so one window is
~4.5 h and the whole scan is ~22 GPU-days serial — but it parallelises
perfectly: `--array=0-119%20` finishes in a bit over a day.  Prep + SMD is one
~5 ns serial job (~30 min).  Trim `pmf.n_windows` / `pmf.window_prod_ns` first
if you want a cheap pilot; 120 windows over ~6 Å is 0.05 Å spacing, which is
dense enough that far fewer windows would still overlap well.

### Running

    cd ligand-1264
    PY=/scratch/midway3/gruschka/metal-toid/bin/python
    # 1. prep + SMD + seeding (one GPU)
    sbatch slurm/run_pmf.sbatch configs/pmf/ni_imid4_a275.yaml runs/pmf/a275 seed
    # 2. the umbrella windows
    sbatch --array=0-119%20 slurm/run_pmf.sbatch configs/pmf/ni_imid4_a275.yaml runs/pmf/a275
    # 3. WHAM (no GPU)
    $PY -m lig1264.wham runs/pmf/a275 --bootstrap 50

`--stop-after {build,prep,smd,seed,windows}` and `--resume` work as in
`run.py`/`ti.py`; `--window N` runs a single window.

Two diagnostics decide whether the run is worth the GPU time, both printed
before any window starts:

* **SMD lag.**  `stage_smd` reports how far the CV falls behind the moving
  restraint.  A soft spring does not break a metal–ligand bond — it just
  stretches, the bond snaps late, and most of the CV range is never visited.
  Measured here on the Ni/imidazole system: at `smd_kappa: 20` the lag grew
  linearly past 1 Å with the Ni–N distance never leaving 2.1 Å, while a stiff
  spring tracked the centre to ±0.1 Å.  Hence the 100 kcal/mol/Å² default and
  the warning when the lag exceeds 1 Å.
* **Window overlap.**  `lig1264.wham` prints the minimum and median
  adjacent-window histogram overlap along with the well depth; below ~0.03
  there is a gap in the reaction coordinate and the PMF is not trustworthy
  across it.

## Provenance of α = 2.75, and what MIRIAM actually did

Checked against the MIRIAM repo (`github.com/lizhen62017/MIRIAM`) and the two
papers behind it.  Four things differ from our setup; none of them are fatal,
but none of the numbers transfer unexamined.

**1. α = 2.75 Å³ is real, inherited, and superseded.**  `TI/lj_1264_pol.dat`
carries the single line `MUTE  2.750    From Gaurav` — Gaurav Sharma is a
coauthor of the earlier single-imidazole parametrization, Li, Song, Sharma,
Koca Fındık & Merz, *Accurate Metal–Imidazole Interactions*, JCTC **2023**,
19, 619 (α of the chelating N fit against experiment for 11 metals in TIP3P,
**SPC/E** and OPC).  The MIRIAM paper (Li et al., *Simulating Metal-Imidazole
Complexes*, JCTC **2024**, 20, 6706) then re-fits with 40 ns/window sampling,
finds the old values overestimate binding once a π-stacked intermediate is
resolved — "therefore, lower polarizability values for nitrogen are needed" —
and tabulates 2.75 in the *one-imidazole-compatible* column of its Table 2,
recommending the lower extended-sampling values for multi-imidazole systems.
Which metal 2.75 belongs to is not determinable from the repo: the TI run uses
the same MUTE = 2.75 for both the Ni²⁺ and the Co²⁺ endpoint, differing only
in the ion's base C4 (`TI/c4_pvdw.txt`: Ni2 212, Co2 204, Cu2 291, Mn2 175,
Zn2 225).  Table 2 is a bitmap in the open-access copy — read the SPC/E column
of the 2023 paper through institutional access before adopting any value.

**2. MUTE is a cloned atom type, not a parameter.**  MIRIAM splits the
chelating nitrogen off its parent type so it can carry its own polarizability:

    change AMBER_ATOM_TYPE @9,18,27,36 MUTE     # the four imidazole N(δ, NB)
    addLJType @%MUTE                            # same 12-6, new C4 row
    add12_6_4 :5,6 c4file c4_pvdw.txt polfile lj_1264_pol.dat

**3. Their C4 is on the ligating nitrogen only** — reproduced by
`c4.zero_other_classes: true` (on in all `configs/pmf/*.yaml`), which sets
`arel = 0` for every ligand class not named in `alpha_by_class` /
`scale_by_class`, and for the counter-ions.  Water and the metal are
untouched, so ion↔water keeps the HFE/IOD-fitted C4 exactly (water O has
`arel = 1` by construction) and Cl⁻ keeps its own −55 kcal·Å⁴/mol to water.
Verified on the built Co/imidazole system: afterwards the only non-zero
`arel` in the whole box outside water are `NG2R50` (0.755) and Co²⁺ (0.310).
The original all-classes behaviour is still the default (`false`), which is
what round 1/2 TI and `runs/pmf/co_imid1_a109.pre_miriam_allclass_40ns`
used — **α fitted under the two conventions is not comparable**, and the TI
configs have deliberately been left on the old one so the round-1 results
stay self-consistent.  Every one of the other 274
entries in `lj_1264_pol.dat` is `0.000` — including `N`, `NA`, `NB`, and `OW`.
ParmEd special-cases water (`add1264.py`: `if attypjs == ['OW']: result = c4`),
so the ion↔water C4 keeps its HFE/IOD-fit value regardless, and every
non-water, non-MUTE pair gets `c4/1.444 * 0 = 0`.  **Our `ions_1264.xml` does
the opposite**: `_pol_element` gives every class a Miller polarizability, so
the ion also sees C4 attraction from the ring carbons, the other nitrogen and
the rest of the solute.  At equal α the two models are not the same
Hamiltonian, and our fitted α will come out lower to compensate.

**4. Different solvent and ligand charges.**  `TI/tleap2.in` is
`solvateoct a OPCBOX 24.0` with ff19SB, `frcmod.opc`, and RESP HID charges
(`cooresp.mol2`, types CC/CV/CR/NA/NB).  We use SPC/E with CGenFF `IMIA` from
`mosic.xml`.  12-6-4 base C4 values are water-model-specific, so our Li/Merz
SPC/E set and their OPC set are different numbers for the same ion.

**Geometry.** MIRIAM did not build an idealized coordination polyhedron.  The
paper takes starting geometries from the lowest-energy PMF structures, and the
deposited `TI/input.pdb` is a relaxed snapshot: M–N 2.19–2.26 Å, N–M–N angles
76/79/91/95/101/165°, metal 0.04–0.23 Å out of each ring plane, with the two
ions superposed at identical coordinates and separated only by
`timask1=":5", timask2=":6"`.  Restraints in the PMF leg are distance-only
(`equi_dis.RST`: flat-bottom at 2.21 Å, rk = 100 kcal/mol/Å²; SMD pulls
2.21 → 11.01 Å) — never angular, so orientation is left to the force field.
`build.place_ligands` instead constructs an ideal octahedron at
`ion.distance_A`, which is a *more* strained start; NPT equilibration should
erase the difference, but that is the likely origin of the 2.55 Å stretched
Ni–N seen at α = 4.50.

## Layout

    lig1264/config.py     YAML -> dataclasses (IonSpec, LigandSpec, C4Spec, BoxSpec, MDSpec)
    lig1264/build.py      force field, ion placement, solvation, createSystem, C4 edits, sanity checks
    lig1264/forces.py     restraint force; find/edit LJ1264C4Force (scale_by_class, alpha_by_class,
                          live-Context scan helpers)
    lig1264/stages.py     minimize_3stage, heat_nvt, run_npt
    lig1264/run.py        free-MD CLI driver with --resume / --stop-after
    lig1264/ti.py         TI driver (A -> B ion swap), --window / --analyze
    lig1264/pmf.py        PMF driver: prep -> SMD -> window seeding -> umbrella windows
    lig1264/wham.py       binned WHAM + block bootstrap + standard-state dG0 -> pmf.{csv,json,png}
    lig1264/fit_alpha.py  Li/Merz secant iteration: one turn of alpha -> dG0 -> next alpha
    lig1264/analysis.py   observables from the DCD
    configs/*.yaml        one file per system (ion, ligand, classes, box, lengths)
                          ni_co_imidazole4.yaml  MIRIAM-style TI: Ni2+ -> Co2+, 4 imidazoles, alpha(NG2R50)=2.75
                                                 (2.75 is inherited, not fitted here -- see Provenance below)
                          ni_co_water.yaml       water leg of the same cycle (count: 0)
                          zn_imidazole.yaml      free MD, Zn2+ / one imidazole
                          zn_ala5.yaml           free MD, Zn2+ / capped Ala5 peptide
    configs/round1/       alpha(NG2R50) scan 1.09 / 2.75 / 4.50 + water leg -- done, see Results
    configs/round2/       alpha 4.00 / 4.50 / 5.00, with ti.traj_ps: 10.0 (window DCDs)
    configs/pmf/          umbrella-sampling PMF configs
                          co_imid1_a{109,275,450}.yaml  Co2+ + 1 imidazole, alpha scan
                          ni_imid4_a275.yaml            Ni2+ + 4 imidazoles (TI-matched system)
    ligands/<name>/       ligand PDB (+ XML if not already in mosic.xml)
                          imidazole/make_imidazole.py builds imidazole.pdb with IMIA names
    slurm/run_one.sbatch  GPU job: free MD run + analysis
    slurm/run_ti.sbatch   GPU job: TI, sequential or --array=0-11 one window per task
    slurm/run_pmf.sbatch  GPU job: PMF prep/SMD/seed, or --array=0-119 one umbrella window
    runs/<name>/          outputs (git-ignored)

## Running

    cd ligand-1264
    PY=/scratch/midway3/gruschka/metal-toid/bin/python
    $PY -m lig1264.run configs/zn_imidazole.yaml --outdir runs/zn_imidazole --platform CUDA
    $PY -m lig1264.analysis runs/zn_imidazole
    # or
    sbatch slurm/run_one.sbatch configs/zn_imidazole.yaml

CPU runs need `--allow-cpu` (deliberately, as in phase1).  On a login node use
`--stop-after build` to check template matching / C4 assignment without a GPU.

## Design notes

* Ion LJ + base C4 come from `ions_1264.xml` (via `12-6-4_base_params.csv`),
  not a hand-written ion XML: mosic.xml keeps LJ in `<LennardJonesForce>`, and
  an ion with LJ in `<NonbondedForce>` would silently lose its ion–water LJ.
* All non-water residues are pinned by name in `residueTemplates` (Zn2+ vs
  CHARMM ZN2 are graph-identical); Modeller's counter-ions `NA`/`CL` are mapped
  onto `Na+`/`Cl-`.  `build.make_system` aborts if the ion ends up with C4 = 0.
* CGenFF is already inside `mosic.xml`: the GROMACS charmm36-feb2021 port
  merges CGenFF 4.4 (`IMIA` imidazole, `IMIM` imidazolium, ... with the
  `CG2R51`/`NG2R50` classes), so small-molecule ligands from CGenFF need only
  a PDB with the right residue/atom names.  Fresh CGenFF molecules (from the
  CGenFF server) still need their own XML, see `ligands/README.md`.
* `Topology.loadBondDefinitions` cannot read force-field XML, so
  `build.register_template_bonds` feeds the template bonds into
  `Topology._standardBonds` before the PDB is parsed; intra-residue bonds
  then need no CONECT records.
* Ligand classes are taken from the same template match `createSystem` used
  (`forces.atom_classes`), so `scale_by_class` keys are exactly the CHARMM
  classes in the XML — `classes.json` in the run dir lists what each atom got.
* Scans over the scale factor can rebuild (one config per value, SLURM array)
  or edit a live Context (`forces.set_class_scale`) for cheaper reweighting /
  TI-style work later.
* Restarting an umbrella window from a DCD frame has to rebuild the box with
  `openmm.app.internal.unitcell.computePeriodicBoxVectors`.  A DCD stores the
  cell as lengths + angles in float32, and for this truncated octahedron
  |c_y| is *exactly* b_y/2, so mdtraj's reconstructed vectors land a rounding
  error outside OpenMM's reduced form and `setPeriodicBoxVectors` rejects
  them.  Rebuilding from lengths/angles gives the reduced representation of
  the same lattice (here c' = c + a - b), so unwrapped positions and
  minimum-image distances are unchanged.  A cubic box never hits this.
* `pmf.py` sets `PLUMED_MAXBACKUP=0`.  Switching the barostat on calls
  `Context.reinitialize()`, which rebuilds the `PlumedForce`; PLUMED then
  re-opens its `PRINT` file and renames the old one to `bck.0.<name>`.  The
  live file holds the post-reinitialize data either way, but two empty
  backups per window x 120 windows is pure clutter.

## Results — round 1: α(NG2R50) scan, Ni²⁺ → Co²⁺ with 4 imidazoles

`configs/round1/`, `runs/round1/`, 2026-09-18.  Four TI legs, 12 windows × 5 ns
each (62 ns per leg), one V100 each, ~12 h wall per leg.  α only enters through
ion↔NG2R50 pairs, so the single `count: 0` water leg is the common reference for
all three ligand legs and does **not** need repeating per α.

| leg          | α (Å³) | ΔG(Ni²⁺→Co²⁺) | block SEM |
|--------------|--------|-----------------|-----------|
| water (0 lig)| —      | 15.283          | 0.008     |
| 4× imidazole | 1.09   | 12.662          | 0.005     |
| 4× imidazole | 2.75   | 14.977          | 0.006     |
| 4× imidazole | 4.50   | 18.125          | 0.006     |

kcal/mol.  ΔΔG = ΔG(complex) − ΔG(water) = ΔG_bind(Co) − ΔG_bind(Ni):

| α (Å³) | ΔΔG (kcal/mol) |
|--------|-----------------|
| 1.09   | −2.621 ± 0.009  |
| 2.75   | −0.305 ± 0.010  |
| 4.50   | +2.842 ± 0.009  |

**Calibration curve** (3 points, so the parabola is exact — but the linear fit
leaves a 0.23 kcal/mol residual, ~25× the statistical error, so the curvature is
real; C4 is 1/r⁴ and a larger α also shortens Ni–N, so sensitivity grows):

    ΔΔG(α) = 0.1182 α² + 0.9412 α − 3.7874       (kcal/mol, α in Å³)
    dΔΔG/dα = 1.20 (α=1.09) → 1.59 (α=2.75) → 2.01 (α=4.50) kcal/mol per Å³

Inverted:  ΔΔG = +1.0 → α = 3.53;  +2.0 → 4.07;  +3.0 → 4.58;  +4.0 → 5.06.
The experimental target is ΔΔG = −RT ln(β₄(Co)/β₄(Ni)) = 1.364 × (log β₄(Ni) −
log β₄(Co)) kcal/mol, still to be taken from a real compilation (Martell &
Smith); a Δlog β₄ of 2.0 would give α ≈ 4.4, of 2.4 → α ≈ 4.7.

**Main finding: α = 2.75 Å³ gives ΔΔG ≈ −0.3, i.e. essentially zero
Ni/Co selectivity for imidazole, with the sign marginally favouring Co.**
(2.75 is the inherited single-imidazole/OPC value, applied here under our
all-classes C4 convention — see Provenance below; it is not a prediction of
MIRIAM's published model.)  That
contradicts Irving–Williams (Ni²⁺ clearly above Co²⁺ for N-donors), so the
target should be positive and the fitted α well above 2.75.

Quality checks, all passed:

* **Convergence** — integrating the first vs second half of each window's
  post-discard dU/dλ shifts ΔG by ≤ 0.009 kcal/mol in every leg.  No drift;
  5 ns/window with `discard_frac: 0.2` is ample.
* **Quadrature** — ⟨dU/dλ⟩ is monotonic and gently curved over all 12 nodes in
  every leg (water: 17.71 → 13.21), so 12-point GL is not the bottleneck.
* **Coordination** — every ligand leg ends (`npt_equil_final.pdb`) as a clean
  octahedral [Ni(im)₄(H₂O)₂]²⁺: 4 × Ni–NE2 at 2.0–2.2 Å + 2 × Ni–O_w at ~2.1 Å;
  the water leg is 6 × Ni–O_w at 2.0–2.1 Å.  Nothing dissociated despite the
  imidazoles being unrestrained.
  *These PDBs are written wrapped (`enforcePeriodicBox=True`), and the box is
  only ~49 Å across, so naive Cartesian distances show spurious 50–62 Å
  "dissociation" — apply triclinic minimum image before believing any of it.*
* **Composition** — 1 Ni²⁺ + 4 IMI + 4040 HOH + 2 Cl⁻, 1.00 g/mL, 300 K; the
  4040 waters match MIRIAM's ~4180, so `diameter_A: 44.5` reproduced the
  reference box.

Caveat: at α = 4.50 one Ni–N is already stretched to 2.55 Å at the end of NPT.
Higher α puts more of the binding on the C4 term, so the coordination shell
needs watching — round 1 saved only checkpoints for the windows, which is why
`ti.traj_ps` was added.

## Next steps

* **Fitting is now the PMF, not TI.**  Scan `c4.alpha_by_class` with
  `configs/pmf/*.yaml` (one config per α, `slurm/run_pmf.sbatch`) and fit the
  WHAM well depth to the experimental binding free energy for a single
  imidazole.  Fit Co²⁺ and Ni²⁺ separately (one single-imidazole PMF each;
  parameters are per metal–ligand pair).  TI runs only after both are
  fitted, as a consistency check of the Ni/Co difference with each end state
  at its own pair α — see the TI section and the round-1 results below.
* **Experimental target, still the blocker.**  For the PMF the relevant number
  is the *stepwise* association constant for one imidazole on the aquo ion
  (log K, plus the standard-state correction), not the Δlog β₄ that the TI
  ΔΔG needs.  Both tables still have to be collected (Martell & Smith).
* **Pilot before the full scan.**  One α at the paper's settings is 5.04 μs
  (~22 GPU-days, ~1 day wall at `--array=...%20`).  Run one α with
  `pmf.n_windows: 40` and `pmf.window_prod_ns: 10` first (~0.4 μs) to check
  the CV range, the window overlap and the plateau, then commit to the full
  protocol.
* **Unrestrained orientation.**  The CV is a bare distance: nothing stops the
  pulled imidazole rotating or drifting sideways at large r, so the plateau
  region converges slowly and the bootstrap bars there are optimistic.  If it
  misbehaves, the standard fix is an angular/positional restraint on the
  leaving ligand plus the matching analytic correction.
* **`configs/round2/`** (TI, α = 4.00/4.50/5.00 with window trajectories) is
  written but now optional — run it only if the PMF-fitted α lands in that
  range and you want the TI cross-check there.
* Ligand set beyond imidazole: other CGenFF residues in `mosic.xml`
  (`grep 'Residue name=' mosic.xml`) vs peptoid sidechain fragments.
* Whether `alpha_by_class` overrides should be written back into
  `make_1264_xml.py`'s class table once fitted.
