# Co(II)–imidazole 12-6-4 PMF — run `co_imid1_a109`

Umbrella-sampling potential of mean force for dissociation of a single
imidazole from Co(II), run to extract a standard-state binding free energy and
so calibrate the 12-6-4 ion↔aromatic-nitrogen polarizability.

**Status: rejected.** The sampling completed in full, but the umbrella windows
failed to cover the barrier separating the bound and unbound states, leaving
the two halves of the reaction coordinate thermodynamically disconnected. No
binding free energy can be extracted from this data set. Diagnosis and
required changes are in [Results](#results) and [Conclusion](#conclusion).

- Run date: 22–27 September 2026
- Iteration 1 of the Li/Merz secant loop; α(NG2R50) = 1.09 Å³
- Target: ΔG° = −3.36 kcal/mol (stepwise log K₁ = 2.46 at 298 K)

---

## System

| | |
|---|---|
| Solute | Co²⁺ + 1 imidazole (CGenFF `IMIA`), 2 Cl⁻ counter-ions |
| Solvent | SPC/E water, 4049 molecules |
| Cell | truncated octahedron, 44.5 Å face-to-face (124,622 Å³), 12,159 atoms |
| Ion concentration | 13.3 mM |
| Force field | MoSiC (CHARMM36-based) + 12-6-4 ion model |

The ion was placed 2.1 Å from the imidazole N_ε2 — the unprotonated aromatic
nitrogen, CHARMM class `NG2R50`, which is the coordinating atom.

### 12-6-4 parameters

Co²⁺ carries its HFE/IOD-fitted 12-6 terms (R_min/2 = 1.457 Å,
ε = 0.0271 kcal/mol) and base C4 = 209.2 kcal Å⁴/mol, all held fixed. Following
the MIRIAM convention, the C4 term survives only on ion↔water and on
ion↔`NG2R50`; the polarizability was zeroed on every other ligand class present
(`CG2R51`, `CG2R53`, `HGP1`, `HGR52`, `NG2R51`) and on the chloride
counter-ions. The single fitted parameter is therefore α(`NG2R50`), set here to
1.09 Å³ — not a scan point but the default Miller polarizability the code
assigns to any aromatic nitrogen, i.e. the unmodified 12-6-4 starting point.

---

## Simulation method

All dynamics: 2 fs timestep, 10 Å cutoff, SHAKE/SETTLE on bonds to hydrogen,
Langevin thermostat at 300 K with 1 ps⁻¹ friction, Monte Carlo barostat at
1 bar for the constant-pressure stages.

**1. Preparation (3.1 ns).** Three-stage energy minimisation (solute
restrained → all but ligand hydrogens restrained → unrestrained), then 1 ns of
NVT heating from 0 to 300 K, 1.1 ns of NPT equilibration and 1.0 ns of NPT
production. Throughout these stages the ion↔N_ε2 distance was held at its bound
value by a harmonic restraint (10 kcal mol⁻¹ Å⁻²). This is necessary rather
than cosmetic: the complex being characterised is worth only a few kcal/mol, so
its residence time is well under a nanosecond and an unrestrained equilibration
lets the ion diffuse away before the pull begins. The restraint acts on the
starting structure only and is absent from every stage below, so it enters no
free-energy estimate.

**2. Steered MD (2 ns).** A moving harmonic restraint
(100 kcal mol⁻¹ Å⁻²) drew the ion↔N_ε2 distance from its equilibrated value of
2.16 Å out to 10.0 Å at a constant rate, generating a continuous dissociation
path. The pull serves only to seed the umbrella windows; its work is not used
as a free energy.

**3. Umbrella sampling (720 ns).** 120 windows spaced 0.066 Å apart spanned
2.14–10.00 Å. Each was started from the steered-MD frame closest to its centre,
held there by a harmonic restraint of **10 kcal mol⁻¹ Å⁻²**, and run for 2 ns
of equilibration followed by 4 ns of production, sampling the distance every
1 ps. Total production: 480 ns across the ladder, 725 ns including all stages.

---

## Analysis

The 120 production time series were combined by binned WHAM (200 bins, solved
by self-consistent iteration) to give W(r), the free energy along the radial
coordinate. Because W(r) retains the 2k_BT ln r volume term, the
orientation-averaged pair PMF used for binding is W(r) + 2k_BT ln r. The
standard-state binding free energy follows from integrating the bound region
against the 1 M reference volume of 1660.5 Å³. Statistical errors are block
bootstraps over each window's own time series, which capture correlation along
the coordinate but not slow motions orthogonal to it.

Three diagnostics were computed alongside the PMF and are plotted in
`figures/`:

1. the population of each window along the coordinate, and the overlap of every
   adjacent pair;
2. pooled sampling density over the whole coordinate, and each window's mean
   sampled distance against the centre it was restrained to;
3. the resulting PMF, with the unsupported region marked.

---

## Results

### Preparation and seeding succeeded

The restrained preparation held the complex bound throughout — the ion↔N_ε2
distance read 2.20 Å after heating, 2.18 Å after equilibration and 2.16 Å after
production. The steered pull tracked its moving restraint with a maximum lag of
0.38 Å and reached 10.16 Å, and window seeding placed every umbrella centre
within 0.011 Å of a real frame. All 120 windows ran to completion with 4000
production samples each, 480,000 in total.

### The umbrella ladder broke in two

![window overlap](figures/fig1_window_overlap.png)

The window populations show one isolated peak at 2.2 Å — windows 0–7 piled on
top of one another — then nothing until 3.2 Å, where the remaining 112 windows
form a proper ladder. Adjacent-window overlap is healthy across that ladder
(median 0.89) but collapses to **0.0002** at the single pair bridging 2.6 Å.

![sampling gap](figures/fig2_sampling_gap.png)

Pooled over all 480,000 frames, the region 2.4–3.2 Å received **665 frames**,
against roughly 5,000 in any *single* 0.05 Å bin of the bulk. The lower panel
shows why: 26 windows never sampled the centre they were assigned. Windows 5–7
slid back into the contact ion pair at 2.25 Å; windows 8–27 fell outward into
the solvent-separated pair near 3.6 Å. Nothing remained on the barrier between
them.

The cause is the window force constant. At 10 kcal mol⁻¹ Å⁻² a window is
√(k_BT/κ) = 0.24 Å wide — wider than the 0.066 Å spacing, and far too soft to
pin the system on a barrier top. An umbrella holds there only if its curvature
exceeds the negative curvature of the underlying free-energy surface; below that
threshold the window becomes bistable and drains into whichever basin it was
seeded nearest, which is exactly the pattern observed.

### The resulting PMF is unusable

![disconnected PMF](figures/fig3_pmf_disconnected.png)

WHAM converged in 1386 iterations and returned:

| quantity | value |
|---|---|
| ΔG° (binding) | −0.10 ± 0.01 kcal/mol |
| pair-PMF well depth | 0.03 kcal/mol at 8.73 Å |
| global minimum | 10.00 Å (the edge of the sampled range) |
| plateau slope | +0.033 kcal/mol/Å |
| adjacent overlap | median 0.89, **min 0.0002** |

These numbers are artifacts, not measurements. With one adjacent pair at
essentially zero overlap the window graph is disconnected, so the free-energy
offset between the contact branch and everything beyond it is not determined by
the data. WHAM nonetheless assigns one, placing the bound state 7.7 kcal/mol
*above* bulk behind a 12.7 kcal/mol barrier — for a complex expected to bind at
−3.36 kcal/mol. A minimum at the outer edge of the sampled range and a binding
free energy of −0.10 kcal/mol are both signatures of the same defect. The
±0.01 kcal/mol error bar is bootstrap noise computed within each branch and
carries no information about the offset between them.

---

## Conclusion

The protocol executed as designed and the data are complete; the parameter
choice for the umbrella restraint was wrong. A force constant of
10 kcal mol⁻¹ Å⁻² cannot hold windows on the contact-to-solvent-separated
barrier, and the resulting hole in the coordinate leaves the PMF
thermodynamically disconnected. No value of α can be fitted from this run, and
the iteration-1 result must be discarded rather than fed to the next secant
step.

**Required for a valid repeat**

1. Raise the umbrella force constant. The window width must stay at or above
   the spacing, which caps the constant at k_BT/Δ² = 136 kcal mol⁻¹ Å⁻² for this
   120-window ladder; 100 kcal mol⁻¹ Å⁻² gives a 0.077 Å width against 0.066 Å
   spacing and is the value the steered pull already demonstrated is sufficient
   to cross this barrier. Note that stiffness and spacing are not independent —
   if a hole persists at 100, the remedy is more windows, not a larger constant.
2. Preparation, steering and seeding are unaffected by this change and need not
   be repeated; only the 120 windows must be rerun.
3. Reject any future run on the overlap diagnostic before interpreting its
   binding free energy. The failure here was visible in `overlap_min` the moment
   WHAM finished, several days before the number it printed alongside was
   examined.

**Separate caveat, still outstanding.** Even once the gap is closed, 4 ns per
window is the cheap setting. The reference study resolved a π-stacked
intermediate only at 40 ns per window, and missing such a state overestimates
binding and biases the fitted α low. The barrier region is where short sampling
is least trustworthy, so the repeat should not be treated as a converged fit.

---

## Files

| path | contents |
|---|---|
| `config.yaml` | run configuration as executed |
| `pmf_provenance.json` | protocol parameters and documented deviations |
| `windows.json` | umbrella centres, force constant, seed frames |
| `window_NNN_colvar.dat` | per-window production time series (×120) |
| `pmf.csv` / `pmf.json` | WHAM profile and diagnostics (**rejected — see above**) |
| `figures/fig1_window_overlap.png` | window populations and adjacent overlap |
| `figures/fig2_sampling_gap.png` | pooled coverage; sampled mean vs. assigned centre |
| `figures/fig3_pmf_disconnected.png` | resulting PMF with the unsupported region marked |
