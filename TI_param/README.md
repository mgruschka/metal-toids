# TI_param — bound-state TI of the 12-6-4 polarizability

Secondary route to α(metal–ligand class), next to the umbrella + MBAR
reweighting in `ligand-1264/` (`lig1264.pmf` → `lig1264.mbar`). It computes

    ΔG_bound(α₀ → α)

for the bound complex by thermodynamic integration, with λ switching the
polarizability of one ligand atom class inside `LJ1264C4Force`.

## λ definition

`lig1264.forces.make_alpha_alchemical` rebuilds the C4 force with
`arel_j → arel_j + λ·darel_j` on the class atoms (e.g. NE2 = `NG2R50`), where
`darel_j = arel_j(α₀)·(α₁/α₀ − 1)`. Because arel ∝ α:

* α(λ) = α₀ + λ(α₁ − α₀). Every window λ_k **is** a physical α_k.
* U is exactly linear in λ: dU/dλ = (α₁ − α₀)·V, where V = dU/dα is the same
  quantity `lig1264.mbar` reweights with (verified to 1e-12 relative).
* The ion's own C4 and the ion↔water C4 are untouched.

A uniform ladder therefore gives ΔG_bound(α₀ → α_k) at **every** window:
cumulative trapezoid TI, plus MBAR over λ. MBAR needs only the dU/dλ samples,
because u_k(x) = u₀(x) + β λ_k dU/dλ(x).

## Binding cycle

    ΔG⁰(α) = ΔG⁰(α₀) + ΔG_bound(α₀→α) − ΔG_unbound(α₀→α),   ΔG_unbound ≈ 0

The class atom's arel couples only to the ion (zero_other_classes: true) and
only within the 1 nm cutoff, so the unbound leg is negligible. With ΔG⁰(α₀)
from the α₀ PMF, this gives the full ΔG⁰(α) curve. Compare it with
`python -m lig1264.mbar <pmf run> --alphas …` (`analyze` prints the matching
grid). For alphas where MBAR reweighting fails (memory: the α' minimum leaves
the sampled range above α ≈ 2), the TI curve is the independent estimate.

## Bound-state definition — read this

The windows are **unrestrained**. Prep (minimize/heat/NPT) holds d(ion,
`cv_atom`) harmonically at `ion.distance_A`, the same restraint pmf.py uses,
so window 0 starts bound. During production the CV is logged in every
`dudl_wNN.csv`. `analyze` reports the fraction of frames with
CV > `bound_r_max_A` and warns above `max_unbound_frac`. A complex worth only
a few kcal/mol can dissociate within a 2 ns window, especially at small α. A
warning there means that window's ⟨dU/dλ⟩ is not a bound-state average.
Set `bound_r_max_A` to the PMF's `r_bound`.

## Running (`metal-toid` python, from `TI_param/`)

```bash
python run_TI.py configs/co_imid1_alpha_ti.yaml --outdir runs/co_imid1_alpha_ti --stop-after build --allow-cpu --platform CPU   # check
sbatch slurm/run_alpha_ti.sbatch configs/co_imid1_alpha_ti.yaml runs/co_imid1_alpha_ti npt   # prep
sbatch --array=0-10 slurm/run_alpha_ti.sbatch configs/co_imid1_alpha_ti.yaml                  # windows
python run_TI.py configs/co_imid1_alpha_ti.yaml --outdir runs/co_imid1_alpha_ti --analyze
```

Without `--window`, the windows run sequentially (window k starts from
window k−1's end state). `--resume` skips finished stages and windows. Each
window is a fresh Context seeded with `seed + 1 + k`, starting from a State
XML. Checkpoints are deliberately avoided, because the prep Context carries
the CV restraint and the window Contexts do not.

Outputs: `dudl_wNN.csv` (time_ps, dudl_kJmol, cv_A), `window_NN.xml`,
`alpha_ti_results.json`, `alpha_ti_dG_vs_alpha.csv` (α, λ, ΔG_TI ± sem,
ΔG_MBAR ± sem, unbound_frac).

## Code layout / sharing with ligand-1264

`run_TI.py` adds `ligand-1264/` to `sys.path` and reuses `lig1264` as-is:
`build` (system), `stages` (minimize/heat/NPT), `pmf` (pull_pair, state I/O,
CV), and `run.get_platform`. The additions in `lig1264` are purely additive:

* `forces.make_alpha_alchemical`, `forces.LAMBDA_ALPHA`, and
  `forces.dudl(context, name=LAMBDA_ION)`. The default name keeps ti.py
  unchanged.
* `lambda_windows.py`: schedules, window loop, cumulative TI, MBAR over λ.
* `config.AlphaTISpec`: the `alpha_ti:` YAML block. Configs without it are
  unaffected.

`pmf.py`, `mbar.py`, `wham.py`, `ti.py`, `run.py`, `stages.py` and `build.py`
are unchanged.

## Known gaps (2026-10-14, deferred)

Found while reviewing the first co_imid1 run (`runs/co_imid1_alpha_ti`):

1. **Plateau correction: implemented 2026-10-14.** Set `alpha_ti.pmf_run`
   to the α₀ PMF run, and `--analyze` subtracts ΔG_unbound(α) =
   −kT ln⟨exp(−βΔα·c/r⁴)⟩ over the plateau shell. c and the binned range come
   from that run's `mbar.json`, so this is the same plateau shift the MBAR
   reweighting contains. It adds `dG_unbound_kcal`, `ddG0_ti_kcal` and
   `ddG0_mbar_kcal` (the CSV gains three columns at the end). For co_imid1
   the shell is 8.82–10.0 Å and ΔG_unbound = −0.0632 kcal/mol at α = 4.5,
   linear in Δα. Original note: `analyze` reported ΔG_bound only and treated
   ΔG_unbound as 0. The C4 term still reaches the PMF plateau, inside
   the 10 Å cutoff. From window 0, V·r⁴ = −606 kJ mol⁻¹ Å⁴ per Å³ (constant
   to 0.3%, so V depends only on the Co–N distance). That gives
   ΔG_unbound ≈ Δα·c/r_plat⁴ = −0.05 (10 Å) to −0.12 (8 Å) kcal/mol at α = 4.5,
   linear in Δα. Then ΔG⁰(α) − ΔG⁰(α₀) = ΔG_bound − ΔG_unbound, i.e. ΔG_bound
   is too negative by this amount. Fix: subtract Δα·c/r_plat⁴, with r_plat
   taken from the same plateau definition `mbar.py` uses (`--plateau-frac`).
2. **First-shell water count: done 2026-10-14 — no change with α.**
   Windows 0, 5 and 10 (α 1.09 / 2.79 / 4.50) all have exactly 5 waters + 1
   N in every frame (cutoff 2.98 Å). No count change and no water exchange
   (5 distinct shell waters each) in 2 ns, as expected for Co²⁺ (µs exchange).
   The first shell stays octahedral, so the TI is not missing a coordination
   change. As a seed replicate, ⟨dU/dλ⟩ differs from the forward run by
   +0.099 / −0.017 / +0.050 kcal/mol. Window 0's 0.10 is about 2.5× its
   block SEM, so the 5-block SEMs may be optimistic by ~2×. Its effect on ΔG
   is < 0.01 (end-window trapezoid weight 0.05). Setup:
   `configs/co_imid1_alpha_ti_wcn.yaml` reruns forward windows 0, 5 and 10
   (same system and start, seed 4026, DCD every 2 ps) into
   `runs/co_imid1_alpha_ti_wcn`. `python water_cn.py <run> --ref-run
   runs/co_imid1_alpha_ti` reports the Co–O(water) and Co–N first-shell counts
   per window: distribution, drift by quarter, changes, and distinct waters.
   It uses one cutoff for all windows, the middle of the window-0
   first-shell depletion gap (2.94 Å at α₀). Note that `first_min` picks the
   first empty bin, 2.31 Å, which is too tight. Baseline from the α₀ NPT
   trajectory: 5 waters + 1 N (octahedral), no change in 2 ns. Original
   note: The 0.6 ps correlation time of
   dU/dλ only shows that the Co–N distance relaxes quickly. Co²⁺ water
   exchange (sub-µs), a change in the number of first-shell waters as the N
   pulls in at high α, or imidazole reorientation would not show up in 2 ns
   windows that all start from the α₀ snapshot. `traj_ps: 0` saves no DCD,
   so this can't be checked afterwards. Fix: rerun windows 0, 5 and 10 with
   `traj_ps: 1`, compute the Co–O(water) and Co–N counts per window, and
   extend the windows whose counts change with α.
3. **Hysteresis test (option B): done 2026-10-14 — no hysteresis.**
   Reverse minus forward ⟨dU/dλ⟩ over all 11 windows: mean −0.0085 kcal/mol,
   χ² 8.5 for 11 windows, max |z| 1.7. The relax run at α₁ is flat
   (−26.216 → −26.238 over 10 ns). α* on the MBAR anchor: forward 3.0353,
   reverse 3.0354. Overlap matrices: `figures/fig5_ti_overlap{,_rev}.png`.
   Setup: Same system
   (`--seed-from runs/co_imid1_alpha_ti`), relaxed 10 ns at α₁ from the
   forward `window_10.xml` (`relax.xml`, with dU/dλ and the CV recorded in
   `relax_dudl.csv`). Then an 11-window array starting from that state,
   `configs/co_imid1_alpha_ti_rev.yaml` (only name and seed differ) →
   `runs/co_imid1_alpha_ti_rev`. Compare its ⟨dU/dλ⟩ per window with the
   forward run, especially at high α. A gap there means slow first-shell
   relaxation, and then gap 2 matters.

   ```bash
   sbatch --dependency=afterok:<fwd window 10> slurm/run_alpha_ti.sbatch configs/co_imid1_alpha_ti_rev.yaml \
       runs/co_imid1_alpha_ti_rev relax --seed-from runs/co_imid1_alpha_ti \
       --start-state runs/co_imid1_alpha_ti/window_10.xml --relax-ns 10 --relax-lambda 1
   sbatch --dependency=afterok:<relax job> --array=0-10 slurm/run_alpha_ti.sbatch configs/co_imid1_alpha_ti_rev.yaml \
       runs/co_imid1_alpha_ti_rev windows --start-state runs/co_imid1_alpha_ti_rev/relax.xml
   ```

Also note that for co_imid1 TI and `mbar.py` are not independent physics
checks. V is a function of the CV alone, so both rest on the same
ΔG⁰(α₀), force field and box. TI's added value is sampling the shifted well
at each α, beyond MBAR's α ≲ 2 coverage limit.

## Next: bidentate case (planned, not started; 2026-10-14)

Order agreed with the user: first finish and validate the co_imid1 forward +
hysteresis runs (analysis, TI vs MBAR, forward vs reverse, the gaps above).
Then move to a bidentate ligand, most likely **2,2′-bipyridine**.

### Ligand parameters (available; corrected 2026-10-25)
CGenFF has `RESI 22BPY` (`toppar_c36_jul21/top_all36_cgenff.rtf:28821`), and
`mosic.xml` carries it. It includes a dedicated four-term N1–C2–C8–N7
torsion (`par_all36_cgenff.prm:7456`). An earlier search for `BIPY`/`BPY`
missed it. Both donor N are class `NG2R60`, so a single α class covers
both. The template matches a free molecule only since the 2026-10-25
`mosic.xml` rebuild (a converter bug had given it a phantom C2 external
bond). `ligand-1264/ligands/bipyridine/` holds the cis (chelating) and trans
PDBs. In the gas phase, minimized trans (180°) lies 6.3 kcal/mol below a
twisted cis minimum (~50°), and planar cis is ~10 kcal/mol above trans.
Spot-check that cis/trans balance against QM before relying on it.

### H-REMD across the λ_α ladder (proposed fix for slow bound sub-states)
- A multidentate bound state has slow sub-states: denticity n versus n−1
  (with a water in the gap), ligand conformers, and peptoid amide cis/trans
  later on. Shifting α changes their populations. 2 ns windows started from
  one snapshot cannot re-equilibrate them, which shows up as forward/reverse
  hysteresis.
- U is linear in λ_α, so a replica swap i↔j is accepted with
  min(1, exp(−β(λᵢ−λⱼ)[dU/dλ(xⱼ) − dU/dλ(xᵢ)])). That needs only the dU/dλ
  already recorded, with no extra energy evaluations. Low-α replicas bind more
  weakly and exchange donors and waters faster, and those configurations
  diffuse up the ladder. MBAR over the replicas is unchanged
  (`lambda_windows.mbar_over_lambda`).
- Implementation sketch: one process, 11 Contexts sharing 1–2 GPUs. The
  STACK benchmark showed total throughput holds with 4 Contexts per GPU.
  Swap attempts every ~1 ps (neighbour pairs, alternating even/odd). Log the
  replica↔λ trajectory and check that replicas actually travel end to end.
  Rough cost: 2–4 h for 11 × 2.2 ns (estimate, not measured).
  openmmtools' `ReplicaExchangeSampler` is an alternative; it is not verified
  to be installed in `metal-toid`.
- Peptoids later: H-REMD over α does not flip amides. Add REST2 on the
  chelator, or a fixed, pre-converged bias on the ω torsions applied
  identically in every window and reweighted out.
- Several donor classes (one α per metal–class pair) need one
  `lambda_alpha_<cls>` parameter per class, each with its own energy
  derivative, so that per-class dU/dλ is recorded. `make_alpha_alchemical`
  currently takes a single class.

### CVs: watch out for non-planar / rigid ligands
The single ion–donor distance used for imidazole will not carry over.
Points to settle before any bidentate PMF or bound-state monitor:
- **Required conformational change.** Free 2,2′-bipyridine is mostly
  *trans*-planar (N's anti), but chelation needs *cis*. Unbinding therefore
  couples to the N–C–C–N torsion. A distance-only CV puts that barrier
  orthogonal to the bias, giving hysteresis and a path-dependent PMF.
  Include the torsion as a CV, or at least monitor it per window.
- **Rigid ligand: donors are not independent.** Both N's sit on one rigid
  frame, so "one donor off" also means a tilt or rotation of the whole
  ligand. Use the ion–donor-centroid distance plus a donor coordination
  number (`COORDINATION` with a switching function), and/or an orientation
  CV (angle between the ion→centroid vector and the ligand plane / N–N
  axis), not two independent ion–N distances.
- **Non-planar ligands** in general (puckered rings, twisted biaryls,
  flexible chelate arms) need the relevant ring or torsion coordinate checked
  the same way. Before choosing CVs, ask which internal coordinate has to
  change for the ligand to bind.
- **Ion–water coordination number** is needed in any case. First-shell water
  entry is usually the slow step of donor exchange.
- The bound-state monitor (`unbound_frac`) should use the donor coordination
  number, not one distance.
- Available here: PLUMED 2.9 in `metal-toid` has `METAD` and `COORDINATION`;
  the **OPES and funnel modules are compiled out**. A funnel or
  cylindrical restraint can be written by hand (`UPPER_WALLS` on a
  cylindrical-distance CV, or an OpenMM `CustomCentroidBondForce`).
- For high-denticity chelators the PMF anchor gets harder with each donor.
  A dummy-atom (double-decoupling) ΔG⁰(α₀) combined with the H-REMD α-ladder
  may scale better (see the discussion of the 5-step dummy-atom cycle).
