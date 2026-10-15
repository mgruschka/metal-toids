"""
YAML config -> dataclasses.  One file per system; everything the pipeline
needs to be reproducible is in here, nothing is hidden in CLI flags except
--outdir / --platform / --resume.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class IonSpec:
    element: str
    charge: int
    # 12-6 + base (water) C4 come from 12-6-4_base_params.csv via ions_1264.xml.
    # Residue name in that XML is Amber-style: Zn2+, Ce3+, Na+, Cl- ...
    resname: Optional[str] = None
    # Initial placement: `distance_A` from ligand atom `near_atom`, pointing
    # away from the ligand centre of mass.  Ignored when the ligand PDB
    # already contains the ion (complex_pdb mode).
    near_atom: Optional[str] = None
    near_resid: Optional[str] = None          # disambiguates when the atom name repeats
    distance_A: float = 2.1

    def __post_init__(self):
        if self.resname is None:
            mag = abs(self.charge)
            sign = "+" if self.charge > 0 else "-"
            self.resname = f"{self.element}{mag if mag > 1 else ''}{sign}"


@dataclass
class LigandSpec:
    pdb: Optional[str] = None                 # ligand (or ion+ligand complex)
    resnames: List[str] = field(default_factory=list)   # residues that count as "ligand"
    xml: List[str] = field(default_factory=list)   # extra FF XMLs (CGenFF-derived etc.)
    contains_ion: bool = False                # True -> pdb is the full complex
    # number of copies placed around the ion (0 -> bare ion, the TI water
    # leg).  >1 arranges them on octahedral directions with `ion.near_atom`
    # pointing at the ion (MIRIAM: 4 imidazoles).
    count: int = 1


@dataclass
class TISpec:
    """
    MIRIAM-style thermodynamic integration: linear mixing of the ion's
    sigma/epsilon/C4 (and charge) from `ion` (lambda=0) to `ion_b`
    (lambda=1) in the presence of the ligands.  Windows are NVT, run in
    sequence from the NPT end state, Gauss-Legendre quadrature over lambda.
    """
    element_b: str = "Co"
    charge_b: int = 2
    n_windows: int = 12                       # Amber's 12-point Gaussian quadrature
    ns_per_window: float = 5.0
    dt_fs: float = 1.0                        # MIRIAM runs TI at 1 fs
    heat_ns: float = 0.36
    npt_ns: float = 2.0
    dudl_every_ps: float = 0.5                # ntave=500 steps at 1 fs
    discard_frac: float = 0.2                 # of each window, before averaging
    traj_ps: float = 0.0                      # window_NN.dcd stride; 0 = no trajectories
    min_max_iter: List[int] = field(default_factory=lambda: [2000, 2000, 10000])


@dataclass
class AlphaTISpec:
    """
    Bound-state TI of the polarizability (TI_param/run_TI.py): arel on ligand
    class `cls` goes linearly from alpha0 (lambda=0) to alpha1 (lambda=1)
    through forces.make_alpha_alchemical.  Every window is itself a physical
    alpha_k = alpha0 + lambda_k*(alpha1-alpha0), so a uniform ladder gives
    dG_bound(alpha0 -> alpha_k) at every k (cumulative TI + MBAR).
    Prep (minimize/heat/NPT) holds the CV harmonically at the bound distance;
    the windows are unrestrained, and the CV is logged so a dissociation shows.
    """
    cls: str = "NG2R50"
    alpha0: Optional[float] = None            # None -> c4.alpha_by_class[cls]
    alpha1: float = 4.0
    n_windows: int = 11
    schedule: str = "uniform"                 # uniform | gauss_legendre
    dt_fs: float = 2.0
    min_max_iter: List[int] = field(default_factory=lambda: [20000, 20000, 20000])
    heat_ns: float = 1.0
    npt_ns: float = 2.0
    window_equil_ps: float = 200.0            # per window, before dU/dl is recorded
    ns_per_window: float = 2.0
    dudl_every_ps: float = 0.5
    discard_frac: float = 0.1                 # of each window's production
    cv_atom: str = "NE2"                      # d(ion, cv_atom) is logged per sample
    cv_resid: Optional[str] = None            # None -> the copy nearest the ion
    bound_r_max_A: float = 3.0                # monitor only: frames beyond are "unbound"
    max_unbound_frac: float = 0.05            # analyze warns above this
    prep_restraint_kappa: float = 10.0        # kcal/mol/A^2; 0 disables
    prep_restraint_A: Optional[float] = None  # None -> ion.distance_A
    traj_ps: float = 0.0                      # window_NN.dcd stride; 0 = off
    # Plateau (unbound-leg) correction, analysis only.  The PMF run at the
    # same alpha0 whose mbar.json supplies V(r) = c/r^4 and the bin edges;
    # dG_unbound(alpha) is the same plateau shift lig1264.mbar applies
    # (last plateau_frac of the binned range).  None -> not applied.
    pmf_run: Optional[str] = None
    plateau_frac: float = 0.15


@dataclass
class PMFSpec:
    """
    Umbrella-sampling PMF for pulling one ligand off the ion, biased through
    PLUMED (lig1264/pmf.py).  Lengths follow the AMBER protocol being
    reproduced: 2 fs, 10 A cutoff, 1 ns heating, 550k/500k NPT steps.
    """
    dt_fs: float = 2.0
    # "10,000 SD + 10,000 CG" per stage -> 20000 L-BFGS iterations (see pmf.py)
    min_max_iter: List[int] = field(default_factory=lambda: [20000, 20000, 20000])
    heat_ns: float = 1.0                      # NVT 0 -> T
    npt_equil_steps: int = 550_000            # 1.1 ns at 2 fs
    npt_prod_steps: int = 500_000             # 1.0 ns at 2 fs
    # --- collective variable: d(ion, pull_atom of one ligand copy)
    pull_atom: str = "NE2"
    pull_resid: Optional[str] = None          # None -> the copy nearest the ion
    r_start_A: Optional[float] = None         # None -> the equilibrated distance
    r_end_A: float = 8.0
    # --- prep restraint: hold the CV at the bound distance through
    # minimize/heat/equil/prod.  Without it a weakly bound complex (and a
    # few-kcal/mol complex is exactly what is being fitted) dissociates in
    # the unbiased 2 ns and the SMD starts from a random bulk separation --
    # the pull then runs inwards and every umbrella centre collapses onto
    # r_end_A.  Only the starting structure is affected; the SMD and the
    # windows never see this force.  0 disables it (pre-2026-09 behaviour).
    prep_restraint_kappa: float = 10.0        # kcal/mol/A^2
    # Centre, A.  None -> r_start_A if set, else ion.distance_A (the
    # placement distance), so it is declared in the config, not measured
    # from a trajectory that may already have fallen apart.
    prep_restraint_A: Optional[float] = None
    # --- steered MD (seeds the windows; not used as a free energy)
    smd_ns: float = 2.0
    # stiff enough that the ligand tracks the moving centre: a soft spring
    # just stretches and the bond snaps late, leaving most of the CV range
    # unsampled.  Check the lag printed by stage_smd.
    smd_kappa: float = 100.0                  # kcal/mol/A^2
    smd_traj_ps: float = 1.0                  # smd.dcd stride; must give >> n_windows frames
    # --- umbrella windows
    n_windows: int = 120
    # kcal/mol/A^2.  Two constraints fix this, and they pull opposite ways:
    #   holding  kappa > max(-W'') or the window is bistable on a barrier top
    #            and drains into a basin (this is what killed co_imid1_a109 at
    #            kappa = 10: the contact/solvent-separated barrier has
    #            max(-W'') ~ 30-60, estimated from the SMD mean force, so
    #            windows 0-29 left their centres inside 1 ps).
    #   overlap  the window width sqrt(kT/kappa) must stay >= the window
    #            spacing, which caps kappa at kT/spacing^2 = 136 for 120
    #            windows over 2.1-10 A (242 over 2.1-8 A).
    # 100 clears the barrier with ~2x margin and sits at 73% of the ceiling
    # (width 0.077 A vs 0.066 A spacing).  If a hole survives at 100, add
    # windows -- stiffness and spacing are not independent knobs.
    window_kappa: float = 100.0
    window_equil_ns: float = 2.0
    window_prod_ns: float = 4.0               # MIRIAM's cheap setting; 40.0 is
                                              # the converged one (pi-stack)
    # --- early drift gate (pmf.stage_window).  After drift_check_ps of each
    # window's equilibration, the mean CV over (5 ps, drift_check_ps] is
    # compared with the centre; if it is off by more than drift_max_spacings
    # window spacings AND drift_max_widths widths sqrt(kT/kappa), the window
    # writes ABORT and the whole pool stops.  Both, because each alone
    # misfires: spacings alone trips on thermal noise of a wide window (at
    # kappa 100 a 15 ps mean scatters ~0.6 spacings -> ~10% false aborts per
    # 120-window run); widths alone scales with the very softness it should
    # catch.  Replayed on co_imid1_a109 (kappa 10) this trips windows 8-15
    # and nothing on the plateau.  Either limit 0 disables the gate.
    drift_check_ps: float = 20.0
    drift_max_spacings: float = 2.0
    drift_max_widths: float = 3.0
    colvar_ps: float = 1.0                    # COLVAR stride
    traj_ps: float = 0.0                      # per-window DCD; 0 = off (they are large)
    discard_frac: float = 0.0                 # of each window COLVAR, before WHAM


@dataclass
class C4Spec:
    # The parameters being fitted.  Keys are CHARMM atom *classes* on the
    # ligand (OC, O, NH3, ...); value multiplies the polarizability-derived
    # arel for ion<->that-class pairs.  1.0 == plain Li/Merz mixing rule.
    scale_by_class: Dict[str, float] = field(default_factory=dict)
    # Hard overrides of the polarizability (A^3) itself, if a class is
    # mis-assigned by the element/hybridisation fallback in ions_1264.xml.
    alpha_by_class: Dict[str, float] = field(default_factory=dict)
    # MIRIAM convention: zero arel for every ligand class NOT named in
    # alpha_by_class / scale_by_class, so C4 survives only on ion<->water
    # (arel_O = 1 by construction, giving exactly the HFE/IOD-fitted C4) and
    # ion<->the class under study.  Their TI/lj_1264_pol.dat is all 0.000
    # except the cloned 'MUTE' nitrogen.  Default False = ParmEd behaviour,
    # Miller polarizability on every class.  See README "Provenance of
    # alpha = 2.75".
    zero_other_classes: bool = False
    lambda_c4: float = 1.0                    # global switch, 0 == pure 12-6


@dataclass
class BoxSpec:
    shape: str = "octahedron"                 # cube | dodecahedron | octahedron
    # at most one of (default: buffer_A = 24):
    #   diameter_A  fixed face-to-face (minimum-image) width.  44.5 A matches
    #               tLEaP `solvateoct ... 24.0` as used in MIRIAM (a = 62.9 A
    #               cube edge, 124,400 A^3, ~4180 waters); 48 A gives the
    #               ~5100 waters / 10 mM quoted in the paper.
    #   buffer_A    solute-to-face distance: face-to-face = solute diameter
    #               + 2*buffer.  Stricter than tLEaP for small solutes (it
    #               gives ~7100 waters for the same system), so prefer
    #               diameter_A when matching a published box.
    buffer_A: Optional[float] = None
    diameter_A: Optional[float] = None
    water: str = "spce"
    neutralize: bool = True
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"

    def __post_init__(self):
        if self.diameter_A is None and self.buffer_A is None:
            self.buffer_A = 24.0
        if self.diameter_A is not None and self.buffer_A is not None:
            raise ValueError("box: set only one of buffer_A or diameter_A")


@dataclass
class MDSpec:
    temperature_K: float = 300.0
    pressure_bar: float = 1.0
    dt_fs: float = 2.0
    friction_per_ps: float = 1.0
    cutoff_nm: float = 1.0
    constraints: str = "HBonds"               # None | HBonds
    # three minimization stages, see stages.minimize_3stage
    min_max_iter: List[int] = field(default_factory=lambda: [2000, 2000, 10000])
    min_restraint_k: float = 1000.0           # kJ/mol/nm^2 on restrained atoms
    heat_ns: float = 1.0                      # NVT, T ramped 0 -> temperature_K
    heat_steps: int = 50                      # number of temperature increments
    npt_equil_ns: float = 1.0
    npt_prod_ns: float = 10.0
    report_ps: float = 10.0                   # DCD / StateData interval
    checkpoint_ps: float = 500.0


@dataclass
class Config:
    name: str
    ion: IonSpec
    ligand: LigandSpec
    c4: C4Spec = field(default_factory=C4Spec)
    box: BoxSpec = field(default_factory=BoxSpec)
    md: MDSpec = field(default_factory=MDSpec)
    ti: Optional[TISpec] = None
    pmf: Optional[PMFSpec] = None
    alpha_ti: Optional[AlphaTISpec] = None
    # base force-field files, relative to repo root
    forcefield: List[str] = field(default_factory=lambda: [
        "mosic.xml", "spce_ions.xml", "ions_1264.xml"])
    platform: str = "CUDA"
    seed: int = 0

    def ff_paths(self) -> List[str]:
        paths = [str(REPO_ROOT / p) for p in self.forcefield]
        paths += [str(_resolve(p)) for p in self.ligand.xml]
        return paths


def _resolve(p: str) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), PKG_ROOT, REPO_ROOT):
        if (base / p).exists():
            return base / p
    return PKG_ROOT / p


def load_config(path) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(
        name=raw["name"],
        ion=IonSpec(**raw["ion"]),
        ligand=LigandSpec(**raw["ligand"]),
        c4=C4Spec(**raw.get("c4", {})),
        box=BoxSpec(**raw.get("box", {})),
        md=MDSpec(**raw.get("md", {})),
        ti=TISpec(**raw["ti"]) if "ti" in raw else None,
        pmf=PMFSpec(**raw["pmf"]) if "pmf" in raw else None,
        alpha_ti=AlphaTISpec(**raw["alpha_ti"]) if "alpha_ti" in raw else None,
        forcefield=raw.get("forcefield", Config.__dataclass_fields__["forcefield"].default_factory()),
        platform=raw.get("platform", "CUDA"),
        seed=raw.get("seed", 0),
    )
    if cfg.ligand.pdb is not None:
        cfg.ligand.pdb = str(_resolve(cfg.ligand.pdb))
    elif cfg.ligand.count:
        raise ValueError("ligand.pdb is required when ligand.count > 0")
    at = cfg.alpha_ti
    if at is not None:
        a_c4 = cfg.c4.alpha_by_class.get(at.cls)
        if at.alpha0 is None:
            if a_c4 is None:
                raise ValueError(f"alpha_ti.alpha0 not set and c4.alpha_by_class has no {at.cls}")
            at.alpha0 = a_c4
        elif a_c4 is not None and abs(a_c4 - at.alpha0) > 1e-9:
            raise ValueError(f"alpha_ti.alpha0 = {at.alpha0} but c4.alpha_by_class"
                             f"[{at.cls}] = {a_c4}: the built system is at the latter")
    return cfg
