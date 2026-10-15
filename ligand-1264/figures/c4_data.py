"""
Dump the numbers behind figures/fig2_c4_convention.png to c4_data.json, so the
plot script needs no OpenMM and the figure is reproducible from a small file.

Builds the Co(II) + 1 imidazole system twice -- once with the ParmEd default
(Miller polarizability on every class) and once with the MIRIAM convention
(c4.zero_other_classes) -- reads `arel` per ligand class out of LJ1264C4Force,
then scans the ion along the NE2 lone-pair axis and evaluates the total
ion<->ligand C4 energy under each convention.

    /scratch/midway3/gruschka/metal-toid/bin/python figures/c4_data.py
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lig1264.config import load_config
from lig1264 import build, forces

CONFIG = "configs/pmf/co_imid1_a109.yaml"
OUT = os.path.join(os.path.dirname(__file__), "c4_data.json")
KCAL_A4 = 4.184e-4          # kJ/mol nm^4 per kcal/mol A^4 (LJ1264C4Force units)


def arel_and_geometry(cfg, ff, modeller):
    system, groups, classes = build.make_system(modeller, ff, cfg)
    f = forces.find_force(system, forces.C4_FORCE_NAME)
    arel = {i: f.getParticleParameters(i)[1] for i in groups["ligand"]}
    by_class = {}
    for i in groups["ligand"]:
        by_class.setdefault(classes[i], arel[i])
    ion = groups["ion"][0]
    c4_ion = f.getParticleParameters(ion)[0] / KCAL_A4        # kcal/mol A^4
    return by_class, arel, groups, classes, c4_ion


def main():
    cfg = load_config(CONFIG)
    ff = build.load_forcefield(cfg)
    modeller = build.solvate(build.load_solute(cfg), ff, cfg)

    default_spec = copy.deepcopy(cfg.c4)
    default_spec.zero_other_classes = False
    miriam_spec = copy.deepcopy(cfg.c4)
    miriam_spec.zero_other_classes = True

    out = {}
    per_atom = {}
    for name, spec in (("default", default_spec), ("miriam", miriam_spec)):
        c = copy.deepcopy(cfg)
        c.c4 = spec
        by_class, arel, groups, classes, c4_ion = arel_and_geometry(c, ff, modeller)
        out[name] = {"arel_by_class": by_class, "c4_ion_kcal_A4": c4_ion}
        per_atom[name] = arel

    # ion <-> ligand C4 energy as the ion slides along the Co...NE2 axis
    xyz = np.array([[v.x, v.y, v.z] for v in modeller.positions.value_in_unit(
        __import__("openmm").unit.nanometer)]) * 10.0            # Angstrom
    ion_i = groups["ion"][0]
    lig = sorted(groups["ligand"])
    anchor = next(a.index for a in modeller.topology.atoms()
                  if a.index in set(lig) and a.name == cfg.ion.near_atom)
    axis = xyz[anchor] - xyz[ion_i]
    r0 = float(np.linalg.norm(axis))
    axis = axis / r0

    rs = np.linspace(1.8, 6.0, 120)
    curves = {}
    for name in ("default", "miriam"):
        c4_ion = out[name]["c4_ion_kcal_A4"]
        e = []
        for r in rs:
            # rigid ligand, translated so the anchor sits r from the ion
            shifted = xyz[lig] + axis * (r - r0)
            d = np.linalg.norm(shifted - xyz[ion_i], axis=1)
            e.append(float(-np.sum(c4_ion * np.array([per_atom[name][i] for i in lig]) / d ** 4)))
        curves[name] = e
    out["scan"] = {"r_A": rs.tolist(), "bound_r_A": r0,
                   "energy_kcal": curves,
                   "background_kcal": (np.array(curves["default"]) -
                                       np.array(curves["miriam"])).tolist()}
    out["meta"] = {"config": CONFIG, "ion": f"{cfg.ion.element}{cfg.ion.charge}+",
                   "alpha_NG2R50": cfg.c4.alpha_by_class.get("NG2R50")}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)
    at_bound = np.interp(r0, rs, out["scan"]["background_kcal"])
    print(f"background C4 removed at the bound distance ({r0:.2f} A): {at_bound:+.2f} kcal/mol")


if __name__ == "__main__":
    main()
