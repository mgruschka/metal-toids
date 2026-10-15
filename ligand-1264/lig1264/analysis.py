#!/usr/bin/env python
"""
Post-processing for one run directory: ion <-> ligand-class distances,
ion-water RDF / coordination number, and a JSON summary that the fitting
step can consume.

    python -m lig1264.analysis runs/zn_ala5 [--phase prod] [--rmax 0.6]

Produces <outdir>/analysis_<phase>.json and .npz (time series, RDFs).
"""
import argparse
import json
import os

import numpy as np
import mdtraj as md

from .build import WATER_RES


def load(outdir, phase):
    top = os.path.join(outdir, "topology.pdb")
    # free MD writes npt_<phase>.dcd; ti.py writes window_NN.dcd, so
    # --phase window_05 also resolves
    dcd = os.path.join(outdir, f"npt_{phase}.dcd")
    if not os.path.exists(dcd):
        dcd = os.path.join(outdir, f"{phase}.dcd")
    groups = json.load(open(os.path.join(outdir, "groups.json")))
    classes = {int(k): v for k, v in json.load(open(os.path.join(outdir, "classes.json"))).items()}
    return md.load(dcd, top=top), groups, classes


def rdf_cn(traj, ion, targets, rmax, nbins):
    """Radial distribution and running coordination number ion -> targets."""
    pairs = np.array([[ion, j] for j in targets])
    r, g = md.compute_rdf(traj, pairs, r_range=(0.0, rmax), n_bins=nbins)
    # running CN: integrate 4 pi r^2 rho g(r), rho = N_targets / <V>
    rho = len(targets) / np.mean(traj.unitcell_volumes)
    dr = r[1] - r[0]
    cn = np.cumsum(4 * np.pi * r ** 2 * g * rho * dr)
    return r, g, cn


def first_min(r, g, r_lo=0.15):
    """Position of the first minimum of g(r) after its first peak."""
    m = r > r_lo
    rr, gg = r[m], g[m]
    if not len(rr):
        return np.nan
    ipk = np.argmax(gg)
    tail = gg[ipk:]
    if len(tail) < 3:
        return np.nan
    return float(rr[ipk + np.argmin(tail)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir")
    ap.add_argument("--phase", default="prod")
    ap.add_argument("--rmax", type=float, default=0.6, help="nm")
    ap.add_argument("--nbins", type=int, default=300)
    ap.add_argument("--skip-frac", type=float, default=0.0,
                    help="discard this fraction of frames from the start")
    args = ap.parse_args(argv)

    traj, groups, classes = load(args.outdir, args.phase)
    traj = traj[int(args.skip_frac * traj.n_frames):]
    ion = groups["ion"][0]
    out = {"phase": args.phase, "n_frames": traj.n_frames,
           "time_ns": float(traj.time[-1] - traj.time[0]) / 1000.0}
    arrays = {}

    # ion <-> each ligand class: min distance per frame + RDF/CN
    by_class = {}
    for i in groups["ligand"]:
        by_class.setdefault(classes[i], []).append(i)
    out["ligand_classes"] = {}
    for cls, idx in sorted(by_class.items()):
        d = md.compute_distances(traj, np.array([[ion, j] for j in idx]))
        dmin = d.min(axis=1)
        r, g, cn = rdf_cn(traj, ion, idx, args.rmax, args.nbins)
        rmin1 = first_min(r, g)
        out["ligand_classes"][cls] = {
            "n_atoms": len(idx),
            "min_dist_mean_A": float(dmin.mean() * 10),
            "min_dist_std_A": float(dmin.std() * 10),
            "frac_bound_2p5A": float((dmin < 0.25).mean()),
            "first_min_A": rmin1 * 10 if np.isfinite(rmin1) else None,
            "cn_at_first_min": float(np.interp(rmin1, r, cn)) if np.isfinite(rmin1) else None,
        }
        arrays[f"dmin_{cls}"] = dmin
        arrays[f"rdf_{cls}"] = np.vstack([r, g, cn])

    # ion <-> water oxygen: IOD and CN, the solvent-side sanity check
    wO = [a.index for a in traj.topology.atoms
          if a.residue.name in WATER_RES and a.element.symbol == "O"]
    r, g, cn = rdf_cn(traj, ion, wO, args.rmax, args.nbins)
    ipk = np.argmax(g)
    rmin1 = first_min(r, g)
    out["water"] = {
        "IOD_A": float(r[ipk] * 10),
        "first_min_A": rmin1 * 10 if np.isfinite(rmin1) else None,
        "CN": float(np.interp(rmin1, r, cn)) if np.isfinite(rmin1) else None,
    }
    arrays["rdf_water_O"] = np.vstack([r, g, cn])

    json.dump(out, open(os.path.join(args.outdir, f"analysis_{args.phase}.json"), "w"), indent=2)
    np.savez(os.path.join(args.outdir, f"analysis_{args.phase}.npz"), **arrays)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
