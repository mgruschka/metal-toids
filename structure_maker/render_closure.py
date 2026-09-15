"""
Render a ring-closure trajectory (make_cyclic.py --trajectory) to a movie with
matplotlib only - no VMD, no X display needed.

    python render_closure.py cyc6_closure.pdb                 # -> cyc6_closure.mp4
    python render_closure.py cyc6_closure.pdb --gif           # -> cyc6_closure.gif
    python render_closure.py cyc6_closure.pdb --fps 30 --hold 2 --spin 90

MP4 uses the ffmpeg bundled with imageio-ffmpeg (pip); GIF needs only Pillow.
Backbone bonds are drawn as sticks coloured by residue, N/O atoms as small
spheres, and the two atoms the optimizer is driving together - N of residue 1
and the virtual N of residue n+1 (residue GAP) - as large blue/red spheres with
the remaining gap printed in the corner.  After the ring closes the last frame
is held while the camera spins.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import mdtraj as md

BACKBONE_BONDS = [("N", "CA"), ("CA", "C"), ("C", "O")]
PRO_BONDS = [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")]
ELEMENT_COLOR = {"N": "#3050f8", "O": "#ff0d0d", "C": "#909090"}


def bond_list(top):
    """(i, j, residue_index) for backbone + proline ring + inter-residue C-N bonds."""
    res = [r for r in top.residues if r.name != "GAP"]
    n = len(res)
    idx = {(r.index, a.name): a.index for r in res for a in r.atoms}
    bonds = []
    for k, r in enumerate(res):
        pairs = BACKBONE_BONDS + (PRO_BONDS if r.name == "PRO" else [])
        for a, b in pairs:
            if (r.index, a) in idx and (r.index, b) in idx:
                bonds.append((idx[(r.index, a)], idx[(r.index, b)], k))
        nxt = res[(k + 1) % n]
        if k + 1 < n:                                   # C(i)-N(i+1); closing bond is the gap
            bonds.append((idx[(r.index, "C")], idx[(nxt.index, "N")], k))
    return bonds, n


def render(pdb, out, fps=20, hold=1.5, spin=120, dpi=110, size=6.0):
    t = md.load(pdb, standard_names=False)
    xyz = t.xyz * 10.0                                 # Angstrom
    top = t.top
    bonds, n = bond_list(top)
    gap_idx = top.select("resname GAP and name N")
    n1_idx = top.select("resSeq 1 and name N")[0]
    gap_idx = gap_idx[0] if len(gap_idx) else None
    heavy = [a for a in top.atoms if a.residue.name != "GAP"]

    # view box: sized to the largest single frame, following a smoothed
    # centroid so the molecule stays centred without jittering
    heavy_idx = [a.index for a in heavy]
    cents = xyz[:, heavy_idx].mean(axis=1)
    half = max(np.abs(xyz[f, heavy_idx] - cents[f]).max() for f in range(t.n_frames)) * 1.05 + 1.0
    smooth = np.copy(cents)
    for f in range(1, len(smooth)):
        smooth[f] = 0.85 * smooth[f - 1] + 0.15 * cents[f]
    cmap = plt.get_cmap("viridis")
    res_color = [cmap(k / max(n - 1, 1)) for k in range(n)]

    n_fold = t.n_frames
    n_hold = int(round(hold * fps))
    total = n_fold + n_hold
    az0 = -60.0

    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def draw(frame):
        ax.cla()
        f = min(frame, n_fold - 1)
        p = xyz[f]
        for i, j, k in bonds:
            ax.plot(*zip(p[i], p[j]), color=res_color[k], lw=3.2, solid_capstyle="round")
        for a in heavy:
            if a.element.symbol in ("N", "O"):
                ax.scatter(*p[a.index], s=34, color=ELEMENT_COLOR[a.element.symbol],
                           depthshade=True, edgecolors="k", linewidths=0.3)
        ax.scatter(*p[n1_idx], s=160, color=ELEMENT_COLOR["N"], edgecolors="k", linewidths=0.6)
        if gap_idx is not None:
            ax.scatter(*p[gap_idx], s=160, color="#ff9900", edgecolors="k", linewidths=0.6)
            ax.plot(*zip(p[n1_idx], p[gap_idx]), color="#ff9900", lw=1.2, ls="--")
            gap = np.linalg.norm(p[n1_idx] - p[gap_idx])
            label = f"gap {gap:5.2f} Å"
        else:
            label = ""
        stage = f"closing  {f + 1}/{n_fold}" if frame < n_fold else "closed"
        ax.text2D(0.03, 0.95, f"{stage}\n{label}", transform=ax.transAxes,
                  fontsize=11, family="monospace", va="top")
        centre = smooth[f]
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.set_axis_off()
        # slow spin throughout, faster once closed
        az = az0 + spin * frame / max(total - 1, 1) * (1.0 if frame < n_fold else 1.6)
        ax.view_init(elev=22, azim=az)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=total, blit=False)
    if out.endswith(".gif"):
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    else:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
        anim.save(out, writer=writer)
    plt.close(fig)
    print(f"wrote {out}: {total} frames ({n_fold} closing + {n_hold} hold) at {fps} fps")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb", help="multi-MODEL closure PDB from make_cyclic.py --trajectory")
    p.add_argument("--out", help="output file (.mp4 default, or .gif)")
    p.add_argument("--gif", action="store_true", help="write a GIF instead of MP4")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--hold", type=float, default=1.5, help="seconds to hold the closed ring")
    p.add_argument("--spin", type=float, default=120, help="total camera azimuth sweep (deg)")
    p.add_argument("--dpi", type=int, default=110)
    a = p.parse_args()
    out = a.out or os.path.splitext(a.pdb)[0] + (".gif" if a.gif else ".mp4")
    render(a.pdb, out, fps=a.fps, hold=a.hold, spin=a.spin, dpi=a.dpi)


if __name__ == "__main__":
    main()
