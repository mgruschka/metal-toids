"""
alpha* figures for Co(II)-imidazole (co_imid1).

    python TI_param/figures/make_alpha_figures.py      # numpy + matplotlib only

  fig3_alpha_star.png    dG0(alpha) = anchor + dG_bound(alpha0 -> alpha): the TI
                         ladder on both PMF anchors (MBAR / WHAM dG0 at alpha0)
                         with error bars and linear best fits, the MBAR
                         reweighting of the PMF (+- SEM band) for reference, and
                         alpha* (interpolated) where each meets the target
  fig4_pmf_overlap.png   adjacent-window histogram overlap of the umbrella run
                         (the same overlap wham.py / mbar.py gate on), plus the
                         window histograms with the worst pair highlighted
  fig5_ti_overlap.png    MBAR overlap matrix of the 11 TI lambda windows (forward
                         ladder), from lambda_windows.mbar_over_lambda -- the
                         same estimator and subsampling as run_TI --analyze
  fig5_ti_overlap_rev.png  the same for the reverse (hysteresis) ladder
  fig6_ti_vs_reweight.png  plateau-corrected TI on the MBAR anchor vs the PMF
                         reweighted by MBAR: overlay, and their difference
                         (with the uncorrected TI for contrast)
  fig3_alpha_star.csv, fig4_pmf_overlap.csv, fig5_ti_overlap.csv, fig6_ti_vs_reweight.csv
                         the plotted numbers (table view)

Style: the house style of ligand-1264/figures/make_figures.py (dataviz reference
palette, light mode).  Slots 1-3 (blue/orange/aqua) validated --pairs all;
aqua is < 3:1 on the surface, so its series is direct-labelled.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "ligand-1264", "figures"))
sys.path.insert(0, os.path.join(REPO, "ligand-1264"))
import make_figures as house                      # noqa: E402  (sets rcParams)
from make_figures import plt, tidy, INK, INK_2, INK_MUTED, BLUE, ORANGE, AQUA  # noqa: E402
from lig1264.wham import load_windows, adjacent_overlap, MIN_OVERLAP           # noqa: E402

PMF = os.path.join(REPO, "ligand-1264", "runs", "pmf", "co_imid1_a109")
TI = os.path.join(REPO, "TI_param", "runs", "co_imid1_alpha_ti")
TARGET = -3.36            # kcal/mol, -RT ln K at 298 K, log K1 = 2.46


def crossing(x, y, target):
    """x where the piecewise-linear y(x) first crosses target (y decreasing)."""
    i = int(np.argmax(y <= target))
    return x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])


# ----------------------------------------------------------------- figure 3
def fig_alpha_star(out_png, out_csv):
    ti = json.load(open(os.path.join(TI, "alpha_ti_results.json")))
    w = ti["windows"]
    a = np.array([x["alpha"] for x in w])
    # plateau-corrected ddG0 = dG_bound - dG_unbound when --analyze had pmf_run
    corrected = all("ddG0_ti_kcal" in x for x in w)
    dg = np.array([x["ddG0_ti_kcal" if corrected else "dG_ti_kcal"] for x in w])
    dg_sem = np.array([x["dG_ti_sem_kcal"] for x in w])

    mb = json.load(open(os.path.join(PMF, "mbar.json")))
    wh = json.load(open(os.path.join(PMF, "pmf.json")))["binding"]
    anchors = [("MBAR", mb["dG0_alpha0_kcal"], mb["dG0_alpha0_sem_kcal"], BLUE),
               ("WHAM", wh["dG0_kcal"], wh["dG0_sem_kcal"], ORANGE)]
    rw = np.genfromtxt(os.path.join(PMF, "mbar_dG0_vs_alpha.csv"), delimiter=",", names=True)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.36)

    # MBAR reweighting of the PMF alone, with its bootstrap band: solid where
    # trusted, faded where the bound state runs below the lowest window (EDGE)
    j = int(np.argmax(rw["edge"] == 1))             # first flagged point
    for sl, al in ((slice(0, j + 1), 1.0), (slice(j, None), 0.35)):
        x, y, e = rw["alpha"][sl], rw["dG0_kcal"][sl], rw["dG0_sem_kcal"][sl]
        ax.fill_between(x, y - e, y + e, color=AQUA, alpha=0.18 * al, lw=0, zorder=2)
        ax.plot(x, y, color=AQUA, lw=1.8, alpha=al, zorder=2.5)
    a_mbar, a_mbar_sem = mb["alpha_target"], mb["alpha_target_sem"]
    ax.plot([], [], color=AQUA, lw=1.8,
            label=f"PMF reweighted by MBAR ± SEM;  α* = {a_mbar:.3f} ± {a_mbar_sem:.3f}\n"
                  "(faded: EDGE-flagged, bound state truncated, biased high)")
    ax.annotate("PMF + MBAR\nreweighting", xy=(1.6, np.interp(1.6, rw["alpha"], rw["dG0_kcal"])),
                xytext=(1.95, 9.2), fontsize=7.5, color=INK_2, ha="left",
                arrowprops=dict(arrowstyle="-", color=AQUA, lw=0.9, shrinkA=2, shrinkB=2))

    rows, summary = [], {}
    xs = np.linspace(0.95, 4.65, 50)
    for name, g0, s0, col in anchors:
        y = g0 + dg
        err = np.sqrt(s0 ** 2 + dg_sem ** 2)
        k, b = np.polyfit(a, y, 1)
        resid = y - (k * a + b)
        a_int, a_fit = crossing(a, y, TARGET), (TARGET - b) / k
        summary[name] = dict(anchor=g0, anchor_sem=s0, slope=k, intercept=b,
                             max_resid=float(np.abs(resid).max()),
                             alpha_star_interp=a_int, alpha_star_fit=a_fit)
        ax.plot(xs, k * xs + b, color=col, lw=1.1, alpha=0.75, zorder=3)
        ax.errorbar(a, y, yerr=err, color=col, marker="o", ms=4.5, mew=0, lw=0,
                    elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=4)
        rows += [(name, ai, yi, ei, k * ai + b) for ai, yi, ei in zip(a, y, err)]
        ax.errorbar([], [], yerr=[], color=col, marker="o", ms=4.5, mew=0, lw=1.1,
                    elinewidth=1.2, capsize=3.5,
                    label=f"TI on {name} anchor ({g0:+.2f} ± {s0:.2f}) + linear fit;  "
                          f"α* = {a_int:.3f}")

    ax.axhline(TARGET, color=INK_MUTED, lw=1.0, zorder=1)
    ax.annotate(f"target ΔG⁰ = {TARGET:.2f}  (log K₁ = 2.46)", xy=(4.6, TARGET),
                xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                fontsize=7.5, color=INK_2)

    ax.set_xlim(0.95, 4.65)
    ax.set_ylim(-16, 13)
    ax.set_xlabel(r"$\alpha$(NG2R50)  (Å$^3$)")
    ax.set_ylabel(r"$\Delta G^\circ(\alpha)$  (kcal/mol)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=1, handlelength=2.0)
    tidy(ax)

    sm, sw = summary["MBAR"], summary["WHAM"]
    fig.suptitle("α*(Co²⁺–imidazole N):  ΔG⁰ = anchor + TI ΔG_bound(α₀→α)"
                 + (" − ΔG_unbound" if corrected else ""),
                 x=0.12, ha="left", fontsize=11.5, color=INK, y=0.955)
    fig.text(0.12, 0.018,
             "TI: 11 uniform windows × 2.2 ns, α 1.09→4.50.  Anchor ΔG⁰(α₀ = 1.09): 120 umbrella "
             "windows × 4 ns,\nbound region r < 2.63 Å"
             + ("; TI plateau-corrected (same plateau shift as the MBAR reweighting)" if corrected
                else "; TI not plateau-corrected") + ".  Error bars: anchor ⊕ TI SEM, "
             f"±{min(r[3] for r in rows):.2f}–"
             f"{max(r[3] for r in rows):.2f} kcal/mol (≈ marker size).\n"
             f"Linear fit: slope {sm['slope']:.2f} kcal/mol per Å³, max |residual| "
             f"{sm['max_resid']:.2f} (ΔG⁰ slightly convex): fit α* {sm['alpha_star_fit']:.3f} / "
             f"{sw['alpha_star_fit']:.3f};\nthe α* in the legend is interpolated between "
             "windows and is the reference.",
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", linespacing=1.35)
    fig.savefig(out_png)
    with open(out_csv, "w") as fh:
        fh.write("anchor,alpha,dG0_kcal,dG0_err_kcal,linear_fit_kcal\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
    print("wrote", out_png, out_csv)
    return summary, (a_mbar, a_mbar_sem)


# ----------------------------------------------------------------- figure 4
def fig_overlap(out_png, out_csv, bins=200):
    meta, centres, samples = load_windows(PMF)      # discard 0, as the run used
    c = centres * 10
    s = [x * 10 for x in samples]
    ov = adjacent_overlap(samples, bins)            # identical to the gate
    mid = 0.5 * (c[:-1] + c[1:])
    i_min = int(np.argmin(ov))

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(9.6, 6.4),
                                   gridspec_kw={"height_ratios": [1, 1.1]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.88, bottom=0.15, hspace=0.42)

    axa.plot(mid, ov, color=BLUE, lw=1.6, marker="o", ms=3.2, mew=0, zorder=3)
    axa.axhline(MIN_OVERLAP, color=INK_MUTED, lw=1.0, zorder=1)
    axa.annotate(f"gate {MIN_OVERLAP} (WHAM/MBAR refuse below)", xy=(10.0, MIN_OVERLAP),
                 xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                 fontsize=7.5, color=INK_2)
    axa.annotate(f"min {ov[i_min]:.3f}\nwindows {i_min}–{i_min + 1}",
                 xy=(mid[i_min], ov[i_min]), xytext=(mid[i_min] + 0.55, 0.12),
                 fontsize=7.5, color=INK_2, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8, shrinkA=2, shrinkB=3))
    axa.text(0.995, 0.96, f"median {np.median(ov):.3f}", transform=axa.transAxes,
             ha="right", va="top", fontsize=7.5, color=INK_2)
    axa.set_xlim(2.0, 10.1)
    axa.set_ylim(0, 1.0)
    axa.set_xlabel("midpoint of the window pair's centres  (Å)")
    axa.set_ylabel("histogram overlap")
    axa.set_title("a   adjacent-window overlap  Σ min(Pᵢ, Pᵢ₊₁)", loc="left", color=INK, pad=8)
    tidy(axa)

    edges = np.linspace(min(x.min() for x in s), max(x.max() for x in s), bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for k, x in enumerate(s):
        h = np.histogram(x, bins=edges)[0] / len(x)
        if k in (i_min, i_min + 1):
            continue
        axb.plot(ctr, h, color=INK_MUTED, lw=0.6, alpha=0.55, zorder=2)
    for k, col in ((i_min, BLUE), (i_min + 1, ORANGE)):
        h = np.histogram(s[k], bins=edges)[0] / len(s[k])
        axb.plot(ctr, h, color=col, lw=1.8, zorder=4,
                 label=f"window {k} (centre {c[k]:.2f} Å)")
    axb.plot([], [], color=INK_MUTED, lw=0.8, label="all other windows")
    axb.set_xlim(2.0, 10.1)
    axb.set_ylim(bottom=0)
    axb.set_xlabel("Co–N(E2) distance  (Å)")
    axb.set_ylabel("fraction of samples per bin")
    axb.set_title(f"b   window histograms ({bins} bins, as the gate uses); worst pair highlighted",
                  loc="left", color=INK, pad=8)
    axb.legend(loc="upper right", ncol=1, handlelength=1.6)
    tidy(axb)

    fig.suptitle("Umbrella sampling coverage — Co²⁺–imidazole, α₀ = 1.09",
                 x=0.075, ha="left", fontsize=11.5, color=INK, y=0.965)
    fig.text(0.075, 0.02,
             f"{len(c)} windows, κ = {meta['kappa_kcal_per_A2']:.0f} kcal/mol/Å², 4 ns production "
             "each, COLVAR every 1 ps.  Narrow windows near 2.2 Å sit on the steep inner wall.\n"
             "The minimum is set by window 7 on the barrier top (2.63 Å): it is bimodal, part of its "
             "sampling falls back into the contact well.",
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
    fig.savefig(out_png)
    with open(out_csv, "w") as fh:
        fh.write("pair,window_i,window_j,centre_i_A,centre_j_A,midpoint_A,overlap\n")
        for i, o in enumerate(ov):
            fh.write(f"{i},{i},{i + 1},{c[i]:.4f},{c[i + 1]:.4f},{mid[i]:.4f},{o:.4f}\n")
    print("wrote", out_png, out_csv)
    return dict(min=float(ov[i_min]), i_min=i_min, at_A=float(mid[i_min]),
                median=float(np.median(ov)))


# ----------------------------------------------------------------- figure 5
RAMP_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fig_ti_overlap(out_png, out_csv, run=TI, label="forward ladder"):
    from matplotlib.colors import LinearSegmentedColormap
    from lig1264 import lambda_windows as LW
    from lig1264.config import load_config
    at = load_config(os.path.join(run, "config.yaml")).alpha_ti
    res = json.load(open(os.path.join(run, "alpha_ti_results.json")))
    lam = np.array([w["lam"] for w in res["windows"]])
    alpha = np.array([w["alpha"] for w in res["windows"]])
    samples = [LW.load_window(os.path.join(run, f"dudl_w{k:02d}.csv"), at.discard_frac)[0]
               for k in range(len(lam))]
    kT = 0.008314462618 * 300.0
    *_, adj, O = LW.mbar_over_lambda(lam, samples, kT, return_matrix=True)
    n = len(lam)

    cmap = LinearSegmentedColormap.from_list("blue_seq", RAMP_SEQ)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    fig.subplots_adjust(left=0.17, right=0.93, top=0.86, bottom=0.17)
    vmax = float(np.ceil(O.max() * 20) / 20)          # data range, to the next 0.05
    im = ax.imshow(O, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
    # values only where they carry the story: the diagonal and its neighbours
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 1:
                v = O[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.6 * vmax else INK)
    ticks = [f"{l:.1f}\n{a:.2f}" for l, a in zip(lam, alpha)]
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{l:.1f} | {a:.2f}" for l, a in zip(lam, alpha)],
                                                fontsize=7)
    ax.set_xlabel("window j:  λ  /  α (Å³)")
    ax.set_ylabel("window i:  λ | α (Å³)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0, colors=INK_2)
    cb.set_label("MBAR overlap  O$_{ij}$", fontsize=8, color=INK_2)

    fig.suptitle(f"TI λ-window overlap — {label}, α 1.09 → 4.50",
                 x=0.17, ha="left", fontsize=11.5, color=INK, y=0.955)
    fig.text(0.17, 0.905, f"adjacent minimum {adj.min():.3f}  (windows {int(np.argmin(adj))}–"
             f"{int(np.argmin(adj)) + 1});  rows sum to 1", fontsize=8, color=INK_2, ha="left")
    fig.text(0.17, 0.02,
             f"pymbar overlap matrix over the {n} uniform-λ windows, u_k = β λ_k dU/dλ; each window's "
             f"dU/dλ\nseries after dropping the first {at.discard_frac:.0%} and subsampling to its "
             "statistical inefficiency (as run_TI --analyze).\nValues printed on the diagonal and first "
             f"off-diagonals only; full matrix in {os.path.basename(out_csv)}.",
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", linespacing=1.35)
    fig.savefig(out_png)
    with open(out_csv, "w") as fh:
        fh.write("i\\j," + ",".join(f"w{j}" for j in range(n)) + "\n")
        for i in range(n):
            fh.write(f"w{i}," + ",".join(f"{O[i, j]:.4f}" for j in range(n)) + "\n")
    print("wrote", out_png, out_csv)
    return dict(adj_min=float(adj.min()), adj=adj.round(3).tolist(),
                second_neighbour_min=float(min(O[i, i + 2] for i in range(n - 2))))


# ----------------------------------------------------------------- figure 6
def fig_ti_vs_reweight(out_png, out_csv):
    w = json.load(open(os.path.join(TI, "alpha_ti_results.json")))["windows"]
    if not all("ddG0_ti_kcal" in x for x in w):
        raise SystemExit("alpha_ti_results.json has no ddG0_ti_kcal: rerun --analyze with alpha_ti.pmf_run")
    a = np.array([x["alpha"] for x in w])
    corr = np.array([x["ddG0_ti_kcal"] for x in w])
    raw = np.array([x["dG_ti_kcal"] for x in w])
    sem = np.array([x["dG_ti_sem_kcal"] for x in w])
    mb = json.load(open(os.path.join(PMF, "mbar.json")))
    g0 = mb["dG0_alpha0_kcal"]
    rw = np.genfromtxt(os.path.join(PMF, "mbar_dG0_vs_alpha.csv"), delimiter=",", names=True)
    rw_at = np.interp(a, rw["alpha"], rw["dG0_kcal"])          # 0.05 grid, error ~1e-4
    edge_at = np.interp(a, rw["alpha"], rw["edge"]) > 0.5
    a_edge = float(rw["alpha"][int(np.argmax(rw["edge"] == 1))])
    d_corr, d_raw = (g0 + corr) - rw_at, (g0 + raw) - rw_at

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.30, wspace=0.27)

    # --- a: overlay
    j = int(np.argmax(rw["edge"] == 1))
    axa.plot(rw["alpha"][: j + 1], rw["dG0_kcal"][: j + 1], color=AQUA, lw=1.8, zorder=2)
    axa.plot(rw["alpha"][j:], rw["dG0_kcal"][j:], color=AQUA, lw=1.8, alpha=0.35, zorder=2)
    axa.errorbar(a, g0 + corr, yerr=np.hypot(mb["dG0_alpha0_sem_kcal"], sem), color=BLUE,
                 marker="o", ms=4.5, mew=0, lw=0, elinewidth=1.2, capsize=3.5, zorder=4)
    axa.plot([], [], color=AQUA, lw=1.8, label="PMF reweighted by MBAR (faded: EDGE-flagged)")
    axa.errorbar([], [], yerr=[], color=BLUE, marker="o", ms=4.5, mew=0, lw=0, elinewidth=1.2,
                 capsize=3.5, label="TI on MBAR anchor, plateau-corrected")
    axa.axhline(TARGET, color=INK_MUTED, lw=1.0, zorder=1)
    axa.annotate(f"target {TARGET:.2f}", xy=(4.6, TARGET), xytext=(0, 4), textcoords="offset points",
                 ha="right", va="bottom", fontsize=7.5, color=INK_2)
    axa.set_xlim(0.95, 4.65)
    axa.set_ylim(-16, 13)
    axa.set_xlabel(r"$\alpha$(NG2R50)  (Å$^3$)")
    axa.set_ylabel(r"$\Delta G^\circ(\alpha)$  (kcal/mol)")
    axa.set_title("a   both routes, same anchor (+%.2f)" % g0, loc="left", color=INK, pad=8)
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, handlelength=2.0)
    tidy(axa)

    # --- b: difference
    axb.axvspan(a_edge, 4.65, color=AQUA, alpha=0.08, lw=0, zorder=0)
    axb.text(a_edge + 0.04, -0.285, "shaded: reweighting EDGE-flagged\n(bound state below 2.18 Å)",
             fontsize=7.5, color=INK_2, ha="left", va="bottom")
    axb.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
    axb.plot(a, d_raw, color=INK_MUTED, lw=1.2, marker="o", ms=3.5, mew=0, zorder=3,
             label="TI before plateau correction")
    axb.errorbar(a, d_corr, yerr=sem, color=BLUE, lw=1.6, marker="o", ms=4.5, mew=0,
                 elinewidth=1.2, capsize=3, zorder=4, label="TI, plateau-corrected")
    k = int(np.argmin(np.abs(a - 2.795)))
    axb.annotate(f"{d_corr[k]:+.3f}", xy=(a[k], d_corr[k]), xytext=(8, 6),
                 textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=INK_2)
    axb.annotate(f"{d_corr[-1]:+.3f}", xy=(a[-1], d_corr[-1]), xytext=(-6, -2),
                 textcoords="offset points", ha="right", va="top", fontsize=7.5, color=INK_2)
    axb.set_xlim(0.95, 4.65)
    axb.set_ylim(-0.30, 0.05)
    axb.set_xlabel(r"$\alpha$(NG2R50)  (Å$^3$)")
    axb.set_ylabel(r"$\Delta G^\circ_\mathrm{TI} - \Delta G^\circ_\mathrm{reweight}$  (kcal/mol)")
    axb.set_title("b   difference", loc="left", color=INK, pad=8)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, handlelength=2.0)
    tidy(axb)

    fig.suptitle("Plateau-corrected TI vs MBAR reweighting of the α₀ PMF — Co²⁺–imidazole",
                 x=0.075, ha="left", fontsize=11.5, color=INK, y=0.955)
    fig.text(0.075, 0.018,
             "Both curves start from the same MBAR ΔG⁰(α₀), so the anchor cancels in panel b; its "
             "error bars are the TI SEM only.  Plateau correction:\nΔG_unbound = −kT ln⟨exp(−βΔα c/r⁴)⟩ "
             "over the reweighting's own plateau shell (8.82–10.0 Å).  Reweighted curve interpolated "
             "to the TI α (0.05 Å³ grid).",
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", linespacing=1.35)
    fig.savefig(out_png)
    with open(out_csv, "w") as fh:
        fh.write("alpha,dG0_ti_corrected_kcal,dG0_ti_uncorrected_kcal,ti_sem_kcal,dG0_reweight_kcal,"
                 "diff_corrected_kcal,diff_uncorrected_kcal,reweight_edge\n")
        for row in zip(a, g0 + corr, g0 + raw, sem, rw_at, d_corr, d_raw, edge_at):
            fh.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(int(v)) for v in row) + "\n")
    print("wrote", out_png, out_csv)
    return dict(alpha=a.round(3).tolist(), diff_corrected=d_corr.round(4).tolist(),
                diff_raw=d_raw.round(4).tolist(), edge_from=a_edge)


if __name__ == "__main__":
    summ, mbar_star = fig_alpha_star(os.path.join(HERE, "fig3_alpha_star.png"),
                                     os.path.join(HERE, "fig3_alpha_star.csv"))
    for k, v in summ.items():
        print(f"{k} anchor {v['anchor']:+.3f}: slope {v['slope']:.3f}, max resid "
              f"{v['max_resid']:.3f}, alpha* interp {v['alpha_star_interp']:.4f}, "
              f"linear fit {v['alpha_star_fit']:.4f}")
    print(f"PMF+MBAR alpha* {mbar_star[0]:.4f} +/- {mbar_star[1]:.4f}")
    print("overlap:", fig_overlap(os.path.join(HERE, "fig4_pmf_overlap.png"),
                                  os.path.join(HERE, "fig4_pmf_overlap.csv")))
    print("TI vs reweight:", fig_ti_vs_reweight(os.path.join(HERE, "fig6_ti_vs_reweight.png"),
                                                os.path.join(HERE, "fig6_ti_vs_reweight.csv")))
    print("TI overlap:", fig_ti_overlap(os.path.join(HERE, "fig5_ti_overlap.png"),
                                        os.path.join(HERE, "fig5_ti_overlap.csv")))
    rev = os.path.join(REPO, "TI_param", "runs", "co_imid1_alpha_ti_rev")
    if os.path.exists(os.path.join(rev, "alpha_ti_results.json")):
        print("TI overlap (reverse):", fig_ti_overlap(
            os.path.join(HERE, "fig5_ti_overlap_rev.png"), os.path.join(HERE, "fig5_ti_overlap_rev.csv"),
            run=rev, label="reverse ladder"))
