"""
Weekly-report figures.

    python figures/make_figures.py            # needs only numpy + matplotlib

  fig1_ti_round1.png       round-1 TI: <dU/dlambda> per window, and ddG(alpha)
                           read from runs/round1/*/ti_results.json
  fig2_c4_convention.png   ParmEd-default vs MIRIAM C4 convention, read from
                           figures/c4_data.json (regenerate with c4_data.py)

Palette: dataviz reference instance, light mode.  alpha is a magnitude, so the
three complex legs use the ordinal blue ramp (steps 250/450/650, validated
--ordinal); the water leg is a different kind of thing and takes categorical
slot 2 (orange).  Figure 2 uses categorical slots 1-3, validated --pairs all.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {1.09: "#86b6ef", 2.75: "#2a78d6", 4.50: "#104281"}   # ordinal, light->dark

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": "#d6d5d0", "axes.linewidth": 0.8,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.dpi": 200,
})


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.9)


def load_ti():
    runs = {}
    for name in ("ni_co_water", "ni_co_imid4_a109", "ni_co_imid4_a275", "ni_co_imid4_a450"):
        p = os.path.join(ROOT, "runs", "round1", name, "ti_results.json")
        runs[name] = json.load(open(p))
    return runs


# ----------------------------------------------------------------- figure 1
def fig_ti(runs, out):
    # explicit layout: tight_layout fights the below-axes legends and caption
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.4))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.30, wspace=0.26)

    series = [("ni_co_water", "water leg\n(no ligand)", ORANGE),
              ("ni_co_imid4_a109", r"$\alpha$ = 1.09", RAMP[1.09]),
              ("ni_co_imid4_a275", r"$\alpha$ = 2.75", RAMP[2.75]),
              ("ni_co_imid4_a450", r"$\alpha$ = 4.50", RAMP[4.50])]

    # the water and alpha=2.75 curves nearly coincide at lambda=1, so their
    # end labels are nudged apart by hand
    nudge = {"ni_co_water": 7, "ni_co_imid4_a275": -7}
    for key, label, colour in series:
        w = runs[key]["windows"]
        lam = np.array([x["lam"] for x in w])
        mean = np.array([x["mean"] for x in w]) / 4.184        # kJ -> kcal/mol
        sem = np.array([x["sem"] for x in w]) / 4.184
        axa.errorbar(lam, mean, yerr=sem, color=colour, lw=1.8, marker="o",
                     ms=4.5, mew=0, capsize=0, label=label.replace("\n", " "),
                     zorder=3)
        axa.annotate(label.replace("\n", " "), xy=(lam[-1], mean[-1]),
                     xytext=(5, nudge.get(key, 0)), textcoords="offset points",
                     va="center", ha="left", fontsize=8, color=colour)

    axa.set_xlabel(r"$\lambda$   (Ni$^{2+}$ $\rightarrow$ Co$^{2+}$)")
    axa.set_ylabel(r"$\langle \partial U/\partial \lambda \rangle$   (kcal/mol)")
    axa.set_title("a   TI integrand, 12 Gauss–Legendre windows", loc="left",
                  color=INK, pad=8)
    axa.set_xlim(-0.04, 1.42)
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
               columnspacing=1.1, handlelength=1.5)
    tidy(axa)

    # --- panel b: ddG(alpha)
    dG_water = runs["ni_co_water"]["dG_kcal"]
    sem_water = runs["ni_co_water"]["dG_sem_kcal"]
    alphas, ddg, err = [], [], []
    for key in ("ni_co_imid4_a109", "ni_co_imid4_a275", "ni_co_imid4_a450"):
        r = runs[key]
        alphas.append(r["alpha"]["NG2R50"])
        ddg.append(r["dG_kcal"] - dG_water)
        err.append((r["dG_sem_kcal"] ** 2 + sem_water ** 2) ** 0.5)
    alphas, ddg, err = np.array(alphas), np.array(ddg), np.array(err)

    slope, intercept = np.polyfit(alphas, ddg, 1)
    xs = np.linspace(0.6, 5.4, 50)
    axb.axhspan(0, 4.6, color="#f2f5f9", zorder=0)
    axb.axhline(0, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axb.plot(xs, slope * xs + intercept, color=INK_MUTED, lw=1.2, ls="-",
             zorder=2, label=f"linear fit, slope {slope:.2f} kcal/mol per Å$^3$")
    for a, y, e in zip(alphas, ddg, err):
        axb.errorbar([a], [y], yerr=[e], color=RAMP[a], marker="o", ms=7, mew=0,
                     capsize=0, zorder=4, lw=0)
        dy = -16 if y > 2 else 9
        axb.annotate(f"{y:+.2f}", xy=(a, y), xytext=(0, dy), textcoords="offset points",
                     ha="center", fontsize=8, color=RAMP[a])
    axb.plot([], [], color=RAMP[2.75], marker="o", ms=7, mew=0, lw=0,
             label=r"TI $\Delta\Delta G$ (complex $-$ water leg)")

    x0 = -intercept / slope
    axb.annotate("sign change\n" + r"at $\alpha$ = " + f"{x0:.2f} Å$^3$",
                 xy=(x0, 0), xytext=(3.45, -2.55), fontsize=8, color=INK_2,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8,
                                 shrinkA=2, shrinkB=4))
    axb.text(0.72, 4.45, "shaded: $\\Delta\\Delta G > 0$ expected\n"
                         "(Irving–Williams, N-donors);\ntarget not yet fixed",
             fontsize=7.5, color=INK_2, ha="left", va="top", linespacing=1.35)
    axb.set_xlabel(r"$\alpha$(NG2R50), imidazole-N polarizability  (Å$^3$)")
    axb.set_ylabel(r"$\Delta\Delta G$   (kcal/mol)")
    axb.set_title("b   Ni/Co selectivity for 4 imidazoles", loc="left", color=INK, pad=8)
    axb.set_xlim(0.6, 5.4)
    axb.set_ylim(-3.6, 4.6)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
               columnspacing=1.1, handlelength=1.5)
    tidy(axb)

    fig.suptitle("Round 1 thermodynamic integration — Ni$^{2+}$ → Co$^{2+}$, 4 imidazoles, SPC/E",
                 x=0.075, ha="left", fontsize=11.5, color=INK, y=0.955)
    fig.text(0.075, 0.035,
             "12 Gauss–Legendre windows × 5 ns NVT at 1 fs per leg; error bars are block SEM. "
             "Water leg = the same alchemical swap with no ligand present.",
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
    fig.savefig(out)
    print("wrote", out)
    return dict(alphas=alphas.tolist(), ddg=ddg.tolist(), slope=slope, zero_at=x0)


# ----------------------------------------------------------------- figure 2
def fig_c4(out):
    d = json.load(open(os.path.join(HERE, "c4_data.json")))
    order = ["NG2R50", "NG2R51", "CG2R51", "CG2R53", "HGR52", "HGP1"]
    pretty = {"NG2R50": "NG2R50\n(ligating N)", "NG2R51": "NG2R51\n(N–H)",
              "CG2R51": "CG2R51\n(ring C)", "CG2R53": "CG2R53\n(ring C)",
              "HGR52": "HGR52\n(ring H)", "HGP1": "HGP1\n(polar H)"}
    dflt = [d["default"]["arel_by_class"][c] for c in order]
    mrm = [d["miriam"]["arel_by_class"][c] for c in order]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.33, wspace=0.26)
    x = np.arange(len(order))
    w = 0.38
    axa.bar(x - w / 2 - 0.012, dflt, w, color=BLUE, label="ParmEd default (Miller α on every class)")
    axa.bar(x + w / 2 + 0.012, mrm, w, color=ORANGE, label="MIRIAM convention (ligating atom only)")
    for xi, v in zip(x, dflt):
        axa.annotate(f"{v:.2f}", xy=(xi - w / 2, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5, color=INK_2)
    axa.annotate(f"{mrm[0]:.2f}", xy=(w / 2, mrm[0]), xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=7.5, color=INK_2)
    for xi in x[1:]:                      # the zeroed classes have no visible bar
        axa.annotate("0", xy=(xi + w / 2, 0), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5, color=ORANGE)
    axa.set_xticks(x)
    axa.set_xticklabels([pretty[c] for c in order], fontsize=7.5)
    axa.set_ylabel(r"$a_{\mathrm{rel}} = \alpha_j\,/\,1.444$")
    axa.set_title("a   C$_4$ mixing weight per imidazole atom class", loc="left",
                  color=INK, pad=8)
    axa.set_ylim(0, 1.06)
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1,
               handlelength=1.5)
    tidy(axa)

    s = d["scan"]
    r = np.array(s["r_A"])
    e_def = np.array(s["energy_kcal"]["default"])
    e_mrm = np.array(s["energy_kcal"]["miriam"])
    bg = np.array(s["background_kcal"])
    rb = s["bound_r_A"]

    axb.axhline(0, color="#d6d5d0", lw=0.8)
    axb.plot(r, e_def, color=BLUE, lw=2.0, label="ParmEd default", zorder=3)
    axb.plot(r, e_mrm, color=ORANGE, lw=2.0, label="MIRIAM convention", zorder=3)
    axb.plot(r, bg, color=AQUA, lw=2.0, ls=(0, (5, 2)), zorder=3,
             label="difference = background removed")
    for y, c, lab in ((e_def, BLUE, "default"), (e_mrm, ORANGE, "MIRIAM"),
                      (bg, AQUA, "background")):
        k = 6                             # label where the curves are separated
        off = (4, 10) if lab == "background" else (7, -1)   # clear of the leader line
        axb.annotate(lab, xy=(r[k], y[k]), xytext=off, textcoords="offset points",
                     va="center", ha="left", fontsize=8, color=c)

    axb.axvline(rb, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    at_bound = float(np.interp(rb, r, bg))
    axb.annotate(f"at the bound distance {rb:.1f} Å\nthe background carried "
                 f"\u2212{abs(at_bound):.1f} kcal/mol",
                 xy=(rb, at_bound), xytext=(rb + 1.5, -13.0), fontsize=8, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    axb.set_xlabel(r"Co$^{2+}\cdots$N$_{\varepsilon}$ distance  (Å)")
    axb.set_ylabel(r"ion$\leftrightarrow$ligand C$_4$ energy  (kcal/mol)")
    axb.set_title("b   What the convention is worth, rigid imidazole", loc="left",
                  color=INK, pad=8)
    axb.set_xlim(1.8, 6.6)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1,
               handlelength=1.8)
    tidy(axb)
    axb.grid(True, axis="x", alpha=0.6)

    fig.suptitle("12-6-4 C$_4$ convention: Co$^{2+}$ + imidazole, "
                 r"$\alpha$(NG2R50) = 1.09 Å$^3$ in both",
                 x=0.075, ha="left", fontsize=11.5, color=INK, y=0.955)
    fig.text(0.075, 0.02,
             "Panel b: rigid imidazole slid along the Co···N$_\\varepsilon$ axis, "
             "summing $-C_4^{ij}/r^4$ over all ligand atoms. "
             "The ion↔water C$_4$ is identical under both conventions.",
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
    fig.savefig(out)
    print("wrote", out)
    return at_bound


if __name__ == "__main__":
    ti = fig_ti(load_ti(), os.path.join(HERE, "fig1_ti_round1.png"))
    bg = fig_c4(os.path.join(HERE, "fig2_c4_convention.png"))
    print(f"\nfig1: ddG = {['%+.2f' % v for v in ti['ddg']]} kcal/mol at alpha = "
          f"{ti['alphas']}, slope {ti['slope']:.2f}, sign change at {ti['zero_at']:.2f}")
    print(f"fig2: background C4 at contact {bg:+.2f} kcal/mol")
