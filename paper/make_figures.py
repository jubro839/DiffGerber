"""Generate the paper's figures from experiments/results/ (corrected protocol)."""

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = pathlib.Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.4, "font.family": "DejaVu Sans",
})
BLUE, RED, GREEN, GRAY, ORANGE = "#2E5EAA", "#C0392B", "#1E7A53", "#6B7480", "#D98218"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=180)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def softplus(x):
    return math.log1p(math.exp(x)) if x < 30 else x


def learned_path(tag):
    s = json.loads((RES / f"scalar_thresholds_{tag}.json").read_text())
    d = {int(k.split("_")[1]): softplus(s[k]["raw"])
         for k in s if k.startswith("tglobal_")}
    return pd.Series(d).sort_index()


# ───────────────────── Fig 1: pipeline ─────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    ax.set_xlim(-0.15, 10.75); ax.set_ylim(-0.35, 2.6)
    ax.axis("off"); ax.grid(False)
    boxes = [
        (0.05, "returns\n$r_t$", GRAY),
        (2.15, "soft Gerber\n$G(\\theta)$", BLUE),
        (4.25, "covariance\n$\\Sigma = D\\,G\\,D$", GRAY),
        (6.35, "GMV layer\n$w(\\Sigma)$", BLUE),
        (8.45, "realized\nvariance", GREEN),
    ]
    TOP, BOT = 2.30, 1.62
    for x, label, col in boxes:
        ax.add_patch(FancyBboxPatch((x, BOT), 2.0, TOP - BOT,
                                    boxstyle="round,pad=0.04",
                                    fc="#F2F5F9", ec=col, lw=1.3))
        ax.text(x + 1.0, (TOP + BOT) / 2, label, ha="center", va="center",
                fontsize=8, color=col)
        if x < 8.4:
            ax.add_patch(FancyArrowPatch((x + 2.02, (TOP + BOT) / 2),
                                         (x + 2.13, (TOP + BOT) / 2),
                                         arrowstyle="-|>", mutation_scale=9,
                                         color="#9AA6B4", lw=1.1))
    # backward pass: clearly below the boxes, bulging downward
    ax.add_patch(FancyArrowPatch((9.45, BOT - 0.10), (3.15, BOT - 0.10),
                                 arrowstyle="-|>", mutation_scale=10, color=RED,
                                 lw=1.2, linestyle=(0, (4, 2)),
                                 connectionstyle="arc3,rad=-0.30"))
    ax.text(6.3, -0.28, "backpropagation: the investment outcome defines the threshold",
            ha="center", fontsize=7.5, color=RED)
    ax.text(3.15, 2.46, "only $\\theta$ (thresholds) is learned; the volatility "
                        "scale $D$ is fixed",
            ha="center", fontsize=7.5, color=BLUE)
    save(fig, "fig1_pipeline")


# ─────────────── Fig 2: threshold sensitivity (headline) ───────────────
def fig_threshold():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    for ax, (tag, title) in zip(axes, [("gs2022", "Large-cap universe ($N{=}100 < T$)"),
                                       ("gs2022_top400", "High-dim universe ($N{=}400 > T$)")]):
        df = pd.read_csv(RES / f"threshold_grid_{tag}.csv")
        grid = df[df["threshold"].str.startswith("c=")].copy()
        grid["c"] = grid["threshold"].str.replace("c=", "").astype(float)
        grid = grid.sort_values("c")
        y = grid["ann_vol_net"] * 100
        ax.plot(grid["c"], y, "-o", color=GRAY, ms=3, label="fixed threshold $c$")
        i_or = y.idxmin()
        ax.scatter([grid.loc[i_or, "c"]], [y[i_or]], s=52, marker="*", color=GREEN,
                   zorder=5, label=f"ex-post oracle ($c^*$={grid.loc[i_or, 'c']:.1f})")
        row = lambda k: df.loc[df["threshold"] == k, "ann_vol_net"].iloc[0] * 100
        ax.axhline(row("learned"), color=BLUE, lw=1.2, ls="-",
                   label=f"learned (ours) {row('learned'):.2f}%")
        ax.axhline(row("val_select"), color=ORANGE, lw=1.0, ls="--",
                   label=f"grid + validation {row('val_select'):.2f}%")
        c05 = grid.loc[np.isclose(grid["c"], 0.5)].iloc[0]
        ax.scatter([0.5], [c05["ann_vol_net"] * 100], s=42, marker="X", color=RED,
                   zorder=5, label="published default $c{=}0.5$")
        ax.set_xlabel("threshold coefficient $c$")
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper center", handlelength=1.6)
        lo, hi = y.min(), np.percentile(y, 85)
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.75 * (hi - lo))
    axes[0].set_ylabel("OOS annualized volatility (%)")
    save(fig, "fig2_threshold_sensitivity")


# ─────────────── Fig 3: learned paths, equities vs crypto ───────────────
def fig_paths():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    e100, e400 = learned_path("gs2022"), learned_path("gs2022_top400")
    cr = pd.read_csv(RES / "crypto_learned_c.csv", index_col=0).iloc[:, 0]
    cr.index = cr.index.astype(int)
    ax.plot(e100.index, e100.values, "-o", ms=3, color=BLUE, label="equities, $N{=}100$")
    ax.plot(e400.index, e400.values, "-s", ms=3, color="#7FA3D6", label="equities, $N{=}400$")
    ax.plot(cr.index, cr.values, "-^", ms=3, color=RED, label="crypto, $N{=}20$")
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.0)
    ax.text(2022.9, 0.54, "published default $c{=}0.5$", fontsize=6.5, color=GRAY)
    ax.set_xlabel("refit year"); ax.set_ylabel("learned threshold $c$")
    ax.set_ylim(0, 1.5)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig3_learned_paths")


# ─────────────── Fig 4: simulation ───────────────
def fig_simulation():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    # (a) Gerber vs shrinkage across tail heaviness
    df = pd.read_csv(RES / "simulation_threshold_vs_tails.csv")
    x = np.arange(len(df))
    lab = [("$\\infty$" if not np.isfinite(float(v)) else f"{float(v):.0f}")
           for v in df["df"]]
    ax = axes[0]
    ax.plot(x, df["excess_at_cstar"], "-o", ms=3.5, color=BLUE, label="Gerber (best $c$)")
    ax.plot(x, df["excess_ledoit_wolf"], "-s", ms=3, color=GRAY, label="Ledoit--Wolf")
    ax.plot(x, df["excess_ans"], "-^", ms=3, color=ORANGE, label="nonlinear shrinkage")
    ax.plot(x, df["excess_sample"], ":", color="#B8C0CA", label="sample")
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_xlabel("Student-$t$ degrees of freedom  (heavier tails $\\leftarrow$)")
    ax.set_ylabel("excess variance vs oracle")
    ax.set_title("(a) Robustness crossover", loc="left")
    ax.legend(frameon=False)
    # (b) two targets
    ax = axes[1]
    t = pd.read_csv(RES / "simulation_two_targets.csv")
    lv = [s.split(",")[0].replace("p=", "") for s in t["contamination"]]
    lv = [f"{float(v) * 100:.0f}%" for v in lv]
    xx = np.arange(len(t))
    ax.plot(xx, t["c_star_statistical"], "-o", ms=3.5, color=GREEN,
            label="statistical target\n(recover clean $\\Sigma$)")
    ax.plot(xx, t["c_star_decision"], "-o", ms=3.5, color=BLUE,
            label="decision target\n(realized variance)")
    ax.fill_between(xx, t["c_star_statistical"], t["c_star_decision"],
                    color=BLUE, alpha=0.10)
    ax.set_xticks(xx); ax.set_xticklabels(lv)
    ax.set_xlabel("outlier contamination rate")
    ax.set_ylabel("optimal threshold $c^*$")
    ax.set_title("(b) The two targets diverge", loc="left")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig4_simulation")


# ─────── Fig 5: estimation objective vs decision objective (R4) ───────
def fig_objectives():
    df = pd.read_csv(RES / "reinforcement_frobenius.csv")
    df = df[df.year != "METRICS"].copy()
    df["year"] = df["year"].astype(int)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(df.year, df.c_decision, "-o", ms=3.5, color=BLUE,
            label="decision objective\n(realized portfolio variance)")
    ax.plot(df.year, df.c_frobenius, "-s", ms=3.5, color=ORANGE,
            label="estimation objective\n(Frobenius to realized $\\Sigma$)")
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.0)
    ax.text(2023.4, 0.435, "published default $c{=}0.5$", fontsize=6.5,
            color=GRAY)
    ax.set_xlabel("refit year"); ax.set_ylabel("learned threshold $c$")
    ax.set_ylim(0.3, 1.4)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    save(fig, "fig5_objectives")


# ─────────────── Fig 6: mechanism (R6) ───────────────
def fig_mechanism():
    df = pd.read_csv(RES / "reinforcement_mechanism.csv")
    LABEL = {"top100": ("S&P top-100", 0.006, -0.045, "left"),
             "sector30": ("S&P sector 30", -0.007, -0.02, "right"),
             "sector50": ("S&P sector 50", 0.007, -0.035, "left"),
             "sector100": ("S&P sector 100", -0.007, 0.035, "right"),
             "exsp30": ("ex-S&P 30", -0.004, 0.045, "right"),
             "exsp50": ("ex-S&P 50", 0.005, -0.055, "left")}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.scatter(df.mean_corr, df.learned_c, s=36, color=BLUE, zorder=5)
    for _, r in df.iterrows():
        lab, dx, dy, ha = LABEL.get(r.universe, (r.universe, 0.006, 0.015, "left"))
        ax.annotate(lab, (r.mean_corr, r.learned_c),
                    xytext=(r.mean_corr + dx, r.learned_c + dy),
                    fontsize=6.5, color=GRAY, ha=ha)
    rho = np.corrcoef(df.mean_corr, df.learned_c)[0, 1]
    b, a = np.polyfit(df.mean_corr, df.learned_c, 1)
    xs = np.linspace(df.mean_corr.min() - 0.02, df.mean_corr.max() + 0.02, 10)
    ax.plot(xs, a + b * xs, "--", color=GRAY, lw=1.0)
    ax.axhline(0.5, color=RED, ls=":", lw=0.9)
    ax.text(0.30, 0.53, "published default", fontsize=6.5, color=RED)
    ax.set_xlabel("mean pairwise correlation of the universe")
    ax.set_ylabel("learned threshold $c$ (mean, 2019+)")
    ax.set_title(f"$\\rho = {rho:+.2f}$ across equity universes", loc="left")
    save(fig, "fig6_mechanism")


# ─────────────── Fig 7: cost sensitivity (T6 companion) ───────────────
def fig_cost():
    cc = pd.read_csv(RES / "economic_value_cost_curve.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = {30: BLUE, 50: GREEN, 100: ORANGE}
    for N in (30, 50, 100):
        d = []
        for cb in (0, 10, 25, 50):
            a = cc[(cc.N == N) & (cc.method == "A3_hrp_gerber")
                   & (cc.cost_bps == cb)].iloc[0].ann_ret_net
            h = cc[(cc.N == N) & (cc.method == "hrp")
                   & (cc.cost_bps == cb)].iloc[0].ann_ret_net
            d.append((a - h) * 100)
        ax.plot([0, 10, 25, 50], d, "-o", ms=3.5, color=colors[N],
                label=f"$N{{=}}{N}$")
    ax.axhline(0.0, color=GRAY, lw=1.0)
    ax.set_xlabel("one-way transaction cost (bp)")
    ax.set_ylabel("$\\Delta$ann.\\ return, DG-HRP $-$ HRP (pp)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig7_cost")


# ───── Fig 3+5+6 merged: everything the fitted threshold tells us ─────
def fig_thresholds_combined():
    """Three panels across the full text width, replacing the separate
    objectives / learned-paths / mechanism figures."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.42))

    # (a) estimation vs decision objective
    ax = axes[0]
    df = pd.read_csv(RES / "reinforcement_frobenius.csv")
    df = df[df.year != "METRICS"].copy()
    df["year"] = df["year"].astype(int)
    ax.plot(df.year, df.c_decision, "-o", ms=2.5, color=BLUE, label="decision")
    ax.plot(df.year, df.c_frobenius, "-s", ms=2.5, color=ORANGE, label="estimation")
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.0)
    ax.set_ylim(0.3, 1.45)
    ax.set_xlabel("refit year"); ax.set_ylabel("learned threshold $c$")
    ax.set_title("(a) Objective", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)

    # (b) mechanism
    ax = axes[1]
    me = pd.read_csv(RES / "reinforcement_mechanism.csv")
    ax.scatter(me.mean_corr, me.learned_c, s=28, color=BLUE, zorder=5)
    b, a = np.polyfit(me.mean_corr, me.learned_c, 1)
    xs = np.linspace(me.mean_corr.min() - 0.02, me.mean_corr.max() + 0.02, 10)
    ax.plot(xs, a + b * xs, "--", color=GRAY, lw=1.0)
    ax.axhline(0.5, color=RED, ls=":", lw=0.9)
    rho = np.corrcoef(me.mean_corr, me.learned_c)[0, 1]
    ax.annotate("mid-cap", (me.mean_corr.min(), 0.5),
                xytext=(me.mean_corr.min() + 0.008, 0.60), fontsize=6.2,
                color=GRAY)
    ax.annotate("large-cap", (me.mean_corr.max(), me.learned_c.max()),
                xytext=(me.mean_corr.max() - 0.115, 1.25), fontsize=6.2,
                color=GRAY)
    ax.set_xlabel("mean pairwise corr.")
    ax.set_title(f"(b) Mechanism ($\\rho={rho:+.2f}$)", loc="left")
    save(fig, "fig_thresholds_combined")


# ─────────── Fig: cumulative wealth, top-100 (net of 10 bp) ───────────
def fig_wealth():
    w = pd.read_csv(RES / "wealth_paths.csv", index_col=0, parse_dates=True)
    show = [("DG-Shrink", BLUE, "-"), ("DG-HRP", GREEN, "-"),
            ("equal_weight", RED, "--"), ("hrp", GRAY, "-"),
            ("ans", ORANGE, "-"), ("gerber_c0.5", "#B8C0CA", ":")]
    LBL = {"equal_weight": "equal weight $1/N$", "hrp": "HRP",
           "ans": "nonlinear shrinkage", "gerber_c0.5": "Gerber $c{=}0.5$"}
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    for name, col, ls in show:
        s_ = w[f"top100|{name}"]
        ax.plot(s_.index, s_.values, ls, color=col, lw=1.3,
                label=f"{LBL.get(name, name)} ({s_.iloc[-1]:.2f})")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("cumulative wealth (net)")
    ax.set_xlabel("")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2, handlelength=1.6)
    save(fig, "fig_wealth")


if __name__ == "__main__":
    fig_pipeline()
    fig_threshold()
    fig_paths()
    fig_simulation()
    fig_objectives()
    fig_mechanism()
    fig_cost()
    fig_thresholds_combined()
    fig_wealth()
    print("figures done.")
