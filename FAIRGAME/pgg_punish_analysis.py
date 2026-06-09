"""Analyze PGG-with-punishment results and compare LLM agents to Herrmann,
Thoeni & Gaechter (2008) humans.

Usage:
    python pgg_punish_analysis.py [path/to/pgg_results.json] [--outdir DIR]

Reads the list of result dicts saved by the Kaggle runner (pgg_results.json),
prints the human-comparison tables (baseline = English, neutral), the language /
personality / society breakdowns, the deviation decomposition of punishment
(Fig 1 / Table 2: prosocial 'punishment of free riding' vs antisocial punishment),
the cross-societal antisocial<->cooperation correlation (Fig 2B), and regenerates
the contribution trajectories (Figs 2A/3), the Fig-1 deviation bars, and the
cross-societal scatter as PNGs.
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src" / "pgg_punish"))
import pgg_results  # noqa: E402

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8 if possible.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def load_results(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline(results):
    return [r for r in results if r["language"] == "en"
            and r["personality_condition"] == "neutral" and r["society"] == "none"]


def print_human_comparison(results):
    base = _baseline(results)
    summary = pgg_results.summarize(base)            # keyed by treatment "N"/"P"
    hum = pgg_results.HUMAN_BENCHMARK
    hum_mean = {
        "N": sum(hum["N_mean_contribution"].values()) / len(hum["N_mean_contribution"]),
        "P": sum(hum["P_mean_contribution"].values()) / len(hum["P_mean_contribution"]),
    }
    print("\n" + "=" * 80)
    print("BASELINE (English, neutral agents)  vs  HUMAN grand-mean (Herrmann et al. 2008)")
    print("=" * 80)
    print(f"{'treat':>5} | {'mean contrib LLM/HUM':>22} | {'antisocial share':>16} | "
          f"{'prosoc/antisoc mean':>20} | parse_fb(c/p)")
    print("-" * 100)
    for tr in ("N", "P"):
        s = summary.get(tr)
        if not s:
            continue
        split = s["antisocial_prosocial_split"]
        fb = s["parse_fallback_rate"]
        print(f"{tr:>5} | {s['mean_contribution']:6.2f} / {hum_mean[tr]:6.2f}            | "
              f"{split['antisocial_share']:14.1%} | "
              f"{split['prosocial_mean']:8.3f} / {split['antisocial_mean']:8.3f}   | "
              f"{fb['contrib']:.1%}/{fb['punish']:.1%}")

    # The headline comparative static: does punishment raise cooperation (P > N)?
    if "N" in summary and "P" in summary:
        dN, dP = summary["N"]["mean_contribution"], summary["P"]["mean_contribution"]
        print("-" * 100)
        print(f"Punishment effect on cooperation: LLM P-N = {dP - dN:+.2f}  "
              f"(HUMAN grand-mean P-N = {hum_mean['P'] - hum_mean['N']:+.2f})")
        # Formal test: are the per-group contributions in P significantly above N?
        n_groups = [pgg_results.mean_contribution(g) for g in base if g["treatment"] == "N"]
        p_groups = [pgg_results.mean_contribution(g) for g in base if g["treatment"] == "P"]
        U, pval = pgg_results.mann_whitney_u(n_groups, p_groups)
        if pval == pval:  # not nan
            print(f"   Mann-Whitney U (P vs N group contributions): U={U:.1f}, p={pval:.4f}  "
                  f"(n_N={len(n_groups)}, n_P={len(p_groups)})")
        print("   Herrmann: punishment raised cooperation in most pools, but the size varied "
              "hugely with antisocial punishment.")
    if any(summary[t]["parse_fallback_rate"]["contrib"] > 0.05 for t in summary):
        print("\n⚠️  contrib parse_fallback_rate > 5% somewhere: low contributions may partly "
              "reflect model non-compliance (full 0..20 range is noisier than a 3-way choice).")


def print_deviation_decomposition(results):
    """Fig 1 / Table 2: mean punishment expenditure by deviation bin (P games)."""
    pgames = [r for r in _baseline(results) if r["treatment"] == "P"]
    if not pgames:
        print("\n(no English-neutral P games — skipping Fig 1 deviation decomposition)")
        return
    bins = pgg_results.deviation_binned_punishment(pgames)
    print("\n" + "=" * 80)
    print("PUNISHMENT BY DEVIATION  (Fig 1 / Table 2) — English, neutral, treatment P")
    print("deviation = target's contribution - punisher's contribution")
    print("=" * 80)
    boston = pgg_results.HUMAN_FIG1_BOSTON
    print(f"{'bin':>10} | {'type':>22} | {'LLM mean exp':>13} | {'HUM (Boston)':>12} | {'n pairs':>8}")
    print("-" * 76)
    for label, _ in pgg_results.DEVIATION_BINS:
        b = bins[label]
        kind = "ANTISOCIAL" if b["is_antisocial"] else "punishment of free riding"
        hum = boston.get(label)
        hum_str = f"{hum:12.2f}" if hum is not None else f"{'~0 / n.a.':>12}"
        print(f"{label:>10} | {kind:>22} | {b['mean_expenditure']:13.3f} | {hum_str} | "
              f"{b['n_pairs']:>8}")
    split = pgg_results.antisocial_prosocial_split(pgames)
    print("-" * 76)
    print(f"Antisocial share of all punishment: {split['antisocial_share']:.1%}  "
          f"(prosocial total {split['prosocial_total']:.0f} vs antisocial total "
          f"{split['antisocial_total']:.0f})")
    # Monotonicity check: humans punish free riding HARDER the deeper it is
    # (Boston: [-20,-11]=2.74 > [-10,-1]=0.96). Does the LLM reproduce that slope?
    deep, shallow = bins["[-20,-11]"]["mean_expenditure"], bins["[-10,-1]"]["mean_expenditure"]
    verdict = "yes" if deep > shallow else "NO"
    print(f"Free-riding punishment rises with deviation depth?  LLM {deep:.2f} vs {shallow:.2f} "
          f"-> {verdict}   (human Boston: 2.74 > 0.96 = yes)")
    print("Note: full per-pool Fig-1 bars are in Herrmann's SOM tables S3-S4; only the Boston "
          "anchor is in the main text, so HUM column is Boston-only (a low-antisocial pool).")


def print_breakdown(results, title, key):
    summary = pgg_results.summarize(results, key=key)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'group':>26} | {'n':>3} | {'mean contrib':>12} | {'antisoc share':>13} | parse_fb(c/p)")
    print("-" * 80)
    for gkey in sorted(summary, key=str):
        s = summary[gkey]
        anti = s["antisocial_prosocial_split"]["antisocial_share"]
        fb = s["parse_fallback_rate"]
        print(f"{str(gkey):>26} | {s['n_games']:>3} | "
              f"{s['mean_contribution']:7.2f}±{s['mean_contribution_sem']:.2f} | "
              f"{anti:12.1%} | {fb['contrib']:.1%}/{fb['punish']:.1%}")


def print_cross_societal(results):
    """Fig 2B: per-society antisocial punishment vs mean contribution + Spearman."""
    soc_games = [r for r in results if r["society"] != "none" and r["treatment"] == "P"]
    if not soc_games:
        print("\n(no society-persona P games — skipping cross-societal analysis)")
        return
    summary = pgg_results.summarize(soc_games, key=lambda r: r["society"])
    hum = pgg_results.HUMAN_BENCHMARK["P_mean_contribution"]
    print("\n" + "=" * 80)
    print("CROSS-SOCIETAL (treatment P) — antisocial punishment vs cooperation (Fig 2B)")
    print("=" * 80)
    print(f"{'society':>16} | {'LLM contrib':>11} | {'HUM contrib':>11} | {'antisocial pts':>14}")
    print("-" * 64)
    anti_vals, coop_vals = [], []
    for soc in sorted(summary, key=lambda s: -summary[s]["mean_contribution"]):
        s = summary[soc]
        anti = s["antisocial_prosocial_split"]["antisocial_total"]
        anti_vals.append(anti)
        coop_vals.append(s["mean_contribution"])
        print(f"{soc:>16} | {s['mean_contribution']:11.2f} | "
              f"{hum.get(soc, float('nan')):11.1f} | {anti:14.0f}")
    rho_llm = pgg_results.spearman(anti_vals, coop_vals)
    # rank correlation of LLM contribution vs human contribution (does ordering match?)
    socs = list(summary.keys())
    llm_c = [summary[s]["mean_contribution"] for s in socs]
    hum_c = [hum.get(s, float("nan")) for s in socs]
    valid = [(a, b) for a, b in zip(llm_c, hum_c) if b == b]
    rho_vs_human = (pgg_results.spearman([a for a, _ in valid], [b for _, b in valid])
                    if len(valid) >= 2 else float("nan"))
    # Paired level test: do LLM and human contributions differ pool-by-pool?
    W, pval = pgg_results.wilcoxon_signed_rank([a for a, _ in valid], [b for _, b in valid])
    mean_gap = (sum(a - b for a, b in valid) / len(valid)) if valid else float("nan")
    print("-" * 64)
    print(f"Spearman(antisocial, contribution)  LLM = {rho_llm:+.2f}   (HUMAN ~ -0.90, Table 1 "
          f"antisocial coef = {pgg_results.HUMAN_TABLE1_OLS['antisocial_punishment']})")
    print(f"Spearman(LLM contribution, HUMAN contribution) across societies = {rho_vs_human:+.2f}")
    print("   (positive = the model reproduces Herrmann's cross-societal RANKING of cooperation.)")
    if pval == pval:  # not nan
        print(f"Wilcoxon signed-rank (LLM vs HUMAN contribution, n={len(valid)} pools): "
              f"W={W:.1f}, p={pval:.4f}; mean LLM-HUM gap = {mean_gap:+.2f}")
        print("   (significant + nonzero gap = right ranking but wrong LEVEL — the model is "
              "systematically more/less cooperative than humans across the board.)")


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def plot_contribution_trajectories(results, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = _baseline(results)
    summary = pgg_results.summarize(base)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"N": "tab:red", "P": "tab:blue"}
    markers = {"N": "s", "P": "o"}
    for tr in ("N", "P"):
        s = summary.get(tr)
        if not s:
            continue
        traj = s["mean_contribution_by_period"]
        ax.plot(range(1, len(traj) + 1), traj, marker=markers[tr], color=colors[tr],
                label=f"treatment {tr}")
    ax.axhline(20, color="grey", lw=0.8, ls="--", label="full endowment (20)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Mean contribution (out of 20)")
    ax.set_ylim(0, 20)
    ax.set_title("Figs 2A/3 (replication): mean contribution per period\nLLM, English, neutral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pgg_fig_contribution_by_period.png", dpi=150)
    plt.close(fig)


def plot_deviation_bins(results, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pgames = [r for r in _baseline(results) if r["treatment"] == "P"]
    if not pgames:
        return
    bins = pgg_results.deviation_binned_punishment(pgames)
    labels = [lab for lab, _ in pgg_results.DEVIATION_BINS]
    means = [bins[lab]["mean_expenditure"] for lab in labels]
    colors = ["tab:green" if not bins[lab]["is_antisocial"] else "tab:red" for lab in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, means, color=colors, edgecolor="black", label="LLM")
    # Overlay the human Boston anchor (the only per-bin Fig-1 values in the text).
    boston = pgg_results.HUMAN_FIG1_BOSTON
    hx = [i for i, lab in enumerate(labels) if lab in boston]
    hy = [boston[labels[i]] for i in hx]
    if hx:
        ax.scatter(hx, hy, color="black", marker="D", zorder=5,
                   label="human (Boston, text)")
    ax.set_xlabel("Deviation (target contribution - punisher contribution)")
    ax.set_ylabel("Mean punishment expenditure")
    ax.set_title("Fig 1 (replication): punishment of free riding (green) vs\n"
                 "antisocial punishment (red) — LLM, English, neutral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pgg_fig1_deviation.png", dpi=150)
    plt.close(fig)


def plot_cross_societal(results, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    soc_games = [r for r in results if r["society"] != "none" and r["treatment"] == "P"]
    if not soc_games:
        return
    summary = pgg_results.summarize(soc_games, key=lambda r: r["society"])
    socs = list(summary.keys())
    anti = [summary[s]["antisocial_prosocial_split"]["antisocial_total"] for s in socs]
    coop = [summary[s]["mean_contribution"] for s in socs]
    rho = pgg_results.spearman(anti, coop)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(anti, coop, color="tab:purple")
    for x, y, s in zip(anti, coop, socs):
        ax.annotate(s, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Mean antisocial punishment (total points per society)")
    ax.set_ylabel("Mean contribution (out of 20)")
    ax.set_title(f"Fig 2B (replication): antisocial punishment vs cooperation\n"
                 f"LLM societies, treatment P — Spearman rho = {rho:+.2f} (human ~ -0.90)")
    fig.tight_layout()
    fig.savefig(outdir / "pgg_fig2b_cross_societal.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="pgg_punish_results/pgg_results.json")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        sys.exit(f"Results file not found: {path}")
    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    results = load_results(path)
    print(f"Loaded {len(results)} games from {path}")

    print_human_comparison(results)
    print_deviation_decomposition(results)
    print_breakdown(
        [r for r in results if r["personality_condition"] == "neutral" and r["society"] == "none"],
        "LANGUAGE BIAS (neutral agents) — group = <treatment>_<lang>",
        key=lambda r: f"{r['treatment']}_{r['language']}")
    print_breakdown(
        [r for r in results if r["language"] == "en" and r["society"] == "none"],
        "PERSONALITY BIAS (English) — group = <treatment>_<personality>",
        key=lambda r: f"{r['treatment']}_{r['personality_condition']}")
    print_cross_societal(results)

    plot_contribution_trajectories(results, outdir)
    plot_deviation_bins(results, outdir)
    plot_cross_societal(results, outdir)
    print(f"\n📊 Figures saved to {outdir}/pgg_fig_contribution_by_period.png, "
          "pgg_fig1_deviation.png, pgg_fig2b_cross_societal.png")


if __name__ == "__main__":
    main()
