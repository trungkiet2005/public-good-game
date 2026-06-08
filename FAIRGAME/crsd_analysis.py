"""Analyze CRSD results and compare LLM agents to Milinski et al. (2008) humans.

Usage:
    python crsd_analysis.py [path/to/crsd_results.json] [--outdir DIR]

Reads the list of result dicts saved by the Kaggle runner (crsd_results.json),
prints the human-comparison tables (baseline = English, neutral) plus the
language and personality breakdowns, runs light statistical tests (scipy if
available), and regenerates Milinski's Fig 2 (cumulative trajectory) and Fig 3
(acts by game half) as PNGs.
"""

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src" / "crsd"))
import crsd_results  # noqa: E402

# Windows consoles default to cp1252 and choke on €/±/emoji; force UTF-8 if possible.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# Milinski 2008 human results with standard errors (n=10 groups per treatment).
HUMAN_FULL = {
    90: {"success": 5, "n_groups": 10, "mean": 118.2, "se": 1.9, "fair": 3.3},
    50: {"success": 1, "n_groups": 10, "mean": 92.2,  "se": 9.0, "fair": 2.1},
    10: {"success": 0, "n_groups": 10, "mean": 73.0,  "se": 4.4, "fair": 1.1},
}

try:
    from scipy import stats as _scipy_stats
except Exception:  # noqa: BLE001
    _scipy_stats = None


def _welch(mean1, sd1, n1, mean2, sd2, n2):
    """Welch t and (if scipy) two-sided p-value."""
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    se = math.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    if se == 0:
        return float("inf"), 0.0
    t = (mean1 - mean2) / se
    num = (sd1 ** 2 / n1 + sd2 ** 2 / n2) ** 2
    den = (sd1 ** 2 / n1) ** 2 / (n1 - 1) + (sd2 ** 2 / n2) ** 2 / (n2 - 1)
    df = num / den if den else float("nan")
    p = (2 * (1 - _scipy_stats.t.cdf(abs(t), df))) if _scipy_stats else float("nan")
    return t, p


def load_results(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def print_human_comparison(results):
    baseline = [r for r in results
                if r["language"] == "en" and r["personality_condition"] == "neutral"]
    summary = crsd_results.summarize(baseline)
    print("\n" + "=" * 78)
    print("BASELINE (English, neutral agents)  vs  HUMAN (Milinski et al. 2008)")
    print("=" * 78)
    print(f"{'p%':>4} | {'success LLM / HUM':>20} | {'mean total LLM / HUM':>24} | "
          f"{'fair-sharers LLM / HUM':>24} | {'parse_fb':>8}")
    print("-" * 100)
    for p in (90, 50, 10):
        s = summary.get(p)
        if not s:
            continue
        h = HUMAN_FULL[p]
        ft = s["final_total"]
        sd_h = h["se"] * math.sqrt(h["n_groups"])
        t, pval = _welch(ft["mean"], ft["std"], ft["n"], h["mean"], sd_h, h["n_groups"])
        pstr = f"(Welch t={t:.2f}, p={pval:.3f})" if not math.isnan(pval) else f"(Welch t={t:.2f})"
        print(f"{p:>4} | {s['success_rate']*100:5.0f}% / {h['success']*10:5.0f}%        | "
              f"{ft['mean']:6.1f}±{ft['sem']:4.1f} / {h['mean']:5.1f}±{h['se']:.1f}   | "
              f"{s['fair_sharers_per_group']:5.2f} / {h['fair']:.2f}              | "
              f"{s['parse_fallback_rate']:6.1%}")
        print(f"       round1 dist (0/2/4): {s['round1_distribution']}   "
              f"acts 1st half {s['acts_by_half']['first']}  2nd half {s['acts_by_half']['second']}   {pstr}")
    if any(crsd_results.summarize(baseline)[p]["parse_fallback_rate"] > 0.05
           for p in summary):
        print("\n⚠️  parse_fallback_rate > 5% in some cell: low contributions may partly reflect")
        print("    model non-compliance/miscounting (faithful visibility on a 7B model).")


def print_breakdown(results, title, key):
    summary = crsd_results.summarize(results, key=key)
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'group':>22} | {'n':>3} | {'success':>8} | {'mean total':>12} | {'fair-sh':>7} | parse_fb")
    print("-" * 78)
    for gkey in sorted(summary, key=str):
        s = summary[gkey]
        print(f"{str(gkey):>22} | {s['n_games']:>3} | {s['success_rate']*100:6.0f}% | "
              f"{s['final_total']['mean']:8.1f}±{s['final_total']['sem']:.1f} | "
              f"{s['fair_sharers_per_group']:6.2f} | {s['parse_fallback_rate']:.1%}")


def plot_fig2_cumulative(results, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = [r for r in results
                if r["language"] == "en" and r["personality_condition"] == "neutral"]
    summary = crsd_results.summarize(baseline)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {90: "tab:blue", 50: "tab:green", 10: "tab:red"}
    for p in (90, 50, 10):
        if p not in summary:
            continue
        traj = summary[p]["cumulative_trajectory"]
        rounds = list(range(1, len(traj) + 1))
        ax.plot(rounds, traj, marker="o", color=colors[p], label=f"{p}% loss risk")
    n_rounds = len(summary[next(iter(summary))]["cumulative_trajectory"])
    ax.axhline(120, color="black", lw=1.2, ls="--", label="target €120")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative group contribution (€)")
    ax.set_title("Fig 2 (replication): cumulative contribution per round\nLLM, English, neutral")
    ax.set_xticks(range(1, n_rounds + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "crsd_fig2_cumulative.png", dpi=150)
    plt.close(fig)


def plot_fig3_acts(results, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = [r for r in results
                if r["language"] == "en" and r["personality_condition"] == "neutral"]
    summary = crsd_results.summarize(baseline)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, p in zip(axes, (90, 50, 10)):
        if p not in summary:
            continue
        acts = summary[p]["acts_by_half"]
        labels = ["€0", "€2", "€4"]
        first = [acts["first"][k] for k in (0, 2, 4)]
        second = [acts["second"][k] for k in (0, 2, 4)]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], first, width=0.4, label="rounds 1-5")
        ax.bar([i + 0.2 for i in x], second, width=0.4, label="rounds 6-10")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title(f"{p}% loss risk")
        ax.set_xlabel("contribution")
    axes[0].set_ylabel("number of acts")
    axes[0].legend()
    fig.suptitle("Fig 3 (replication): selfish/fair/altruist acts by game half — LLM, English, neutral")
    fig.tight_layout()
    fig.savefig(outdir / "crsd_fig3_acts.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="crsd_results/crsd_results.json")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        sys.exit(f"Results file not found: {path}")
    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    results = load_results(path)
    print(f"Loaded {len(results)} games from {path}")
    if _scipy_stats is None:
        print("(scipy not available — Welch p-values omitted; install scipy for them.)")

    print_human_comparison(results)
    print_breakdown([r for r in results if r["personality_condition"] == "neutral"],
                    "LANGUAGE BIAS (neutral agents) — group = p<loss>_<lang>",
                    key=lambda r: f"p{r['treatment_loss_prob']}_{r['language']}")
    print_breakdown([r for r in results if r["language"] == "en"],
                    "PERSONALITY BIAS (English) — group = p<loss>_<personality>",
                    key=lambda r: f"p{r['treatment_loss_prob']}_{r['personality_condition']}")

    plot_fig2_cumulative(results, outdir)
    plot_fig3_acts(results, outdir)
    print(f"\n📊 Figures saved to {outdir}/crsd_fig2_cumulative.png and crsd_fig3_acts.png")


if __name__ == "__main__":
    main()
