"""Reasoning-trace analysis for the CRSD paper (manipulation check).

Question: do the agents REPRESENT the catastrophe risk and the target
arithmetic in their free-form reasoning, even though their contributions do
not RESPOND to the risk manipulation? If a model verbalises "90%"/"10%" yet
contributes identically across treatments, the risk insensitivity cannot be
dismissed as a failure to read the brief — the information is represented but
not acted on. Likewise, if the model states the correct running total, the
self-summation confound (faithful no-running-total design) loses force.

Input:  FAIRGAME/results/crsd_results/<model>/crsd_results.json
        (the per-game JSON saved by the Kaggle notebook; players[i].reasonings
        holds the raw reply for every round). English / neutral / climate
        baseline only — the regexes are English.
Output: analysis/trace_stats.json + analysis/trace_tables.md

Run:    python analysis/trace_analysis.py [--root <dir with <model>/crsd_results.json>]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1] / "FAIRGAME" / "results" / "crsd_results"

TREATMENTS = (90, 50, 10)

# "90%", "90 %", "90 percent", "90-percent"
def prob_regex(p):
    return re.compile(rf"\b{p}\s*(?:%|‐?percent|\s*per\s*cent)", re.IGNORECASE)


RISK_WORDS = re.compile(
    r"\b(risk\w*|probabilit\w*|chance\w*|odds|gambl\w*|lotter\w*|disaster\w*|catastroph\w*)\b",
    re.IGNORECASE)


def number_mention(text, value):
    """True if the integer `value` appears as a standalone number (or €value)."""
    return re.search(rf"(?<![\d.]){value}(?![\d.])", text) is not None


def analyse_model(results):
    """Per-decision trace metrics for one model's EN/neutral/climate baseline."""
    out = {}
    for p in TREATMENTS:
        games = [r for r in results
                 if r["language"] == "en" and r["personality_condition"] == "neutral"
                 and r.get("framing", "neutral") == "climate"
                 and r["treatment_loss_prob"] == p]
        prob_re = prob_regex(p)
        rows = []   # one per decision: dict(round, contrib, prob, riskword, cum_ok, remaining_ok)
        for g in games:
            target = g["target"]
            round_totals = g["round_totals"]
            for pl in g["players"]:
                for r_idx, (contrib, reason) in enumerate(
                        zip(pl["contributions"], pl["reasonings"])):
                    text = reason or ""
                    cum = sum(round_totals[:r_idx])          # true total BEFORE this round
                    remaining = target - cum
                    rows.append({
                        "round": r_idx + 1,
                        "contrib": int(contrib),
                        "prob_mention": bool(prob_re.search(text)),
                        "riskword": bool(RISK_WORDS.search(text)),
                        # arithmetic metrics only meaningful from round 2 with nonzero totals
                        "cum_ok": bool(r_idx >= 1 and cum > 0 and number_mention(text, cum)),
                        "remaining_ok": bool(r_idx >= 1 and 0 < remaining
                                             and number_mention(text, remaining)),
                        "arith_eligible": bool(r_idx >= 1 and cum > 0),
                    })
        n = len(rows)
        eligible = [r for r in rows if r["arith_eligible"]]
        with_prob = [r for r in rows if r["prob_mention"]]
        without_prob = [r for r in rows if not r["prob_mention"]]
        out[p] = {
            "n_games": len(games),
            "n_decisions": n,
            "pct_prob_mention": (sum(r["prob_mention"] for r in rows) / n) if n else float("nan"),
            "pct_riskword": (sum(r["riskword"] for r in rows) / n) if n else float("nan"),
            "pct_correct_cum": (sum(r["cum_ok"] for r in eligible) / len(eligible))
                               if eligible else float("nan"),
            "pct_correct_remaining": (sum(r["remaining_ok"] for r in eligible) / len(eligible))
                                     if eligible else float("nan"),
            "mean_contrib_prob_mentioned": (np.mean([r["contrib"] for r in with_prob])
                                            if with_prob else float("nan")),
            "mean_contrib_prob_not_mentioned": (np.mean([r["contrib"] for r in without_prob])
                                                if without_prob else float("nan")),
            "_contribs_prob_mentioned": [r["contrib"] for r in with_prob],
        }
    # The decisive cross-treatment test: among decisions that explicitly verbalise
    # their own treatment's loss probability, do contributions differ across 90/50/10?
    samples = [out[p]["_contribs_prob_mentioned"] for p in TREATMENTS]
    if all(len(s) >= 5 for s in samples):
        H, pv = sps.kruskal(*samples)
        kw = {"H": float(H), "p": float(pv),
              "n": [len(s) for s in samples],
              "means": [float(np.mean(s)) for s in samples]}
    else:
        kw = {"H": float("nan"), "p": float("nan"),
              "n": [len(s) for s in samples],
              "means": [float(np.mean(s)) if s else float("nan") for s in samples]}
    for p in TREATMENTS:
        del out[p]["_contribs_prob_mentioned"]
    return {"by_treatment": {str(p): out[p] for p in TREATMENTS},
            "kruskal_prob_mentioned_only": kw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="dir holding <model>/crsd_results.json")
    args = ap.parse_args()

    model_dirs = sorted(d for d in args.root.iterdir()
                        if (d / "crsd_results.json").is_file())
    if not model_dirs:
        raise SystemExit(f"No <model>/crsd_results.json under {args.root}")

    stats, lines = {}, [
        "# CRSD reasoning-trace analysis (EN / neutral / climate baseline)\n",
        "Manipulation check: is the loss probability REPRESENTED in the agents'",
        "reasoning even though contributions do not respond to it? And does the",
        "agent track the running total / remaining distance to the EUR120 target",
        "(self-summation check)? Arithmetic metrics are over rounds >= 2.\n",
        "| model | p | decisions | mentions own p | risk words | correct cum total "
        "| correct remaining | contrib (p mentioned) | contrib (not) |",
        "|---|---|---|---|---|---|---|---|---|"]
    for d in model_dirs:
        model = d.name
        results = json.loads((d / "crsd_results.json").read_text(encoding="utf-8"))
        m = analyse_model(results)
        stats[model] = m
        for p in TREATMENTS:
            s = m["by_treatment"][str(p)]
            lines.append(
                f"| {model} | {p}% | {s['n_decisions']} "
                f"| {s['pct_prob_mention']:.1%} | {s['pct_riskword']:.1%} "
                f"| {s['pct_correct_cum']:.1%} | {s['pct_correct_remaining']:.1%} "
                f"| {s['mean_contrib_prob_mentioned']:.2f} "
                f"| {s['mean_contrib_prob_not_mentioned']:.2f} |")
        kw = m["kruskal_prob_mentioned_only"]
        lines.append(
            f"\n**{model}** — Kruskal–Wallis across treatments, restricted to decisions "
            f"that explicitly mention their own loss probability: H = {kw['H']:.2f}, "
            f"p = {kw['p']:.3g} (n = {kw['n']}, means = "
            f"{[round(x, 2) for x in kw['means']]}). A non-significant H here means the "
            f"risk is verbalised but does not move the contribution.\n")

    (HERE / "trace_stats.json").write_text(
        json.dumps(stats, indent=1, default=float), encoding="utf-8")
    (HERE / "trace_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("->", HERE / "trace_stats.json")
    print("->", HERE / "trace_tables.md")


if __name__ == "__main__":
    main()
