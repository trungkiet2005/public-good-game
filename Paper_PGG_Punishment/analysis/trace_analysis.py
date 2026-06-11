"""Reasoning-trace analysis for the PGG-punishment paper (motive check).

Question: WHAT DO THE AGENTS SAY they are punishing for, and does the stated
motive match the actual target? Humans punish free-riders and say so; our
headline result is that 54-59% of LLM deduction points hit equal-or-higher
contributors (antisocial). This script classifies the free-text reasoning of
every punishment decision into motive categories (free-riding enforcement,
revenge, fairness/equality talk, deterrence) and cross-tabulates the stated
motive against where the points actually landed. A decision that CLAIMS
free-riding enforcement while spending most of its points on equal-or-higher
contributors is mis-targeted rhetoric — direct mechanistic evidence that the
conditional-cooperation logic is absent.

Input:  FAIRGAME/results/pgg_punish_results/<model>/pgg_results.json
        (players[i].punish_reasonings = one raw reply per period;
         punishment_matrix_by_period[t][i][j] = points i -> j, true indices).
        English / neutral / no-society baseline, treatment P only.
Output: analysis/trace_stats.json + analysis/trace_tables.md

Run:    python analysis/trace_analysis.py [--root <dir with <model>/pgg_results.json>]
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1] / "FAIRGAME" / "results" / "pgg_punish_results"

MOTIVES = {
    "freeride": re.compile(
        r"(free.?rid\w*"
        r"|contribut\w*\s+(?:less|little|nothing|least|only|zero|0)"
        r"|low(?:est)?\s+contribut\w*"
        r"|didn'?t\s+contribut\w*|did\s+not\s+contribut\w*"
        r"|under.?contribut\w*"
        r"|below\s+(?:the\s+)?(?:average|group)"
        r"|less\s+than\s+(?:me|my|mine|i\s+did|others|everyone))",
        re.IGNORECASE),
    "revenge": re.compile(
        r"(reveng\w*|retaliat\w*|payback|pay\s+back"
        r"|punish\w*\s+me|deduct\w*\s+(?:i|I)\s+received"
        r"|received\s+\d+\s+deduction|points?\s+(?:i|I)\s+received"
        r"|got\s+deducted|someone\s+deducted)",
        re.IGNORECASE),
    "fairness": re.compile(r"\b(fair\w*|unfair\w*|equal\w*|equit\w*|justice)\b",
                           re.IGNORECASE),
    "deterrence": re.compile(
        r"(deter\w*|discourag\w*|encourag\w+\s+(?:them\s+|him\s+|her\s+)?to\s+contribut"
        r"|incentiv\w*|send\s+a\s+(?:message|signal)|signal\w*|enforce\w*|norm\w*)",
        re.IGNORECASE),
}


def analyse_model(results):
    games = [r for r in results
             if r["treatment"] == "P" and r["language"] == "en"
             and r["personality_condition"] == "neutral" and r["society"] == "none"]
    rows = []   # one per (punisher, period) WITH points assigned
    n_zero = 0
    for g in games:
        contribs = g["contributions_by_period"]              # [t][player]
        mats = g["punishment_matrix_by_period"]              # [t][i][j]
        for i, pl in enumerate(g["players"]):
            reasons = pl["punish_reasonings"]
            for t, mat in enumerate(mats):
                pts = mat[i]
                total = sum(pts)
                if total == 0:
                    n_zero += 1
                    continue
                # where did the points actually land?
                anti_pts = sum(pts[j] for j in range(len(pts))
                               if j != i and contribs[t][j] >= contribs[t][i])
                text = reasons[t] if t < len(reasons) else ""
                motives = {k: bool(rx.search(text or "")) for k, rx in MOTIVES.items()}
                rows.append({
                    "anti_share": anti_pts / total,
                    "anti_dominant": anti_pts / total > 0.5,
                    **motives,
                })
    n = len(rows)
    if n == 0:
        return {"n_games": len(games), "n_punish_decisions": 0}

    def pct(key):
        return sum(r[key] for r in rows) / n

    claimed_fr = [r for r in rows if r["freeride"]]
    out = {
        "n_games": len(games),
        "n_punish_decisions": n,
        "n_zero_punish_decisions": n_zero,
        "pct_anti_dominant": pct("anti_dominant"),
        "motive_rates": {k: pct(k) for k in MOTIVES},
        # the rhetoric-vs-target cross-tab
        "pct_freeride_claimed": len(claimed_fr) / n,
        "pct_anti_dominant_given_freeride_claimed":
            (sum(r["anti_dominant"] for r in claimed_fr) / len(claimed_fr))
            if claimed_fr else float("nan"),
        "mean_anti_share_given_freeride_claimed":
            (sum(r["anti_share"] for r in claimed_fr) / len(claimed_fr))
            if claimed_fr else float("nan"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="dir holding <model>/pgg_results.json")
    args = ap.parse_args()

    model_dirs = sorted(d for d in args.root.iterdir()
                        if (d / "pgg_results.json").is_file())
    if not model_dirs:
        raise SystemExit(f"No <model>/pgg_results.json under {args.root}")

    stats, lines = {}, [
        "# PGG punishment reasoning-trace analysis (EN / neutral / treatment P)\n",
        "Motive classification of every nonzero punishment decision, cross-tabulated",
        "against where the deduction points actually landed (antisocial = target",
        "contributed >= punisher). 'mis-targeted rhetoric' = the reply claims",
        "free-riding enforcement while >50% of its points hit equal-or-higher",
        "contributors.\n",
        "| model | punish decisions | anti-dominant | claims free-riding | claims revenge "
        "| claims fairness | claims deterrence | anti-dominant GIVEN free-riding claimed |",
        "|---|---|---|---|---|---|---|---|"]
    for d in model_dirs:
        model = d.name
        results = json.loads((d / "pgg_results.json").read_text(encoding="utf-8"))
        m = analyse_model(results)
        stats[model] = m
        if not m.get("n_punish_decisions"):
            lines.append(f"| {model} | 0 | – | – | – | – | – | – |")
            continue
        mr = m["motive_rates"]
        lines.append(
            f"| {model} | {m['n_punish_decisions']} "
            f"| {m['pct_anti_dominant']:.1%} "
            f"| {mr['freeride']:.1%} | {mr['revenge']:.1%} "
            f"| {mr['fairness']:.1%} | {mr['deterrence']:.1%} "
            f"| {m['pct_anti_dominant_given_freeride_claimed']:.1%} |")
    lines.append(
        "\nReading: if 'anti-dominant GIVEN free-riding claimed' stays high, the model "
        "verbalises the human prosocial motive while aiming at cooperators — the stated "
        "logic and the targeting are decoupled.")

    (HERE / "trace_stats.json").write_text(
        json.dumps(stats, indent=1, default=float), encoding="utf-8")
    (HERE / "trace_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("->", HERE / "trace_stats.json")
    print("->", HERE / "trace_tables.md")


if __name__ == "__main__":
    main()
