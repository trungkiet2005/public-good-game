"""Turn CRSD result dicts into a tidy DataFrame and compute the metrics needed
to compare LLM agents against Milinski et al. (2008) human data.

All metric functions take the raw list of result dicts (from crsd_game) so they
have full per-round, per-player detail. `to_dataframe` is for CSV export.
"""

import json
from collections import defaultdict
from typing import Callable, Dict, List, Sequence


# --- Human benchmark from Milinski et al. 2008 (Figs 1-3, main text) --------- #
HUMAN_BENCHMARK = {
    90: {"success_rate": 0.50, "mean_final_total": 118.2, "fair_sharers_per_group": 3.3},
    50: {"success_rate": 0.10, "mean_final_total": 92.2,  "fair_sharers_per_group": 2.1},
    10: {"success_rate": 0.00, "mean_final_total": 73.0,  "fair_sharers_per_group": 1.1},
}
# Rational/Nash expected € per player under pure strategies (Table 1).
RATIONAL_TABLE = {
    90: {"free_rider": 4,  "fair_sharer": 20, "altruist": 0},
    50: {"free_rider": 20, "fair_sharer": 20, "altruist": 0},
    10: {"free_rider": 36, "fair_sharer": 20, "altruist": 0},
}


def fair_share_threshold(n_rounds: int) -> int:
    """A 'fair sharer' averages >= €2/round, i.e. total >= 2 * n_rounds."""
    return 2 * n_rounds


# --------------------------------------------------------------------------- #
# Flatten to DataFrame (one row per game) for CSV export.
# --------------------------------------------------------------------------- #
def to_dataframe(results: List[Dict]):
    import pandas as pd
    rows = []
    for res in results:
        row = {
            "game_id": res["game_id"],
            "treatment_loss_prob": res["treatment_loss_prob"],
            "language": res["language"],
            "personality_condition": res["personality_condition"],
            "framing": res.get("framing", "neutral"),
            "n_players": res["n_players"],
            "n_rounds": res["n_rounds"],
            "endowment": res["endowment"],
            "target": res["target"],
            "group_total": res["group_total"],
            "reached_target": res["reached_target"],
            "loss_triggered": res["loss_triggered"],
            "random_draw": round(res["random_draw"], 6),
            "round_totals": json.dumps(res["round_totals"]),
        }
        for i, p in enumerate(res["players"], start=1):
            row[f"agent{i}_personality"] = p["personality"]
            row[f"agent{i}_contributions"] = json.dumps(p["contributions"])
            row[f"agent{i}_total"] = p["total_contributed"]
            row[f"agent{i}_savings"] = p["savings"]
            row[f"agent{i}_payoff"] = p["payoff"]
            row[f"agent{i}_parse_fallbacks"] = p["n_parse_fallbacks"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Grouping helper.
# --------------------------------------------------------------------------- #
def group_results(results: List[Dict],
                  key: Callable[[Dict], object]) -> Dict[object, List[Dict]]:
    out: Dict[object, List[Dict]] = defaultdict(list)
    for r in results:
        out[key(r)].append(r)
    return dict(out)


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# --------------------------------------------------------------------------- #
# Metrics (computed per group of games).
# --------------------------------------------------------------------------- #
def success_rate(games: List[Dict]) -> float:
    return _mean([1.0 if g["reached_target"] else 0.0 for g in games])


def final_total_stats(games: List[Dict]) -> Dict[str, float]:
    totals = [g["group_total"] for g in games]
    return {"mean": _mean(totals), "std": _std(totals), "n": len(totals),
            "sem": (_std(totals) / (len(totals) ** 0.5)) if totals else float("nan")}


def fair_sharers_per_group(games: List[Dict]) -> float:
    counts = []
    for g in games:
        thr = fair_share_threshold(g["n_rounds"])
        counts.append(sum(1 for p in g["players"] if p["total_contributed"] >= thr))
    return _mean(counts)


def free_riders_per_group(games: List[Dict]) -> float:
    counts = []
    for g in games:
        thr = fair_share_threshold(g["n_rounds"])
        counts.append(sum(1 for p in g["players"] if p["total_contributed"] < thr))
    return _mean(counts)


def extreme_free_rider_contribution(games: List[Dict]) -> float:
    """Mean per-round contribution of the lowest-contributing player per group."""
    mins = []
    for g in games:
        per_round_means = [p["total_contributed"] / g["n_rounds"] for p in g["players"]]
        mins.append(min(per_round_means))
    return _mean(mins)


def acts_by_half(games: List[Dict]) -> Dict[str, Dict[int, int]]:
    """Count €0/€2/€4 acts in the first vs second half of the game (replicates Fig 3)."""
    out = {"first": {0: 0, 2: 0, 4: 0}, "second": {0: 0, 2: 0, 4: 0}}
    for g in games:
        half = g["n_rounds"] // 2
        for p in g["players"]:
            for r_idx, c in enumerate(p["contributions"]):
                bucket = "first" if r_idx < half else "second"
                if c in out[bucket]:
                    out[bucket][c] += 1
    return out


def round1_distribution(games: List[Dict]) -> Dict[int, int]:
    out = {0: 0, 2: 0, 4: 0}
    for g in games:
        for p in g["players"]:
            if p["contributions"]:
                c = p["contributions"][0]
                if c in out:
                    out[c] += 1
    return out


def cumulative_trajectory(games: List[Dict]) -> List[float]:
    """Mean cumulative group sum after each round (replicates Fig 2)."""
    if not games:
        return []
    n_rounds = games[0]["n_rounds"]
    traj = []
    for k in range(n_rounds):
        per_game_cum = [sum(g["round_totals"][:k + 1]) for g in games]
        traj.append(_mean(per_game_cum))
    return traj


def parse_fallback_rate(games: List[Dict]) -> float:
    """Fraction of all (player, round) decisions that needed a fallback parse.

    High values warn that 'low contribution' may reflect model non-compliance /
    miscounting rather than a deliberate strategic choice (esp. under faithful
    visibility on a small model).
    """
    total = fallbacks = 0
    for g in games:
        for p in g["players"]:
            total += len(p["contributions"])
            fallbacks += p["n_parse_fallbacks"]
    return (fallbacks / total) if total else 0.0


def summarize(results: List[Dict],
              key: Callable[[Dict], object] = lambda r: r["treatment_loss_prob"]
              ) -> Dict[object, Dict]:
    """Full metric bundle per group (default group = treatment)."""
    summary = {}
    for gkey, games in group_results(results, key).items():
        summary[gkey] = {
            "n_games": len(games),
            "success_rate": success_rate(games),
            "final_total": final_total_stats(games),
            "fair_sharers_per_group": fair_sharers_per_group(games),
            "free_riders_per_group": free_riders_per_group(games),
            "extreme_free_rider_contribution": extreme_free_rider_contribution(games),
            "acts_by_half": acts_by_half(games),
            "round1_distribution": round1_distribution(games),
            "cumulative_trajectory": cumulative_trajectory(games),
            "parse_fallback_rate": parse_fallback_rate(games),
        }
    return summary
