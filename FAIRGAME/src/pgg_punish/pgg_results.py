"""Turn PGG-with-punishment result dicts into a tidy DataFrame and compute the
metrics needed to compare LLM agents against Herrmann, Thoeni & Gaechter (2008).

All metric functions take the raw list of result dicts (from pgg_game.finalize)
so they have full per-period, per-player detail. `to_dataframe` is for CSV export.

The analytical heart of the replication is the deviation decomposition (Fig 1 /
Table 2): for every (period, punisher a, target b) pair in a punishment game,
    deviation   = contribution_b - contribution_a   (target minus punisher)
    expenditure = points a assigned to b
Deviation < 0  => the target contributed LESS  => "punishment of free riding".
Deviation >= 0 => the target contributed EQUAL-OR-MORE => "ANTISOCIAL punishment".
"""

import json
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple


# --- Human benchmark from Herrmann et al. 2008 (Fig 2A = P, Fig 3 = N) ------- #
HUMAN_BENCHMARK = {
    "P_mean_contribution": {  # mean contribution (out of 20), P experiment
        "Boston": 18.0, "Copenhagen": 17.7, "St.Gallen": 16.7, "Zurich": 16.2,
        "Nottingham": 15.0, "Seoul": 14.7, "Bonn": 14.5, "Melbourne": 14.1,
        "Chengdu": 13.9, "Minsk": 12.9, "Samara": 11.7, "Dnipropetrovsk": 10.9,
        "Muscat": 9.9, "Istanbul": 7.1, "Riyadh": 6.9, "Athens": 5.7,
    },
    "N_mean_contribution": {  # mean contribution (out of 20), N experiment
        "Copenhagen": 11.5, "Dnipropetrovsk": 10.6, "Minsk": 10.5, "St.Gallen": 10.1,
        "Muscat": 10.0, "Samara": 9.7, "Zurich": 9.3, "Boston": 9.3, "Bonn": 9.2,
        "Chengdu": 8.0, "Seoul": 7.9, "Riyadh": 7.6, "Nottingham": 6.9,
        "Athens": 6.4, "Istanbul": 5.4, "Melbourne": 4.9,
    },
    # Spearman rank correlation between mean antisocial punishment and mean
    # contribution across the 16 pools (Fig 2B).
    "antisocial_vs_contribution_spearman_rho": -0.90,
}

# --- Human Fig 1 anchor (deviation-binned mean punishment expenditure) -------- #
# Herrmann's Fig 1 plots mean punishment expenditure per deviation bin for each
# of the 16 pools; the per-pool bar values live in the figure / SOM tables S3-S4,
# NOT in the main-text PDF. The ONLY explicit Fig-1 numbers in the text are the
# Boston example, so that is the single faithful numeric anchor we encode. Use it
# as a reference point (Boston is a LOW-antisocial pool) and rely on the
# qualitative pattern + Table 1/2 coefficients for the rest.
HUMAN_FIG1_BOSTON = {  # mean money units expended, Boston pool (text, p.1363)
    "[-20,-11]": 2.74,
    "[-10,-1]": 0.96,
    # [0], [1,10], [11,20]: near-zero in Boston (little antisocial punishment).
}

# Table 1 (OLS): group avg contributions periods 2-10 regressed on period-1
# contribution + group avg punishment of free riding + group avg antisocial
# punishment. Model 1 (no pool dummies). These are clean, fully-in-text human
# coefficients for the punishment -> cooperation link.
HUMAN_TABLE1_OLS = {
    "period1_contribution": 0.779,      # ***
    "punishment_of_free_riding": 0.521,  # **  (prosocial punishment RAISES coop)
    "antisocial_punishment": -2.247,     # *** (antisocial punishment LOWERS coop)
    "constant": 5.057,
    "adjusted_r2": 0.60,
    "n_group_obs": 273,
}

# Deviation bins for Fig 1 (label -> inclusive (lo, hi)).
DEVIATION_BINS: List[Tuple[str, Tuple[int, int]]] = [
    ("[-20,-11]", (-20, -11)),
    ("[-10,-1]", (-10, -1)),
    ("[0]", (0, 0)),
    ("[1,10]", (1, 10)),
    ("[11,20]", (11, 20)),
]


def _bin_label(deviation: int) -> str:
    for label, (lo, hi) in DEVIATION_BINS:
        if lo <= deviation <= hi:
            return label
    return "[other]"


def is_antisocial_bin(label: str) -> bool:
    """Deviation >= 0 (target contributed equal-or-more) is antisocial."""
    return label in ("[0]", "[1,10]", "[11,20]")


# --------------------------------------------------------------------------- #
# Small stats helpers (no hard scipy/numpy dependency).
# --------------------------------------------------------------------------- #
def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _rankdata(xs: Sequence[float]) -> List[float]:
    """Average-rank of each element (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rho via scipy if available, else a manual rank-Pearson fallback.

    Returns nan for <2 points or a constant input (correlation is undefined and
    scipy would otherwise emit a ConstantInputWarning)."""
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    try:  # pragma: no cover - optional fast path
        from scipy.stats import spearmanr
        return float(spearmanr(xs, ys).correlation)
    except Exception:
        return _pearson(_rankdata(xs), _rankdata(ys))


def _normal_sf(z: float) -> float:
    """Upper-tail P(Z > z) of the standard normal, via erfc (no scipy needed)."""
    import math
    return 0.5 * math.erfc(z / (2 ** 0.5))


def mann_whitney_u(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U (unpaired). scipy fast-path, else a tie- and
    continuity-corrected normal approximation. Returns (U, p). Use to test whether
    two independent samples (e.g. LLM contributions in N vs P) differ in level."""
    from collections import Counter
    n1, n2 = len(xs), len(ys)
    if n1 < 1 or n2 < 1:
        return float("nan"), float("nan")
    try:  # pragma: no cover - optional exact small-n path
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(list(xs), list(ys), alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        pass
    combined = list(xs) + list(ys)
    ranks = _rankdata(combined)
    R1 = sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U = min(U1, n1 * n2 - U1)
    n = n1 + n2
    mu = n1 * n2 / 2.0
    tie_term = sum(t ** 3 - t for t in Counter(combined).values())
    sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if sigma2 <= 0:
        return U, float("nan")
    z = (abs(U - mu) - 0.5) / sigma2 ** 0.5          # continuity correction
    return U, min(1.0, 2 * _normal_sf(z))


def wilcoxon_signed_rank(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank for PAIRED xs, ys (e.g. LLM vs human mean
    contribution across the SAME 16 societies). scipy fast-path, else a tie- and
    continuity-corrected normal approximation; zero differences are dropped.
    Returns (W, p)."""
    from collections import Counter
    pairs = [(a, b) for a, b in zip(xs, ys) if a == a and b == b]   # drop nan
    nz = [a - b for a, b in pairs if (a - b) != 0]
    if len(nz) < 1:
        return float("nan"), float("nan")
    try:  # pragma: no cover - optional exact small-n path
        from scipy.stats import wilcoxon
        res = wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
        return float(res.statistic), float(res.pvalue)
    except Exception:
        pass
    mags = [abs(d) for d in nz]
    ranks = _rankdata(mags)
    Wpos = sum(r for r, d in zip(ranks, nz) if d > 0)
    W = min(Wpos, sum(ranks) - Wpos)
    n = len(nz)
    mu = n * (n + 1) / 4.0
    tie_term = sum(t ** 3 - t for t in Counter(mags).values())
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if sigma2 <= 0:
        return W, float("nan")
    z = (abs(W - mu) - 0.5) / sigma2 ** 0.5          # continuity correction
    return W, min(1.0, 2 * _normal_sf(z))


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    if vx == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx


# --------------------------------------------------------------------------- #
# Grouping helper.
# --------------------------------------------------------------------------- #
def group_results(results: List[Dict],
                  key: Callable[[Dict], object]) -> Dict[object, List[Dict]]:
    out: Dict[object, List[Dict]] = defaultdict(list)
    for r in results:
        out[key(r)].append(r)
    return dict(out)


# --------------------------------------------------------------------------- #
# Contribution / earnings metrics.
# --------------------------------------------------------------------------- #
def mean_contribution(game: Dict) -> float:
    """Mean per-player per-period contribution for one game."""
    rows = game["contributions_by_period"]
    flat = [c for row in rows for c in row]
    return _mean(flat)


def mean_contribution_by_period(games: List[Dict]) -> List[float]:
    """Fig 2A / Fig 3 trajectory: mean contribution per period across games."""
    if not games:
        return []
    T = games[0]["n_periods"]
    out = []
    for t in range(T):
        vals = [c for g in games for c in g["contributions_by_period"][t]]
        out.append(_mean(vals))
    return out


def mean_earnings_by_period(games: List[Dict]) -> List[float]:
    if not games:
        return []
    T = games[0]["n_periods"]
    return [_mean([e for g in games for e in g["earnings_by_period"][t]]) for t in range(T)]


# --------------------------------------------------------------------------- #
# Punishment metrics (P games only; N games are skipped automatically).
# --------------------------------------------------------------------------- #
def _iter_punishment_pairs(games: List[Dict]):
    """Yield (deviation, expenditure) for every (period, punisher a, target b!=a)."""
    for g in games:
        mats = g.get("punishment_matrix_by_period") or []
        contribs = g["contributions_by_period"]
        n = g["n_players"]
        for t, P in enumerate(mats):
            if not P:
                continue
            c = contribs[t]
            for a in range(n):
                for b in range(n):
                    if a == b:
                        continue
                    yield (c[b] - c[a], P[a][b])


def deviation_binned_punishment(games: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Fig 1 / Table 2: mean punishment expenditure per deviation bin (includes
    non-punishers, i.e. averaged over all punisher-target pairs in the bin)."""
    totals = {label: 0.0 for label, _ in DEVIATION_BINS}
    counts = {label: 0 for label, _ in DEVIATION_BINS}
    for dev, exp in _iter_punishment_pairs(games):
        label = _bin_label(dev)
        if label in totals:
            totals[label] += exp
            counts[label] += 1
    out = {}
    for label, _ in DEVIATION_BINS:
        n = counts[label]
        out[label] = {
            "mean_expenditure": (totals[label] / n) if n else 0.0,
            "total_expenditure": totals[label],
            "n_pairs": n,
            "is_antisocial": is_antisocial_bin(label),
        }
    return out


def antisocial_prosocial_split(games: List[Dict]) -> Dict[str, float]:
    """Total / mean punishment expenditure split into prosocial (dev<0) vs
    antisocial (dev>=0), plus the antisocial share of all punishment."""
    pro_tot = pro_n = anti_tot = anti_n = 0
    for dev, exp in _iter_punishment_pairs(games):
        if dev < 0:
            pro_tot += exp
            pro_n += 1
        else:
            anti_tot += exp
            anti_n += 1
    total = pro_tot + anti_tot
    return {
        "prosocial_total": pro_tot,
        "prosocial_mean": (pro_tot / pro_n) if pro_n else 0.0,
        "antisocial_total": anti_tot,
        "antisocial_mean": (anti_tot / anti_n) if anti_n else 0.0,
        "antisocial_share": (anti_tot / total) if total else 0.0,
    }


def antisocial_punishment_per_group(game: Dict) -> float:
    """Total antisocial (dev>=0) deduction points assigned in one game."""
    total = 0
    for dev, exp in _iter_punishment_pairs([game]):
        if dev >= 0:
            total += exp
    return float(total)


def antisocial_vs_cooperation_corr(games: List[Dict]) -> float:
    """Spearman rho between per-group antisocial punishment and per-group mean
    contribution. Herrmann's human value is ~ -0.90."""
    pgames = [g for g in games if g["treatment"] == "P"]
    if len(pgames) < 2:
        return float("nan")
    anti = [antisocial_punishment_per_group(g) for g in pgames]
    coop = [mean_contribution(g) for g in pgames]
    return spearman(anti, coop)


def vengeance_table(games: List[Dict]) -> Dict[str, object]:
    """Punishment ASSIGNED in period t+1 as a function of punishment RECEIVED in
    period t — the revenge channel. Returns mean assigned-next per received bin
    plus an OLS slope of assigned_{t+1} on received_t over all (player, t) pairs.
    """
    received_bins = [("0", (0, 0)), ("1-3", (1, 3)), ("4-6", (4, 6)), ("7+", (7, 10_000))]
    bin_tot = {lab: 0.0 for lab, _ in received_bins}
    bin_n = {lab: 0 for lab, _ in received_bins}
    xs, ys = [], []
    for g in games:
        if g["treatment"] != "P":
            continue
        for p in g["players"]:
            recv = p["punishment_received"]
            assigned = [sum(row) for row in p["punishment_assigned"]]
            for t in range(len(recv) - 1):
                r = recv[t]
                a_next = assigned[t + 1] if t + 1 < len(assigned) else 0
                xs.append(r)
                ys.append(a_next)
                for lab, (lo, hi) in received_bins:
                    if lo <= r <= hi:
                        bin_tot[lab] += a_next
                        bin_n[lab] += 1
                        break
    mean_assigned_next = {lab: (bin_tot[lab] / bin_n[lab]) if bin_n[lab] else 0.0
                          for lab, _ in received_bins}
    return {"mean_assigned_next_by_received_bin": mean_assigned_next,
            "slope_assigned_on_received": _ols_slope(xs, ys),
            "n_pairs": len(xs)}


def parse_fallback_rate(games: List[Dict]) -> Dict[str, float]:
    """Fraction of decisions that needed a fallback parse, for each stage. High
    values warn that low contribution / odd punishment may reflect model
    non-compliance rather than strategy."""
    c_tot = c_fb = p_tot = p_fb = 0
    for g in games:
        for p in g["players"]:
            c_tot += len(p["contributions"])
            c_fb += p["n_contrib_fallbacks"]
            p_tot += len(p["punishment_assigned"])
            p_fb += p["n_punish_fallbacks"]
    return {"contrib": (c_fb / c_tot) if c_tot else 0.0,
            "punish": (p_fb / p_tot) if p_tot else 0.0}


# --------------------------------------------------------------------------- #
# Flatten to DataFrame (one row per game) for CSV export.
# --------------------------------------------------------------------------- #
def to_dataframe(results: List[Dict]):
    import pandas as pd
    rows = []
    for res in results:
        anti = antisocial_punishment_per_group(res) if res["treatment"] == "P" else 0.0
        row = {
            "game_id": res["game_id"],
            "treatment": res["treatment"],
            "language": res["language"],
            "personality_condition": res["personality_condition"],
            "society": res["society"],
            "n_players": res["n_players"],
            "n_periods": res["n_periods"],
            "endowment": res["endowment"],
            "mpcr": res["mpcr"],
            "mean_contribution": round(mean_contribution(res), 4),
            "mean_contribution_by_period": json.dumps(
                [round(x, 4) for x in res["mean_contribution_by_period"]]),
            "antisocial_punishment_total": anti,
        }
        for i, p in enumerate(res["players"], start=1):
            row[f"agent{i}_personality"] = p["personality"]
            row[f"agent{i}_contributions"] = json.dumps(p["contributions"])
            row[f"agent{i}_total_contributed"] = p["total_contributed"]
            row[f"agent{i}_total_earnings"] = p["total_earnings"]
            row[f"agent{i}_contrib_fallbacks"] = p["n_contrib_fallbacks"]
            row[f"agent{i}_punish_fallbacks"] = p["n_punish_fallbacks"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary bundle.
# --------------------------------------------------------------------------- #
def summarize(results: List[Dict],
              key: Callable[[Dict], object] = lambda r: r["treatment"]
              ) -> Dict[object, Dict]:
    """Full metric bundle per group (default group = treatment)."""
    summary = {}
    for gkey, games in group_results(results, key).items():
        contribs = [mean_contribution(g) for g in games]
        summary[gkey] = {
            "n_games": len(games),
            "mean_contribution": _mean(contribs),
            "mean_contribution_sem": (_std(contribs) / (len(contribs) ** 0.5)) if contribs else float("nan"),
            "mean_contribution_by_period": mean_contribution_by_period(games),
            "mean_earnings_by_period": mean_earnings_by_period(games),
            "deviation_binned_punishment": deviation_binned_punishment(games),
            "antisocial_prosocial_split": antisocial_prosocial_split(games),
            "antisocial_vs_cooperation_corr": antisocial_vs_cooperation_corr(games),
            "vengeance": vengeance_table(games),
            "parse_fallback_rate": parse_fallback_rate(games),
        }
    return summary
