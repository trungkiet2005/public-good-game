"""Post-hoc human--LLM alignment analyses requiring no new model inference.

The script compares the raw Herrmann et al. (2008) replication data with the
existing English/neutral FAIRGAME runs. It deliberately uses matched units:
player-period contributions, punisher-target decisions, group-level mechanism
regressions, and city-level normalized punishment rates.

Run from anywhere with::

    python Paper_PGG_Punishment/analysis/human_llm_alignment.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "FAIRGAME" / "results" / "pgg_punish_results"
HUMAN_CSV = (PROJECT / "4969858_extracted" / "HerrmannThoeniGaechter" /
             "HerrmannThoeniGaechterDATA.csv")
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen25-7b-instruct", "gemma2-9b-it", "llama-3-1-8b"]
LABELS = {
    "qwen25-7b-instruct": "Qwen2.5-7B",
    "gemma2-9b-it": "Gemma-2-9B",
    "llama-3-1-8b": "Llama-3.1-8B",
}
COUNTRIES = {
    "Boston": "United States", "Melbourne": "Australia",
    "Nottingham": "United Kingdom", "St.Gallen": "Switzerland",
    "Zurich": "Switzerland", "Copenhagen": "Denmark", "Bonn": "Germany",
    "Seoul": "South Korea", "Chengdu": "China", "Minsk": "Belarus",
    "Samara": "Russia", "Dnipropetrovsk": "Ukraine", "Istanbul": "Turkey",
    "Athens": "Greece", "Riyadh": "Saudi Arabia", "Muscat": "Oman",
}
BINS = [(-20, -11, "[-20,-11]"), (-10, -1, "[-10,-1]"),
        (0, 0, "[0]"), (1, 10, "[1,10]"), (11, 20, "[11,20]")]


def clean_city(value: str) -> str:
    return {"St. Gallen": "St.Gallen", "Dnipropetrovs'k": "Dnipropetrovsk"}.get(
        value, value)


def finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def rho(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return None
    x2, y2 = zip(*pairs)
    if len(set(x2)) < 2 or len(set(y2)) < 2:
        return None
    return finite(stats.spearmanr(x2, y2).statistic)


def load_human():
    df = pd.read_csv(HUMAN_CSV, skiprows=3)
    df["city"] = df["city"].map(clean_city)
    numeric = [c for c in df.columns if c not in {"sessionid", "p", "city"}]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df["treatment"] = df["p"].map({"N-experiment": "N", "P-experiment": "P"})
    df["deviation"] = df["otherscontribution"] - df["senderscontribution"]
    decisions = df.drop_duplicates(["subjectid", "treatment", "period"]).copy()
    return df, decisions


def load_llm():
    out = {}
    for model in MODELS:
        games = json.loads((RESULTS / model / "pgg_results.json").read_text(encoding="utf-8"))
        out[model] = games
    return out


def baseline_games(games, treatment):
    return [g for g in games if g["treatment"] == treatment and g["language"] == "en"
            and g["personality_condition"] == "neutral" and g["society"] == "none"]


def society_games(games, treatment="P"):
    return [g for g in games if g["treatment"] == treatment and g["society"] != "none"]


def llm_pairs(games):
    rows = []
    for g in games:
        for t, (contrib, matrix) in enumerate(zip(
                g["contributions_by_period"], g["punishment_matrix_by_period"]), start=1):
            for a in range(g["n_players"]):
                for b in range(g["n_players"]):
                    if a != b:
                        rows.append({"game": g["game_id"], "period": t, "punisher": a,
                                     "target": b, "sender": contrib[a], "other": contrib[b],
                                     "deviation": contrib[b] - contrib[a],
                                     "punishment": matrix[a][b], "society": g["society"]})
    return pd.DataFrame(rows)


def llm_decisions(games):
    rows = []
    for g in games:
        received = g.get("punishment_received_by_period") or [[0] * g["n_players"]] * g["n_periods"]
        for t, contrib in enumerate(g["contributions_by_period"], start=1):
            for player, value in enumerate(contrib):
                rows.append({"game": g["game_id"], "player": player, "period": t,
                             "contribution": value, "received": received[t - 1][player]})
    return pd.DataFrame(rows)


def bin_profile(pairs):
    result = {}
    for lo, hi, label in BINS:
        part = pairs[pairs["deviation"].between(lo, hi)]
        result[label] = {"mean": finite(part["punishment"].mean()), "n": int(len(part)),
                         "total": finite(part["punishment"].sum())}
    return result


def profile_alignment(human, model):
    hv = np.array([human[label]["mean"] for _, _, label in BINS], dtype=float)
    mv = np.array([model[label]["mean"] for _, _, label in BINS], dtype=float)
    return {"rmse": float(np.sqrt(np.mean((hv - mv) ** 2))),
            "pearson": finite(stats.pearsonr(hv, mv).statistic),
            "spearman": rho(hv.tolist(), mv.tolist())}


def group_mechanism_human(rows, decisions):
    pdec = decisions[decisions["treatment"] == "P"]
    contrib = pdec.groupby(["groupid", "period"])["senderscontribution"].mean().reset_index()
    c1 = contrib[contrib["period"] == 1].set_index("groupid")["senderscontribution"]
    c2 = contrib[contrib["period"] >= 2].groupby("groupid")["senderscontribution"].mean()
    pairs = rows[rows["treatment"] == "P"]
    pro = pairs[pairs["deviation"] < 0].groupby("groupid")["punishment"].mean()
    anti = pairs[pairs["deviation"] >= 0].groupby("groupid")["punishment"].mean()
    meta = pairs.groupby("groupid").agg(mgroupid=("mgroupid", "first"), city=("city", "first"))
    return pd.concat([c1.rename("c1"), c2.rename("c2to10"), pro.rename("prosocial"),
                      anti.rename("antisocial"), meta], axis=1).fillna(0).reset_index()


def group_mechanism_llm(games):
    rows = []
    for g in games:
        contrib = np.asarray(g["contributions_by_period"], dtype=float)
        pairs = llm_pairs([g])
        rows.append({"groupid": g["game_id"], "c1": contrib[0].mean(),
                     "c2to10": contrib[1:].mean(),
                     "prosocial": pairs.loc[pairs.deviation < 0, "punishment"].mean(),
                     "antisocial": pairs.loc[pairs.deviation >= 0, "punishment"].mean()})
    return pd.DataFrame(rows).fillna(0)


def mechanism_regression(frame, cluster=None):
    x = sm.add_constant(frame[["c1", "prosocial", "antisocial"]])
    model = sm.OLS(frame["c2to10"], x)
    if cluster and frame[cluster].nunique() > 1:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster]})
    else:
        fit = model.fit(cov_type="HC1")
    return {"n": int(fit.nobs), "r2": finite(fit.rsquared),
            "coefficients": {name: {"b": finite(fit.params[name]),
                                     "se": finite(fit.bse[name]),
                                     "p": finite(fit.pvalues[name])}
                             for name in fit.params.index}}


def distribution_summary(values, raw):
    values = np.asarray(values, dtype=float)
    raw = np.asarray(raw, dtype=float)
    return {"n_units": int(len(values)), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "q10": float(np.quantile(values, .1)), "median": float(np.median(values)),
            "q90": float(np.quantile(values, .9)),
            "raw_zero_share": float(np.mean(raw == 0)),
            "raw_full_share": float(np.mean(raw == 20))}


def reaction_summary(decisions, id_cols, contribution="contribution", received="received"):
    d = decisions.sort_values(id_cols + ["period"]).copy()
    d["next_contribution"] = d.groupby(id_cols)[contribution].shift(-1)
    d["delta_next"] = d["next_contribution"] - d[contribution]
    d = d.dropna(subset=["delta_next", received])
    fit = sm.OLS(d["delta_next"], sm.add_constant(d[[received]])).fit(cov_type="HC1")
    adjusted = sm.OLS(
        d["delta_next"], sm.add_constant(d[[received, contribution, "period"]])
    ).fit(cov_type="HC1")
    cuts = [-.1, .1, 3.1, 6.1, np.inf]
    labels = ["0", "1-3", "4-6", "7+"]
    d["received_bin"] = pd.cut(d[received], cuts, labels=labels)
    means = d.groupby("received_bin", observed=False)["delta_next"].mean()
    return {"n": int(len(d)), "slope": finite(fit.params[received]),
            "se": finite(fit.bse[received]), "p": finite(fit.pvalues[received]),
            "adjusted_slope": finite(adjusted.params[received]),
            "adjusted_se": finite(adjusted.bse[received]),
            "adjusted_p": finite(adjusted.pvalues[received]),
            "mean_delta_by_received": {label: finite(means.get(label)) for label in labels}}


def city_human(rows):
    p = rows[rows["treatment"] == "P"].copy()
    out = []
    for city, part in p.groupby("city"):
        anti = part[part["deviation"] >= 0]
        total = part["punishment"].sum()
        out.append({"city": city, "country": COUNTRIES.get(city),
                    "contribution": float(part["senderscontribution"].mean()),
                    "anti_mean_per_opportunity": float(anti["punishment"].mean()),
                    "anti_share": float(anti["punishment"].sum() / total) if total else 0.0,
                    "civic": finite(part["civic"].mean()),
                    "ruleoflaw": finite(part["ruleoflaw"].mean())})
    return pd.DataFrame(out).sort_values("city")


def city_llm(games):
    out = []
    for city, grouped in pd.DataFrame([{"city": g["society"], "game": g} for g in games]).groupby("city"):
        gs = grouped["game"].tolist()
        pairs = llm_pairs(gs)
        anti = pairs[pairs["deviation"] >= 0]
        total = pairs["punishment"].sum()
        contrib = [x for g in gs for row in g["contributions_by_period"] for x in row]
        out.append({"city": city, "contribution": float(np.mean(contrib)),
                    "anti_mean_per_opportunity": float(anti["punishment"].mean()),
                    "anti_share": float(anti["punishment"].sum() / total) if total else 0.0})
    return pd.DataFrame(out).sort_values("city")


def city_alignment(human, model):
    d = human.merge(model, on="city", suffixes=("_human", "_llm"))
    return {"n_cities": int(len(d)),
            "contribution_llm_vs_human": rho(d.contribution_llm.tolist(), d.contribution_human.tolist()),
            "anti_mean_llm_vs_human": rho(d.anti_mean_per_opportunity_llm.tolist(),
                                           d.anti_mean_per_opportunity_human.tolist()),
            "anti_share_llm_vs_human": rho(d.anti_share_llm.tolist(), d.anti_share_human.tolist()),
            "llm_contribution_vs_civic": rho(d.contribution_llm.tolist(), d.civic.tolist()),
            "llm_contribution_vs_ruleoflaw": rho(d.contribution_llm.tolist(), d.ruleoflaw.tolist()),
            "llm_anti_mean_vs_civic": rho(d.anti_mean_per_opportunity_llm.tolist(), d.civic.tolist()),
            "llm_anti_mean_vs_ruleoflaw": rho(d.anti_mean_per_opportunity_llm.tolist(),
                                               d.ruleoflaw.tolist())}


def trajectory(decisions, treatment, contribution_col):
    d = decisions[decisions["treatment"] == treatment]
    return d.groupby("period")[contribution_col].mean().reindex(range(1, 11)).to_numpy()


def analyse():
    human_rows, human_decisions = load_human()
    llms = load_llm()
    hpairs = human_rows[human_rows["treatment"] == "P"].rename(
        columns={"senderscontribution": "sender", "otherscontribution": "other"})
    hprofile = bin_profile(hpairs)
    hgroups = group_mechanism_human(human_rows, human_decisions)
    hcity = city_human(human_rows)
    human_reg = mechanism_regression(hgroups, "mgroupid")

    hreact = human_decisions[human_decisions["treatment"] == "P"][
        ["subjectid", "period", "senderscontribution", "recpun"]].rename(
            columns={"senderscontribution": "contribution", "recpun": "received"})
    out = {
        "notes": {
            "country": "Country is derived metadata; the source CSV identifies city pools only.",
            "units": "Punishment means include zero assignments and are per punisher-target opportunity.",
            "independence": "Human regression clusters by matching group; LLM regressions use HC1 with n=10 games.",
        },
        "countries": COUNTRIES,
        "human": {
            "conditional_punishment": hprofile,
            "mechanism_regression": human_reg,
            "reaction": reaction_summary(hreact, ["subjectid"]),
            "city": hcity.to_dict(orient="records"),
            "city_correlations": {
                "contribution_vs_civic": rho(hcity.contribution.tolist(), hcity.civic.tolist()),
                "contribution_vs_ruleoflaw": rho(hcity.contribution.tolist(), hcity.ruleoflaw.tolist()),
                "anti_mean_vs_civic": rho(hcity.anti_mean_per_opportunity.tolist(), hcity.civic.tolist()),
                "anti_mean_vs_ruleoflaw": rho(hcity.anti_mean_per_opportunity.tolist(), hcity.ruleoflaw.tolist()),
            },
        },
        "models": {},
    }

    human_unit = human_decisions.groupby(["subjectid", "treatment"])[
        "senderscontribution"].mean().reset_index()
    for treatment in ("N", "P"):
        vals = human_unit.loc[human_unit.treatment == treatment, "senderscontribution"]
        raw = human_decisions.loc[human_decisions.treatment == treatment, "senderscontribution"]
        out["human"].setdefault("distribution", {})[treatment] = distribution_summary(vals, raw)
        out["human"].setdefault("trajectory", {})[treatment] = trajectory(
            human_decisions.rename(columns={"senderscontribution": "contribution"}),
            treatment, "contribution").tolist()

    for model, games in llms.items():
        entry = {}
        pbase = baseline_games(games, "P")
        ppairs = llm_pairs(pbase)
        profile = bin_profile(ppairs)
        entry["conditional_punishment"] = profile
        entry["conditional_alignment"] = profile_alignment(hprofile, profile)
        entry["mechanism_regression"] = mechanism_regression(group_mechanism_llm(pbase))

        pdec = llm_decisions(pbase)
        entry["reaction"] = reaction_summary(pdec, ["game", "player"])
        entry["distribution"] = {}
        entry["trajectory"] = {}
        for treatment in ("N", "P"):
            gs = baseline_games(games, treatment)
            dec = llm_decisions(gs)
            units = dec.groupby(["game", "player"])["contribution"].mean()
            entry["distribution"][treatment] = distribution_summary(units, dec.contribution)
            hvals = human_unit.loc[human_unit.treatment == treatment, "senderscontribution"]
            entry["distribution"][treatment]["wasserstein_vs_human"] = float(
                stats.wasserstein_distance(hvals, units))
            tr = dec.groupby("period")["contribution"].mean().reindex(range(1, 11)).to_numpy()
            htr = np.asarray(out["human"]["trajectory"][treatment])
            entry["trajectory"][treatment] = {
                "means": tr.tolist(), "rmse_vs_human": float(np.sqrt(np.mean((tr - htr) ** 2))),
                "pearson_vs_human": finite(stats.pearsonr(tr, htr).statistic),
            }
        lcity = city_llm(society_games(games))
        entry["city"] = lcity.to_dict(orient="records")
        entry["city_alignment"] = city_alignment(hcity, lcity)
        out["models"][model] = entry
    return out


def fmt(value, digits=2):
    return "NA" if value is None else f"{value:.{digits}f}"


def write_markdown(result):
    hreg = result["human"]["mechanism_regression"]["coefficients"]
    lines = ["# Human-LLM alignment from existing PGG runs", "",
             "No new LLM inference is used. Human punishment and LLM punishment are both",
             "measured per punisher-target opportunity, including zero assignments.", "",
             "## Country coverage", "",
             "The source data has 16 city pools across 15 countries; Switzerland has two pools.",
             "Country labels are derived metadata, not a column in the original CSV.", "",
             "## Mechanism regression", "",
             "Group-level OLS: mean contribution in periods 2-10 ~ period-1 contribution +",
             "mean punishment of lower contributors + mean antisocial punishment.", "",
             "| source | n groups | c1 | prosocial punishment | antisocial punishment | R2 |",
             "|---|---:|---:|---:|---:|---:|",
             f"| Human | {result['human']['mechanism_regression']['n']} | "
             f"{fmt(hreg['c1']['b'])} | {fmt(hreg['prosocial']['b'])} | "
             f"{fmt(hreg['antisocial']['b'])} | {fmt(result['human']['mechanism_regression']['r2'])} |"]
    for model in MODELS:
        reg = result["models"][model]["mechanism_regression"]
        c = reg["coefficients"]
        lines.append(f"| {LABELS[model]} | {reg['n']} | {fmt(c['c1']['b'])} | "
                     f"{fmt(c['prosocial']['b'])} | {fmt(c['antisocial']['b'])} | {fmt(reg['r2'])} |")

    lines += ["", "## Conditional punishment alignment", "",
              "| model | profile RMSE | Pearson | Spearman |",
              "|---|---:|---:|---:|"]
    for model in MODELS:
        a = result["models"][model]["conditional_alignment"]
        lines.append(f"| {LABELS[model]} | {fmt(a['rmse'])} | {fmt(a['pearson'])} | {fmt(a['spearman'])} |")

    lines += ["", "## Distribution and trajectory alignment", "",
              "Wasserstein distance uses player-level ten-period mean contributions.", "",
              "| model | W-dist N | W-dist P | trajectory RMSE N | trajectory RMSE P |",
              "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        e = result["models"][model]
        lines.append(f"| {LABELS[model]} | {fmt(e['distribution']['N']['wasserstein_vs_human'])} | "
                     f"{fmt(e['distribution']['P']['wasserstein_vs_human'])} | "
                     f"{fmt(e['trajectory']['N']['rmse_vs_human'])} | "
                     f"{fmt(e['trajectory']['P']['rmse_vs_human'])} |")

    lines += ["", "## Reaction to received punishment", "",
              "Adjusted slope predicts next-period contribution change from points received this period,",
              "controlling current contribution and period.", "",
              "| source | n player-periods | raw slope | adjusted slope | robust SE | p |",
              "|---|---:|---:|---:|---:|---:|"]
    hr = result["human"]["reaction"]
    lines.append(f"| Human | {hr['n']} | {fmt(hr['slope'], 3)} | {fmt(hr['adjusted_slope'], 3)} | "
                 f"{fmt(hr['adjusted_se'], 3)} | {fmt(hr['adjusted_p'], 3)} |")
    for model in MODELS:
        r = result["models"][model]["reaction"]
        lines.append(f"| {LABELS[model]} | {r['n']} | {fmt(r['slope'], 3)} | "
                     f"{fmt(r['adjusted_slope'], 3)} | {fmt(r['adjusted_se'], 3)} | "
                     f"{fmt(r['adjusted_p'], 3)} |")

    lines += ["", "## City-level alignment", "",
              "All punishment quantities below are normalized per opportunity, not totals.", "",
              "| model | contribution vs human | anti-mean vs human | anti-share vs human | "
              "contribution vs civic | contribution vs rule of law | anti-mean vs civic | "
              "anti-mean vs rule of law |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    hc = result["human"]["city_correlations"]
    lines.append(f"| Human benchmark | 1.00 | 1.00 | 1.00 | {fmt(hc['contribution_vs_civic'])} | "
                 f"{fmt(hc['contribution_vs_ruleoflaw'])} | {fmt(hc['anti_mean_vs_civic'])} | "
                 f"{fmt(hc['anti_mean_vs_ruleoflaw'])} |")
    for model in MODELS:
        c = result["models"][model]["city_alignment"]
        lines.append(f"| {LABELS[model]} | {fmt(c['contribution_llm_vs_human'])} | "
                     f"{fmt(c['anti_mean_llm_vs_human'])} | {fmt(c['anti_share_llm_vs_human'])} | "
                     f"{fmt(c['llm_contribution_vs_civic'])} | "
                     f"{fmt(c['llm_contribution_vs_ruleoflaw'])} | "
                     f"{fmt(c['llm_anti_mean_vs_civic'])} | "
                     f"{fmt(c['llm_anti_mean_vs_ruleoflaw'])} |")

    lines += ["", "## Interpretation cautions", "",
              "- Human P and N are sequential for most subjects; LLM P and N are independent games.",
              "- The city prompt is only a location label, so city alignment tests model priors rather than a rich cultural manipulation.",
              "- LLM mechanism regressions have only ten baseline groups and are descriptive.",
              "- Demographic alignment cannot be estimated because the existing LLM runs have no matched demographic personas."]
    (HERE / "human_llm_alignment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(result):
    colors = {MODELS[0]: "#4C72B0", MODELS[1]: "#DD8452", MODELS[2]: "#55A868"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    ax = axes[0, 0]
    labels = [label for _, _, label in BINS]
    ax.plot(labels, [result["human"]["conditional_punishment"][x]["mean"] for x in labels],
            color="black", marker="o", lw=2, label="Human")
    for model in MODELS:
        ax.plot(labels, [result["models"][model]["conditional_punishment"][x]["mean"] for x in labels],
                color=colors[model], marker="o", label=LABELS[model])
    ax.set_title("Conditional punishment function")
    ax.set_ylabel("Mean points per opportunity")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    periods = np.arange(1, 11)
    ax.plot(periods, result["human"]["trajectory"]["N"], color="black", ls="--", label="Human N")
    ax.plot(periods, result["human"]["trajectory"]["P"], color="black", label="Human P")
    for model in MODELS:
        ax.plot(periods, result["models"][model]["trajectory"]["P"]["means"],
                color=colors[model], label=f"{LABELS[model]} P")
    ax.set_title("Contribution trajectories")
    ax.set_xlabel("Period")
    ax.set_ylabel("Contribution")
    ax.set_ylim(0, 20)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    hcity = pd.DataFrame(result["human"]["city"])
    for model in MODELS:
        city = pd.DataFrame(result["models"][model]["city"])
        d = hcity.merge(city, on="city", suffixes=("_human", "_llm"))
        ax.scatter(d.contribution_human, d.contribution_llm, label=LABELS[model],
                   color=colors[model], alpha=.8)
    ax.plot([5, 19], [5, 19], color="gray", ls=":")
    ax.set_title("City contribution alignment")
    ax.set_xlabel("Human city mean")
    ax.set_ylabel("LLM city mean")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    hr = result["human"]["reaction"]["mean_delta_by_received"]
    xlabels = list(hr)
    x = np.arange(len(xlabels))
    ax.plot(x, [hr[z] for z in xlabels], color="black", marker="o", lw=2, label="Human")
    for model in MODELS:
        rr = result["models"][model]["reaction"]["mean_delta_by_received"]
        ax.plot(x, [rr[z] for z in xlabels], color=colors[model], marker="o", label=LABELS[model])
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(x, xlabels)
    ax.set_title("Next-period reaction to received punishment")
    ax.set_xlabel("Points received")
    ax.set_ylabel("Mean contribution change")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_alignment_posthoc.png", dpi=200, bbox_inches="tight")
    try:
        fig.savefig(FIGDIR / "fig_alignment_posthoc.pdf", bbox_inches="tight")
    except (ImportError, OSError, PermissionError):
        # Some managed Windows environments block Matplotlib's optional
        # fontTools/lxml PDF dependency. The raster figure remains complete.
        pass
    plt.close(fig)


def main():
    result = analyse()
    (HERE / "human_llm_alignment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result)
    make_figure(result)
    print(HERE / "human_llm_alignment.md")
    print(HERE / "human_llm_alignment.json")
    print(FIGDIR / "fig_alignment_posthoc.png")


if __name__ == "__main__":
    main()
