"""Consolidated analysis for the Royal Society Interface paper.

Loads the two cross-model result CSVs (crsd_all_models.csv, pgg_all_models.csv)
and the per-model metrics JSONs, computes every table the paper cites across all
three LLMs (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B), runs the formal tests
(treatment ANOVA, Welch t, Fisher exact, Mann-Whitney, Spearman, Wilcoxon), and
writes consolidated stats (JSON + Markdown) plus publication figures.

Run:  python analysis/analyze.py
Outputs: analysis/stats.json, analysis/tables.md, figures/*.png
"""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # Royal_Society_Interface/
RES = ROOT.parent / "FAIRGAME" / "results"
CRSD_CSV = RES / "crsd_results" / "crsd_all_models.csv"
PGG_CSV = RES / "pgg_punish_results" / "pgg_all_models.csv"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

MODELS = ["qwen25-7b-instruct", "gemma2-9b-it", "llama-3-1-8b"]
MODEL_LABEL = {
    "qwen25-7b-instruct": "Qwen2.5-7B",
    "gemma2-9b-it": "Gemma-2-9B",
    "llama-3-1-8b": "Llama-3.1-8B",
}

# ----------------------------------------------------------------------------- #
# Human benchmarks (from the two source papers).
# ----------------------------------------------------------------------------- #
# Milinski et al. 2008 (PNAS), n=10 groups per treatment.
HUMAN_CRSD = {
    90: {"success": 5, "n": 10, "mean": 118.2, "se": 1.9, "fair": 3.3},
    50: {"success": 1, "n": 10, "mean": 92.2, "se": 9.0, "fair": 2.1},
    10: {"success": 0, "n": 10, "mean": 73.0, "se": 4.4, "fair": 1.1},
}
# Herrmann, Thoeni & Gaechter 2008 (Science) mean contributions by pool.
HUMAN_PGG_P = {
    "Boston": 18.0, "Copenhagen": 17.7, "St.Gallen": 16.7, "Zurich": 16.2,
    "Nottingham": 15.0, "Seoul": 14.7, "Bonn": 14.5, "Melbourne": 14.1,
    "Chengdu": 13.9, "Minsk": 12.9, "Samara": 11.7, "Dnipropetrovsk": 10.9,
    "Muscat": 9.9, "Istanbul": 7.1, "Riyadh": 6.9, "Athens": 5.7,
}
HUMAN_PGG_N = {
    "Copenhagen": 11.5, "Dnipropetrovsk": 10.6, "Minsk": 10.5, "St.Gallen": 10.1,
    "Muscat": 10.0, "Samara": 9.7, "Zurich": 9.3, "Boston": 9.3, "Bonn": 9.2,
    "Chengdu": 8.0, "Seoul": 7.9, "Riyadh": 7.6, "Nottingham": 6.9,
    "Athens": 6.4, "Istanbul": 5.4, "Melbourne": 4.9,
}
HUMAN_PGG_N_GRAND = sum(HUMAN_PGG_N.values()) / len(HUMAN_PGG_N)
HUMAN_PGG_P_GRAND = sum(HUMAN_PGG_P.values()) / len(HUMAN_PGG_P)


def parse_list(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(s)


# ----------------------------------------------------------------------------- #
# CRSD.
# ----------------------------------------------------------------------------- #
def load_crsd():
    df = pd.read_csv(CRSD_CSV)
    df["round_totals"] = df["round_totals"].apply(parse_list)
    for i in range(1, 7):
        df[f"agent{i}_contributions"] = df[f"agent{i}_contributions"].apply(parse_list)
    return df


def crsd_cell_stats(sub):
    """sub: rows of one cell. Returns dict of summary stats."""
    n = len(sub)
    if n == 0:
        return None
    totals = sub["group_total"].to_numpy(dtype=float)
    success = sub["reached_target"].mean()
    # fair-sharers: agents with total contribution >= 2*n_rounds (avg >= EUR2/round)
    nr = int(sub["n_rounds"].iloc[0])
    fair_per_group = []
    for _, r in sub.iterrows():
        cnt = sum(1 for i in range(1, 7) if r[f"agent{i}_total"] >= 2 * nr)
        fair_per_group.append(cnt)
    # parse fallback rate = total fallbacks / total decisions
    fb_total = sum(sub[f"agent{i}_parse_fallbacks"].sum() for i in range(1, 7))
    decisions = n * 6 * nr
    # cumulative trajectory (mean across groups)
    traj = np.array([np.cumsum(rt) for rt in sub["round_totals"]])
    cum = traj.mean(axis=0).tolist()
    # acts by half
    acts = {"first": {0: 0, 2: 0, 4: 0}, "second": {0: 0, 2: 0, 4: 0}}
    for _, r in sub.iterrows():
        for i in range(1, 7):
            c = r[f"agent{i}_contributions"]
            for rd, v in enumerate(c):
                half = "first" if rd < nr // 2 else "second"
                if v in acts[half]:
                    acts[half][v] += 1
    return {
        "n": n,
        "success_rate": float(success),
        "success_count": int(round(success * n)),
        "mean_total": float(totals.mean()),
        "std_total": float(totals.std(ddof=1)) if n > 1 else 0.0,
        "sem_total": float(totals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "fair_sharers_per_group": float(np.mean(fair_per_group)),
        "parse_fallback_rate": float(fb_total / decisions) if decisions else 0.0,
        "cumulative_trajectory": cum,
        "acts_by_half": acts,
        "group_totals": totals.tolist(),
    }


def welch(m1, s1, n1, m2, s2, n2):
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return float("inf"), 0.0
    t = (m1 - m2) / se
    num = (s1**2 / n1 + s2**2 / n2) ** 2
    den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    dfree = num / den if den else float("nan")
    p = 2 * (1 - sps.t.cdf(abs(t), dfree))
    return float(t), float(p)


def fisher(a, b, c, d):
    _, p = sps.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(p)


def analyse_crsd(df):
    out = {"baseline": {}, "anova": {}, "personality": {}, "language": {}}
    for model in MODELS:
        md = df[df["model"] == model]
        # baseline = en, neutral, climate
        base = md[(md["language"] == "en") & (md["personality_condition"] == "neutral")
                  & (md["framing"] == "climate")]
        out["baseline"][model] = {}
        group_totals_by_p = {}
        for p in (90, 50, 10):
            cell = base[base["treatment_loss_prob"] == p]
            s = crsd_cell_stats(cell)
            out["baseline"][model][p] = s
            group_totals_by_p[p] = s["group_totals"]
            # tests vs human
            h = HUMAN_CRSD[p]
            sd_h = h["se"] * math.sqrt(h["n"])
            t, pv = welch(s["mean_total"], s["std_total"], s["n"], h["mean"], sd_h, h["n"])
            fp = fisher(s["success_count"], s["n"] - s["success_count"], h["success"],
                        h["n"] - h["success"])
            s["welch_t_vs_human"] = t
            s["welch_p_vs_human"] = pv
            s["fisher_p_success_vs_human"] = fp
        # one-way ANOVA across treatments (risk sensitivity)
        F, pv = sps.f_oneway(group_totals_by_p[90], group_totals_by_p[50],
                             group_totals_by_p[10])
        out["anova"][model] = {"F": float(F), "p": float(pv)}
        # personality (en, climate)
        out["personality"][model] = {}
        pers = md[(md["language"] == "en") & (md["framing"] == "climate")]
        for cond in ("neutral", "cooperative", "selfish", "risk_averse"):
            out["personality"][model][cond] = {}
            for p in (90, 50, 10):
                cell = pers[(pers["personality_condition"] == cond)
                            & (pers["treatment_loss_prob"] == p)]
                if len(cell):
                    out["personality"][model][cond][p] = crsd_cell_stats(cell)
        # language (neutral, climate)
        out["language"][model] = {}
        langs = md[(md["personality_condition"] == "neutral") & (md["framing"] == "climate")]
        for lang in ("en", "fr", "ar", "cn", "vn"):
            out["language"][model][lang] = {}
            for p in (90, 50, 10):
                cell = langs[(langs["language"] == lang)
                             & (langs["treatment_loss_prob"] == p)]
                if len(cell):
                    out["language"][model][lang][p] = crsd_cell_stats(cell)
    return out


# ----------------------------------------------------------------------------- #
# PGG.
# ----------------------------------------------------------------------------- #
def load_pgg():
    df = pd.read_csv(PGG_CSV)
    df["mean_contribution_by_period"] = df["mean_contribution_by_period"].apply(parse_list)
    return df


def load_pgg_metrics():
    m = {}
    for model in MODELS:
        p = RES / "pgg_punish_results" / model / "pgg_metrics.json"
        m[model] = json.loads(p.read_text(encoding="utf-8"))
    return m


def pgg_cell(sub):
    n = len(sub)
    if n == 0:
        return None
    mc = sub["mean_contribution"].to_numpy(dtype=float)
    traj = np.array([np.array(x, dtype=float) for x in sub["mean_contribution_by_period"]])
    # contrib fallback rate
    fb = sum(sub[f"agent{i}_contrib_fallbacks"].sum() for i in range(1, 5))
    nper = int(sub["n_periods"].iloc[0])
    decisions = n * 4 * nper
    return {
        "n": n,
        "mean_contribution": float(mc.mean()),
        "sem_contribution": float(mc.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "mean_by_period": traj.mean(axis=0).tolist(),
        "contrib_fallback_rate": float(fb / decisions) if decisions else 0.0,
        "per_game_means": mc.tolist(),
    }


def analyse_pgg(df, metrics):
    out = {"baseline": {}, "punish_effect": {}, "deviation": {}, "antisocial_split": {},
           "language": {}, "personality": {}, "society": {}, "cross_societal": {}}
    for model in MODELS:
        md = df[df["model"] == model]
        base = md[(md["language"] == "en") & (md["personality_condition"] == "neutral")
                  & (md["society"] == "none")]
        out["baseline"][model] = {}
        for tr in ("N", "P"):
            out["baseline"][model][tr] = pgg_cell(base[base["treatment"] == tr])
        # punishment effect on cooperation + Mann-Whitney on per-group means
        nN = base[base["treatment"] == "N"]["mean_contribution"].to_numpy()
        nP = base[base["treatment"] == "P"]["mean_contribution"].to_numpy()
        U, pv = sps.mannwhitneyu(nP, nN, alternative="two-sided")
        out["punish_effect"][model] = {
            "P_minus_N": float(nP.mean() - nN.mean()),
            "mwu_U": float(U), "mwu_p": float(pv),
            "human_P_minus_N": HUMAN_PGG_P_GRAND - HUMAN_PGG_N_GRAND,
        }
        # deviation decomposition + antisocial split (from metrics, baseline P)
        mb = metrics[model]["baseline_en_neutral_by_treatment"]["P"]
        out["deviation"][model] = mb["deviation_binned_punishment"]
        out["antisocial_split"][model] = mb["antisocial_prosocial_split"]
        # language / personality (from metrics by_treatment_*)
        out["language"][model] = metrics[model]["by_treatment_language"]
        out["personality"][model] = metrics[model]["by_treatment_personality"]
        # society (from metrics by_treatment_society) -> P only for cross-societal
        soc = metrics[model]["by_treatment_society"]
        out["society"][model] = soc
        # cross-societal: per society (treatment P) antisocial total vs contribution
        socs, anti, coop, hum = [], [], [], []
        for key, v in soc.items():
            if not key.startswith("P_"):
                continue
            name = key[2:]
            socs.append(name)
            anti.append(v["antisocial_prosocial_split"]["antisocial_total"])
            coop.append(v["mean_contribution"])
            hum.append(HUMAN_PGG_P.get(name, float("nan")))
        rho_anti = sps.spearmanr(anti, coop).correlation if len(anti) >= 2 else float("nan")
        valid = [(c, h) for c, h in zip(coop, hum) if h == h]
        rho_hum = (sps.spearmanr([c for c, _ in valid], [h for _, h in valid]).correlation
                   if len(valid) >= 2 else float("nan"))
        W, pw = sps.wilcoxon([c for c, _ in valid], [h for _, h in valid])
        out["cross_societal"][model] = {
            "societies": socs, "antisocial_total": anti, "mean_contribution": coop,
            "human_contribution": hum,
            "spearman_antisocial_vs_coop": float(rho_anti),
            "spearman_llm_vs_human_rank": float(rho_hum),
            "wilcoxon_W": float(W), "wilcoxon_p": float(pw),
            "mean_llm_minus_human": float(np.mean([c - h for c, h in valid])),
        }
    return out


# ----------------------------------------------------------------------------- #
# Figures.
# ----------------------------------------------------------------------------- #
def make_figures(crsd, pgg, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "figure.dpi": 200})
    risk_color = {90: "#1f77b4", 50: "#2ca02c", 10: "#d62728"}
    model_color = {"qwen25-7b-instruct": "#4C72B0", "gemma2-9b-it": "#DD8452",
                   "llama-3-1-8b": "#55A868"}
    PRIMARY = "gemma2-9b-it"

    # ---- Figure 1: CRSD risk (in)sensitivity ------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    # (a) cumulative trajectory, primary model, 3 risk levels
    ax = axes[0]
    b = crsd["baseline"][PRIMARY]
    nr = 10
    fair = [120 / nr * r for r in range(1, nr + 1)]
    ax.plot(range(1, nr + 1), fair, color="black", lw=1.2, ls=":", label="fair-share path")
    for p in (90, 50, 10):
        ax.plot(range(1, nr + 1), b[p]["cumulative_trajectory"], marker="o", ms=3.5,
                color=risk_color[p], label=f"{p}% risk")
    ax.axhline(120, color="grey", lw=0.9, ls="--")
    ax.text(1.2, 123, "target €120", fontsize=7, color="grey")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative group contribution (€)")
    ax.set_title(f"(a) {MODEL_LABEL[PRIMARY]} trajectories")
    ax.set_xticks(range(1, nr + 1))
    ax.legend(loc="upper left")
    # (b) final totals: humans decline with risk, LLMs flat & high
    ax = axes[1]
    ps = [90, 50, 10]
    x = np.arange(len(ps))
    w = 0.2
    hum = [HUMAN_CRSD[p]["mean"] for p in ps]
    hum_se = [HUMAN_CRSD[p]["se"] for p in ps]
    ax.bar(x - 1.5 * w, hum, w, yerr=hum_se, capsize=2, color="0.4", label="Human")
    for j, model in enumerate(MODELS):
        means = [crsd["baseline"][model][p]["mean_total"] for p in ps]
        sems = [crsd["baseline"][model][p]["sem_total"] for p in ps]
        ax.bar(x + (j - 0.5) * w, means, w, yerr=sems, capsize=2,
               color=model_color[model], label=MODEL_LABEL[model])
    ax.axhline(120, color="grey", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in ps])
    ax.set_xlabel("Loss probability")
    ax.set_ylabel("Final group total (€)")
    ax.set_title("(b) Final total vs risk")
    ax.legend(ncol=2, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_crsd_risk.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: CRSD moderators (personality, language) ----------------- #
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    # (a) personality -> success rate (avg over treatments), per model
    ax = axes[0]
    conds = ["selfish", "risk_averse", "neutral", "cooperative"]
    cond_lab = {"selfish": "Selfish", "risk_averse": "Risk-averse",
                "neutral": "Neutral", "cooperative": "Cooperative"}
    x = np.arange(len(conds))
    w = 0.25
    for j, model in enumerate(MODELS):
        vals = []
        for c in conds:
            srs = [crsd["personality"][model][c][p]["success_rate"]
                   for p in (90, 50, 10) if p in crsd["personality"][model][c]]
            vals.append(100 * np.mean(srs))
        ax.bar(x + (j - 1) * w, vals, w, color=model_color[model], label=MODEL_LABEL[model])
    ax.set_xticks(x)
    ax.set_xticklabels([cond_lab[c] for c in conds], rotation=20, ha="right")
    ax.set_ylabel("Target reached (%)")
    ax.set_title("(a) Disposition prompt")
    ax.legend(fontsize=6.5)
    # (b) language -> mean total + fallback (primary model)
    ax = axes[1]
    langs = ["en", "fr", "cn", "vn", "ar"]
    lang_lab = {"en": "EN", "fr": "FR", "cn": "ZH", "vn": "VI", "ar": "AR"}
    x = np.arange(len(langs))
    means = [np.mean([crsd["language"][PRIMARY][l][p]["mean_total"] for p in (90, 50, 10)])
             for l in langs]
    fbs = [100 * np.mean([crsd["language"][PRIMARY][l][p]["parse_fallback_rate"]
                          for p in (90, 50, 10)]) for l in langs]
    bars = ax.bar(x, means, 0.55, color="#8172B3", label="Final total (€)")
    ax.axhline(120, color="grey", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([lang_lab[l] for l in langs])
    ax.set_ylabel("Final group total (€)")
    ax.set_title(f"(b) Language — {MODEL_LABEL[PRIMARY]}")
    ax.set_ylim(0, 190)
    ax2 = ax.twinx()
    ax2.plot(x, fbs, marker="D", ms=4, color="#C44E52", label="Parse-fail (%)")
    ax2.set_ylabel("Format non-compliance (%)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, max(15, max(fbs) * 1.3))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_crsd_moderators.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: PGG punishment ------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    # (a) contribution by period, N vs P, primary model
    ax = axes[0]
    bn = pgg["baseline"][PRIMARY]["N"]["mean_by_period"]
    bp = pgg["baseline"][PRIMARY]["P"]["mean_by_period"]
    ax.plot(range(1, 11), bn, marker="s", ms=3.5, color="#C44E52", label="No punishment (N)")
    ax.plot(range(1, 11), bp, marker="o", ms=3.5, color="#4C72B0", label="With punishment (P)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Mean contribution (of 20)")
    ax.set_ylim(0, 20)
    ax.set_xticks(range(1, 11))
    ax.set_title(f"(a) Contribution — {MODEL_LABEL[PRIMARY]}")
    ax.legend(loc="lower left")
    # (b) deviation-binned punishment, primary model
    ax = axes[1]
    bins = pgg["deviation"][PRIMARY]
    order = ["[-20,-11]", "[-10,-1]", "[0]", "[1,10]", "[11,20]"]
    lab = ["≤-11", "-10..-1", "0", "1..10", "11..20"]
    means = [bins[k]["mean_expenditure"] for k in order]
    colors = ["#55A868" if not bins[k]["is_antisocial"] else "#C44E52" for k in order]
    ax.bar(range(len(order)), means, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(lab, rotation=15)
    ax.set_xlabel("Target's contribution − punisher's")
    ax.set_ylabel("Mean punishment assigned")
    share = pgg["antisocial_split"][PRIMARY]["antisocial_share"]
    ax.set_title(f"(b) Punishment — antisocial {share:.0%}")
    ax.legend(handles=[Patch(facecolor="#55A868", edgecolor="black",
                             label="punish free-riding"),
                       Patch(facecolor="#C44E52", edgecolor="black",
                             label="antisocial")], loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_pgg_punishment.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: PGG cross-societal -------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    cs = pgg["cross_societal"][PRIMARY]
    # (a) antisocial vs contribution scatter
    ax = axes[0]
    ax.scatter(cs["antisocial_total"], cs["mean_contribution"], color="#8172B3", s=18)
    for xx, yy, nm in zip(cs["antisocial_total"], cs["mean_contribution"], cs["societies"]):
        ax.annotate(nm, (xx, yy), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Antisocial punishment (total pts)")
    ax.set_ylabel("Mean contribution (of 20)")
    ax.set_title(f"(a) rho={cs['spearman_antisocial_vs_coop']:+.2f} (human −0.90)")
    # (b) LLM vs human contribution ranking
    ax = axes[1]
    valid = [(nm, c, h) for nm, c, h in zip(cs["societies"], cs["mean_contribution"],
                                            cs["human_contribution"]) if h == h]
    hv = [h for _, _, h in valid]
    cv = [c for _, c, h in valid]
    ax.scatter(hv, cv, color="#DD8452", s=18)
    ax.set_xlabel("Human contribution, Herrmann (of 20)")
    ax.set_ylabel("LLM contribution (of 20)")
    ax.set_title(f"(b) rank rho={cs['spearman_llm_vs_human_rank']:+.2f}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_pgg_cross_societal.png", bbox_inches="tight")
    plt.close(fig)
    print("figures ->", FIGDIR)


# ----------------------------------------------------------------------------- #
# Markdown report (numbers I cite in the paper).
# ----------------------------------------------------------------------------- #
def write_tables(crsd, pgg):
    L = []
    L.append("# Consolidated statistics for the paper\n")
    L.append("## CRSD (Study 1) — baseline EN/neutral/climate vs human (Milinski 2008)\n")
    L.append("| model | p | success LLM/HUM | mean total LLM/HUM | fair-sharers LLM/HUM | "
             "Welch p | Fisher p | parse-fail |")
    L.append("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for p in (90, 50, 10):
            s = crsd["baseline"][model][p]
            h = HUMAN_CRSD[p]
            L.append(f"| {MODEL_LABEL[model]} | {p}% | {s['success_rate']*100:.0f}%/{h['success']*10:.0f}% "
                     f"| {s['mean_total']:.1f}±{s['sem_total']:.1f}/{h['mean']:.1f} "
                     f"| {s['fair_sharers_per_group']:.1f}/{h['fair']:.1f} "
                     f"| {s['welch_p_vs_human']:.1e} | {s['fisher_p_success_vs_human']:.1e} "
                     f"| {s['parse_fallback_rate']*100:.1f}% |")
    L.append("\n### Risk-sensitivity ANOVA (group_total across 90/50/10)\n")
    L.append("| model | F | p |  (human: F=13.78, p<0.0001) |")
    L.append("|---|---|---|---|")
    for model in MODELS:
        a = crsd["anova"][model]
        L.append(f"| {MODEL_LABEL[model]} | {a['F']:.2f} | {a['p']:.3g} | |")

    L.append("\n## CRSD personality (EN, climate; success% avg over treatments)\n")
    L.append("| model | selfish | risk_averse | neutral | cooperative |")
    L.append("|---|---|---|---|---|")
    for model in MODELS:
        row = [MODEL_LABEL[model]]
        for c in ("selfish", "risk_averse", "neutral", "cooperative"):
            srs = [crsd["personality"][model][c][p]["success_rate"] for p in (90, 50, 10)]
            mts = [crsd["personality"][model][c][p]["mean_total"] for p in (90, 50, 10)]
            row.append(f"{100*np.mean(srs):.0f}% / €{np.mean(mts):.0f}")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## CRSD language (neutral, climate; mean total avg over treatments / parse-fail)\n")
    L.append("| model | en | fr | cn | vn | ar |")
    L.append("|---|---|---|---|---|---|")
    for model in MODELS:
        row = [MODEL_LABEL[model]]
        for l in ("en", "fr", "cn", "vn", "ar"):
            mts = [crsd["language"][model][l][p]["mean_total"] for p in (90, 50, 10)]
            fbs = [crsd["language"][model][l][p]["parse_fallback_rate"] for p in (90, 50, 10)]
            row.append(f"€{np.mean(mts):.0f} / {100*np.mean(fbs):.1f}%")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## PGG (Study 2) — baseline EN/neutral vs human (Herrmann 2008)\n")
    L.append("| model | mean contrib N | mean contrib P | P−N | MWU p | antisocial share | "
             "(human N/P grand-mean) |")
    L.append("|---|---|---|---|---|---|---|")
    for model in MODELS:
        bN = pgg["baseline"][model]["N"]
        bP = pgg["baseline"][model]["P"]
        pe = pgg["punish_effect"][model]
        sh = pgg["antisocial_split"][model]["antisocial_share"]
        L.append(f"| {MODEL_LABEL[model]} | {bN['mean_contribution']:.2f}±{bN['sem_contribution']:.2f} "
                 f"| {bP['mean_contribution']:.2f}±{bP['sem_contribution']:.2f} "
                 f"| {pe['P_minus_N']:+.2f} | {pe['mwu_p']:.3g} | {sh:.0%} "
                 f"| {HUMAN_PGG_N_GRAND:.1f}/{HUMAN_PGG_P_GRAND:.1f} |")

    L.append("\n### PGG deviation-binned punishment (baseline P) — antisocial share & free-rider slope\n")
    L.append("| model | ≤-11 | -10..-1 | 0 | 1..10 | 11..20 | antisocial share |")
    L.append("|---|---|---|---|---|---|---|")
    for model in MODELS:
        bins = pgg["deviation"][model]
        order = ["[-20,-11]", "[-10,-1]", "[0]", "[1,10]", "[11,20]"]
        vals = [f"{bins[k]['mean_expenditure']:.2f}" for k in order]
        sh = pgg["antisocial_split"][model]["antisocial_share"]
        L.append(f"| {MODEL_LABEL[model]} | " + " | ".join(vals) + f" | {sh:.0%} |")

    L.append("\n### PGG cross-societal (treatment P, 16 societies)\n")
    L.append("| model | Spearman(antisoc,coop) | Spearman(LLM,HUM rank) | Wilcoxon p | mean LLM−HUM |")
    L.append("|---|---|---|---|---|")
    for model in MODELS:
        cs = pgg["cross_societal"][model]
        L.append(f"| {MODEL_LABEL[model]} | {cs['spearman_antisocial_vs_coop']:+.2f} "
                 f"| {cs['spearman_llm_vs_human_rank']:+.2f} | {cs['wilcoxon_p']:.3g} "
                 f"| {cs['mean_llm_minus_human']:+.2f} |")

    (HERE / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print("tables ->", HERE / "tables.md")


def main():
    crsd_df = load_crsd()
    pgg_df = load_pgg()
    pgg_metrics = load_pgg_metrics()
    crsd = analyse_crsd(crsd_df)
    pgg = analyse_pgg(pgg_df, pgg_metrics)
    (HERE / "stats.json").write_text(
        json.dumps({"crsd": crsd, "pgg": pgg}, indent=1, default=float), encoding="utf-8")
    print("stats ->", HERE / "stats.json")
    write_tables(crsd, pgg)
    make_figures(crsd, pgg, pgg_metrics)


if __name__ == "__main__":
    main()
