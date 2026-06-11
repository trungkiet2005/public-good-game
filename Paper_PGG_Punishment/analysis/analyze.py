"""Analysis for the PGG-with-punishment paper.

Loads pgg_all_models.csv plus the per-model pgg_metrics.json, computes the
human-vs-LLM baseline (N vs P) with Hedges g and rank-biserial effect sizes and
Holm-adjusted p-values, the deviation-binned punishment / antisocial split, and
the cross-societal Spearman/Wilcoxon tests (with Fisher-z CIs) across the three
models, then writes:
  analysis/stats.json    -- machine-readable stats
  analysis/tables.md     -- the main-text numbers
  analysis/si_tables.tex -- LaTeX tables for the electronic supplementary material
  figures/fig1_punishment.{pdf,png}, figures/fig2_cross.{pdf,png}  (vector + raster)

Run:  python analysis/analyze.py
"""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # Paper_PGG_Punishment/
RES = ROOT.parent / "FAIRGAME" / "results"
PGG_CSV = RES / "pgg_punish_results" / "pgg_all_models.csv"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

MODELS = ["qwen25-7b-instruct", "gemma2-9b-it", "llama-3-1-8b"]
MODEL_LABEL = {"qwen25-7b-instruct": "Qwen2.5-7B", "gemma2-9b-it": "Gemma-2-9B",
               "llama-3-1-8b": "Llama-3.1-8B"}

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
HUMAN_N_GRAND = sum(HUMAN_PGG_N.values()) / len(HUMAN_PGG_N)
HUMAN_P_GRAND = sum(HUMAN_PGG_P.values()) / len(HUMAN_PGG_P)


def parse_list(s):
    return s if isinstance(s, list) else ast.literal_eval(s)


def load_pgg():
    df = pd.read_csv(PGG_CSV)
    df["mean_contribution_by_period"] = df["mean_contribution_by_period"].apply(parse_list)
    return df


def load_metrics():
    return {m: json.loads((RES / "pgg_punish_results" / m / "pgg_metrics.json").read_text(
        encoding="utf-8")) for m in MODELS}


def cell(sub):
    n = len(sub)
    if n == 0:
        return None
    mc = sub["mean_contribution"].to_numpy(dtype=float)
    traj = np.array([np.array(x, dtype=float) for x in sub["mean_contribution_by_period"]])
    fb = sum(sub[f"agent{i}_contrib_fallbacks"].sum() for i in range(1, 5))
    nper = int(sub["n_periods"].iloc[0]); dec = n * 4 * nper
    return {"n": n, "mean_contribution": float(mc.mean()),
            "sem_contribution": float(mc.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "mean_by_period": traj.mean(axis=0).tolist(),
            "contrib_fallback_rate": float(fb / dec) if dec else 0.0,
            "per_game": mc.tolist()}


def hedges_g(a, b):
    """Hedges g for two independent samples a, b (a-b) with approx 95% CI."""
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan"), float("nan"), float("nan")
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = J * d
    se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2))) * J
    return float(g), float(g - 1.96 * se), float(g + 1.96 * se)


def spearman_ci(rho, n):
    if n < 4 or abs(rho) >= 1:
        return float("nan"), float("nan")
    z = math.atanh(rho)
    se = 1 / math.sqrt(n - 3)
    return float(math.tanh(z - 1.96 * se)), float(math.tanh(z + 1.96 * se))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        a = max(a, prev)
        prev = a
        adj[i] = a
    return adj


def analyse(df, metrics):
    out = {"baseline": {}, "punish_effect": {}, "deviation": {}, "antisocial_split": {},
           "cross_societal": {}, "society": {}, "personality": {}}
    pe_ps = []
    for model in MODELS:
        md = df[df["model"] == model]
        base = md[(md["language"] == "en") & (md["personality_condition"] == "neutral")
                  & (md["society"] == "none")]
        out["baseline"][model] = {tr: cell(base[base["treatment"] == tr]) for tr in ("N", "P")}
        nN = base[base["treatment"] == "N"]["mean_contribution"].to_numpy()
        nP = base[base["treatment"] == "P"]["mean_contribution"].to_numpy()
        U, pv = sps.mannwhitneyu(nP, nN, alternative="two-sided")
        rb = 2 * U / (len(nP) * len(nN)) - 1            # rank-biserial correlation
        g, glo, ghi = hedges_g(nP, nN)
        out["punish_effect"][model] = {
            "P_minus_N": float(nP.mean() - nN.mean()), "mwu_U": float(U), "mwu_p": float(pv),
            "rank_biserial": float(rb), "hedges_g": g, "hedges_g_lo": glo, "hedges_g_hi": ghi,
            "human_P_minus_N": HUMAN_P_GRAND - HUMAN_N_GRAND}
        pe_ps.append(pv)
        mb = metrics[model]["baseline_en_neutral_by_treatment"]["P"]
        out["deviation"][model] = mb["deviation_binned_punishment"]
        out["antisocial_split"][model] = mb["antisocial_prosocial_split"]
        out["society"][model] = metrics[model]["by_treatment_society"]
        out["personality"][model] = metrics[model]["by_treatment_personality"]
        soc = metrics[model]["by_treatment_society"]
        socs, anti, coop, hum = [], [], [], []
        for key, v in soc.items():
            if not key.startswith("P_"):
                continue
            name = key[2:]
            socs.append(name)
            anti.append(v["antisocial_prosocial_split"]["antisocial_total"])
            coop.append(v["mean_contribution"])
            hum.append(HUMAN_PGG_P.get(name, float("nan")))
        rho_anti = sps.spearmanr(anti, coop).correlation
        ra_lo, ra_hi = spearman_ci(rho_anti, len(anti))
        valid = [(c, h) for c, h in zip(coop, hum) if h == h]
        rho_hum = sps.spearmanr([c for c, _ in valid], [h for _, h in valid]).correlation
        rh_lo, rh_hi = spearman_ci(rho_hum, len(valid))
        W, pw = sps.wilcoxon([c for c, _ in valid], [h for _, h in valid])
        out["cross_societal"][model] = {
            "societies": socs, "antisocial_total": anti, "mean_contribution": coop,
            "human_contribution": hum, "spearman_antisocial_vs_coop": float(rho_anti),
            "sac_ci": [ra_lo, ra_hi], "spearman_llm_vs_human_rank": float(rho_hum),
            "shr_ci": [rh_lo, rh_hi], "wilcoxon_W": float(W), "wilcoxon_p": float(pw),
            "mean_llm_minus_human": float(np.mean([c - h for c, h in valid]))}
    adj = holm(pe_ps)
    for i, model in enumerate(MODELS):
        out["punish_effect"][model]["mwu_p_holm"] = adj[i]
    return out


def make_figures(pgg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "figure.dpi": 200,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    PRIMARY = "gemma2-9b-it"

    def save(fig, stem):
        fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGDIR / f"{stem}.png", bbox_inches="tight")
        plt.close(fig)

    # Fig 1: punishment dynamics
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    bn = pgg["baseline"][PRIMARY]["N"]["mean_by_period"]
    bp = pgg["baseline"][PRIMARY]["P"]["mean_by_period"]
    ax.plot(range(1, 11), bn, marker="s", ms=3.5, color="#C44E52", label="No punishment (N)")
    ax.plot(range(1, 11), bp, marker="o", ms=3.5, color="#4C72B0", label="With punishment (P)")
    ax.set_xlabel("Period"); ax.set_ylabel("Mean contribution (of 20)")
    ax.set_ylim(0, 20); ax.set_xticks(range(1, 11))
    ax.set_title(f"(a) Contribution — {MODEL_LABEL[PRIMARY]}"); ax.legend(loc="lower left")
    ax = axes[1]
    bins = pgg["deviation"][PRIMARY]
    order = ["[-20,-11]", "[-10,-1]", "[0]", "[1,10]", "[11,20]"]
    lab = ["≤-11", "-10..-1", "0", "1..10", "11..20"]
    means = [bins[k]["mean_expenditure"] for k in order]
    colors = ["#55A868" if not bins[k]["is_antisocial"] else "#C44E52" for k in order]
    ax.bar(range(len(order)), means, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, rotation=15)
    ax.set_xlabel("Target's contribution − punisher's"); ax.set_ylabel("Mean punishment assigned")
    share = pgg["antisocial_split"][PRIMARY]["antisocial_share"]
    ax.set_title(f"(b) Punishment — antisocial {share:.0%}")
    ax.legend(handles=[Patch(facecolor="#55A868", edgecolor="black", label="punish free-riding"),
                       Patch(facecolor="#C44E52", edgecolor="black", label="antisocial")],
              loc="upper right")
    fig.tight_layout(); save(fig, "fig1_punishment")

    # Fig 2: cross-societal
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    cs = pgg["cross_societal"][PRIMARY]
    ax = axes[0]
    ax.scatter(cs["antisocial_total"], cs["mean_contribution"], color="#8172B3", s=18)
    for xx, yy, nm in zip(cs["antisocial_total"], cs["mean_contribution"], cs["societies"]):
        ax.annotate(nm, (xx, yy), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Antisocial punishment (total pts)"); ax.set_ylabel("Mean contribution (of 20)")
    ax.set_title(f"(a) rho={cs['spearman_antisocial_vs_coop']:+.2f} (human −0.90)")
    ax = axes[1]
    valid = [(c, h) for c, h in zip(cs["mean_contribution"], cs["human_contribution"]) if h == h]
    ax.scatter([h for _, h in valid], [c for c, _ in valid], color="#DD8452", s=18)
    ax.set_xlabel("Human contribution, Herrmann (of 20)"); ax.set_ylabel("LLM contribution (of 20)")
    ax.set_title(f"(b) rank rho={cs['spearman_llm_vs_human_rank']:+.2f}")
    fig.tight_layout(); save(fig, "fig2_cross")
    print("figures ->", FIGDIR)


def write_tables(pgg):
    L = ["# PGG paper - consolidated statistics\n",
         "## Baseline EN/neutral vs human (Herrmann 2008); human N/P grand-mean 8.6/12.9\n",
         "| model | contrib N | contrib P | P−N | Hedges g [95% CI] | rank-biserial | MWU p (Holm) | antisocial share |",
         "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        bN = pgg["baseline"][model]["N"]; bP = pgg["baseline"][model]["P"]
        pe = pgg["punish_effect"][model]; sh = pgg["antisocial_split"][model]["antisocial_share"]
        L.append(f"| {MODEL_LABEL[model]} | {bN['mean_contribution']:.2f}±{bN['sem_contribution']:.2f} "
                 f"| {bP['mean_contribution']:.2f}±{bP['sem_contribution']:.2f} "
                 f"| {pe['P_minus_N']:+.2f} "
                 f"| {pe['hedges_g']:.2f} [{pe['hedges_g_lo']:.2f}, {pe['hedges_g_hi']:.2f}] "
                 f"| {pe['rank_biserial']:+.2f} | {pe['mwu_p']:.3g} ({pe['mwu_p_holm']:.3g}) | {sh:.0%} |")
    L += ["\n### Cross-societal (treatment P, 16 societies); human Spearman(antisoc,coop)=-0.90\n",
          "| model | Spearman(antisoc,coop) [CI] | Spearman(LLM,HUM) [CI] | Wilcoxon p | mean LLM−HUM |",
          "|---|---|---|---|---|"]
    for model in MODELS:
        cs = pgg["cross_societal"][model]
        L.append(f"| {MODEL_LABEL[model]} | {cs['spearman_antisocial_vs_coop']:+.2f} "
                 f"[{cs['sac_ci'][0]:+.2f}, {cs['sac_ci'][1]:+.2f}] "
                 f"| {cs['spearman_llm_vs_human_rank']:+.2f} "
                 f"[{cs['shr_ci'][0]:+.2f}, {cs['shr_ci'][1]:+.2f}] "
                 f"| {cs['wilcoxon_p']:.3g} | {cs['mean_llm_minus_human']:+.2f} |")
    (HERE / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print("tables ->", HERE / "tables.md")


def write_si(pgg):
    rows = []
    # Baseline with effect sizes
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Punishment effect on cooperation with effect sizes: "
                r"Hedges $g$ and rank-biserial correlation $r$ for the P$-$N contrast "
                r"on the ten per-group means, with Holm--Bonferroni adjusted "
                r"Mann--Whitney $p$-values and 95\%\ confidence intervals. Human "
                r"grand-mean gain $\approx+4.3$ tokens \cite{herrmann2008antisocial}.}")
    rows.append(r"\label{tab:si_punish}")
    rows.append(r"\begin{tabular}{lccccc}\toprule")
    rows.append(r"Model & $\Delta$ (P$-$N) & Hedges $g$ [95\% CI] & $r$ & "
                r"MWU $p$ (Holm) & Antisocial \\\midrule")
    for model in MODELS:
        pe = pgg["punish_effect"][model]
        sh = pgg["antisocial_split"][model]["antisocial_share"]
        rows.append(f"{MODEL_LABEL[model]} & {pe['P_minus_N']:+.2f} & "
                    f"{pe['hedges_g']:.2f} [{pe['hedges_g_lo']:.2f}, {pe['hedges_g_hi']:.2f}] & "
                    f"{pe['rank_biserial']:+.2f} & {pe['mwu_p']:.3f} ({pe['mwu_p_holm']:.3f}) & "
                    f"{sh*100:.0f}\\% \\\\")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    # Deviation-binned punishment (full, all models)
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Mean punishment expenditure by contribution deviation "
                r"(target minus punisher) for the English, neutral baseline, "
                r"treatment P. Deviations $<0$ are punishment of free-riding; "
                r"deviations $\geq 0$ are antisocial.}")
    rows.append(r"\label{tab:si_deviation}")
    rows.append(r"\begin{tabular}{lccccc}\toprule")
    rows.append(r"Model & $\leq-11$ & $-10$ to $-1$ & $0$ & $1$ to $10$ & $11$ to $20$ \\\midrule")
    order = ["[-20,-11]", "[-10,-1]", "[0]", "[1,10]", "[11,20]"]
    for model in MODELS:
        b = pgg["deviation"][model]
        rows.append(f"{MODEL_LABEL[model]} & " +
                    " & ".join(f"{b[k]['mean_expenditure']:.2f}" for k in order) + r" \\")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    # Cross-societal with CIs
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Cross-societal structure (treatment P, 16 society "
                r"personas) with Fisher-$z$ 95\%\ confidence intervals. Human "
                r"reference: Spearman$(\text{antisocial},\text{cooperation})"
                r"\approx-0.90$ \cite{herrmann2008antisocial}.}")
    rows.append(r"\label{tab:si_cross}")
    rows.append(r"\begin{tabular}{lcccc}\toprule")
    rows.append(r"Model & $\rho_{\mathrm{anti,coop}}$ [95\% CI] & "
                r"$\rho_{\mathrm{LLM,HUM}}$ [95\% CI] & Wilcoxon $p$ & "
                r"$\overline{\text{LLM}-\text{HUM}}$ \\\midrule")
    for model in MODELS:
        cs = pgg["cross_societal"][model]
        rows.append(f"{MODEL_LABEL[model]} & "
                    f"{cs['spearman_antisocial_vs_coop']:+.2f} "
                    f"[{cs['sac_ci'][0]:+.2f}, {cs['sac_ci'][1]:+.2f}] & "
                    f"{cs['spearman_llm_vs_human_rank']:+.2f} "
                    f"[{cs['shr_ci'][0]:+.2f}, {cs['shr_ci'][1]:+.2f}] & "
                    f"{cs['wilcoxon_p']:.3f} & {cs['mean_llm_minus_human']:+.2f} \\\\")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    # Per-society table (primary model = Gemma) with human reference
    PRIMARY = "gemma2-9b-it"
    soc = pgg["society"][PRIMARY]
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Per-society contributions for %s against the human pools "
                r"of Herrmann \textit{et al.}\ \cite{herrmann2008antisocial}; "
                r"antisocial = total antisocial punishment points (treatment P). "
                r"Societies ordered by human treatment-P contribution.}"
                % MODEL_LABEL[PRIMARY])
    rows.append(r"\label{tab:si_society}")
    rows.append(r"\begin{tabular}{lccccc}\toprule")
    rows.append(r"Society & LLM N & LLM P & Antisocial & Human N & Human P \\\midrule")
    for name in sorted(HUMAN_PGG_P, key=lambda s: -HUMAN_PGG_P[s]):
        nkey, pkey = f"N_{name}", f"P_{name}"
        ln = soc.get(nkey, {}).get("mean_contribution", float("nan"))
        lp = soc.get(pkey, {}).get("mean_contribution", float("nan"))
        anti = soc.get(pkey, {}).get("antisocial_prosocial_split", {}).get("antisocial_total", 0)
        rows.append(f"{name} & {ln:.2f} & {lp:.2f} & {anti:.0f} & "
                    f"{HUMAN_PGG_N[name]:.1f} & {HUMAN_PGG_P[name]:.1f} \\\\")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    # Personality breakdown (treatment P: contribution + antisocial share)
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Dispositional steerability (English, treatment P): mean "
                r"contribution and antisocial share of punishment by disposition "
                r"prompt and model.}")
    rows.append(r"\label{tab:si_personality}")
    rows.append(r"\begin{tabular}{llcc}\toprule")
    rows.append(r"Model & Disposition & Mean contribution & Antisocial share \\\midrule")
    cond_lab = {"neutral": "Neutral", "cooperative": "Cooperative",
                "selfish": "Selfish", "vengeful": "Vengeful"}
    for j, model in enumerate(MODELS):
        per = pgg["personality"][model]
        for ci, c in enumerate(("neutral", "cooperative", "selfish", "vengeful")):
            e = per.get(f"P_{c}")
            if not e:
                continue
            mc = e["mean_contribution"]
            sh = e["antisocial_prosocial_split"]["antisocial_share"]
            lead = (r"\multirow{4}{*}{%s}" % MODEL_LABEL[model]) if ci == 0 else ""
            rows.append(f"{lead} & {cond_lab[c]} & {mc:.2f} & {sh*100:.0f}\\% \\\\")
        if model != MODELS[-1]:
            rows.append(r"\midrule")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    (HERE / "si_tables.tex").write_text("\n".join(rows), encoding="utf-8")
    print("si_tables ->", HERE / "si_tables.tex")


def main():
    pgg = analyse(load_pgg(), load_metrics())
    (HERE / "stats.json").write_text(json.dumps(pgg, indent=1, default=float), encoding="utf-8")
    print("stats ->", HERE / "stats.json")
    write_tables(pgg)
    write_si(pgg)
    make_figures(pgg)


if __name__ == "__main__":
    main()
