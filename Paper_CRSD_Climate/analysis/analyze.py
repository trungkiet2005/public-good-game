"""Analysis for the CRSD (collective-risk climate dilemma) paper.

Loads crsd_all_models.csv, computes the human-vs-LLM baseline (with Hedges g
effect sizes, 95% CIs and Holm-adjusted p-values), the risk-sensitivity ANOVA
(with eta-squared), and the personality / language breakdowns across the three
models, then writes:
  analysis/stats.json   -- machine-readable stats
  analysis/tables.md    -- the main-text numbers
  analysis/si_tables.tex-- LaTeX tables for the electronic supplementary material
  figures/fig1_risk.{pdf,png}, figures/fig2_moderators.{pdf,png}  (vector + raster)

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
ROOT = HERE.parent                                   # Paper_CRSD_Climate/
RES = ROOT.parent / "FAIRGAME" / "results"
CRSD_CSV = RES / "crsd_results" / "crsd_all_models.csv"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

MODELS = ["qwen25-7b-instruct", "gemma2-9b-it", "llama-3-1-8b"]
MODEL_LABEL = {"qwen25-7b-instruct": "Qwen2.5-7B", "gemma2-9b-it": "Gemma-2-9B",
               "llama-3-1-8b": "Llama-3.1-8B"}

# Milinski et al. 2008 (PNAS), n=10 groups per treatment.
HUMAN_CRSD = {
    90: {"success": 5, "n": 10, "mean": 118.2, "se": 1.9, "fair": 3.3},
    50: {"success": 1, "n": 10, "mean": 92.2, "se": 9.0, "fair": 2.1},
    10: {"success": 0, "n": 10, "mean": 73.0, "se": 4.4, "fair": 1.1},
}


def parse_list(s):
    return s if isinstance(s, list) else ast.literal_eval(s)


def load_crsd():
    df = pd.read_csv(CRSD_CSV)
    df["round_totals"] = df["round_totals"].apply(parse_list)
    for i in range(1, 7):
        df[f"agent{i}_contributions"] = df[f"agent{i}_contributions"].apply(parse_list)
    return df


def cell_stats(sub):
    n = len(sub)
    if n == 0:
        return None
    totals = sub["group_total"].to_numpy(dtype=float)
    nr = int(sub["n_rounds"].iloc[0])
    fair = [sum(1 for i in range(1, 7) if r[f"agent{i}_total"] >= 2 * nr)
            for _, r in sub.iterrows()]
    fb = sum(sub[f"agent{i}_parse_fallbacks"].sum() for i in range(1, 7))
    dec = n * 6 * nr
    traj = np.array([np.cumsum(rt) for rt in sub["round_totals"]])
    acts = {"first": {0: 0, 2: 0, 4: 0}, "second": {0: 0, 2: 0, 4: 0}}
    for _, r in sub.iterrows():
        for i in range(1, 7):
            for rd, v in enumerate(r[f"agent{i}_contributions"]):
                half = "first" if rd < nr // 2 else "second"
                if v in acts[half]:
                    acts[half][v] += 1
    return {
        "n": n, "success_rate": float(sub["reached_target"].mean()),
        "success_count": int(round(sub["reached_target"].mean() * n)),
        "mean_total": float(totals.mean()),
        "std_total": float(totals.std(ddof=1)) if n > 1 else 0.0,
        "sem_total": float(totals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "fair_sharers_per_group": float(np.mean(fair)),
        "parse_fallback_rate": float(fb / dec) if dec else 0.0,
        "cumulative_trajectory": traj.mean(axis=0).tolist(),
        "acts_by_half": acts, "group_totals": totals.tolist(),
    }


# --------------------------------------------------------------------------- #
# Statistics: tests + effect sizes + multiple-comparison correction.
# --------------------------------------------------------------------------- #
def welch(m1, s1, n1, m2, s2, n2):
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return float("inf"), 0.0
    t = (m1 - m2) / se
    num = (s1**2 / n1 + s2**2 / n2) ** 2
    den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    return float(t), float(2 * (1 - sps.t.cdf(abs(t), num / den)))


def fisher(a, b, c, d):
    return float(sps.fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])


def hedges_g(m1, s1, n1, m2, s2, n2):
    """Hedges g (bias-corrected standardised mean difference) with approx 95% CI."""
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return float("inf"), float("nan"), float("nan")
    d = (m1 - m2) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = J * d
    se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2))) * J
    return float(g), float(g - 1.96 * se), float(g + 1.96 * se)


def eta_squared(F, k_groups, n_total):
    dfb = k_groups - 1
    dfw = n_total - k_groups
    return float((F * dfb) / (F * dfb + dfw))


def tost_welch(m1, s1, n1, m2, s2, n2, bound):
    """Two-one-sided-tests (TOST) Welch equivalence test of |m1-m2| < bound.

    Returns (diff, p_tost, df) where p_tost is the larger of the two
    one-sided p-values (Lakens et al. 2018); p_tost < .05 establishes
    equivalence within +/- bound.
    """
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    num = (s1**2 / n1 + s2**2 / n2) ** 2
    den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    df = num / den
    diff = m1 - m2
    if se == 0:
        return float(diff), 0.0 if abs(diff) < bound else 1.0, float(df)
    p_lo = 1 - sps.t.cdf((diff + bound) / se, df)   # H1: diff > -bound
    p_hi = sps.t.cdf((diff - bound) / se, df)       # H1: diff < +bound
    return float(diff), float(max(p_lo, p_hi)), float(df)


def bf01_anova_bic(groups):
    """BIC-approximate Bayes factor for the one-way ANOVA null (single mean)
    against the alternative of separate group means (Wagenmakers 2007).

    BF01 > 1 favours the null; e.g. BF01 = 3 means the data are three times
    more likely under "no treatment effect".
    """
    allv = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    n = len(allv)
    rss0 = float(((allv - allv.mean()) ** 2).sum())
    rss1 = float(sum(((np.asarray(g, dtype=float) - np.mean(g)) ** 2).sum()
                     for g in groups))
    if rss1 == 0:
        return float("nan")
    bic0 = n * math.log(rss0 / n) + 1 * math.log(n)
    bic1 = n * math.log(rss1 / n) + len(groups) * math.log(n)
    return float(math.exp((bic1 - bic0) / 2))


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


def analyse(df):
    out = {"baseline": {}, "anova": {}, "personality": {}, "language": {}}
    welch_ps, fisher_ps, keys = [], [], []
    for model in MODELS:
        md = df[df["model"] == model]
        base = md[(md["language"] == "en") & (md["personality_condition"] == "neutral")
                  & (md["framing"] == "climate")]
        out["baseline"][model] = {}
        gt = {}
        for p in (90, 50, 10):
            s = cell_stats(base[base["treatment_loss_prob"] == p])
            out["baseline"][model][p] = s
            gt[p] = s["group_totals"]
            h = HUMAN_CRSD[p]
            sd_h = h["se"] * math.sqrt(h["n"])
            t, pv = welch(s["mean_total"], s["std_total"], s["n"], h["mean"], sd_h, h["n"])
            g, glo, ghi = hedges_g(s["mean_total"], s["std_total"], s["n"],
                                   h["mean"], sd_h, h["n"])
            fp = fisher(s["success_count"], s["n"] - s["success_count"],
                        h["success"], h["n"] - h["success"])
            s.update({"welch_t_vs_human": t, "welch_p_vs_human": pv,
                      "hedges_g": g, "hedges_g_lo": glo, "hedges_g_hi": ghi,
                      "fisher_p_success_vs_human": fp})
            welch_ps.append(pv)
            fisher_ps.append(fp)
            keys.append((model, p))
        F, pv = sps.f_oneway(gt[90], gt[50], gt[10])
        out["anova"][model] = {"F": float(F), "p": float(pv),
                               "eta2": eta_squared(F, 3, sum(len(gt[p]) for p in gt))}
        # Evidence FOR the risk null: TOST equivalence on each pairwise risk
        # contrast (SESOI = half the human 90->10 decline) and a BIC-approximate
        # Bayes factor for the one-way ANOVA null.
        bound = 0.5 * (HUMAN_CRSD[90]["mean"] - HUMAN_CRSD[10]["mean"])
        tost = {}
        for pa, pb in ((90, 50), (50, 10), (90, 10)):
            a = np.asarray(gt[pa], dtype=float)
            b = np.asarray(gt[pb], dtype=float)
            diff, pt, dfree = tost_welch(a.mean(), a.std(ddof=1), len(a),
                                         b.mean(), b.std(ddof=1), len(b), bound)
            tost[f"{pa}v{pb}"] = {"diff": diff, "p": pt, "df": dfree}
        out["anova"][model].update({
            "tost_bound": float(bound), "tost": tost,
            "bf01_bic": bf01_anova_bic([gt[90], gt[50], gt[10]])})
        out["personality"][model] = {}
        pers = md[(md["language"] == "en") & (md["framing"] == "climate")]
        for c in ("neutral", "cooperative", "selfish", "risk_averse"):
            out["personality"][model][c] = {
                p: cell_stats(pers[(pers["personality_condition"] == c)
                                   & (pers["treatment_loss_prob"] == p)])
                for p in (90, 50, 10)}
        out["language"][model] = {}
        langs = md[(md["personality_condition"] == "neutral") & (md["framing"] == "climate")]
        for l in ("en", "fr", "ar", "cn", "vn"):
            out["language"][model][l] = {
                p: cell_stats(langs[(langs["language"] == l)
                                    & (langs["treatment_loss_prob"] == p)])
                for p in (90, 50, 10)}
    # Holm correction across the 9-cell baseline family (totals and success separately).
    welch_adj = holm(welch_ps)
    fisher_adj = holm(fisher_ps)
    out["holm"] = {f"{m}_{p}": {"welch_p_holm": welch_adj[i], "fisher_p_holm": fisher_adj[i]}
                   for i, (m, p) in enumerate(keys)}
    return out


# --------------------------------------------------------------------------- #
# Figures (vector PDF + raster PNG).
# --------------------------------------------------------------------------- #
def make_figures(crsd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "figure.dpi": 200,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    risk_color = {90: "#1f77b4", 50: "#2ca02c", 10: "#d62728"}
    model_color = {"qwen25-7b-instruct": "#4C72B0", "gemma2-9b-it": "#DD8452",
                   "llama-3-1-8b": "#55A868"}
    PRIMARY = "gemma2-9b-it"

    def save(fig, stem):
        fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGDIR / f"{stem}.png", bbox_inches="tight")
        plt.close(fig)

    # Fig 1: risk (in)sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    b = crsd["baseline"][PRIMARY]
    nr = 10
    ax.plot(range(1, nr + 1), [120 / nr * r for r in range(1, nr + 1)], color="black",
            lw=1.2, ls=":", label="fair-share path")
    for p in (90, 50, 10):
        ax.plot(range(1, nr + 1), b[p]["cumulative_trajectory"], marker="o", ms=3.5,
                color=risk_color[p], label=f"{p}% risk")
    ax.axhline(120, color="grey", lw=0.9, ls="--")
    ax.text(1.2, 123, "target €120", fontsize=7, color="grey")
    ax.set_xlabel("Round"); ax.set_ylabel("Cumulative group contribution (€)")
    ax.set_title(f"(a) {MODEL_LABEL[PRIMARY]} trajectories")
    ax.set_xticks(range(1, nr + 1)); ax.legend(loc="upper left")
    ax = axes[1]
    ps = [90, 50, 10]; x = np.arange(len(ps)); w = 0.2
    ax.bar(x - 1.5 * w, [HUMAN_CRSD[p]["mean"] for p in ps], w,
           yerr=[HUMAN_CRSD[p]["se"] for p in ps], capsize=2, color="0.4", label="Human")
    for j, model in enumerate(MODELS):
        ax.bar(x + (j - 0.5) * w, [crsd["baseline"][model][p]["mean_total"] for p in ps], w,
               yerr=[crsd["baseline"][model][p]["sem_total"] for p in ps], capsize=2,
               color=model_color[model], label=MODEL_LABEL[model])
    ax.axhline(120, color="grey", lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"{p}%" for p in ps])
    ax.set_xlabel("Loss probability"); ax.set_ylabel("Final group total (€)")
    ax.set_title("(b) Final total vs risk"); ax.legend(ncol=2, fontsize=6.5)
    fig.tight_layout(); save(fig, "fig1_risk")

    # Fig 2: moderators
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    conds = ["selfish", "risk_averse", "neutral", "cooperative"]
    cond_lab = {"selfish": "Selfish", "risk_averse": "Risk-averse",
                "neutral": "Neutral", "cooperative": "Cooperative"}
    x = np.arange(len(conds)); w = 0.25
    for j, model in enumerate(MODELS):
        vals = [100 * np.mean([crsd["personality"][model][c][p]["success_rate"]
                               for p in (90, 50, 10)]) for c in conds]
        ax.bar(x + (j - 1) * w, vals, w, color=model_color[model], label=MODEL_LABEL[model])
    ax.set_xticks(x); ax.set_xticklabels([cond_lab[c] for c in conds], rotation=20, ha="right")
    ax.set_ylabel("Target reached (%)"); ax.set_title("(a) Disposition prompt")
    ax.legend(fontsize=6.5)
    ax = axes[1]
    langs = ["en", "fr", "cn", "vn", "ar"]
    lang_lab = {"en": "EN", "fr": "FR", "cn": "ZH", "vn": "VI", "ar": "AR"}
    x = np.arange(len(langs))
    means = [np.mean([crsd["language"][PRIMARY][l][p]["mean_total"] for p in (90, 50, 10)])
             for l in langs]
    fbs = [100 * np.mean([crsd["language"][PRIMARY][l][p]["parse_fallback_rate"]
                          for p in (90, 50, 10)]) for l in langs]
    ax.bar(x, means, 0.55, color="#8172B3")
    ax.axhline(120, color="grey", lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([lang_lab[l] for l in langs])
    ax.set_ylabel("Final group total (€)"); ax.set_title(f"(b) Language — {MODEL_LABEL[PRIMARY]}")
    ax.set_ylim(0, 190)
    ax2 = ax.twinx()
    ax2.plot(x, fbs, marker="D", ms=4, color="#C44E52")
    ax2.set_ylabel("Format non-compliance (%)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, max(15, max(fbs) * 1.3))
    fig.tight_layout(); save(fig, "fig2_moderators")
    print("figures ->", FIGDIR)


def write_tables(crsd):
    L = ["# CRSD paper - consolidated statistics\n",
         "## Baseline EN/neutral/climate vs human (Milinski 2008)\n",
         "| model | p | success LLM/HUM | mean total LLM/HUM | fair LLM/HUM | Hedges g [95% CI] | Welch p (Holm) | Fisher p (Holm) | parse-fail |",
         "|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for p in (90, 50, 10):
            s = crsd["baseline"][model][p]; h = HUMAN_CRSD[p]
            hp = crsd["holm"][f"{model}_{p}"]
            L.append(f"| {MODEL_LABEL[model]} | {p}% | {s['success_rate']*100:.0f}%/{h['success']*10:.0f}% "
                     f"| {s['mean_total']:.1f}±{s['sem_total']:.1f}/{h['mean']:.1f} "
                     f"| {s['fair_sharers_per_group']:.1f}/{h['fair']:.1f} "
                     f"| {s['hedges_g']:.1f} [{s['hedges_g_lo']:.1f}, {s['hedges_g_hi']:.1f}] "
                     f"| {s['welch_p_vs_human']:.1e} ({hp['welch_p_holm']:.1e}) "
                     f"| {s['fisher_p_success_vs_human']:.1e} ({hp['fisher_p_holm']:.1e}) "
                     f"| {s['parse_fallback_rate']*100:.1f}% |")
    L += ["\n### Risk-sensitivity ANOVA (human F=13.78, p<0.0001, eta2~0.51)\n",
          "ANOVA + evidence for the null: BIC Bayes factor BF01 (>1 favours no risk effect)",
          "and TOST equivalence on the 90% vs 10% contrast, bounds = +/- half the human",
          "90->10 decline (+/-22.6 EUR).\n",
          "| model | F | p | eta2 | BF01 | diff 90-10 (EUR) | TOST p (90v10) |",
          "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        a = crsd["anova"][model]
        t = a["tost"]["90v10"]
        L.append(f"| {MODEL_LABEL[model]} | {a['F']:.2f} | {a['p']:.3g} | {a['eta2']:.3f} "
                 f"| {a['bf01_bic']:.2f} | {t['diff']:+.1f} | {t['p']:.2g} |")
    (HERE / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print("tables ->", HERE / "tables.md")


# --------------------------------------------------------------------------- #
# Electronic supplementary material: full LaTeX tables.
# --------------------------------------------------------------------------- #
def write_si(crsd):
    rows = []
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Baseline (English, neutral, climate) versus human "
                r"\cite{milinski2008climate}, with bias-corrected effect sizes "
                r"(Hedges $g$ on final group total, LLM$-$human), Holm--Bonferroni "
                r"adjusted $p$-values across the nine-cell family, and 95\%\ "
                r"confidence intervals.}")
    rows.append(r"\label{tab:si_baseline}")
    rows.append(r"\begin{tabular}{llcccc}")
    rows.append(r"\toprule")
    rows.append(r"Loss & Model & Hedges $g$ [95\% CI] & Welch $p$ (Holm) & "
                r"Fisher $p$ (Holm) & parse-fail \\")
    rows.append(r"\midrule")
    for p in (90, 50, 10):
        for j, model in enumerate(MODELS):
            s = crsd["baseline"][model][p]
            hp = crsd["holm"][f"{model}_{p}"]
            lead = (r"\multirow{3}{*}{%d\%%}" % p) if j == 0 else ""
            rows.append(f"{lead} & {MODEL_LABEL[model]} & "
                        f"{s['hedges_g']:.1f} [{s['hedges_g_lo']:.1f}, {s['hedges_g_hi']:.1f}] & "
                        f"{s['welch_p_vs_human']:.0e} ({hp['welch_p_holm']:.0e}) & "
                        f"{s['fisher_p_success_vs_human']:.0e} ({hp['fisher_p_holm']:.0e}) & "
                        f"{s['parse_fallback_rate']*100:.1f}\\% \\\\")
        if p != 10:
            rows.append(r"\midrule")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    # ANOVA with eta^2, Bayes factor and TOST equivalence
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Risk sensitivity per model: one-way ANOVA of the final "
                r"group total across the three loss-probability treatments with "
                r"$\eta^2$ effect size, together with two analyses that quantify "
                r"evidence \emph{for} the null. $\mathrm{BF}_{01}$ is the "
                r"BIC-approximate Bayes factor in favour of the single-mean null "
                r"over the three-mean alternative (values $>1$ favour ``no risk "
                r"effect''). TOST is the two-one-sided-tests Welch equivalence "
                r"test on the extreme contrast (90\%\ vs 10\%\ loss probability) "
                r"with equivalence bounds $\pm$\euro 22.6, i.e.\ half the human "
                r"90\%$\to$10\%\ decline of \euro 45.2; $p<0.05$ establishes that "
                r"the model's risk response is reliably smaller than half the "
                r"human response. Human reference: $F_{2,27}=13.78$, $P<0.0001$.}")
    rows.append(r"\label{tab:si_anova}")
    rows.append(r"\begin{tabular}{lcccccc}\toprule")
    rows.append(r"Model & $F_{2,27}$ & $p$ & $\eta^2$ & $\mathrm{BF}_{01}$ & "
                r"$\Delta_{90-10}$ (\euro) & TOST $p$ \\\midrule")
    for model in MODELS:
        a = crsd["anova"][model]
        t = a["tost"]["90v10"]
        rows.append(f"{MODEL_LABEL[model]} & {a['F']:.2f} & {a['p']:.3f} & "
                    f"{a['eta2']:.3f} & {a['bf01_bic']:.2f} & {t['diff']:+.1f} & "
                    f"{t['p']:.3f} \\\\")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    # Language breakdown (averaged over treatments)
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Language robustness (neutral disposition, climate framing): "
                r"per model and language, mean of the three loss-probability "
                r"treatments. Success is target attainment; total is final group "
                r"investment (\euro); parse-fail is format non-compliance.}")
    rows.append(r"\label{tab:si_language}")
    rows.append(r"\begin{tabular}{ll" + "ccc" * 1 + r"}\toprule")
    rows.append(r"Model & Language & Success & Final total (\euro) & Parse-fail \\\midrule")
    lang_lab = {"en": "English", "fr": "French", "cn": "Chinese", "vn": "Vietnamese", "ar": "Arabic"}
    for j, model in enumerate(MODELS):
        for li, l in enumerate(("en", "fr", "cn", "vn", "ar")):
            cells = [crsd["language"][model][l][p] for p in (90, 50, 10)]
            succ = 100 * np.mean([c["success_rate"] for c in cells])
            tot = np.mean([c["mean_total"] for c in cells])
            fb = 100 * np.mean([c["parse_fallback_rate"] for c in cells])
            lead = (r"\multirow{5}{*}{%s}" % MODEL_LABEL[model]) if li == 0 else ""
            rows.append(f"{lead} & {lang_lab[l]} & {succ:.0f}\\% & {tot:.0f} & {fb:.1f}\\% \\\\")
        if model != MODELS[-1]:
            rows.append(r"\midrule")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    # Personality breakdown (averaged over treatments)
    rows.append(r"\begin{table*}[t]\centering")
    rows.append(r"\caption{Dispositional steerability (English, climate framing): "
                r"per model and disposition prompt, mean of the three "
                r"loss-probability treatments.}")
    rows.append(r"\label{tab:si_personality}")
    rows.append(r"\begin{tabular}{llccc}\toprule")
    rows.append(r"Model & Disposition & Success & Final total (\euro) & Fair-sharers \\\midrule")
    cond_lab = {"neutral": "Neutral", "cooperative": "Cooperative",
                "selfish": "Selfish", "risk_averse": "Risk-averse"}
    for j, model in enumerate(MODELS):
        for ci, c in enumerate(("selfish", "risk_averse", "neutral", "cooperative")):
            cells = [crsd["personality"][model][c][p] for p in (90, 50, 10)]
            succ = 100 * np.mean([x["success_rate"] for x in cells])
            tot = np.mean([x["mean_total"] for x in cells])
            fair = np.mean([x["fair_sharers_per_group"] for x in cells])
            lead = (r"\multirow{4}{*}{%s}" % MODEL_LABEL[model]) if ci == 0 else ""
            rows.append(f"{lead} & {cond_lab[c]} & {succ:.0f}\\% & {tot:.0f} & {fair:.1f} \\\\")
        if model != MODELS[-1]:
            rows.append(r"\midrule")
    rows.append(r"\bottomrule\end{tabular}\end{table*}")
    rows.append("")

    (HERE / "si_tables.tex").write_text("\n".join(rows), encoding="utf-8")
    print("si_tables ->", HERE / "si_tables.tex")


def main():
    crsd = analyse(load_crsd())
    (HERE / "stats.json").write_text(json.dumps(crsd, indent=1, default=float), encoding="utf-8")
    print("stats ->", HERE / "stats.json")
    write_tables(crsd)
    write_si(crsd)
    make_figures(crsd)


if __name__ == "__main__":
    main()
