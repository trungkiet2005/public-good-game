"""Regenerate Slides/figures/fig2_moderators.{pdf,png} for the talk.

Same two-panel CRSD moderators figure as the paper, but
  (a) disposition steerability -> target-reached %, per model (unchanged);
  (b) language robustness -> final group total per language, ALL THREE models,
      WITHOUT the parse-fail / format-non-compliance overlay (dropped for the talk).

Reuses the paper's analysis code so the numbers are identical to the manuscript.
Run:  python Slides/make_fig2_clean.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CRSD_AN = ROOT / "Paper_CRSD_Climate" / "analysis" / "analyze.py"
OUT = ROOT / "Slides" / "figures"
OUT.mkdir(exist_ok=True, parents=True)

spec = importlib.util.spec_from_file_location("crsd_analyze", CRSD_AN)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)            # __main__ guard means main() does NOT run

crsd = A.analyse(A.load_crsd())
MODELS, MODEL_LABEL = A.MODELS, A.MODEL_LABEL
model_color = {"qwen25-7b-instruct": "#4C72B0", "gemma2-9b-it": "#DD8452",
               "llama-3-1-8b": "#55A868"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 200,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

# (a) disposition -> target-reached % (avg over the three loss probabilities)
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

# (b) language -> final group total, ALL three models (no parse-fail overlay)
ax = axes[1]
langs = ["en", "fr", "cn", "vn", "ar"]
lang_lab = {"en": "EN", "fr": "FR", "cn": "ZH", "vn": "VI", "ar": "AR"}
x = np.arange(len(langs)); w = 0.25
for j, model in enumerate(MODELS):
    means = [np.mean([crsd["language"][model][l][p]["mean_total"] for p in (90, 50, 10)])
             for l in langs]
    ax.bar(x + (j - 1) * w, means, w, color=model_color[model], label=MODEL_LABEL[model])
ax.axhline(120, color="grey", lw=0.9, ls="--")
ax.text(len(langs) - 1.6, 124, "target €120", fontsize=7, color="grey")
ax.set_xticks(x); ax.set_xticklabels([lang_lab[l] for l in langs])
ax.set_ylabel("Final group total (€)"); ax.set_title("(b) Language robustness")
ax.set_ylim(0, 205); ax.legend(fontsize=6.5)

fig.tight_layout()
fig.savefig(OUT / "fig2_moderators.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_moderators.png", bbox_inches="tight")
print("wrote", OUT / "fig2_moderators.pdf")
