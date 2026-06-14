# Open-weight LLM agents as human surrogates in social-dilemma experiments

This repository contains the code, data, analysis pipeline and manuscripts for a study
asking a single question: **can open-weight large language model (LLM) agents stand in for
human participants in landmark social-dilemma experiments?**

Using the [FAIRGAME](FAIRGAME/) multi-agent framework, we replicate two canonical human
studies with three open instruction-tuned models (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B),
crossed with five languages, four dispositions and sixteen societies:

1. **Collective-Risk Social Dilemma (CRSD)** — Milinski et al. (2008) climate dilemma, in
   which a group must reach a shared investment target or face a probabilistic catastrophic
   loss.
2. **Public-Goods Game with costly peer punishment (PGG)** — Fehr & Gächter (2002), scaled
   across societies by Herrmann, Thöni & Gächter (2008), which exposes the cross-cultural
   geography of (anti-)social punishment.

**Headline finding.** The models reproduce the *qualitative scaffolding* of cooperation but
violate the *diagnostic comparative statics* that make these experiments scientifically
informative. They reach the CRSD target on 100% of games at every loss probability (humans
50/10/0%) and never show the human investment decline as catastrophe risk falls; punishment
does not raise contributions; and the cross-societal antisocial-punishment signature
collapses (Spearman ρ ≈ −0.15 vs. human −0.90). Current open LLMs behave like a single
high-prosociality, risk-insensitive, prompt-steerable *"average cooperator"* — they mimic
human moves without the underlying preference structure, and are therefore not yet drop-in
human surrogates where the manipulation, not the mean, carries the scientific payload.

## Repository layout

| Path | Contents |
|------|----------|
| [`FAIRGAME/`](FAIRGAME/) | The simulation framework. Forked from the SOM Research Lab / LIST [FAIRGAME](https://github.com/SOM-Research/FAIRGAME) and extended with two dedicated game modules (see below). Runs **local** open-weight models via `transformers` / `vLLM`, including a Kaggle GPU integration. |
| [`FAIRGAME/src/crsd/`](FAIRGAME/src/crsd/) | CRSD game module: agent, game loop, prompt builder, results parser. |
| [`FAIRGAME/src/pgg_punish/`](FAIRGAME/src/pgg_punish/) | PGG-with-punishment game module (mirrors the CRSD module structure). |
| [`AntisocialPunishment_Dataset/`](AntisocialPunishment_Dataset/) | The original Herrmann–Thöni–Gächter (2008) human dataset used as the human baseline for the PGG comparison. |
| [`Royal_Society_Interface/`](Royal_Society_Interface/) | **Main manuscript** (LaTeX, RSIF template): *"Scaffolding without preferences: open-weight language-model agents fail the diagnostic comparative statics of two landmark social-dilemma experiments."* Synthesises both games. |
| [`Paper_CRSD_Climate/`](Paper_CRSD_Climate/) | Standalone CRSD manuscript + supplementary information. |
| [`Paper_PGG_Punishment/`](Paper_PGG_Punishment/) | Standalone PGG-punishment manuscript + supplementary information. |
| [`Literature_review_paper/`](Literature_review_paper/) | Literature review: *LLM agents in repeated public-goods games.* |

Each manuscript directory carries its own `analysis/analyze.py` (the analysis
**source of truth** that turns raw run output into the figures, tables and statistics used
in that paper), a `figures/` folder and a `refs.bib`.

## The FAIRGAME framework

FAIRGAME simulates configurable game-theoretic scenarios with LLM agents of varying
identities, languages and personalities, then quantifies the outcomes. This fork is geared
toward **local / offline open-weight models** (the upstream cloud SDK connectors for OpenAI,
Anthropic and Mistral have been removed) so that runs are reproducible on a single GPU.

Key entry points:

- [`FAIRGAME/main.py`](FAIRGAME/main.py) — core runner with an inline input example.
- [`FAIRGAME/resources/`](FAIRGAME/resources/) — JSON game configs and prompt templates
  (`crsd_config/`, `crsd_templates/`, `pgg_punish_config/`, `pgg_punish_templates/`, …).
- [`FAIRGAME/results/`](FAIRGAME/results/) — raw run output (`crsd_results/`,
  `pgg_punish_results/`).
- [`FAIRGAME/crsd_analysis.py`](FAIRGAME/crsd_analysis.py) and
  [`FAIRGAME/pgg_punish_analysis.py`](FAIRGAME/pgg_punish_analysis.py) — top-level analysis
  helpers.
- Kaggle GPU notebooks: `kaggle_crsd_notebook.py`, `kaggle_pgg_punish_notebook.py` and the
  follow-up runners — see [`FAIRGAME/FOLLOWUP_RUNS.md`](FAIRGAME/FOLLOWUP_RUNS.md) and
  [`FAIRGAME/kaggle_run/README_KAGGLE_API.md`](FAIRGAME/kaggle_run/README_KAGGLE_API.md).

> **Kaggle note.** The Kaggle GPU is an RTX PRO 6000 Blackwell (sm_120 / CUDA 13). If the
> vLLM FlashInfer sampler JIT-crashes, set `VLLM_USE_FLASHINFER_SAMPLER=0`.

## Quick start

```bash
# 1. Install runtime deps (minimal — most live in the Kaggle/Colab base image)
pip install -r FAIRGAME/requirements.txt

# 2. Run a simulation (edit resources/* configs to choose game, models, languages, …)
python FAIRGAME/main.py

# 3. Reproduce a paper's figures/tables/stats from the raw results
python Royal_Society_Interface/analysis/analyze.py
```

Running the full sweep (3 models × 5 languages × 4 dispositions × 16 societies, both games)
requires a GPU; the Kaggle notebooks under `FAIRGAME/` are the supported path for that.

## Building the manuscripts

Each paper compiles with `latexmk` using the bundled `rsproca_new.cls` (Royal Society
ProcA/Interface class) and Biber for the bibliography:

```bash
cd Royal_Society_Interface
latexmk -pdf main.tex
```

## Models & conditions

- **Models:** Qwen2.5-7B-Instruct, Gemma-2-9B-it, Llama-3.1-8B-Instruct (open weights).
- **Languages:** five.
- **Dispositions:** four agent personalities.
- **Societies:** sixteen (matching the Herrmann et al. cross-cultural panel).

## Citation

If you use this code or data, please cite the main manuscript (Royal Society Interface,
in preparation) and the upstream works it builds on: Milinski et al. (2008), Fehr & Gächter
(2002), Herrmann, Thöni & Gächter (2008), and the FAIRGAME framework (SOM Research Lab /
LIST).

## License

The FAIRGAME framework retains its upstream **Apache License 2.0**
(see [`FAIRGAME/LICENSE`](FAIRGAME/LICENSE)). Manuscripts and the
Herrmann–Thöni–Gächter dataset are subject to their respective original terms.
