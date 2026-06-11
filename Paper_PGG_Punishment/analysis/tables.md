# PGG paper - consolidated statistics

## Baseline EN/neutral vs human (Herrmann 2008); human N/P grand-mean 8.6/12.9

| model | contrib N | contrib P | P−N | Hedges g [95% CI] | rank-biserial | MWU p (Holm) | antisocial share |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 14.26±0.48 | 13.90±0.51 | -0.36 | -0.22 [-1.06, 0.62] | -0.11 | 0.705 (1) | 59% |
| Gemma-2-9B | 13.41±0.37 | 9.86±0.37 | -3.55 | -2.93 [-4.20, -1.66] | -0.94 | 0.00044 (0.00132) | 55% |
| Llama-3.1-8B | 10.67±0.45 | 10.75±0.43 | +0.08 | 0.06 [-0.78, 0.90] | +0.04 | 0.91 (1) | 54% |

### Cross-societal (treatment P, 16 societies); human Spearman(antisoc,coop)=-0.90

| model | Spearman(antisoc,coop) [CI] | Spearman(LLM,HUM) [CI] | Wilcoxon p | mean LLM−HUM |
|---|---|---|---|---|
| Qwen2.5-7B | -0.09 [-0.56, +0.43] | +0.33 [-0.20, +0.71] | 0.669 | +0.80 |
| Gemma-2-9B | -0.15 [-0.60, +0.37] | +0.06 [-0.45, +0.54] | 0.00516 | -3.30 |
| Llama-3.1-8B | +0.30 [-0.23, +0.69] | +0.17 [-0.35, +0.62] | 0.025 | -2.47 |