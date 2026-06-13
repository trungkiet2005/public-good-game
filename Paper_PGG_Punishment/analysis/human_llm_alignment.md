# Human-LLM alignment from existing PGG runs

No new LLM inference is used. Human punishment and LLM punishment are both
measured per punisher-target opportunity, including zero assignments.

## Country coverage

The source data has 16 city pools across 15 countries; Switzerland has two pools.
Country labels are derived metadata, not a column in the original CSV.

## Mechanism regression

Group-level OLS: mean contribution in periods 2-10 ~ period-1 contribution +
mean punishment of lower contributors + mean antisocial punishment.

| source | n groups | c1 | prosocial punishment | antisocial punishment | R2 |
|---|---:|---:|---:|---:|---:|
| Human | 280 | 0.77 | 0.55 | -2.25 | 0.62 |
| Qwen2.5-7B | 10 | 0.36 | -0.36 | 0.34 | 0.56 |
| Gemma-2-9B | 10 | 0.29 | -1.21 | -0.36 | 0.55 |
| Llama-3.1-8B | 10 | 0.26 | 0.20 | 0.01 | 0.32 |

## Conditional punishment alignment

| model | profile RMSE | Pearson | Spearman |
|---|---:|---:|---:|
| Qwen2.5-7B | 1.60 | 0.86 | 0.70 |
| Gemma-2-9B | 1.45 | 0.97 | 0.90 |
| Llama-3.1-8B | 1.93 | 0.54 | 0.60 |

## Distribution and trajectory alignment

Wasserstein distance uses player-level ten-period mean contributions.

| model | W-dist N | W-dist P | trajectory RMSE N | trajectory RMSE P |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 5.73 | 3.50 | 6.45 | 0.90 |
| Gemma-2-9B | 5.36 | 5.00 | 6.68 | 3.57 |
| Llama-3.1-8B | 3.18 | 4.31 | 3.26 | 2.61 |

## Reaction to received punishment

Adjusted slope predicts next-period contribution change from points received this period,
controlling current contribution and period.

| source | n player-periods | raw slope | adjusted slope | robust SE | p |
|---|---:|---:|---:|---:|---:|
| Human | 10052 | 0.332 | 0.142 | 0.023 | 0.000 |
| Qwen2.5-7B | 360 | -0.040 | 0.016 | 0.065 | 0.804 |
| Gemma-2-9B | 360 | -0.081 | -0.079 | 0.059 | 0.182 |
| Llama-3.1-8B | 360 | 0.002 | -0.002 | 0.056 | 0.973 |

## City-level alignment

All punishment quantities below are normalized per opportunity, not totals.

| model | contribution vs human | anti-mean vs human | anti-share vs human | contribution vs civic | contribution vs rule of law | anti-mean vs civic | anti-mean vs rule of law |
|---|---:|---:|---:|---:|---:|---:|---:|
| Human benchmark | 1.00 | 1.00 | 1.00 | 0.34 | 0.66 | -0.48 | -0.58 |
| Qwen2.5-7B | 0.33 | -0.29 | -0.26 | 0.15 | 0.37 | -0.09 | 0.06 |
| Gemma-2-9B | 0.06 | 0.52 | 0.34 | -0.11 | -0.05 | -0.89 | -0.30 |
| Llama-3.1-8B | 0.17 | 0.00 | -0.19 | -0.03 | 0.24 | 0.34 | -0.04 |

## Interpretation cautions

- Human P and N are sequential within-subject (two counterbalanced ten-period sequences per participant); LLM P and N are independent between-group games, so the P-N contrast is a within-subject gain for humans but a between-group difference for the models.
- Human punishment here is summarized as raw per-opportunity means, whereas Herrmann et al. used interval (censored) regression clustered by matching group; the means are comparable to the LLM means but do not reproduce the paper's censored coefficients.
- The city prompt is only a location label, so city alignment tests model priors rather than a rich cultural manipulation.
- LLM mechanism regressions have only ten baseline groups and are descriptive.
- Demographic alignment cannot be estimated because the existing LLM runs have no matched demographic personas.
