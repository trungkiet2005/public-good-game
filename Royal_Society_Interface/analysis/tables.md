# Consolidated statistics for the paper

## CRSD (Study 1) — baseline EN/neutral/climate vs human (Milinski 2008)

| model | p | success LLM/HUM | mean total LLM/HUM | fair-sharers LLM/HUM | Welch p | Fisher p | parse-fail |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 90% | 100%/50% | 183.0±2.1/118.2 | 6.0/3.3 | 1.5e-14 | 3.3e-02 | 0.0% |
| Qwen2.5-7B | 50% | 100%/10% | 186.6±1.2/92.2 | 6.0/2.1 | 2.0e-06 | 1.2e-04 | 0.0% |
| Qwen2.5-7B | 10% | 100%/0% | 181.0±2.8/73.0 | 5.9/1.1 | 1.4e-12 | 1.1e-05 | 0.0% |
| Gemma-2-9B | 90% | 100%/50% | 168.2±3.3/118.2 | 5.8/3.3 | 2.6e-09 | 3.3e-02 | 0.0% |
| Gemma-2-9B | 50% | 100%/10% | 162.8±2.9/92.2 | 5.9/2.1 | 1.3e-05 | 1.2e-04 | 0.2% |
| Gemma-2-9B | 10% | 100%/0% | 158.0±3.9/73.0 | 5.8/1.1 | 3.0e-11 | 1.1e-05 | 0.2% |
| Llama-3.1-8B | 90% | 100%/50% | 177.4±2.9/118.2 | 5.8/3.3 | 2.1e-11 | 3.3e-02 | 0.2% |
| Llama-3.1-8B | 50% | 100%/10% | 165.2±3.6/92.2 | 5.6/2.1 | 7.6e-06 | 1.2e-04 | 0.0% |
| Llama-3.1-8B | 10% | 100%/0% | 170.0±3.1/73.0 | 5.7/1.1 | 4.0e-12 | 1.1e-05 | 0.0% |

### Risk-sensitivity ANOVA (group_total across 90/50/10)

| model | F | p |  (human: F=13.78, p<0.0001) |
|---|---|---|---|
| Qwen2.5-7B | 1.76 | 0.191 | |
| Gemma-2-9B | 2.23 | 0.127 | |
| Llama-3.1-8B | 3.64 | 0.0398 | |

## CRSD personality (EN, climate; success% avg over treatments)

| model | selfish | risk_averse | neutral | cooperative |
|---|---|---|---|---|
| Qwen2.5-7B | 0% / €74 | 93% / €132 | 100% / €184 | 100% / €207 |
| Gemma-2-9B | 0% / €33 | 73% / €124 | 100% / €163 | 100% / €159 |
| Llama-3.1-8B | 27% / €113 | 73% / €125 | 100% / €171 | 100% / €179 |

## CRSD language (neutral, climate; mean total avg over treatments / parse-fail)

| model | en | fr | cn | vn | ar |
|---|---|---|---|---|---|
| Qwen2.5-7B | €184 / 0.0% | €179 / 0.0% | €183 / 0.0% | €140 / 0.0% | €176 / 0.0% |
| Gemma-2-9B | €163 / 0.1% | €164 / 2.3% | €169 / 0.2% | €164 / 0.6% | €145 / 13.7% |
| Llama-3.1-8B | €171 / 0.1% | €168 / 0.6% | €141 / 0.3% | €163 / 14.8% | €163 / 0.3% |

## PGG (Study 2) — baseline EN/neutral vs human (Herrmann 2008)

| model | mean contrib N | mean contrib P | P−N | MWU p | antisocial share | (human N/P grand-mean) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 14.26±0.48 | 13.90±0.51 | -0.36 | 0.705 | 59% | 8.6/12.9 |
| Gemma-2-9B | 13.41±0.37 | 9.86±0.37 | -3.55 | 0.00044 | 55% | 8.6/12.9 |
| Llama-3.1-8B | 10.67±0.45 | 10.75±0.43 | +0.08 | 0.91 | 54% | 8.6/12.9 |

### PGG deviation-binned punishment (baseline P) — antisocial share & free-rider slope

| model | ≤-11 | -10..-1 | 0 | 1..10 | 11..20 | antisocial share |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 2.78 | 2.42 | 2.29 | 2.40 | 2.15 | 59% |
| Gemma-2-9B | 3.71 | 2.45 | 1.40 | 2.17 | 2.06 | 55% |
| Llama-3.1-8B | 2.84 | 2.86 | 2.60 | 2.78 | 2.53 | 54% |

### PGG cross-societal (treatment P, 16 societies)

| model | Spearman(antisoc,coop) | Spearman(LLM,HUM rank) | Wilcoxon p | mean LLM−HUM |
|---|---|---|---|---|
| Qwen2.5-7B | -0.09 | +0.33 | 0.669 | +0.80 |
| Gemma-2-9B | -0.15 | +0.06 | 0.00516 | -3.30 |
| Llama-3.1-8B | +0.30 | +0.17 | 0.025 | -2.47 |