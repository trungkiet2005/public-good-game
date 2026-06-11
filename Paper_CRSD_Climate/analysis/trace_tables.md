# CRSD reasoning-trace analysis (EN / neutral / climate baseline)

Manipulation check: is the loss probability REPRESENTED in the agents'
reasoning even though contributions do not respond to it? And does the
agent track the running total / remaining distance to the EUR120 target
(self-summation check)? Arithmetic metrics are over rounds >= 2.

| model | p | decisions | mentions own p | risk words | correct cum total | correct remaining | contrib (p mentioned) | contrib (not) |
|---|---|---|---|---|---|---|---|---|
| gemma2-9b-it | 90% | 600 | 26.5% | 67.2% | 21.3% | 11.5% | 2.98 | 2.74 |
| gemma2-9b-it | 50% | 600 | 17.7% | 68.0% | 24.1% | 18.1% | 2.87 | 2.68 |
| gemma2-9b-it | 10% | 600 | 20.0% | 67.7% | 23.1% | 15.6% | 2.63 | 2.63 |

**gemma2-9b-it** — Kruskal–Wallis across treatments, restricted to decisions that explicitly mention their own loss probability: H = 8.12, p = 0.0172 (n = [159, 106, 120], means = [2.98, 2.87, 2.63]). A non-significant H here means the risk is verbalised but does not move the contribution.

| llama-3-1-8b | 90% | 600 | 14.8% | 43.0% | 19.3% | 15.4% | 3.12 | 2.93 |
| llama-3-1-8b | 50% | 600 | 14.2% | 41.7% | 17.4% | 15.7% | 2.73 | 2.76 |
| llama-3-1-8b | 10% | 600 | 15.3% | 40.3% | 16.3% | 14.3% | 2.93 | 2.81 |

**llama-3-1-8b** — Kruskal–Wallis across treatments, restricted to decisions that explicitly mention their own loss probability: H = 4.11, p = 0.128 (n = [89, 85, 92], means = [3.12, 2.73, 2.93]). A non-significant H here means the risk is verbalised but does not move the contribution.

| qwen25-7b-instruct | 90% | 600 | 8.0% | 72.7% | 27.6% | 24.1% | 3.17 | 3.04 |
| qwen25-7b-instruct | 50% | 600 | 8.0% | 68.0% | 27.4% | 26.9% | 3.33 | 3.09 |
| qwen25-7b-instruct | 10% | 600 | 6.0% | 66.7% | 28.7% | 24.8% | 2.67 | 3.04 |

**qwen25-7b-instruct** — Kruskal–Wallis across treatments, restricted to decisions that explicitly mention their own loss probability: H = 4.92, p = 0.0856 (n = [48, 48, 36], means = [3.17, 3.33, 2.67]). A non-significant H here means the risk is verbalised but does not move the contribution.
