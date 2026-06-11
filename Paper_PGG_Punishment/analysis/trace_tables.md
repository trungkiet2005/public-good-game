# PGG punishment reasoning-trace analysis (EN / neutral / treatment P)

Motive classification of every nonzero punishment decision, cross-tabulated
against where the deduction points actually landed (antisocial = target
contributed >= punisher). 'mis-targeted rhetoric' = the reply claims
free-riding enforcement while >50% of its points hit equal-or-higher
contributors.

| model | punish decisions | anti-dominant | claims free-riding | claims revenge | claims fairness | claims deterrence | anti-dominant GIVEN free-riding claimed |
|---|---|---|---|---|---|---|---|
| gemma2-9b-it | 359 | 51.5% | 38.7% | 28.7% | 17.5% | 29.8% | 33.1% |
| llama-3-1-8b | 339 | 51.6% | 3.2% | 18.0% | 9.4% | 22.4% | 63.6% |
| qwen25-7b-instruct | 383 | 57.7% | 13.8% | 57.4% | 26.1% | 34.5% | 49.1% |

Reading: if 'anti-dominant GIVEN free-riding claimed' stays high, the model verbalises the human prosocial motive while aiming at cooperators — the stated logic and the targeting are decoupled.