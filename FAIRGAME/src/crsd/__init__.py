"""
CRSD: Collective-Risk Social Dilemma module for FAIRGAME.

Implements the Milinski et al. (2008, PNAS) collective-risk social dilemma as an
N-player, T-round, threshold + probabilistic-loss public-goods game. This does
NOT use FAIRGAME's 2-player payoff-matrix engine (which cannot express the
cumulative threshold or the end-of-game probabilistic settlement); instead it is
a dedicated game loop that *reuses* FAIRGAME's local LLM connector (the actual
model call is injected as a `responder`, so the engine here has zero FAIRGAME
imports and is fully unit-testable without a GPU).

Reference: Milinski M, Sommerfeld RD, Krambeck H-J, Reed FA, Marotzke J (2008)
"The collective-risk social dilemma and the prevention of simulated dangerous
climate change." PNAS 105(7):2291-2294.
"""
