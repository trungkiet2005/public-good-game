"""Public Goods Game with Punishment — Herrmann, Thoeni & Gaechter (2008, Science).

Implements the repeated linear public-goods game with and without a costly
peer-punishment stage ("Antisocial Punishment Across Societies"), as an N-player,
T-period game played by LLM agents. Groups of 4 play 10 periods under partner
matching; each period every member is endowed with 20 tokens and contributes an
integer amount to a group project with marginal-per-capita return 0.4. In the P
treatment a second stage lets each member assign 0-10 deduction points to each
other member (cost 1 token to the punisher, -3 tokens to the target).

Like the sibling `crsd` module, this does NOT use FAIRGAME's 2-player payoff-matrix
engine. It is a dedicated game loop that *reuses* FAIRGAME's local LLM connector:
the model call is injected as a `responder(prompts) -> responses`, so the engine
imports nothing from FAIRGAME and is fully unit-testable offline without a GPU.

The headline analyses it supports: contribution levels N vs P (Figs 2-3); the
deviation decomposition of punishment into prosocial "punishment of free riding"
vs "antisocial punishment" of equal-or-higher contributors (Fig 1 / Table 2); the
negative antisocial-punishment <-> cooperation correlation (Fig 2B); and the
vengeance channel (punishment received drives punishment assigned next period).
"""
