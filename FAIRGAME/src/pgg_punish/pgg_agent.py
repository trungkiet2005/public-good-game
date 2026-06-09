"""Player state for the public-goods-game-with-punishment (Herrmann et al. 2008).

A PGGPlayer is deliberately thin (like CRSDPlayer): it records, per period, what
the player contributed and — in the punishment (P) treatment — how many deduction
points it assigned to each other member and how many it received in total. Stage-1
income, punishment cost/impact, and final earnings are derived quantities computed
at settlement time by the game.

Indexing convention
-------------------
`punishment_assigned[t]` is a dense list aligned to the *true* player indices of
the group (length n_players), where entry j is the points this player assigned to
player j; the diagonal (own index) is always 0. This makes the per-period 4x4
punishment matrix trivial to assemble for the deviation/antisocial analysis.

`punishment_received[t]` is only the *total* points received that period (the
anonymous number the human subjects saw); the per-source attribution lives in the
game's punishment matrix and is never shown to agents.
"""

from typing import List, Sequence


class PGGPlayer:
    def __init__(self, player_id: str, personality: str = "none",
                 society: str = "none") -> None:
        """
        Args:
            player_id: Fixed *true* label used internally, e.g. "Member 3". Note
                that what other players SEE is a per-period relabelling, so this id
                is never leaked across periods.
            personality: Disposition key ("none", "cooperative", "selfish",
                "vengeful", "norm-enforcer"). "none" = neutral baseline.
            society: Society-persona key (a city name) or "none". "none" = no
                persona shown (apples-to-apples baseline vs. the human pools).
        """
        self.id: str = player_id
        self.personality: str = personality
        self.society: str = society

        # one entry per completed period
        self.contributions: List[int] = []        # tokens placed in the project, in [0, endowment]
        self.stage1_income: List[float] = []       # income before punishment (audit)
        self.earnings: List[float] = []            # income after punishment (final, per period)
        self.punishment_assigned: List[List[int]] = []   # [period] -> points to each true idx (diag 0)
        self.punishment_received: List[int] = []   # [period] -> total points received (anonymous)

        # raw model text + parse-success flags (audit / non-compliance diagnostics)
        self.contrib_reasonings: List[str] = []
        self.punish_reasonings: List[str] = []
        self.contrib_parse_ok: List[bool] = []
        self.punish_parse_ok: List[bool] = []

    # -- recording ----------------------------------------------------------- #
    def record_contribution(self, contribution: int, reasoning: str,
                            parse_ok: bool) -> None:
        self.contributions.append(int(contribution))
        self.contrib_reasonings.append(reasoning)
        self.contrib_parse_ok.append(bool(parse_ok))

    def record_punishment(self, assigned_row: Sequence[int], received_total: int,
                          reasoning: str, parse_ok: bool) -> None:
        self.punishment_assigned.append([int(x) for x in assigned_row])
        self.punishment_received.append(int(received_total))
        self.punish_reasonings.append(reasoning)
        self.punish_parse_ok.append(bool(parse_ok))

    # -- derived ------------------------------------------------------------- #
    @property
    def total_contributed(self) -> int:
        return sum(self.contributions)

    @property
    def total_earnings(self) -> float:
        return sum(self.earnings)

    @property
    def n_contrib_fallbacks(self) -> int:
        """How many periods needed a fallback contribution parse (non-compliance proxy)."""
        return sum(1 for ok in self.contrib_parse_ok if not ok)

    @property
    def n_punish_fallbacks(self) -> int:
        """How many periods needed a fallback punishment parse (non-compliance proxy)."""
        return sum(1 for ok in self.punish_parse_ok if not ok)
