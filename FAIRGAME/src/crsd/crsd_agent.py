"""CRSD player state.

A CRSDPlayer is deliberately thin: it records what the player contributed each
round plus auxiliary audit data (the model's raw reasoning text and whether the
contribution had to be recovered by a fallback parse). Savings and payoff are
derived quantities and are computed at settlement time by the game.
"""

from typing import List


class CRSDPlayer:
    def __init__(self, player_id: str, personality: str = "none") -> None:
        """
        Args:
            player_id: Fixed anonymous label shown to everyone, e.g. "Player 3".
            personality: Disposition key ("none", "cooperative", "selfish",
                "risk-averse"). "none" = neutral baseline (no disposition shown),
                which is the apples-to-apples condition vs. Milinski's humans.
        """
        self.id: str = player_id
        self.personality: str = personality
        self.contributions: List[int] = []   # one entry per round, each in {0, 2, 4}
        self.reasonings: List[str] = []       # raw model text per round (audit)
        self.parse_ok: List[bool] = []        # True if the CONTRIBUTION token was found cleanly

    def record(self, contribution: int, reasoning: str, parse_ok: bool) -> None:
        self.contributions.append(int(contribution))
        self.reasonings.append(reasoning)
        self.parse_ok.append(bool(parse_ok))

    @property
    def total_contributed(self) -> int:
        return sum(self.contributions)

    def savings(self, endowment: int) -> int:
        """What stays in the private account: endowment minus everything moved in."""
        return endowment - self.total_contributed

    @property
    def n_parse_fallbacks(self) -> int:
        """How many rounds needed a fallback parse (proxy for model non-compliance)."""
        return sum(1 for ok in self.parse_ok if not ok)
