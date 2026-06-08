"""CRSD game engine: round loop, lockstep batched runner, and settlement.

The model call is INJECTED as `responder(prompts: List[str]) -> List[str]`, so
this module imports nothing from FAIRGAME and runs fully offline in tests with a
scripted responder. In production the Kaggle cell wires the responder to
`send_prompts_global` from FAIRGAME's local LLM connector.

Settlement (Milinski 2008):
  group_total = sum of all contributions over all players and all rounds
  reached     = group_total >= target
  if reached:        each player keeps savings = endowment - own_total
  else:              ONE random draw for the whole group; with prob loss_prob%
                     everyone gets 0, otherwise everyone keeps their savings.
"""

import random
from typing import Callable, Dict, List, Optional, Sequence

try:  # works whether imported as a package (FAIRGAME) or flat (tests/Kaggle)
    from .crsd_agent import CRSDPlayer
    from .crsd_prompt import build_prompt, format_history, parse_contribution, personality_block
except ImportError:  # pragma: no cover - import shim
    from crsd_agent import CRSDPlayer
    from crsd_prompt import build_prompt, format_history, parse_contribution, personality_block

Responder = Callable[[List[str]], List[str]]


class CRSDGame:
    """A single 6-player, 10-round collective-risk game instance."""

    def __init__(self, game_id: str, language: str, personality_condition: str,
                 personalities: Sequence[str], template: str, params: Dict) -> None:
        self.game_id = game_id
        self.language = language
        self.personality_condition = personality_condition
        self.params = params
        self.template = template

        self.n_players: int = params["n_players"]
        self.n_rounds: int = params["n_rounds"]
        self.endowment: int = params["endowment"]
        self.target: int = params["target"]
        self.loss_prob: int = params["loss_prob"]            # 90 / 50 / 10
        self.options: Sequence[int] = params.get("contribution_options", (0, 2, 4))

        assert len(personalities) == self.n_players, "need one personality per player"
        self.players: List[CRSDPlayer] = [
            CRSDPlayer(f"Player {i + 1}", personalities[i]) for i in range(self.n_players)
        ]
        self.history: List[List[int]] = []   # completed rounds; each = contribs aligned to players
        self.current_round: int = 0          # 0-based index of the round about to be played
        self.result: Optional[Dict] = None

    # -- lockstep interface -------------------------------------------------- #
    def build_round_prompts(self) -> List[str]:
        player_ids = [p.id for p in self.players]
        prompts = []
        keep_prob = 100 - self.loss_prob
        for p in self.players:
            mapping = {
                "PLAYER_ID": p.id,
                "N_PLAYERS": self.n_players,
                "N_ROUNDS": self.n_rounds,
                "ENDOWMENT": self.endowment,
                "TARGET": self.target,
                "LOSS_PROB": self.loss_prob,
                "KEEP_PROB": keep_prob,
                "CURRENT_ROUND": self.current_round + 1,
                "PERSONALITY_BLOCK": personality_block(self.language, p.personality),
                "HISTORY_BLOCK": format_history(self.history, player_ids, p.id, self.language),
            }
            prompts.append(build_prompt(self.template, mapping))
        return prompts

    def ingest_round(self, contributions: Sequence[int], reasonings: Sequence[str],
                     parse_ok: Sequence[bool]) -> None:
        for p, c, r, ok in zip(self.players, contributions, reasonings, parse_ok):
            p.record(c, r, ok)
        self.history.append([int(c) for c in contributions])
        self.current_round += 1

    # -- settlement ---------------------------------------------------------- #
    def settle(self, rng: random.Random) -> Dict:
        group_total = sum(p.total_contributed for p in self.players)
        reached = group_total >= self.target
        draw = rng.random()                                  # single draw for the whole group
        loss_triggered = (not reached) and (draw < self.loss_prob / 100.0)

        players_out = []
        for p in self.players:
            savings = p.savings(self.endowment)
            if reached:
                payoff = savings
            else:
                payoff = 0 if loss_triggered else savings
            players_out.append({
                "id": p.id,
                "personality": p.personality,
                "contributions": list(p.contributions),
                "total_contributed": p.total_contributed,
                "savings": savings,
                "payoff": payoff,
                "n_parse_fallbacks": p.n_parse_fallbacks,
                "reasonings": list(p.reasonings),
            })

        self.result = {
            "game_id": self.game_id,
            "treatment_loss_prob": self.loss_prob,
            "language": self.language,
            "personality_condition": self.personality_condition,
            "n_players": self.n_players,
            "n_rounds": self.n_rounds,
            "endowment": self.endowment,
            "target": self.target,
            "group_total": group_total,
            "reached_target": reached,
            "loss_triggered": loss_triggered,
            "random_draw": draw,
            "round_totals": [sum(r) for r in self.history],
            "players": players_out,
        }
        return self.result


def _generate_and_parse(prompts: List[str], responder: Responder,
                        options: Sequence[int], max_parse_retries: int):
    """Batched generation + parsing with targeted retries for unparseable replies."""
    responses = list(responder(prompts))
    values: List[int] = [0] * len(prompts)
    parse_ok: List[bool] = [False] * len(prompts)
    pending: List[int] = []
    for i, resp in enumerate(responses):
        v, ok = parse_contribution(resp, options)
        values[i], parse_ok[i] = v, ok
        if not ok:
            pending.append(i)

    attempt = 0
    while pending and attempt < max_parse_retries:
        attempt += 1
        retry_prompts = [prompts[i] for i in pending]
        retry_responses = list(responder(retry_prompts))
        still_pending: List[int] = []
        for j, i in enumerate(pending):
            v, ok = parse_contribution(retry_responses[j], options)
            if ok:
                values[i], parse_ok[i], responses[i] = v, True, retry_responses[j]
            else:
                still_pending.append(i)
        pending = still_pending
    return responses, values, parse_ok


def run_games_lockstep(games: List[CRSDGame], responder: Responder, *,
                       rng: random.Random, max_parse_retries: int = 2,
                       progress: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
    """Step every game forward in lockstep so the LLM sees one large batch per round.

    All games must share n_rounds (they do: fixed 10). Per round we gather
    n_players prompts from every game into a single `responder` call, then
    scatter the parsed contributions back to each game.
    """
    if not games:
        return []
    n_rounds = games[0].n_rounds
    options = games[0].options
    assert all(g.n_rounds == n_rounds for g in games), "all games must share n_rounds for lockstep"

    for r in range(n_rounds):
        prompts: List[str] = []
        index: List[tuple] = []        # (game_idx, player_idx)
        for gi, g in enumerate(games):
            for pi, p in enumerate(g.build_round_prompts()):
                prompts.append(p)
                index.append((gi, pi))

        responses, values, parse_ok = _generate_and_parse(
            prompts, responder, options, max_parse_retries)

        buckets: Dict[int, Dict[int, tuple]] = {}
        for (gi, pi), resp, val, ok in zip(index, responses, values, parse_ok):
            buckets.setdefault(gi, {})[pi] = (val, resp, ok)
        for gi, g in enumerate(games):
            b = buckets[gi]
            contribs = [b[pi][0] for pi in range(g.n_players)]
            reasonings = [b[pi][1] for pi in range(g.n_players)]
            oks = [b[pi][2] for pi in range(g.n_players)]
            g.ingest_round(contribs, reasonings, oks)

        if progress:
            progress(r + 1, n_rounds)

    results = []
    for gi, g in enumerate(games):
        # deterministic per-game draw: seed from the master rng once, per game
        game_rng = random.Random(rng.random())
        results.append(g.settle(game_rng))
    return results
