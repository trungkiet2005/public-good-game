"""Public-goods-game-with-punishment engine (Herrmann, Thoeni & Gaechter 2008).

Like the CRSD engine, the model call is INJECTED as
`responder(prompts: List[str]) -> List[str]`, so this module imports nothing from
FAIRGAME and runs fully offline in tests with a scripted responder. In production
the Kaggle cell wires the responder to `send_prompts_global`.

What differs from CRSD
----------------------
* Each period has TWO decision stages: a contribution stage (both treatments) and
  a deduction/punishment stage (treatment P only). The lockstep runner therefore
  makes up to two batched `responder` calls per period.
* The punishment decision is a VECTOR: each player emits one integer per other
  member. So the scattered payload is a List[int] rather than a scalar.
* Earnings accrue every period (no end-of-game probabilistic draw). The per-game
  RNG is used only for the per-period, per-reader relabelling that blocks
  cross-period identity tracking (faithful to the partner design).

Per-period earnings (treatment P), for player i:
  project      = sum_j c_j
  stage1_i     = (endowment - c_i) + mpcr * project
  assigned_i   = sum_{b!=i} P[i][b]        # points i assigned (cost 1 each)
  received_i   = sum_{a!=i} P[a][i]        # points i received (impact 3 each)
  raw_i        = stage1_i - cost*assigned_i - impact*received_i
  earnings_i   = max(0, raw_i) if floor_earnings else raw_i
Treatment N: earnings_i = stage1_i.
"""

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:  # works whether imported as a package (src.pgg_punish) or flat (tests/Kaggle)
    from .pgg_agent import PGGPlayer
    from .pgg_prompt import (
        CONTRIB_KEYS, PUNISH_KEYS, action_desc, build_prompt, format_contrib_history,
        format_others_block, format_punish_history, parse_contribution,
        parse_punishment, personality_block, received_last_period_block,
        society_block, treatment_block,
    )
except ImportError:  # pragma: no cover - import shim
    from pgg_agent import PGGPlayer
    from pgg_prompt import (
        CONTRIB_KEYS, PUNISH_KEYS, action_desc, build_prompt, format_contrib_history,
        format_others_block, format_punish_history, parse_contribution,
        parse_punishment, personality_block, received_last_period_block,
        society_block, treatment_block,
    )

Responder = Callable[[List[str]], List[str]]


def _slot_label(slot: int) -> str:
    """Temporary display label for the slot-th other member: 0 -> 'Member A'."""
    return f"Member {chr(ord('A') + slot)}"


class PGGGame:
    """A single 4-member, 10-period public-goods game (treatment N or P)."""

    def __init__(self, game_id: str, language: str, personality_condition: str,
                 personalities: Sequence[str], society: str,
                 contrib_template: str, punish_template: str, params: Dict) -> None:
        self.game_id = game_id
        self.language = language
        self.personality_condition = personality_condition
        self.society = society
        self.params = params
        self.contrib_template = contrib_template
        self.punish_template = punish_template

        self.treatment: str = params["treatment"]              # "N" or "P"
        self.n_players: int = params["n_players"]              # 4
        self.n_periods: int = params["n_periods"]             # 10
        self.endowment: int = params["endowment"]             # 20
        self.mpcr: float = params["mpcr"]                     # 0.4
        self.contrib_min: int = params["contrib_min"]         # 0
        self.contrib_max: int = params["contrib_max"]         # 20
        self.options: Sequence[int] = params.get("options") or tuple(
            range(self.contrib_min, self.contrib_max + 1))
        self.max_punish: int = params["max_punish"]           # 10
        self.punish_cost: int = params["punish_cost"]         # 1
        self.punish_impact: int = params["punish_impact"]     # 3
        self.relabel_others: bool = params.get("relabel_others", True)
        self.show_received: bool = params.get("show_received", True)
        self.floor_earnings: bool = params.get("floor_earnings", True)

        assert len(personalities) == self.n_players, "need one personality per player"
        self.players: List[PGGPlayer] = [
            PGGPlayer(f"Member {i + 1}", personalities[i], society)
            for i in range(self.n_players)
        ]

        # phase machine
        self.current_period: int = 0           # 0-based index of the period being played
        self.phase: str = "contrib"            # "contrib" -> "punish" (P) -> settle -> "contrib"

        # transient per-period buffers (cleared each period)
        self._period_contribs: Optional[List[int]] = None     # true-idx aligned
        self._period_project: Optional[int] = None
        self._period_stage1: Optional[List[float]] = None
        self._period_punish: Optional[List[List[int]]] = None  # P[a][b], diag 0

        # completed-period records (for history rendering + result assembly)
        self.contrib_history_raw: List[List[int]] = []        # per period, true-idx aligned
        self.punish_matrix_history: List[List[List[int]]] = []  # per period (P); [] for N
        self._relabel_maps: List[Dict[int, List[int]]] = []   # per period: reader_idx -> [true idx per slot]

        self._rng: random.Random = random.Random(0)           # overridden by the runner
        self.result: Optional[Dict] = None

        # precompute static display strings
        self._action_desc = action_desc(self.language, self.options,
                                        self.contrib_min, self.contrib_max)
        self._other_labels = ", ".join(_slot_label(s) for s in range(self.n_players - 1))

    # -- static display helpers --------------------------------------------- #
    def _others_true_order(self, reader_idx: int) -> List[int]:
        return [j for j in range(self.n_players) if j != reader_idx]

    def _build_relabel_map(self) -> Dict[int, List[int]]:
        m: Dict[int, List[int]] = {}
        for i in range(self.n_players):
            others = self._others_true_order(i)
            if self.relabel_others:
                self._rng.shuffle(others)
            m[i] = others
        return m

    def _ensure_relabel_map(self) -> None:
        """Build (once) the relabelling for the current period."""
        while len(self._relabel_maps) <= self.current_period:
            self._relabel_maps.append(self._build_relabel_map())

    # -- history shaping (per reader, using each period's temporary labels) -- #
    def _contrib_history_for(self, reader_idx: int) -> List[Dict]:
        out = []
        for k in range(self.current_period):          # completed periods only
            relab = self._relabel_maps[k][reader_idx]
            contribs = self.contrib_history_raw[k]
            others = [(_slot_label(s), contribs[true_j]) for s, true_j in enumerate(relab)]
            out.append({
                "others": others,
                "you": contribs[reader_idx],
                "project_total": sum(contribs),
                "my_income": self.players[reader_idx].earnings[k],
            })
        return out

    def _punish_history_for(self, reader_idx: int) -> List[Dict]:
        out = []
        p = self.players[reader_idx]
        for k in range(self.current_period):
            if k >= len(p.punishment_assigned):
                break
            relab = self._relabel_maps[k][reader_idx]
            assigned_row = p.punishment_assigned[k]
            assigned = [(_slot_label(s), assigned_row[true_j]) for s, true_j in enumerate(relab)]
            out.append({"assigned": assigned, "received_total": p.punishment_received[k]})
        return out

    # -- contribution phase -------------------------------------------------- #
    def build_contrib_prompts(self) -> List[str]:
        self._ensure_relabel_map()
        tblock = treatment_block(self.language, self.treatment,
                                 self.punish_cost, self.punish_impact)
        prompts = []
        for i, p in enumerate(self.players):
            mapping = {
                "PLAYER_ID": p.id,
                "N_PLAYERS": self.n_players,
                "N_ROUNDS": self.n_periods,
                "ENDOWMENT": self.endowment,
                "MPCR": self.mpcr,
                "ACTION_DESC": self._action_desc,
                "OTHER_LABELS": self._other_labels,
                "CONTRIB_MIN": self.contrib_min,
                "CONTRIB_MAX": self.contrib_max,
                "CURRENT_ROUND": self.current_period + 1,
                "TREATMENT_BLOCK": tblock,
                "SOCIETY_BLOCK": society_block(self.language, p.society),
                "PERSONALITY_BLOCK": personality_block(self.language, p.personality),
                "HISTORY_BLOCK": format_contrib_history(
                    self._contrib_history_for(i), p.id, self.language),
            }
            prompts.append(build_prompt(self.contrib_template, mapping, CONTRIB_KEYS))
        return prompts

    def ingest_contributions(self, contributions: Sequence[int],
                             reasonings: Sequence[str], parse_ok: Sequence[bool]) -> None:
        clamped = [max(self.contrib_min, min(int(c), self.contrib_max)) for c in contributions]
        for p, c, r, ok in zip(self.players, clamped, reasonings, parse_ok):
            p.record_contribution(c, r, ok)
        self._period_contribs = clamped
        self._period_project = sum(clamped)
        self._period_stage1 = [
            (self.endowment - clamped[i]) + self.mpcr * self._period_project
            for i in range(self.n_players)
        ]
        if self.treatment == "P":
            self._period_punish = [[0] * self.n_players for _ in range(self.n_players)]
            self.phase = "punish"
        else:
            self.settle_period()

    # -- punishment phase (P only) ------------------------------------------ #
    def build_punish_prompts(self) -> List[str]:
        assert self.treatment == "P" and self.phase == "punish"
        relmap = self._relabel_maps[self.current_period]
        prompts = []
        for i, p in enumerate(self.players):
            others = [(_slot_label(s), self._period_contribs[true_j])
                      for s, true_j in enumerate(relmap[i])]
            recv_last = (p.punishment_received[self.current_period - 1]
                         if (self.show_received and self.current_period > 0
                             and len(p.punishment_received) >= self.current_period)
                         else None)
            mapping = {
                "PLAYER_ID": p.id,
                "CURRENT_ROUND": self.current_period + 1,
                "N_ROUNDS": self.n_periods,
                "MY_CONTRIB": self._period_contribs[i],
                "OTHERS_BLOCK": format_others_block(others, self.language),
                "GROUP_PROJECT_TOTAL": self._period_project,
                "MY_INCOME_THIS_PERIOD": f"{self._period_stage1[i]:g}",
                "RECEIVED_LAST_PERIOD": received_last_period_block(self.language, recv_last),
                "PUNISH_COST": self.punish_cost,
                "PUNISH_IMPACT": self.punish_impact,
                "MAX_PUNISH": self.max_punish,
                "SOCIETY_BLOCK": society_block(self.language, p.society),
                "PERSONALITY_BLOCK": personality_block(self.language, p.personality),
                "PUNISH_HISTORY_BLOCK": format_punish_history(
                    self._punish_history_for(i), self.language),
            }
            prompts.append(build_prompt(self.punish_template, mapping, PUNISH_KEYS))
        return prompts

    def ingest_punishment(self, punish_vectors: Sequence[Sequence[int]],
                          reasonings: Sequence[str], parse_ok: Sequence[bool]) -> None:
        assert self.treatment == "P" and self.phase == "punish"
        relmap = self._relabel_maps[self.current_period]
        for i in range(self.n_players):
            vec = punish_vectors[i]
            for slot, true_j in enumerate(relmap[i]):
                pts = 0
                if slot < len(vec):
                    pts = max(0, min(int(vec[slot]), self.max_punish))
                self._period_punish[i][true_j] = pts            # diagonal stays 0 (i never in relmap[i])
        # record per-player rows + anonymous received totals
        for i, p in enumerate(self.players):
            received_total = sum(self._period_punish[a][i] for a in range(self.n_players))
            p.record_punishment(self._period_punish[i], received_total,
                                reasonings[i], parse_ok[i])
        self.settle_period()

    # -- per-period settlement ---------------------------------------------- #
    def settle_period(self) -> None:
        stage1 = self._period_stage1
        if self.treatment == "P":
            P = self._period_punish
            assigned = [sum(P[i][b] for b in range(self.n_players)) for i in range(self.n_players)]
            received = [sum(P[a][i] for a in range(self.n_players)) for i in range(self.n_players)]
        else:
            P = None
            assigned = [0] * self.n_players
            received = [0] * self.n_players

        for i, p in enumerate(self.players):
            raw = stage1[i] - self.punish_cost * assigned[i] - self.punish_impact * received[i]
            earn = max(0.0, raw) if self.floor_earnings else raw
            p.stage1_income.append(float(stage1[i]))
            p.earnings.append(float(earn))

        self.contrib_history_raw.append(list(self._period_contribs))
        self.punish_matrix_history.append([row[:] for row in P] if P is not None else [])

        # advance
        self.current_period += 1
        self.phase = "contrib"
        self._period_contribs = self._period_project = self._period_stage1 = None
        self._period_punish = None

    # -- end-of-game result assembly ---------------------------------------- #
    def finalize(self) -> Dict:
        T = self.n_periods
        players_out = []
        for p in self.players:
            players_out.append({
                "id": p.id,
                "personality": p.personality,
                "society": p.society,
                "contributions": list(p.contributions),
                "stage1_income": [round(x, 4) for x in p.stage1_income],
                "earnings": [round(x, 4) for x in p.earnings],
                "total_contributed": p.total_contributed,
                "total_earnings": round(p.total_earnings, 4),
                "punishment_assigned": [list(r) for r in p.punishment_assigned],
                "punishment_received": list(p.punishment_received),
                "n_contrib_fallbacks": p.n_contrib_fallbacks,
                "n_punish_fallbacks": p.n_punish_fallbacks,
                "contrib_reasonings": list(p.contrib_reasonings),
                "punish_reasonings": list(p.punish_reasonings),
            })

        contributions_by_period = [list(row) for row in self.contrib_history_raw]
        stage1_by_period = [[round(self.players[i].stage1_income[t], 4)
                             for i in range(self.n_players)] for t in range(T)]
        earnings_by_period = [[round(self.players[i].earnings[t], 4)
                               for i in range(self.n_players)] for t in range(T)]
        mean_contribution_by_period = [sum(row) / len(row) for row in contributions_by_period]

        if self.treatment == "P":
            received_by_period = [[self.players[i].punishment_received[t]
                                   for i in range(self.n_players)] for t in range(T)]
            punishment_matrix_by_period = [[list(r) for r in mat]
                                           for mat in self.punish_matrix_history]
        else:
            received_by_period = []
            punishment_matrix_by_period = []

        relabel_maps_serializable = [{str(i): m[i] for i in m} for m in self._relabel_maps]

        self.result = {
            "game_id": self.game_id,
            "treatment": self.treatment,
            "language": self.language,
            "personality_condition": self.personality_condition,
            "society": self.society,
            "n_players": self.n_players,
            "n_periods": self.n_periods,
            "endowment": self.endowment,
            "mpcr": self.mpcr,
            "max_punish": self.max_punish,
            "punish_cost": self.punish_cost,
            "punish_impact": self.punish_impact,
            "floor_earnings": self.floor_earnings,
            "contributions_by_period": contributions_by_period,
            "stage1_income_by_period": stage1_by_period,
            "earnings_by_period": earnings_by_period,
            "punishment_matrix_by_period": punishment_matrix_by_period,
            "punishment_received_by_period": received_by_period,
            "relabel_maps": relabel_maps_serializable,
            "mean_contribution_by_period": mean_contribution_by_period,
            "players": players_out,
        }
        return self.result


# --------------------------------------------------------------------------- #
# Batched generation + parsing with targeted retries (parser injected so the same
# retry machinery serves both the contribution and the punishment stage).
# --------------------------------------------------------------------------- #
def _generate_and_parse(prompts: List[str], responder: Responder,
                        parser: Callable[[str], Tuple[object, bool]],
                        max_parse_retries: int):
    responses = list(responder(prompts)) if prompts else []
    values: List[object] = [None] * len(prompts)
    parse_ok: List[bool] = [False] * len(prompts)
    pending: List[int] = []
    for i, resp in enumerate(responses):
        v, ok = parser(resp)
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
            v, ok = parser(retry_responses[j])
            if ok:
                values[i], parse_ok[i], responses[i] = v, True, retry_responses[j]
            else:
                still_pending.append(i)
        pending = still_pending
    return responses, values, parse_ok


def _bucket(index: List[Tuple[int, int]], responses, values, parse_ok):
    buckets: Dict[int, Dict[int, tuple]] = {}
    for (gi, pi), resp, val, ok in zip(index, responses, values, parse_ok):
        buckets.setdefault(gi, {})[pi] = (val, resp, ok)
    return buckets


def run_games_lockstep(games: List[PGGGame], responder: Responder, *,
                       rng: random.Random, max_parse_retries: int = 2,
                       progress: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
    """Step every game forward in lockstep. Per period: one contribution batch over
    ALL games, then (if any) one punishment batch over the P games only.
    """
    if not games:
        return []
    n_periods = games[0].n_periods
    assert all(g.n_periods == n_periods for g in games), "all games must share n_periods"

    # deterministic per-game RNG (relabelling + reproducibility), mirroring CRSD.
    for g in games:
        g._rng = random.Random(rng.random())

    for t in range(n_periods):
        # ---------- PHASE 1: contribution (all games) ---------- #
        prompts: List[str] = []
        index: List[Tuple[int, int]] = []
        for gi, g in enumerate(games):
            for pi, pr in enumerate(g.build_contrib_prompts()):
                prompts.append(pr)
                index.append((gi, pi))

        responses, values, oks = _generate_and_parse(
            prompts, responder,
            lambda txt: parse_contribution(txt, games[0].options),
            max_parse_retries)
        buckets = _bucket(index, responses, values, oks)
        for gi, g in enumerate(games):
            b = buckets[gi]
            g.ingest_contributions(
                [b[pi][0] for pi in range(g.n_players)],
                [b[pi][1] for pi in range(g.n_players)],
                [b[pi][2] for pi in range(g.n_players)])

        # ---------- PHASE 2: punishment (P games only) ---------- #
        p_games = [(gi, g) for gi, g in enumerate(games) if g.treatment == "P"]
        if p_games:
            n_targets = games[0].n_players - 1
            max_points = games[0].max_punish
            prompts, index = [], []
            for gi, g in p_games:
                for pi, pr in enumerate(g.build_punish_prompts()):
                    prompts.append(pr)
                    index.append((gi, pi))

            responses, values, oks = _generate_and_parse(
                prompts, responder,
                lambda txt: parse_punishment(txt, n_targets, max_points),
                max_parse_retries)
            buckets = _bucket(index, responses, values, oks)
            for gi, g in p_games:
                b = buckets[gi]
                g.ingest_punishment(
                    [b[pi][0] for pi in range(g.n_players)],
                    [b[pi][1] for pi in range(g.n_players)],
                    [b[pi][2] for pi in range(g.n_players)])

        if progress:
            progress(t + 1, n_periods)

    return [g.finalize() for g in games]
