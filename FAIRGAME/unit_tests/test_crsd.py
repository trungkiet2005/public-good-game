"""Unit tests for the CRSD module — pure logic, no GPU/model required.

Run directly:   python unit_tests/test_crsd.py
Or with pytest: pytest unit_tests/test_crsd.py
"""

import random
import sys
from pathlib import Path

_CRSD_DIR = Path(__file__).resolve().parent.parent / "src" / "crsd"
sys.path.insert(0, str(_CRSD_DIR))

import crsd_results  # noqa: E402
from crsd_game import CRSDGame, run_games_lockstep  # noqa: E402
from crsd_prompt import build_prompt, format_history, parse_contribution  # noqa: E402

_TEMPLATE_EN = (Path(__file__).resolve().parent.parent
                / "resources" / "crsd_templates" / "crsd_en.txt").read_text(encoding="utf-8")


class FakeRng:
    """random.Random stand-in returning a fixed draw, for deterministic settlement tests."""
    def __init__(self, value):
        self._value = value

    def random(self):
        return self._value


def _make_game(n_rounds=10, loss_prob=90, personalities=None, gid="g"):
    params = dict(n_players=6, n_rounds=n_rounds, endowment=40, target=120,
                  loss_prob=loss_prob, contribution_options=(0, 2, 4))
    pers = personalities or ["none"] * 6
    return CRSDGame(gid, "en", "neutral", pers, _TEMPLATE_EN, params)


def _ingest_constant(game, per_player_round, n_rounds):
    for _ in range(n_rounds):
        game.ingest_round(list(per_player_round), [""] * 6, [True] * 6)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_primary_token():
    assert parse_contribution(">>> CONTRIBUTION = 4") == (4, True)
    assert parse_contribution("reasoning...\n>>> CONTRIBUTION = 0\n") == (0, True)
    assert parse_contribution("**>>> CONTRIBUTION = 2**") == (2, True)


def test_parse_bare_token_no_prefix():
    # Templates now emit a bare "CONTRIBUTION = X" (the ">>>" prefix was removed);
    # it must still parse as the primary token, not fall back.
    assert parse_contribution("I'll cooperate.\nCONTRIBUTION = 4") == (4, True)
    assert parse_contribution("CONTRIBUTION = 2") == (2, True)
    assert parse_contribution("**CONTRIBUTION = 0**") == (0, True)


def test_parse_last_primary_wins():
    assert parse_contribution("I considered 4 then\n>>> CONTRIBUTION = 2") == (2, True)
    assert parse_contribution(">>> CONTRIBUTION = 4\n>>> CONTRIBUTION = 0") == (0, True)


def test_parse_fallback_and_default():
    # No token, but a usable standalone option -> value with primary_ok False
    assert parse_contribution("I will give 2 this round.") == (2, False)
    # Nothing usable -> default 0, primary_ok False
    assert parse_contribution("no numbers at all") == (0, False)
    # "€20" must NOT be read as 2 or 0 (the classic substring trap)
    assert parse_contribution("I have €20 left") == (0, False)


# --------------------------------------------------------------------------- #
# History rendering (faithful: per-round + round total, no cumulative)
# --------------------------------------------------------------------------- #
def test_history_empty():
    assert format_history([], [f"Player {i}" for i in range(1, 7)], "Player 3", "en") \
        == "(no rounds have been played yet)"


def test_history_marks_you_and_round_total():
    ids = [f"Player {i}" for i in range(1, 7)]
    block = format_history([[2, 2, 0, 4, 2, 2]], ids, "Player 3", "en")
    assert "Round 1:" in block
    assert "Player 3 (you): €0" in block
    assert "(round total: €12)" in block
    assert "cumulative" not in block.lower()  # faithful: no running total leaked


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def test_build_prompt_fills_all_placeholders():
    g = _make_game()
    prompts = g.build_round_prompts()
    p = prompts[0]
    assert "Player 1" in p
    assert "{" not in p and "}" not in p  # no leftover placeholders
    assert "round 1 of 10" in p
    assert "probability 90%" in p and "probability 10%" in p  # loss/keep split


def test_framing_recorded_and_defaults_neutral():
    params = dict(n_players=6, n_rounds=2, endowment=40, target=120, loss_prob=90,
                  contribution_options=(0, 2, 4))
    gc = CRSDGame("gc", "en", "neutral", ["none"] * 6, _TEMPLATE_EN, params, framing="climate")
    _ingest_constant(gc, [2] * 6, 2)
    rc = gc.settle(FakeRng(0.5))
    assert rc["framing"] == "climate"
    # framing is optional and defaults to neutral (backward compatible with old callers)
    gn = CRSDGame("gn", "en", "neutral", ["none"] * 6, _TEMPLATE_EN, params)
    _ingest_constant(gn, [2] * 6, 2)
    rn = gn.settle(FakeRng(0.5))
    assert rn["framing"] == "neutral"
    df = crsd_results.to_dataframe([rc, rn])
    assert list(df["framing"]) == ["climate", "neutral"]


def test_climate_template_renders_with_same_placeholders():
    climate_en = (Path(__file__).resolve().parent.parent / "resources"
                  / "crsd_templates" / "crsd_climate_en.txt").read_text(encoding="utf-8")
    params = dict(n_players=6, n_rounds=10, endowment=40, target=120, loss_prob=90,
                  contribution_options=(0, 2, 4))
    g = CRSDGame("g", "en", "neutral", ["none"] * 6, climate_en, params, framing="climate")
    p = g.build_round_prompts()[0]
    assert "{" not in p and "}" not in p          # every placeholder resolved
    assert "climate" in p.lower()                  # climate cover story present
    assert "CONTRIBUTION = X" in p and ">>>" not in p


# --------------------------------------------------------------------------- #
# Settlement
# --------------------------------------------------------------------------- #
def test_settle_all_fair_reaches_target():
    g = _make_game(loss_prob=90)
    _ingest_constant(g, [2] * 6, 10)
    res = g.settle(FakeRng(0.0))
    assert res["group_total"] == 120
    assert res["reached_target"] is True
    assert res["loss_triggered"] is False
    assert all(p["payoff"] == 20 for p in res["players"])  # 40 - 20


def test_settle_all_altruist_reaches_but_zero_savings():
    g = _make_game()
    _ingest_constant(g, [4] * 6, 10)
    res = g.settle(FakeRng(0.0))
    assert res["group_total"] == 240 and res["reached_target"] is True
    assert all(p["payoff"] == 0 for p in res["players"])  # 40 - 40


def test_settle_all_free_rider_loss_vs_keep():
    # Fail target; draw 0.5 < 0.9 -> loss everyone 0
    g_loss = _make_game(loss_prob=90)
    _ingest_constant(g_loss, [0] * 6, 10)
    res_loss = g_loss.settle(FakeRng(0.5))
    assert res_loss["group_total"] == 0 and res_loss["reached_target"] is False
    assert res_loss["loss_triggered"] is True
    assert all(p["payoff"] == 0 for p in res_loss["players"])

    # Same but draw 0.95 >= 0.9 -> no loss, keep full 40
    g_keep = _make_game(loss_prob=90)
    _ingest_constant(g_keep, [0] * 6, 10)
    res_keep = g_keep.settle(FakeRng(0.95))
    assert res_keep["loss_triggered"] is False
    assert all(p["payoff"] == 40 for p in res_keep["players"])


def test_settle_threshold_exact_and_just_miss():
    g_exact = _make_game()
    _ingest_constant(g_exact, [2] * 6, 10)
    assert g_exact.settle(FakeRng(1.0))["reached_target"] is True   # exactly 120

    g_miss = _make_game(loss_prob=10)
    for r in range(9):
        g_miss.ingest_round([2] * 6, [""] * 6, [True] * 6)          # 108
    g_miss.ingest_round([2, 2, 2, 2, 2, 0], [""] * 6, [True] * 6)   # +10 = 118
    res = g_miss.settle(FakeRng(0.05))                              # 0.05 < 0.10 -> loss
    assert res["group_total"] == 118 and res["reached_target"] is False
    assert res["loss_triggered"] is True


# --------------------------------------------------------------------------- #
# Lockstep runner + scatter correctness
# --------------------------------------------------------------------------- #
def test_run_games_lockstep_scatter():
    # Two games; responder returns by batch position: first 6 -> game0 (=4), next 6 -> game1 (=0)
    def responder(prompts):
        out = []
        for i in range(len(prompts)):
            val = 4 if (i // 6) % 2 == 0 else 0
            out.append(f"reasoning\n>>> CONTRIBUTION = {val}")
        return out

    games = [_make_game(n_rounds=2, gid="g0"), _make_game(n_rounds=2, gid="g1")]
    results = run_games_lockstep(games, responder, rng=random.Random(1))
    by_id = {r["game_id"]: r for r in results}
    assert by_id["g0"]["group_total"] == 4 * 6 * 2   # 48
    assert by_id["g1"]["group_total"] == 0
    assert all(p["total_contributed"] == 8 for p in by_id["g0"]["players"])
    assert by_id["g0"]["reached_target"] is False     # 48 < 120
    # every decision parsed cleanly -> no fallbacks
    assert all(p["n_parse_fallbacks"] == 0 for p in by_id["g0"]["players"])


def test_run_games_lockstep_parse_retry_then_fallback():
    calls = {"n": 0}

    def flaky_responder(prompts):
        calls["n"] += 1
        # First call: garbage (forces retries); later calls: valid "2"
        if calls["n"] == 1:
            return ["I am thinking about it"] * len(prompts)
        return [">>> CONTRIBUTION = 2"] * len(prompts)

    g = _make_game(n_rounds=1)
    res = run_games_lockstep([g], flaky_responder, rng=random.Random(0), max_parse_retries=2)[0]
    # retry recovered a clean "2" for everyone
    assert all(p["contributions"][0] == 2 for p in res["players"])
    assert all(p["n_parse_fallbacks"] == 0 for p in res["players"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _result_with(loss_prob, contribs_per_player):
    """Build a settled result dict directly (contribs_per_player: list of 6 lists)."""
    g = CRSDGame("x", "en", "neutral", ["none"] * 6, _TEMPLATE_EN,
                 dict(n_players=6, n_rounds=len(contribs_per_player[0]), endowment=40,
                      target=120, loss_prob=loss_prob, contribution_options=(0, 2, 4)))
    n_rounds = len(contribs_per_player[0])
    for r in range(n_rounds):
        g.ingest_round([contribs_per_player[pi][r] for pi in range(6)], [""] * 6, [True] * 6)
    return g.settle(FakeRng(0.99))


def test_metrics_basic():
    # Game A: all fair (total 120, reached). Game B: all free (total 0).
    a = _result_with(90, [[2] * 10 for _ in range(6)])
    b = _result_with(90, [[0] * 10 for _ in range(6)])
    summary = crsd_results.summarize([a, b])
    s = summary[90]
    assert s["n_games"] == 2
    assert s["success_rate"] == 0.5
    assert abs(s["final_total"]["mean"] - 60.0) < 1e-9
    # fair-share threshold = 20; game A has 6 fair-sharers, game B has 0 -> mean 3.0
    assert abs(s["fair_sharers_per_group"] - 3.0) < 1e-9
    # acts: game A = 60 acts of '2'; game B = 60 acts of '0' (30 each half per game)
    assert s["acts_by_half"]["first"][2] == 30 and s["acts_by_half"]["second"][2] == 30
    assert s["acts_by_half"]["first"][0] == 30
    assert s["round1_distribution"][2] == 6 and s["round1_distribution"][0] == 6
    assert s["parse_fallback_rate"] == 0.0
    # cumulative trajectory of game A alone would be 12,24,...; averaged with B (0): 6,12,...
    assert abs(s["cumulative_trajectory"][0] - 6.0) < 1e-9
    assert abs(s["cumulative_trajectory"][-1] - 60.0) < 1e-9


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"PASS  {fn.__name__}")
            passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)


# --------------------------------------------------------------------------- #
# Follow-up: running-total control (self-summation confound)
# --------------------------------------------------------------------------- #
def test_format_history_running_total():
    h = [[4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2, 2]]
    ids = [f"Player {i + 1}" for i in range(6)]
    faithful = format_history(h, ids, "Player 1", "en")
    assert "cumulative total" not in faithful          # default stays faithful
    shown = format_history(h, ids, "Player 1", "en", show_running_total=True)
    assert "cumulative total invested so far" in shown
    assert "36" in shown.splitlines()[-1]              # 24 + 12


def test_game_show_running_total_param():
    params = dict(n_players=6, n_rounds=10, endowment=40, target=120,
                  loss_prob=90, contribution_options=(0, 2, 4),
                  show_running_total=True)
    g = CRSDGame("g", "en", "neutral", ["none"] * 6, _TEMPLATE_EN, params)
    g.ingest_round([4] * 6, [""] * 6, [True] * 6)
    prompt = g.build_round_prompts()[0]
    assert "cumulative total invested so far" in prompt
    assert "24" in prompt
    _ingest_constant(g, [4] * 6, 9)
    res = g.settle(FakeRng(0.999))
    assert res["show_running_total"] is True


def test_game_default_hides_running_total():
    g = _make_game()
    g.ingest_round([4] * 6, [""] * 6, [True] * 6)
    assert "cumulative total invested so far" not in g.build_round_prompts()[0]
    _ingest_constant(g, [4] * 6, 9)
    assert g.settle(FakeRng(0.999))["show_running_total"] is False
