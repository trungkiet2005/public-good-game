"""Unit tests for the pgg_punish module — pure logic, no GPU/model required.

Run directly:   python unit_tests/test_pgg_punish.py
Or with pytest: pytest unit_tests/test_pgg_punish.py
"""

import random
import sys
from pathlib import Path

_PGG_DIR = Path(__file__).resolve().parent.parent / "src" / "pgg_punish"
sys.path.insert(0, str(_PGG_DIR))

import pgg_results  # noqa: E402
from pgg_game import PGGGame, run_games_lockstep  # noqa: E402
from pgg_prompt import parse_contribution, parse_punishment  # noqa: E402

_TPL_DIR = Path(__file__).resolve().parent.parent / "resources" / "pgg_punish_templates"
_CONTRIB_EN = (_TPL_DIR / "pgg_contrib_en.txt").read_text(encoding="utf-8")
_PUNISH_EN = (_TPL_DIR / "pgg_punish_en.txt").read_text(encoding="utf-8")


def _params(treatment="P", relabel=True, floor=True, max_punish=10):
    return dict(treatment=treatment, n_players=4, n_periods=10, endowment=20, mpcr=0.4,
                contrib_min=0, contrib_max=20, options=None, max_punish=max_punish,
                punish_cost=1, punish_impact=3, relabel_others=relabel,
                show_received=True, floor_earnings=floor)


def _make_game(treatment="P", relabel=True, floor=True, n_periods=10, gid="g",
               personalities=None, society="none"):
    p = _params(treatment, relabel, floor)
    p["n_periods"] = n_periods
    pers = personalities or ["none"] * 4
    return PGGGame(gid, "en", "neutral", pers, society, _CONTRIB_EN, _PUNISH_EN, p)


# --------------------------------------------------------------------------- #
# Parsing — contribution (full 0..20 range)
# --------------------------------------------------------------------------- #
def test_parse_contribution_token():
    assert parse_contribution(">>> CONTRIBUTION = 17") == (17, True)
    assert parse_contribution("reason\n>>> CONTRIBUTION = 0\n") == (0, True)
    assert parse_contribution("**>>> CONTRIBUTION = 20**") == (20, True)
    # last legal token wins
    assert parse_contribution(">>> CONTRIBUTION = 4\n>>> CONTRIBUTION = 9") == (9, True)


def test_parse_contribution_fallback_and_range():
    # no token but a usable in-range int -> fallback
    assert parse_contribution("I will give 12 this round.") == (12, False)
    # nothing usable -> default 0
    assert parse_contribution("no numbers here") == (0, False)
    # out-of-range token (25) is rejected; no usable fallback -> default 0
    assert parse_contribution(">>> CONTRIBUTION = 25") == (0, False)


# --------------------------------------------------------------------------- #
# Parsing — punishment (vector of 3 in [0,10])
# --------------------------------------------------------------------------- #
def test_parse_punishment_token():
    assert parse_punishment(">>> DEDUCT: A=0 B=5 C=10") == ([0, 5, 10], True)
    assert parse_punishment("**>>> DEDUCT: A=2 B=3 C=4**") == ([2, 3, 4], True)
    assert parse_punishment(">>> DEDUCT: A=1, B=2, C=3") == ([1, 2, 3], True)


def test_parse_bare_token_no_prefix():
    # Templates now emit bare tokens (the ">>>" prefix was removed); both stages
    # must still parse as the primary token, not fall back.
    assert parse_contribution("reasoning...\nCONTRIBUTION = 14") == (14, True)
    assert parse_punishment("plan\nDEDUCT: A=2 B=0 C=1") == ([2, 0, 1], True)
    assert parse_punishment("DEDUCT: Member A=1 Member B=2 Member C=3") == ([1, 2, 3], True)


def test_parse_punishment_fallback_clamp_default():
    # missing >>> token, but A=/B=/C= present -> fallback
    assert parse_punishment("Member A=5 Member B=0 Member C=0") == ([5, 0, 0], False)
    # wrong count (no C) -> primary fails, fallback fills C=0
    assert parse_punishment(">>> DEDUCT: A=1 B=2") == ([1, 2, 0], False)
    # out of range -> clamped, not primary_ok
    assert parse_punishment(">>> DEDUCT: A=15 B=0 C=0") == ([10, 0, 0], False)
    # nothing parseable -> all zero (no fabricated punishment)
    assert parse_punishment("I prefer not to deduct anything") == ([0, 0, 0], False)


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def test_contrib_prompt_fills_placeholders_and_treatment_block():
    gp = _make_game("P")
    prompts = gp.build_contrib_prompts()
    p0 = prompts[0]
    assert "{" not in p0 and "}" not in p0
    assert "Member A, Member B, Member C" in p0
    assert "period 1 of 10" in p0
    assert "CONTRIBUTION = X" in p0          # bare machine token (no ">>>" prefix)
    assert ">>>" not in p0                    # decoration/prefix removed from template
    assert "deduction points" in p0          # treatment notice present in P

    gn = _make_game("N")
    n0 = gn.build_contrib_prompts()[0]
    assert "{" not in n0 and "}" not in n0
    assert "deduction points" not in n0       # no punishment notice in N


def test_punish_prompt_built_after_contributions():
    gp = _make_game("P", relabel=False)
    gp._rng = random.Random(0)
    gp.build_contrib_prompts()
    gp.ingest_contributions([4, 8, 12, 16], [""] * 4, [True] * 4)
    pp = gp.build_punish_prompts()[0]
    assert "{" not in pp and "}" not in pp
    assert "DEDUCT: A=a B=b C=c" in pp        # bare machine token (no ">>>" prefix)
    assert ">>>" not in pp
    assert "Member A" in pp


# --------------------------------------------------------------------------- #
# Settlement math
# --------------------------------------------------------------------------- #
def test_stage1_income_and_N_earnings():
    g = _make_game("N", relabel=False, n_periods=1)
    g._rng = random.Random(0)
    g.build_contrib_prompts()
    g.ingest_contributions([10, 10, 10, 10], [""] * 4, [True] * 4)   # settles immediately (N)
    res = g.finalize()
    # project=40, stage1 = (20-10)+0.4*40 = 26 for everyone; N: earnings == stage1
    assert res["stage1_income_by_period"][0] == [26, 26, 26, 26]
    assert res["earnings_by_period"][0] == [26, 26, 26, 26]
    assert res["punishment_matrix_by_period"] == []


def test_P_punishment_settlement():
    g = _make_game("P", relabel=False, n_periods=1)
    g._rng = random.Random(0)
    g.build_contrib_prompts()
    g.ingest_contributions([10, 10, 10, 10], [""] * 4, [True] * 4)   # phase -> punish
    # reader0 (others=[1,2,3]) assigns 5 to slot0 (player1); all others assign nothing
    vectors = [[5, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    g.ingest_punishment(vectors, [""] * 4, [True] * 4)
    res = g.finalize()
    # stage1 = 26 each; p0 spent 5 (->21); p1 received 5 (->26-15=11); p2,p3 = 26
    assert res["earnings_by_period"][0] == [21, 11, 26, 26]
    assert res["punishment_matrix_by_period"][0][0][1] == 5
    assert res["punishment_received_by_period"][0] == [0, 5, 0, 0]


def test_floor_rule():
    # contributions all 0 -> stage1 = 20 each. player0 hits player1 with 10 -> -10 raw.
    for floor, expected_p1 in [(True, 0.0), (False, -10.0)]:
        g = _make_game("P", relabel=False, floor=floor, n_periods=1)
        g._rng = random.Random(0)
        g.build_contrib_prompts()
        g.ingest_contributions([0, 0, 0, 0], [""] * 4, [True] * 4)
        g.ingest_punishment([[10, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [""] * 4, [True] * 4)
        res = g.finalize()
        assert res["earnings_by_period"][0][1] == expected_p1
        assert res["earnings_by_period"][0][0] == 10.0   # 20 - 10 assigned


# --------------------------------------------------------------------------- #
# Relabelling roundtrip
# --------------------------------------------------------------------------- #
def test_relabel_roundtrip_maps_slots_to_true_targets():
    g = _make_game("P", relabel=True, n_periods=1)
    g._rng = random.Random(123)
    g.build_contrib_prompts()                     # builds relabel map for period 0
    relmap = {i: list(v) for i, v in g._relabel_maps[0].items()}
    g.ingest_contributions([0, 5, 10, 15], [""] * 4, [True] * 4)
    # each reader assigns [1,2,3] in slot order
    g.ingest_punishment([[1, 2, 3]] * 4, [""] * 4, [True] * 4)
    mat = g.finalize()["punishment_matrix_by_period"][0]
    for i in range(4):
        assert mat[i][i] == 0                     # no self-punishment
        for slot, true_j in enumerate(relmap[i]):
            assert mat[i][true_j] == slot + 1     # slot 0->1, 1->2, 2->3


def test_relabel_disabled_is_identity_order():
    g = _make_game("P", relabel=False, n_periods=1)
    g._rng = random.Random(7)
    g.build_contrib_prompts()
    assert g._relabel_maps[0][0] == [1, 2, 3]
    assert g._relabel_maps[0][1] == [0, 2, 3]
    assert g._relabel_maps[0][3] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Deviation binning / antisocial split
# --------------------------------------------------------------------------- #
def test_deviation_binning_and_split():
    g = _make_game("P", relabel=False, n_periods=1)
    g._rng = random.Random(0)
    g.build_contrib_prompts()
    # contributions: punisher p0=4; targets p1=18 (dev +14), p2=0 (dev -4), p3=4 (dev 0)
    g.ingest_contributions([4, 18, 0, 4], [""] * 4, [True] * 4)
    # p0 assigns to [1,2,3] = [2,3,1]; others assign nothing
    g.ingest_punishment([[2, 3, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [""] * 4, [True] * 4)
    res = g.finalize()
    bins = pgg_results.deviation_binned_punishment([res])
    assert bins["[11,20]"]["total_expenditure"] == 2     # p0 -> p1 (dev +14), antisocial
    assert bins["[-10,-1]"]["total_expenditure"] == 3    # p0 -> p2 (dev -4), prosocial
    assert bins["[0]"]["total_expenditure"] == 1         # p0 -> p3 (dev 0), antisocial
    assert bins["[11,20]"]["is_antisocial"] is True
    assert bins["[-10,-1]"]["is_antisocial"] is False
    split = pgg_results.antisocial_prosocial_split([res])
    assert split["antisocial_total"] == 3                # 2 + 1
    assert split["prosocial_total"] == 3                 # the single -4 deviation


# --------------------------------------------------------------------------- #
# Lockstep runner — two phases, N + P together
# --------------------------------------------------------------------------- #
def test_run_lockstep_two_phase_scatter():
    def responder(prompts):
        out = []
        for p in prompts:
            if "DEDUCT" in p:
                out.append(">>> DEDUCT: A=1 B=0 C=0")     # 1 point to slot-0 member
            else:
                out.append(">>> CONTRIBUTION = 8")
        return out

    # N game FIRST with max_punish=0 (as in pgg_N.json): regression guard that the
    # runner derives punishment params from a P game, not games[0].
    gN = PGGGame("N0", "en", "neutral", ["none"] * 4, "none",
                 _CONTRIB_EN, _PUNISH_EN, _params("N", relabel=False, max_punish=0))
    gP = PGGGame("P0", "en", "neutral", ["none"] * 4, "none",
                 _CONTRIB_EN, _PUNISH_EN, _params("P", relabel=False))
    gN.n_periods = gP.n_periods = 2
    results = run_games_lockstep([gN, gP], responder, rng=random.Random(1))
    by = {r["game_id"]: r for r in results}

    # N: earnings == stage1 (24.8 each); no punishment
    assert by["N0"]["earnings_by_period"][0] == [24.8, 24.8, 24.8, 24.8]
    assert by["N0"]["punishment_matrix_by_period"] == []
    # P (relabel off): slot-0 targets are p1,p0,p0,p0 -> received p0=3, p1=1
    #   stage1 = (20-8)+0.4*32 = 24.8; assigned 1 each.
    assert by["P0"]["earnings_by_period"][0] == [14.8, 20.8, 23.8, 23.8]
    assert by["P0"]["punishment_received_by_period"][0] == [3, 1, 0, 0]
    assert all(p["n_contrib_fallbacks"] == 0 for p in by["P0"]["players"])
    assert all(p["n_punish_fallbacks"] == 0 for p in by["P0"]["players"])


def test_run_lockstep_retry_then_recover():
    calls = {"n": 0}

    def flaky(prompts):
        calls["n"] += 1
        garbage = calls["n"] == 1
        out = []
        for p in prompts:
            if garbage:
                out.append("still thinking, no token here")
            elif "DEDUCT" in p:
                out.append(">>> DEDUCT: A=0 B=0 C=0")
            else:
                out.append(">>> CONTRIBUTION = 5")
        return out

    g = PGGGame("g", "en", "neutral", ["none"] * 4, "none",
                _CONTRIB_EN, _PUNISH_EN, _params("P", relabel=False))
    g.n_periods = 1
    res = run_games_lockstep([g], flaky, rng=random.Random(0), max_parse_retries=2)[0]
    assert all(p["contributions"][0] == 5 for p in res["players"])
    assert all(p["n_contrib_fallbacks"] == 0 for p in res["players"])
    assert all(p["n_punish_fallbacks"] == 0 for p in res["players"])


def test_run_lockstep_retry_exhausted_falls_back():
    def always_garbage(prompts):
        return ["I won't comply"] * len(prompts)

    g = PGGGame("g", "en", "neutral", ["none"] * 4, "none",
                _CONTRIB_EN, _PUNISH_EN, _params("P", relabel=False))
    g.n_periods = 1
    res = run_games_lockstep([g], always_garbage, rng=random.Random(0), max_parse_retries=1)[0]
    # contributions default to 0 with a fallback flag; punishment defaults to all-zero
    assert all(p["contributions"][0] == 0 for p in res["players"])
    assert all(p["n_contrib_fallbacks"] == 1 for p in res["players"])
    assert all(p["n_punish_fallbacks"] == 1 for p in res["players"])
    assert res["punishment_received_by_period"][0] == [0, 0, 0, 0]


# --------------------------------------------------------------------------- #
# Metrics: vengeance slope and antisocial<->cooperation correlation
# --------------------------------------------------------------------------- #
def test_vengeance_slope_positive():
    # synthetic P game dicts: received_t drives assigned_{t+1}
    def player(recv0, assigned1_sum):
        return {"punishment_received": [recv0, 0],
                "punishment_assigned": [[0, 0, 0], _row(assigned1_sum)]}

    def _row(total):                       # spread `total` across 3 targets
        return [total, 0, 0]

    game = {"treatment": "P",
            "players": [player(0, 0), player(6, 7), player(3, 3), player(9, 9)]}
    v = pgg_results.vengeance_table([game])
    assert v["slope_assigned_on_received"] > 0
    assert v["n_pairs"] == 4


def test_antisocial_vs_cooperation_corr_negative():
    hi = {"treatment": "P", "n_players": 4,
          "contributions_by_period": [[15, 15, 15, 15]],
          "punishment_matrix_by_period": [[[0] * 4 for _ in range(4)]]}
    lo = {"treatment": "P", "n_players": 4,
          "contributions_by_period": [[2, 2, 2, 2]],
          "punishment_matrix_by_period": [[[0, 5, 0, 0], [0, 0, 0, 0],
                                           [0, 0, 0, 0], [0, 0, 0, 0]]]}
    corr = pgg_results.antisocial_vs_cooperation_corr([hi, lo])
    assert corr < 0


def test_mann_whitney_u_separates_and_handles_edges():
    # Clearly separated samples -> small two-sided p.
    lo = [1, 2, 3, 4, 5]
    hi = [10, 11, 12, 13, 14]
    U, p = pgg_results.mann_whitney_u(lo, hi)
    assert p < 0.05
    # Identical samples -> not significant.
    _, p_same = pgg_results.mann_whitney_u([5, 5, 5], [5, 5, 5])
    assert p_same != p_same or p_same > 0.5      # nan (no variance) or large p
    # Degenerate input -> nan, no crash.
    u2, p2 = pgg_results.mann_whitney_u([], [1, 2])
    assert u2 != u2 and p2 != p2


def test_wilcoxon_signed_rank_paired():
    # LLM consistently below human by a fixed offset -> all diffs same sign.
    hum = [18, 16, 14, 12, 10, 8, 6, 5, 4, 3]
    llm = [h - 3 for h in hum]
    W, p = pgg_results.wilcoxon_signed_rank(llm, hum)
    assert p < 0.05                              # systematic level difference
    assert W == 0.0                              # no positive-sign ranks
    # No differences -> nan (test undefined), no crash.
    w2, p2 = pgg_results.wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    assert w2 != w2 and p2 != p2
    # nan pairs are dropped, not propagated.
    w3, p3 = pgg_results.wilcoxon_signed_rank([1, float("nan"), 3], [0, 9, 1])
    assert p3 == p3                              # finite p from the 2 valid pairs


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
            import traceback
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
        else:
            print(f"PASS  {fn.__name__}")
            passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
