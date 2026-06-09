"""Prompt building, history rendering, and robust action parsing for CRSD.

Design notes
------------
* The action set is fixed to {0, 2, 4} (Milinski's €0/€2/€4), so the action list
  is baked into each localized template rather than injected, which avoids having
  to translate the word "or".
* Each model reply must end with a LANGUAGE-AGNOSTIC machine token:
      CONTRIBUTION = X            (X in {0, 2, 4})
  Anchoring on the literal "CONTRIBUTION =" keyword (not a bare number) keeps
  parsing identical across all 5 languages and avoids FAIRGAME's default
  substring match (where "€20" would spuriously match "2"/"0"). A legacy ">>>"
  prefix is still accepted but no longer required (templates no longer emit it).
* "Faithful" visibility (Milinski): each player sees every player's per-round
  contribution and that round's total, but NEVER the cumulative-to-target sum
  (humans tracked that on paper). So `format_history` reports per-round totals
  only; there is no running-total placeholder.
"""

import re
from typing import Dict, List, Sequence, Tuple

# ----------------------------------------------------------------------------- #
# Placeholders filled into a template. ACTION list is intentionally NOT here
# (baked into templates). Keys are replaced by exact-string substitution.
# ----------------------------------------------------------------------------- #
PLACEHOLDER_KEYS = (
    "PLAYER_ID", "N_PLAYERS", "N_ROUNDS", "ENDOWMENT", "TARGET",
    "LOSS_PROB", "KEEP_PROB", "CURRENT_ROUND", "PERSONALITY_BLOCK", "HISTORY_BLOCK",
)

# Localized labels used when rendering the contribution history block.
LABELS: Dict[str, Dict[str, str]] = {
    "en": {"round": "Round", "round_total": "round total", "you": "(you)",
           "no_rounds": "(no rounds have been played yet)"},
    "fr": {"round": "Tour", "round_total": "total du tour", "you": "(vous)",
           "no_rounds": "(aucun tour n'a encore été joué)"},
    "vn": {"round": "Vòng", "round_total": "tổng vòng", "you": "(bạn)",
           "no_rounds": "(chưa có vòng nào được chơi)"},
    "cn": {"round": "轮次", "round_total": "本轮合计", "you": "（你）",
           "no_rounds": "（尚未进行任何轮次）"},
    "ar": {"round": "الجولة", "round_total": "مجموع الجولة", "you": "(أنت)",
           "no_rounds": "(لم تُلعب أي جولة بعد)"},
}

# Optional disposition sentence (the personality manipulation). "none" => "".
PERSONALITY_BLOCKS: Dict[str, Dict[str, str]] = {
    "en": {
        "none": "",
        "cooperative": "Your disposition: you tend to act cooperatively, and you care about the group reaching its target.",
        "selfish": "Your disposition: you tend to act selfishly, and you prioritize keeping your own money.",
        "risk-averse": "Your disposition: you are strongly risk-averse, and you strongly dislike any chance of losing your money.",
    },
    "fr": {
        "none": "",
        "cooperative": "Votre tempérament : vous avez tendance à coopérer, et vous tenez à ce que le groupe atteigne son objectif.",
        "selfish": "Votre tempérament : vous avez tendance à être égoïste, et vous donnez la priorité à la conservation de votre propre argent.",
        "risk-averse": "Votre tempérament : vous êtes fortement averse au risque, et vous détestez toute possibilité de perdre votre argent.",
    },
    "vn": {
        "none": "",
        "cooperative": "Tính cách của bạn: bạn có xu hướng hợp tác, và bạn quan tâm đến việc cả nhóm đạt được mục tiêu.",
        "selfish": "Tính cách của bạn: bạn có xu hướng ích kỷ, và bạn ưu tiên giữ lại tiền của riêng mình.",
        "risk-averse": "Tính cách của bạn: bạn rất ngại rủi ro, và bạn cực kỳ không thích bất kỳ khả năng nào làm mất tiền của mình.",
    },
    "cn": {
        "none": "",
        "cooperative": "你的性格：你倾向于合作，并且你在意小组能否达成目标。",
        "selfish": "你的性格：你倾向于自私，并且你优先保住自己的钱。",
        "risk-averse": "你的性格：你极度厌恶风险，并且你非常不愿意承担任何损失金钱的可能性。",
    },
    "ar": {
        "none": "",
        "cooperative": "طبيعتك: تميل إلى التعاون، ويهمّك أن تبلغ المجموعة هدفها.",
        "selfish": "طبيعتك: تميل إلى الأنانية، وتعطي الأولوية للاحتفاظ بأموالك.",
        "risk-averse": "طبيعتك: أنت شديد النفور من المخاطرة، وتكره بشدة أي احتمال لخسارة أموالك.",
    },
}

# Primary machine token. Tolerant of spacing, a leading €, bold markdown, and an
# OPTIONAL legacy ">>>" prefix (templates now emit a bare "CONTRIBUTION = X").
_PRIMARY = re.compile(r"(?:>>>\s*)?\**\s*CONTRIBUTION\s*\**\s*=\s*\**\s*€?\s*(0|2|4)\b", re.IGNORECASE)


def personality_block(language: str, personality: str) -> str:
    table = PERSONALITY_BLOCKS.get(language, PERSONALITY_BLOCKS["en"])
    return table.get(personality, "")


def parse_contribution(text: str, options: Sequence[int] = (0, 2, 4),
                       default: int = 0) -> Tuple[int, bool]:
    """Extract the chosen contribution from a model reply.

    Returns (value, primary_ok). `primary_ok` is True only when the explicit
    `CONTRIBUTION = X` token was found; the caller treats `not primary_ok`
    as "needs a retry". When the token is absent we still try a best-effort
    fallback (last standalone allowed integer) so a usable value survives even
    if retries are exhausted.
    """
    if text:
        matches = list(_PRIMARY.finditer(text))
        if matches:
            return int(matches[-1].group(1)), True
        opts = set(int(o) for o in options)
        toks = [int(t) for t in re.findall(r"(?<![\d.])(\d+)(?![\d.])", text) if int(t) in opts]
        if toks:
            return toks[-1], False
    return int(default), False


def format_history(history: List[List[int]], player_ids: Sequence[str],
                   you_id: str, language: str = "en") -> str:
    """Render the faithful contribution history (per-round, per-player + round total).

    Args:
        history: list (one entry per *completed* round) of contribution lists,
            aligned to `player_ids`.
        player_ids: fixed labels, e.g. ["Player 1", ..., "Player 6"].
        you_id: the reading player's id (marked "(you)").
        language: label localization.
    """
    lab = LABELS.get(language, LABELS["en"])
    if not history:
        return lab["no_rounds"]
    lines = []
    for r_idx, row in enumerate(history, start=1):
        parts = []
        for pid, contrib in zip(player_ids, row):
            tag = f" {lab['you']}" if pid == you_id else ""
            parts.append(f"{pid}{tag}: €{contrib}")
        rt = sum(row)
        lines.append(f"{lab['round']} {r_idx}:  " + "   ".join(parts)
                     + f"   ({lab['round_total']}: €{rt})")
    return "\n".join(lines)


def build_prompt(template: str, mapping: Dict[str, object]) -> str:
    """Fill a template by exact substitution of each `{KEY}`.

    Uses sequential str.replace (NOT str.format) so stray braces, currency
    symbols, or unmatched keys never raise — important because the rendered
    history and personality text may contain arbitrary characters.
    """
    out = template
    for key in PLACEHOLDER_KEYS:
        out = out.replace("{" + key + "}", str(mapping.get(key, "")))
    return out
