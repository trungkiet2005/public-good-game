"""Prompt building, history rendering, and robust action parsing for the
public-goods-game-with-punishment (Herrmann, Thoeni & Gaechter 2008).

Design notes (mirrors crsd_prompt.py, generalised for a TWO-STAGE period)
------------------------------------------------------------------------
* There are two decision stages per period, so there are two templates and two
  parsers: a CONTRIBUTION stage (both treatments) and a DEDUCTION/punishment
  stage (treatment P only).
* The contribution action set is the full integer range [0, 20] by default
  (faithful to the paper), but `options` is configurable (e.g. a coarse discrete
  set for small models). Because the range is wide, the action description is
  INJECTED as {ACTION_DESC} (not baked in) and the parser is range-validated.
* Each model reply must end with a LANGUAGE-AGNOSTIC, ASCII machine token:
      CONTRIBUTION = X              (X an integer in [CONTRIB_MIN, CONTRIB_MAX])
      DEDUCT: A=a B=b C=c            (a,b,c integers in [0, MAX_PUNISH])
  Keeping the tokens in English/ASCII makes parsing identical across all 5
  languages and robust (e.g. "20 tokens" mid-reasoning will not be mis-grabbed).
  A legacy ">>>" prefix is still accepted but no longer required (templates no
  longer emit it).
* Anti-reputation: the three other members are shown under per-period, per-reader
  temporary labels (Member A/B/C). The history is rendered with those temporary
  labels too, so a fixed opponent can never be tracked across periods (this is
  the deliberate divergence from CRSD's fixed-label format_history).
* Wording is kept neutral. The agent-facing text uses "deduction points / reduce
  income" (the paper's own neutral wording), never "punish", so it does NOT prime
  antisocial punishment of cooperators — the very confound the paper measures.
"""

import re
from typing import Dict, List, Sequence, Tuple

# ----------------------------------------------------------------------------- #
# Placeholder keys for the two templates. Replaced by exact-string substitution
# (NOT str.format), so stray braces / currency symbols in rendered history never
# raise. Each template gets its own key tuple.
# ----------------------------------------------------------------------------- #
CONTRIB_KEYS = (
    "PLAYER_ID", "N_PLAYERS", "N_ROUNDS", "ENDOWMENT", "MPCR", "ACTION_DESC",
    "OTHER_LABELS", "CONTRIB_MIN", "CONTRIB_MAX", "CURRENT_ROUND", "TREATMENT_BLOCK",
    "SOCIETY_BLOCK", "PERSONALITY_BLOCK", "HISTORY_BLOCK",
)

PUNISH_KEYS = (
    "PLAYER_ID", "CURRENT_ROUND", "N_ROUNDS", "MY_CONTRIB", "OTHERS_BLOCK",
    "GROUP_PROJECT_TOTAL", "MY_INCOME_THIS_PERIOD", "RECEIVED_LAST_PERIOD",
    "PUNISH_COST", "PUNISH_IMPACT", "MAX_PUNISH", "SOCIETY_BLOCK",
    "PERSONALITY_BLOCK", "PUNISH_HISTORY_BLOCK",
)

# ----------------------------------------------------------------------------- #
# Localized labels used when rendering history / others blocks.
# ----------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, str]] = {
    "en": {"period": "Period", "member": "Member", "you": "You",
           "project_total": "group project total", "your_income": "your income",
           "placed": "placed", "tokens_in_project": "tokens in the project",
           "assigned": "deduction points you assigned", "to": "to",
           "received": "total deduction points you received",
           "no_periods": "(no periods have been played yet)"},
    "fr": {"period": "Période", "member": "Membre", "you": "Vous",
           "project_total": "total du projet de groupe", "your_income": "votre revenu",
           "placed": "a placé", "tokens_in_project": "jetons dans le projet",
           "assigned": "points de déduction que vous avez attribués", "to": "à",
           "received": "total des points de déduction que vous avez reçus",
           "no_periods": "(aucune période n'a encore été jouée)"},
    "vn": {"period": "Kỳ", "member": "Thành viên", "you": "Bạn",
           "project_total": "tổng quỹ chung", "your_income": "thu nhập của bạn",
           "placed": "đã góp", "tokens_in_project": "token vào quỹ chung",
           "assigned": "điểm trừ bạn đã gán", "to": "cho",
           "received": "tổng điểm trừ bạn đã nhận",
           "no_periods": "(chưa có kỳ nào được chơi)"},
    "cn": {"period": "周期", "member": "成员", "you": "你",
           "project_total": "团队项目总额", "your_income": "你的收入",
           "placed": "投入了", "tokens_in_project": "个代币到项目",
           "assigned": "你分配的扣分", "to": "给",
           "received": "你收到的扣分总数",
           "no_periods": "（尚未进行任何周期）"},
    "ar": {"period": "الفترة", "member": "العضو", "you": "أنت",
           "project_total": "إجمالي المشروع الجماعي", "your_income": "دخلك",
           "placed": "وضع", "tokens_in_project": "رمزًا في المشروع",
           "assigned": "نقاط الخصم التي خصّصتها", "to": "إلى",
           "received": "إجمالي نقاط الخصم التي تلقيتها",
           "no_periods": "(لم تُلعب أي فترة بعد)"},
}

# ----------------------------------------------------------------------------- #
# Optional disposition sentence (the personality manipulation). "none" => "".
# Wording is general and NEVER references punishing high contributors / punishers,
# so it does not pre-bake the antisocial-punishment finding.
# ----------------------------------------------------------------------------- #
PERSONALITY_BLOCKS: Dict[str, Dict[str, str]] = {
    "en": {
        "none": "",
        "cooperative": "Your disposition: you tend to act cooperatively, and you care about the group doing well as a whole.",
        "selfish": "Your disposition: you tend to act in your own interest, and you prioritize your own token income.",
        "vengeful": "Your disposition: you tend to respond in kind, and you are inclined to answer how others have treated you.",
        "norm-enforcer": "Your disposition: you care about fairness, and you are inclined to hold members accountable when they do not contribute their share.",
    },
    "fr": {
        "none": "",
        "cooperative": "Votre tempérament : vous avez tendance à coopérer, et le bien-être du groupe dans son ensemble vous tient à cœur.",
        "selfish": "Votre tempérament : vous avez tendance à agir dans votre propre intérêt, et vous donnez la priorité à votre propre revenu en jetons.",
        "vengeful": "Votre tempérament : vous avez tendance à rendre la pareille, et vous êtes enclin à répondre à la façon dont les autres vous ont traité.",
        "norm-enforcer": "Votre tempérament : l'équité vous importe, et vous êtes enclin à demander des comptes aux membres qui ne contribuent pas leur juste part.",
    },
    "vn": {
        "none": "",
        "cooperative": "Tính cách của bạn: bạn có xu hướng hợp tác, và bạn quan tâm đến việc cả nhóm cùng tốt lên.",
        "selfish": "Tính cách của bạn: bạn có xu hướng hành động vì lợi ích bản thân, và bạn ưu tiên thu nhập token của riêng mình.",
        "vengeful": "Tính cách của bạn: bạn có xu hướng đáp lại tương xứng, và bạn nghiêng về việc đáp lại cách người khác đã đối xử với bạn.",
        "norm-enforcer": "Tính cách của bạn: bạn coi trọng sự công bằng, và bạn nghiêng về việc buộc các thành viên chịu trách nhiệm khi họ không đóng góp phần của mình.",
    },
    "cn": {
        "none": "",
        "cooperative": "你的性格：你倾向于合作，并且你在意整个团队的整体表现。",
        "selfish": "你的性格：你倾向于从自身利益出发，并且优先考虑自己的代币收入。",
        "vengeful": "你的性格：你倾向于以同样方式回应，并且会根据别人如何对待你来作出回应。",
        "norm-enforcer": "你的性格：你重视公平，并倾向于在成员未尽其本分时让他们承担责任。",
    },
    "ar": {
        "none": "",
        "cooperative": "طبيعتك: تميل إلى التعاون، ويهمّك أن تكون حال المجموعة ككل جيدة.",
        "selfish": "طبيعتك: تميل إلى التصرف وفق مصلحتك الخاصة، وتعطي الأولوية لدخلك من الرموز.",
        "vengeful": "طبيعتك: تميل إلى الرد بالمثل، وتنزع إلى الرد على الطريقة التي عاملك بها الآخرون.",
        "norm-enforcer": "طبيعتك: تهتم بالإنصاف، وتنزع إلى محاسبة الأعضاء عندما لا يسهمون بنصيبهم.",
    },
}

# ----------------------------------------------------------------------------- #
# Society-persona sentence. A NEUTRAL location label only — no behavioural
# stereotype is injected. The point is to probe whatever priors the model itself
# holds about each pool, then compare to Herrmann's human ranking. "none" => "".
# (One table reused for all languages: the persona is stated in-language via the
# city/country names, which are kept in Latin script for parsing safety.)
# ----------------------------------------------------------------------------- #
SOCIETY_COUNTRY: Dict[str, str] = {
    "Boston": "the United States", "Melbourne": "Australia", "Nottingham": "the United Kingdom",
    "St.Gallen": "Switzerland", "Zurich": "Switzerland", "Copenhagen": "Denmark",
    "Bonn": "Germany", "Seoul": "South Korea", "Chengdu": "China", "Minsk": "Belarus",
    "Samara": "Russia", "Dnipropetrovsk": "Ukraine", "Istanbul": "Turkey",
    "Athens": "Greece", "Riyadh": "Saudi Arabia", "Muscat": "Oman",
}

_SOCIETY_SENTENCE: Dict[str, str] = {
    "en": "You are a participant from {city}, {country}.",
    "fr": "Vous êtes un participant de {city}, {country}.",
    "vn": "Bạn là một người tham gia đến từ {city}, {country}.",
    "cn": "你是来自{city}（{country}）的一名参与者。",
    "ar": "أنت مشارك من {city}، {country}.",
}

# ----------------------------------------------------------------------------- #
# NORM-LADEN personas (POSITIVE CONTROL for the cross-cultural null).
# Used when the society key is "norm:<City>". Unlike the neutral label above,
# these carry explicit civic-norm content modelled on the constructs Herrmann
# et al. (2008) link to cross-societal punishment differences (norms of civic
# cooperation, rule of law, trust in institutions, attitudes to sanctioning and
# revenge). They describe the SOCIETY the participant comes from; they never
# mention this game, contributions, deduction points, or any target behaviour,
# so they prime cultural context without demand effects. If even these rich
# personas fail to move behaviour, the geography of cooperation is genuinely
# out of reach for the model; if they succeed, the neutral-label null is a
# statement about labels, not about the model's ceiling.
# English only by design: the norm-persona block is run in the English cell so
# that persona content, not translation quality, is the manipulated variable.
# 4 strong-civic-norm pools (human P-contributions 16-18 of 20) and 4
# weak-civic-norm pools (human P-contributions 6-10 of 20).
NORM_PERSONAS_EN: Dict[str, str] = {
    "Boston": ("You are a participant from Boston, the United States. You grew up in a society "
               "where civic cooperation is strong: people generally pay their taxes, respect "
               "public rules, and expect courts and institutions to work. Sanctioning someone "
               "who breaks a shared rule is widely seen as legitimate and necessary, and "
               "retaliating against the person who sanctioned you is considered unacceptable."),
    "Copenhagen": ("You are a participant from Copenhagen, Denmark. You grew up in a society "
                   "with very strong civic norms: people follow rules even when nobody is "
                   "watching, trust strangers and public institutions, and treat tax evasion or "
                   "fare dodging as shameful. Holding rule-breakers accountable is regarded as a "
                   "civic duty, and taking revenge on someone who held you accountable is "
                   "strongly frowned upon."),
    "St.Gallen": ("You are a participant from St. Gallen, Switzerland. You grew up in a society "
                  "with strong civic cooperation and a deeply rooted respect for common rules: "
                  "people pay what they owe, institutions are trusted and effective, and "
                  "communities expect every member to do their share. Sanctions against those "
                  "who dodge their obligations are accepted as fair, and answering a justified "
                  "sanction with retaliation is socially condemned."),
    "Zurich": ("You are a participant from Zurich, Switzerland. You grew up in a society where "
               "shared rules are respected and enforced: public services function reliably, "
               "people trust one another and the state, and doing one's part for common causes "
               "is the norm. Penalties against those who exploit the community are viewed as "
               "legitimate, and revenge against a legitimate penalty is not tolerated."),
    "Athens": ("You are a participant from Athens, Greece. You grew up in a society where many "
               "people are sceptical of public institutions and the courts, where tax evasion "
               "and fare dodging are widespread and usually go without consequences, and where "
               "being exploited by people who shirk their obligations is a familiar experience. "
               "Many people see formal rules as something to get around, and answering a "
               "perceived slight or sanction with payback is socially understandable."),
    "Istanbul": ("You are a participant from Istanbul, Turkey. You grew up in a society where "
                 "trust in strangers and in public institutions is limited, where bending the "
                 "rules is common when enforcement is weak, and where personal honour matters "
                 "greatly. Being publicly sanctioned can feel like an insult, and standing up "
                 "for yourself against whoever penalised you is widely seen as a natural "
                 "response."),
    "Riyadh": ("You are a participant from Riyadh, Saudi Arabia. You grew up in a society where "
               "cooperation is anchored in family, kinship and personal networks rather than in "
               "impersonal civic rules, and where trust in strangers outside one's own circle "
               "is low. A sanction coming from an anonymous stranger carries little legitimacy, "
               "and responding in kind to whoever harmed your standing is widely seen as "
               "defending your honour."),
    "Muscat": ("You are a participant from Muscat, Oman. You grew up in a society where social "
               "life is organised around family ties and personal loyalty rather than abstract "
               "civic duty, where strangers are treated with reserve, and where formal "
               "institutions matter less than personal relationships. An anonymous penalty from "
               "a stranger commands little respect, and repaying ill treatment in kind is a "
               "common reaction."),
}

# ----------------------------------------------------------------------------- #
# Treatment notice (injected into {TREATMENT_BLOCK} of the contribution prompt).
# Non-empty only in treatment P, so the agent contributes KNOWING a deduction
# stage follows (faithful: human P subjects knew this before contributing).
# {cost}/{impact} are filled in by treatment_block().
# ----------------------------------------------------------------------------- #
TREATMENT_NOTICE: Dict[str, str] = {
    "en": "After everyone has contributed and the contributions are revealed, there is a second stage: you may assign deduction points to the other members (this is explained in full at that stage). Each deduction point you assign costs you {cost} token and reduces the receiving member's income that period by {impact} tokens. Every member can do this, so deduction points may also be assigned to you.",
    "fr": "Une fois que tout le monde a contribué et que les contributions sont révélées, il y a une seconde étape : vous pouvez attribuer des points de déduction aux autres membres (ceci est expliqué en détail à cette étape). Chaque point de déduction que vous attribuez vous coûte {cost} jeton et réduit le revenu du membre concerné pour cette période de {impact} jetons. Chaque membre peut le faire, donc des points de déduction peuvent aussi vous être attribués.",
    "vn": "Sau khi mọi người đã đóng góp và các mức đóng góp được công bố, có một giai đoạn thứ hai: bạn có thể gán điểm trừ cho các thành viên khác (sẽ được giải thích đầy đủ ở giai đoạn đó). Mỗi điểm trừ bạn gán khiến bạn tốn {cost} token và làm giảm thu nhập của thành viên bị gán trong kỳ đó {impact} token. Mọi thành viên đều có thể làm điều này, nên điểm trừ cũng có thể bị gán cho bạn.",
    "cn": "在所有人都完成投入并公布各自的投入之后，会有第二个阶段：你可以给其他成员分配扣分（届时会详细说明）。你每分配一个扣分会让你损失 {cost} 个代币，并使被分配成员该周期的收入减少 {impact} 个代币。每位成员都可以这样做，因此你也可能被分配扣分。",
    "ar": "بعد أن يساهم الجميع وتُكشف المساهمات، هناك مرحلة ثانية: يمكنك تخصيص نقاط خصم للأعضاء الآخرين (سيُشرح ذلك بالكامل في تلك المرحلة). كل نقطة خصم تخصّصها تكلّفك {cost} رمزًا وتقلّل دخل العضو المعني في تلك الفترة بمقدار {impact} رموز. يمكن لكل عضو أن يفعل ذلك، لذا قد تُخصَّص لك نقاط خصم أيضًا.",
}

# Anonymous 'points received last period' sentence ({RECEIVED_LAST_PERIOD} of the
# punishment prompt). Enables the vengeance channel; empty in period 1.
RECEIVED_NOTICE: Dict[str, str] = {
    "en": "In the previous period you received a total of {n} deduction points.",
    "fr": "Lors de la période précédente, vous avez reçu au total {n} points de déduction.",
    "vn": "Ở kỳ trước, bạn đã nhận tổng cộng {n} điểm trừ.",
    "cn": "在上一周期，你总共收到了 {n} 个扣分。",
    "ar": "في الفترة السابقة تلقيت ما مجموعه {n} نقطة خصم.",
}


_ACTION_DESC_FULL: Dict[str, str] = {
    "en": "an integer number of tokens from {lo} to {hi}",
    "fr": "un nombre entier de jetons de {lo} à {hi}",
    "vn": "một số nguyên token từ {lo} đến {hi}",
    "cn": "{lo} 到 {hi} 之间的整数个代币",
    "ar": "عددًا صحيحًا من الرموز من {lo} إلى {hi}",
}
_ACTION_DESC_COARSE: Dict[str, str] = {
    "en": "one of these whole numbers of tokens: {opts}",
    "fr": "l'un de ces nombres entiers de jetons : {opts}",
    "vn": "một trong các số token sau: {opts}",
    "cn": "以下整数个代币之一：{opts}",
    "ar": "أحد هذه الأعداد الصحيحة من الرموز: {opts}",
}


def action_desc(language: str, options: Sequence[int], lo: int, hi: int) -> str:
    """Localized phrase describing the contribution action space.

    Full integer range -> 'an integer number of tokens from <lo> to <hi>';
    a coarse discrete set -> 'one of these whole numbers of tokens: 0, 5, 10, ...'.
    """
    full = tuple(range(lo, hi + 1))
    if tuple(options) == full:
        tmpl = _ACTION_DESC_FULL.get(language, _ACTION_DESC_FULL["en"])
        return tmpl.format(lo=lo, hi=hi)
    opts = ", ".join(str(o) for o in options)
    tmpl = _ACTION_DESC_COARSE.get(language, _ACTION_DESC_COARSE["en"])
    return tmpl.format(opts=opts)


def treatment_block(language: str, treatment: str, punish_cost: int,
                    punish_impact: int) -> str:
    if treatment != "P":
        return ""
    tmpl = TREATMENT_NOTICE.get(language, TREATMENT_NOTICE["en"])
    return tmpl.format(cost=punish_cost, impact=punish_impact)


def received_last_period_block(language: str, n) -> str:
    """`n` is None in period 1 (nothing received yet) -> empty string."""
    if n is None:
        return ""
    tmpl = RECEIVED_NOTICE.get(language, RECEIVED_NOTICE["en"])
    return tmpl.format(n=n)


# ----------------------------------------------------------------------------- #
# Machine tokens. Tolerant of spacing, leading bold markdown, optional "Member",
# and an OPTIONAL legacy ">>>" prefix (templates now emit bare tokens).
# ----------------------------------------------------------------------------- #
_CONTRIB_PRIMARY = re.compile(
    r"(?:>>>\s*)?\**\s*CONTRIBUTION\s*\**\s*=\s*\**\s*€?\s*(\d{1,2})\b", re.IGNORECASE)

_DEDUCT_PRIMARY = re.compile(
    r"(?:>>>\s*)?\**\s*DEDUCT\s*\**\s*:?\s*"
    r"(?:Member\s*)?A\s*=\s*(\d{1,2})\b[\s,]*"
    r"(?:Member\s*)?B\s*=\s*(\d{1,2})\b[\s,]*"
    r"(?:Member\s*)?C\s*=\s*(\d{1,2})\b", re.IGNORECASE)

_DEDUCT_KEYVAL = {
    "A": re.compile(r"\bA\s*=\s*(\d{1,2})\b", re.IGNORECASE),
    "B": re.compile(r"\bB\s*=\s*(\d{1,2})\b", re.IGNORECASE),
    "C": re.compile(r"\bC\s*=\s*(\d{1,2})\b", re.IGNORECASE),
    "D": re.compile(r"\bD\s*=\s*(\d{1,2})\b", re.IGNORECASE),
    "E": re.compile(r"\bE\s*=\s*(\d{1,2})\b", re.IGNORECASE),
}


# --------------------------------------------------------------------------- #
# Block helpers.
# --------------------------------------------------------------------------- #
def personality_block(language: str, personality: str) -> str:
    table = PERSONALITY_BLOCKS.get(language, PERSONALITY_BLOCKS["en"])
    return table.get(personality, "")


def society_block(language: str, society: str) -> str:
    """Render the society persona, or ''.

    Society keys:
      "none" / ""        -> no persona.
      "<City>"           -> NEUTRAL label: "You are a participant from <City>, <Country>."
      "norm:<City>"      -> NORM-LADEN persona (positive control), English only —
                            rich civic-norm content from NORM_PERSONAS_EN.
    """
    if not society or society == "none":
        return ""
    if society.startswith("norm:"):
        city = society[len("norm:"):]
        text = NORM_PERSONAS_EN.get(city)
        if text is None:
            raise KeyError(f"No norm-laden persona defined for {city!r} "
                           f"(available: {sorted(NORM_PERSONAS_EN)})")
        return text
    country = SOCIETY_COUNTRY.get(society, "")
    sentence = _SOCIETY_SENTENCE.get(language, _SOCIETY_SENTENCE["en"])
    return sentence.format(city=society, country=country)


# --------------------------------------------------------------------------- #
# Parsing. Both parsers follow CRSD's contract: return (value, primary_ok);
# primary_ok is True only when the explicit CONTRIBUTION/DEDUCT token is present AND valid, and
# the caller treats `not primary_ok` as "needs a retry". A best-effort fallback
# still yields a usable value if retries are exhausted.
# --------------------------------------------------------------------------- #
def parse_contribution(text: str, options: Sequence[int] = tuple(range(0, 21)),
                       default: int = 0) -> Tuple[int, bool]:
    """Extract the chosen contribution from a model reply.

    Returns (value, primary_ok). primary_ok is True only when the explicit
    `CONTRIBUTION = X` token is found AND X is a legal option. Fallback is the
    last standalone integer that is a legal option (primary_ok=False). The
    `\\d{1,2}` cap stops a 3+ digit number masquerading as the answer.
    """
    opts = set(int(o) for o in options)
    if text:
        matches = list(_CONTRIB_PRIMARY.finditer(text))
        for m in reversed(matches):                 # take the LAST legal token
            v = int(m.group(1))
            if v in opts:
                return v, True
        toks = [int(t) for t in re.findall(r"(?<![\d.])(\d+)(?![\d.])", text) if int(t) in opts]
        if toks:
            return toks[-1], False
    return int(default), False


def parse_punishment(text: str, n_targets: int = 3, max_points: int = 10,
                     default: int = 0) -> Tuple[List[int], bool]:
    """Extract per-target deduction points for Member A..(A+n_targets-1).

    Returns (list_of_ints aligned to A, B, C, ..., primary_ok). Values are clamped
    to [0, max_points]. Degrades to all-`default` (all-zero, i.e. NO punishment) if
    nothing parseable — a parse failure must never be silently recorded as
    "this agent punished", which would contaminate the antisocial measure.
    """
    labels = [chr(ord("A") + i) for i in range(n_targets)]
    zero = [int(default)] * n_targets

    def clamp(v: int) -> int:
        return max(0, min(int(v), max_points))

    if not text:
        return zero, False

    # Primary strict token (defined for the standard 3-target case).
    if n_targets == 3:
        hits = list(_DEDUCT_PRIMARY.finditer(text))
        if hits:
            last = hits[-1]
            raw = [int(last.group(i + 1)) for i in range(3)]
            in_range = all(0 <= r <= max_points for r in raw)
            return [clamp(r) for r in raw], bool(in_range)

    # Fallback: last "X=number" per label, anywhere in the text.
    found = {}
    for lab in labels:
        pat = _DEDUCT_KEYVAL.get(lab)
        if pat is None:
            continue
        kv = list(pat.finditer(text))
        if kv:
            found[lab] = clamp(int(kv[-1].group(1)))
    if found:
        return [found.get(lab, int(default)) for lab in labels], False

    return zero, False


# --------------------------------------------------------------------------- #
# History / context rendering.
# --------------------------------------------------------------------------- #
def format_others_block(others_contribs: Sequence[Tuple[str, int]],
                        language: str = "en") -> str:
    """Render the current period's other-member contributions (punishment stage).

    Args:
        others_contribs: ordered list of (slot_label, contribution), e.g.
            [("Member A", 12), ("Member B", 0), ("Member C", 8)].
    """
    lab = LABELS.get(language, LABELS["en"])
    lines = []
    for slot_label, c in others_contribs:
        lines.append(f"- {slot_label} {lab['placed']} {c} {lab['tokens_in_project']}.")
    return "\n".join(lines)


def format_contrib_history(history: Sequence[Dict], you_id: str,
                           language: str = "en") -> str:
    """Render the contribution history using each period's TEMPORARY relabelling.

    Args:
        history: one dict per completed period, shaped for THIS reader:
            {
              "others": [("Member A", 12), ("Member B", 0), ("Member C", 8)],
              "you": 6,
              "project_total": 26,
              "my_income": 24.4,
            }
        you_id: unused for display (kept for signature parity / debugging).
        language: label localization.
    """
    lab = LABELS.get(language, LABELS["en"])
    if not history:
        return lab["no_periods"]
    lines = []
    for k, h in enumerate(history, start=1):
        parts = [f"{slot}: {c}" for slot, c in h["others"]]
        parts.append(f"{lab['you']}: {h['you']}")
        income = h["my_income"]
        income_str = f"{income:g}"
        lines.append(
            f"{lab['period']} {k}:  " + "   ".join(parts)
            + f"   ({lab['project_total']}: {h['project_total']}, {lab['your_income']}: {income_str})")
    return "\n".join(lines)


def format_punish_history(punish_history: Sequence[Dict],
                          language: str = "en") -> str:
    """Render this reader's own deduction history (punishment stage, P only).

    Args:
        punish_history: one dict per completed period:
            {"assigned": [("Member A", 0), ("Member B", 4), ("Member C", 0)],
             "received_total": 6}
    """
    lab = LABELS.get(language, LABELS["en"])
    if not punish_history:
        return lab["no_periods"]
    lines = []
    for k, h in enumerate(punish_history, start=1):
        parts = [f"{slot}: {pts}" for slot, pts in h["assigned"]]
        lines.append(
            f"{lab['period']} {k}:  {lab['assigned']} -> " + "   ".join(parts)
            + f"   ({lab['received']}: {h['received_total']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Template assembly.
# --------------------------------------------------------------------------- #
def build_prompt(template: str, mapping: Dict[str, object],
                 keys: Sequence[str]) -> str:
    """Fill a template by exact substitution of each `{KEY}` in `keys`.

    Uses sequential str.replace (NOT str.format) so stray braces, currency
    symbols, or unmatched keys never raise — important because the rendered
    history / persona text may contain arbitrary characters.
    """
    out = template
    for key in keys:
        out = out.replace("{" + key + "}", str(mapping.get(key, "")))
    return out
