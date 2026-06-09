"""
=====================================================================
FAIRGAME × PGG-Punishment — Kaggle OFFLINE notebook (Internet OFF, GPU ON)
=====================================================================
Replicate Herrmann, Thoeni & Gaechter (2008) "Antisocial Punishment Across
Societies" — public goods game WITH/WITHOUT punishment — bằng LLM agents, chạy
local trên GPU Kaggle. Một file duy nhất: nạp model (Cell 1–5) rồi chạy PGG±P +
lưu kết quả (Cell 6–11). Song song với CRSD; KHÔNG đụng tới module CRSD.

CÁCH CHẠY:
  1. Tạo notebook Kaggle mới — GPU: ON, Internet: OFF.
  2. + Add Input:
       a) Code: repo public-good-game (read-only), CHỨA src/pgg_punish/ +
          resources/pgg_punish_*.
       b) Model: Qwen2.5-7B-Instruct (mount ở /kaggle/input/<...>).
  3. Copy file này vào notebook, chia cell theo "# CELL N".
  4. Sửa MODEL_PATH + KAGGLE_CODE_INPUT ở Cell 1/3 cho đúng path của bạn.
  5. Run lần lượt Cell 1 → 11.

Output: /kaggle/working/pgg_punish_results/  (pgg_results.json, pgg_all_games.csv,
pgg_metrics.json, per-treatment CSV) + pgg_punish_results.zip ở Output tab.

LƯU Ý TẢI: treatment P gọi LLM 2 lần/kỳ (đóng góp + trừng phạt). Full design có thể
~28k generation — dùng các flag RUN_*_BLOCK + cắt SOCIETIES + giảm groupsPerCondition
để vừa quota GPU.
=====================================================================
"""

# =====================================================================
# CELL 1: CẤU HÌNH — SỬA Ở ĐÂY
# =====================================================================

# --- Model (đã add làm Kaggle input). Chạy "!ls /kaggle/input/" để xem path thực. ---
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
MODEL_SHORT_NAME = "qwen25-7b-instruct"

# --- Engine + tham số sinh ---
ENGINE = "transformers"   # "transformers" (ổn định offline) | "vllm" (nếu image có sẵn)
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.8         # 0.7–1.0: cần >0 để 4 agent khác nhau
MAX_TOKENS = 512          # đủ reasoning ngắn + dòng ">>> CONTRIBUTION/DEDUCT"
GPU_UTIL = 0.90           # chỉ dùng cho vllm
TP_SIZE = 1               # tensor parallel (vllm); single GPU = 1
FAIRGAME_VERBOSE_LOGS = "0"

# --- PGG runner ---
import random  # noqa: E402
from pathlib import Path  # noqa: E402

SEED = 20080307                # reproducibility (ngày publish paper: 7 Mar 2008)
BATCH_SIZE = 256               # prompts/forward; 0 = cả lượt 1 batch. 7B+96GB: 256 an toàn.
MAX_PARSE_RETRIES = 2          # re-hỏi riêng reply không parse được

TREATMENTS = ["N", "P"]                       # cả hai điều kiện (không trừng phạt / có)
RUN_LANGUAGE_BLOCK = True                      # {N,P} × 5 langs × neutral × G
RUN_PERSONALITY_BLOCK = True                   # {N,P} × en × {coop,selfish,vengeful} × G
RUN_SOCIETY_BLOCK = True                       # {N,P} × en × 16 societies × G  (NẶNG nhất)
PERSONALITY_LANG = "en"
SOCIETY_LANG = "en"
PERSONALITY_CONDITIONS = ["cooperative", "selfish", "vengeful"]
# Cắt bớt để chạy nhanh; None = dùng toàn bộ "societies" trong config (16 pool).
SOCIETIES = None
OUTPUT_DIR = Path("/kaggle/working/pgg_punish_results")

# =====================================================================
# CELL 2: Chuẩn bị path + helpers (Internet OFF — không pip)
# =====================================================================
import os
import sys

os.environ["FAIRGAME_VERBOSE_LOGS"] = FAIRGAME_VERBOSE_LOGS

WORK_COPY = Path("/kaggle/working/FAIRGAME")
MARKER_ROOT = Path("/kaggle/working/.fairgame_project_root")


def ensure_src_importable():
    """chdir + sys.path tới thư mục có src/ (marker từ Cell 3 hoặc quét WORK_COPY)."""
    if MARKER_ROOT.exists():
        root = Path(MARKER_ROOT.read_text(encoding="utf-8").strip())
    else:
        root = None
        for c in (WORK_COPY, WORK_COPY / "FAIRGAME", WORK_COPY / "fairgame"):
            if c.is_dir() and (c / "src").is_dir():
                root = c
                break
        if root is None and WORK_COPY.is_dir():
            for ch in WORK_COPY.iterdir():
                if ch.is_dir() and (ch / "src").is_dir():
                    root = ch
                    break
        if root is None:
            raise RuntimeError("Không thấy src/: chạy Cell 3 trước.")
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def import_local_connector():
    """Import local_vllm_connector dù repo dùng prefix nào (legacy.FAIRGAME.src / src / FAIRGAME.src)."""
    ensure_src_importable()
    last = None
    for modpath in ("legacy.FAIRGAME.src.llm_connectors.local_vllm_connector",
                    "src.llm_connectors.local_vllm_connector",
                    "FAIRGAME.src.llm_connectors.local_vllm_connector"):
        try:
            return __import__(modpath, fromlist=["init_local_llm"])
        except Exception as e:  # noqa: BLE001
            last = e
    raise ImportError(f"Không import được local_vllm_connector: {last}")


print("Internet OFF: dùng thư viện có sẵn trên image Kaggle.")

# =====================================================================
# CELL 3: Setup FAIRGAME source (copy repo + patch game_round)
# =====================================================================
import shutil

# Repo đã add làm Code input (read-only). Sửa nếu path khác.
KAGGLE_CODE_INPUT = Path(
    "/kaggle/input/notebooks/trungkiet/git-public-good-game/public-good-game/"
)
FAIRGAME_WORK = Path("/kaggle/working/FAIRGAME")


def resolve_fairgame_root(base: Path) -> Path:
    for c in (base, base / "FAIRGAME", base / "fairgame"):
        if c.is_dir() and (c / "src").is_dir():
            return c.resolve()
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "src").is_dir():
                return child.resolve()
    hint = [p.name for p in base.iterdir()] if base.is_dir() else []
    raise FileNotFoundError(f"Không thấy src/ dưới {base}. Mục con: {hint}")


def apply_game_round_no_retry_patch(project_root: Path) -> None:
    """Ghi đè src/game_round.py bản không cần PyPI ``retry`` (an toàn với snapshot cũ)."""
    import base64
    dest = project_root / "src" / "game_round.py"
    bundled = project_root / "offline_patch_assets" / "game_round.py"
    if bundled.is_file():
        shutil.copyfile(bundled, dest)
        print("✅ patch: src/game_round.py ← offline_patch_assets/")
        return
    text = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    if "from retry import retry" not in text:
        return
    b64_path = project_root / "kaggle_game_round_patch.b64"
    if b64_path.is_file():
        dest.write_bytes(base64.b64decode(b64_path.read_text(encoding="ascii").strip()))
        print("✅ patch: src/game_round.py ← kaggle_game_round_patch.b64")
        return
    raise RuntimeError("game_round.py còn import ``retry`` mà thiếu patch asset — cập nhật Code Input.")


if FAIRGAME_WORK.exists():
    shutil.rmtree(FAIRGAME_WORK)
shutil.copytree(KAGGLE_CODE_INPUT, FAIRGAME_WORK)

FAIRGAME_ROOT = resolve_fairgame_root(FAIRGAME_WORK)
apply_game_round_no_retry_patch(FAIRGAME_ROOT)
os.chdir(FAIRGAME_ROOT)
if str(FAIRGAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIRGAME_ROOT))
(FAIRGAME_ROOT / "resources" / "results").mkdir(parents=True, exist_ok=True)
MARKER_ROOT.write_text(str(FAIRGAME_ROOT), encoding="utf-8")

# Kiểm tra các file PGG-Punishment có mặt trong repo đã upload chưa
_need = [FAIRGAME_ROOT / "src" / "pgg_punish" / "pgg_game.py",
         FAIRGAME_ROOT / "resources" / "pgg_punish_templates" / "pgg_contrib_en.txt",
         FAIRGAME_ROOT / "resources" / "pgg_punish_templates" / "pgg_punish_en.txt",
         FAIRGAME_ROOT / "resources" / "pgg_punish_config" / "pgg_P.json"]
_missing = [str(p) for p in _need if not p.exists()]
print(f"✅ Code ready — project root: {FAIRGAME_ROOT}")
print(f"📁 Model path: {MODEL_PATH} | exists={Path(MODEL_PATH).exists()}")
if _missing:
    print("❌ THIẾU file PGG-Punishment trong repo (push & re-add Code input):")
    for m in _missing:
        print("   -", m)
else:
    print("✅ pgg_punish files present (src/pgg_punish + resources/pgg_punish_*).")

# =====================================================================
# CELL 4: Load model vào GPU
# =====================================================================
conn = import_local_connector()
print(f"🚀 Loading model ({ENGINE})...")
if ENGINE == "vllm":
    conn.init_local_llm(
        MODEL_PATH, engine="vllm", max_model_len=MAX_MODEL_LEN,
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        gpu_memory_utilization=GPU_UTIL, tensor_parallel_size=TP_SIZE,
    )
else:
    conn.init_local_llm(
        MODEL_PATH, engine="transformers", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
print("✅ Model loaded!")

# =====================================================================
# CELL 5: Test nhanh (model đã sống chưa)
# =====================================================================
conn = import_local_connector()
_test = conn.LocalVLLMConnector(provider_model="test")
print("🧪 2+2 =", _test.send_prompt("What is 2+2? Answer with just the number."))

# =====================================================================
# CELL 6: Import pgg_punish modules (anchor theo __file__ của connector)
# =====================================================================
import json

conn = import_local_connector()
send_prompts_global = conn.send_prompts_global

SRC_DIR = Path(conn.__file__).resolve().parent.parent          # .../src
FAIRGAME_BASE = SRC_DIR.parent                                  # repo root chứa src/ + resources/
sys.path.insert(0, str(SRC_DIR / "pgg_punish"))

import pgg_results                                              # noqa: E402
from pgg_game import PGGGame, run_games_lockstep                # noqa: E402
from pgg_prompt import parse_contribution, parse_punishment     # noqa: E402

TEMPLATE_DIR = FAIRGAME_BASE / "resources" / "pgg_punish_templates"
CONFIG_DIR = FAIRGAME_BASE / "resources" / "pgg_punish_config"
templates_contrib = {p.stem.replace("pgg_contrib_", ""): p.read_text(encoding="utf-8")
                     for p in TEMPLATE_DIR.glob("pgg_contrib_*.txt")}
templates_punish = {p.stem.replace("pgg_punish_", ""): p.read_text(encoding="utf-8")
                    for p in TEMPLATE_DIR.glob("pgg_punish_*.txt")}
print(f"✅ pgg_punish imported. langs={sorted(templates_contrib)} | configs dir={CONFIG_DIR}")


def load_config(treatment):
    return json.loads((CONFIG_DIR / f"pgg_{treatment}.json").read_text(encoding="utf-8"))


def params_from_config(cfg):
    opts = cfg.get("contributionOptions")
    return dict(treatment=cfg["treatment"], n_players=cfg["nPlayers"], n_periods=cfg["nPeriods"],
                endowment=cfg["endowment"], mpcr=cfg["mpcr"],
                contrib_min=cfg["contributionMin"], contrib_max=cfg["contributionMax"],
                options=tuple(opts) if opts else None,
                max_punish=cfg["maxPunishPerTarget"], punish_cost=cfg["punishCostRatio"],
                punish_impact=cfg["punishImpactRatio"], relabel_others=cfg["relabelOthers"],
                show_received=cfg["showReceivedPunishment"], floor_earnings=cfg["floorPeriodEarnings"])


# =====================================================================
# CELL 7: Smoke test — model có tuân thủ token ">>> CONTRIBUTION/DEDUCT"?
# =====================================================================
_cfgP = load_config("P")
_pp = params_from_config(_cfgP)
_probe = PGGGame("probe", "en", "neutral", _cfgP["personalityConditions"]["neutral"], "none",
                 templates_contrib["en"], templates_punish["en"], _pp)
_probe._rng = random.Random(SEED)
_cprompts = _probe.build_contrib_prompts()
_cresp = send_prompts_global(_cprompts[:1], batch_size=0)
print("🧪 Contribution reply (Member 1, period 1) — 500 ký tự cuối:\n", _cresp[0][-500:])
print("🧪 Parsed contribution:", parse_contribution(_cresp[0], _probe.options),
      "(primary_ok=True nghĩa là model tuân thủ token)")

# đẩy 1 lượt đóng góp giả để vào giai đoạn trừng phạt, rồi test token DEDUCT
_probe.ingest_contributions([10, 10, 10, 10], [""] * 4, [True] * 4)
_pprompts = _probe.build_punish_prompts()
_presp = send_prompts_global(_pprompts[:1], batch_size=0)
print("\n🧪 Deduction reply (Member 1, period 1) — 500 ký tự cuối:\n", _presp[0][-500:])
print("🧪 Parsed punishment:", parse_punishment(_presp[0], _probe.n_players - 1, _probe.max_punish))

# =====================================================================
# CELL 8: Build games theo run plan
# =====================================================================
games = []
for treatment in TREATMENTS:
    cfg = load_config(treatment)
    params = params_from_config(cfg)
    G = cfg["groupsPerCondition"]
    conds = cfg["personalityConditions"]
    societies = SOCIETIES if SOCIETIES is not None else cfg.get("societies", [])

    if RUN_LANGUAGE_BLOCK:                       # Block A+B: langs × neutral
        for lang in cfg["languages"]:
            ct, pt = templates_contrib[lang], templates_punish[lang]
            for k in range(G):
                games.append(PGGGame(f"{treatment}_{lang}_neutral_{k}", lang, "neutral",
                                     conds["neutral"], "none", ct, pt, params))

    if RUN_PERSONALITY_BLOCK:                    # Block C: en × dispositions
        ct, pt = templates_contrib[PERSONALITY_LANG], templates_punish[PERSONALITY_LANG]
        for cond in PERSONALITY_CONDITIONS:
            for k in range(G):
                games.append(PGGGame(f"{treatment}_{PERSONALITY_LANG}_{cond}_{k}", PERSONALITY_LANG,
                                     cond, conds[cond], "none", ct, pt, params))

    if RUN_SOCIETY_BLOCK:                        # Block D: en × societies (persona xã hội)
        ct, pt = templates_contrib[SOCIETY_LANG], templates_punish[SOCIETY_LANG]
        for soc in societies:
            for k in range(G):
                games.append(PGGGame(f"{treatment}_{SOCIETY_LANG}_soc-{soc}_{k}", SOCIETY_LANG,
                                     "neutral", conds["neutral"], soc, ct, pt, params))

_n_p = sum(1 for g in games if g.treatment == "P")
print(f"🎮 Tổng số games: {len(games)}  (P games: {_n_p})")
print(f"   ≈ {len(games) * 4 * 10} generation đóng góp + {_n_p * 4 * 10} generation trừng phạt")

# =====================================================================
# CELL 9: Chạy lockstep (mỗi kỳ: 1 batch đóng góp + 1 batch trừng phạt cho game P)
# =====================================================================
def responder(prompts):
    return send_prompts_global(prompts, batch_size=BATCH_SIZE)


def _progress(done, total):
    print(f"   period {done}/{total} xong  ({len(games)} games; game P gọi 2 batch/kỳ)")


print("🚀 Bắt đầu chạy PGG±Punishment...")
results = run_games_lockstep(games, responder, rng=random.Random(SEED),
                             max_parse_retries=MAX_PARSE_RETRIES, progress=_progress)
print("✅ Hoàn tất tất cả games.")

# =====================================================================
# CELL 10: Lưu kết quả + bảng so với human (Herrmann 2008)
# =====================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "pgg_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

df = pgg_results.to_dataframe(results)
df.to_csv(OUTPUT_DIR / "pgg_all_games.csv", index=False)
for (tr, lang), sub in df.groupby(["treatment", "language"]):
    sub.to_csv(OUTPUT_DIR / f"pgg_{tr}_{lang}.csv", index=False)


def _ser(summary):
    return {str(k): v for k, v in summary.items()}


baseline = [r for r in results if r["language"] == "en"
            and r["personality_condition"] == "neutral" and r["society"] == "none"]
society_games = [r for r in results if r["society"] != "none"]

metrics = {
    "baseline_en_neutral_by_treatment": _ser(pgg_results.summarize(baseline)),
    "by_treatment_language": _ser(pgg_results.summarize(
        [r for r in results if r["personality_condition"] == "neutral" and r["society"] == "none"],
        key=lambda r: f"{r['treatment']}_{r['language']}")),
    "by_treatment_personality": _ser(pgg_results.summarize(
        [r for r in results if r["language"] == PERSONALITY_LANG and r["society"] == "none"],
        key=lambda r: f"{r['treatment']}_{r['personality_condition']}")),
    "by_treatment_society": _ser(pgg_results.summarize(
        society_games, key=lambda r: f"{r['treatment']}_{r['society']}")),
    "human_benchmark": pgg_results.HUMAN_BENCHMARK,
}
(OUTPUT_DIR / "pgg_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# --- Bảng: contribution LLM (en, neutral) vs HUMAN grand-mean, theo treatment ---
bsum = pgg_results.summarize(baseline)
hum = pgg_results.HUMAN_BENCHMARK
hum_mean = {"N": sum(hum["N_mean_contribution"].values()) / len(hum["N_mean_contribution"]),
            "P": sum(hum["P_mean_contribution"].values()) / len(hum["P_mean_contribution"])}
print("\n=== LLM (en, neutral) vs HUMAN grand-mean (Herrmann 2008) ===")
print(f"{'treat':>5} | {'mean contrib LLM/HUM':>22} | {'antisocial share':>16} | parse_fb(c/p)")
for tr in TREATMENTS:
    s = bsum.get(tr)
    if not s:
        continue
    anti = s["antisocial_prosocial_split"]["antisocial_share"]
    fb = s["parse_fallback_rate"]
    print(f"{tr:>5} | {s['mean_contribution']:6.2f} / {hum_mean[tr]:6.2f}            | "
          f"{anti:14.1%} | {fb['contrib']:.1%}/{fb['punish']:.1%}")

# --- Cross-societal: LLM antisocial ↔ contribution (so rho người ~ -0.90) ---
if society_games:
    soc_sum = pgg_results.summarize(
        [r for r in society_games if r["treatment"] == "P"],
        key=lambda r: r["society"])
    anti_vals, coop_vals = [], []
    print("\n=== Cross-societal (treatment P): mean contribution per society ===")
    for soc in sorted(soc_sum, key=lambda s: -soc_sum[s]["mean_contribution"]):
        s = soc_sum[soc]
        anti = s["antisocial_prosocial_split"]["antisocial_total"]
        h = hum["P_mean_contribution"].get(soc, float("nan"))
        anti_vals.append(anti)
        coop_vals.append(s["mean_contribution"])
        print(f"   {soc:>16}: LLM {s['mean_contribution']:5.2f} | HUM {h:4.1f} | "
              f"antisocial_pts {anti:6.0f}")
    rho = pgg_results.spearman(anti_vals, coop_vals)
    print(f"\n   Spearman(antisocial, contribution) LLM = {rho:+.2f}  (HUMAN ~ -0.90)")

print(f"\n📦 Output → {OUTPUT_DIR}")
print("ℹ️  parse_fb cao (>5%): đóng góp/trừng phạt bất thường có thể do model không tuân token "
      "(0..20 nhiễu hơn). Cân nhắc đặt contributionOptions = tập rời rạc cho model nhỏ.")

# =====================================================================
# CELL 11: Zip để download
# =====================================================================
import zipfile

zip_path = Path("/kaggle/working/pgg_punish_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"✅ {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — tải ở Output tab.")
print("➡️  Sau khi tải về: python pgg_punish_analysis.py pgg_punish_results/pgg_results.json")
