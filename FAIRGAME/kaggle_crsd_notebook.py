"""
=====================================================================
FAIRGAME × CRSD — Kaggle OFFLINE notebook (Internet OFF, GPU ON)  ── ALL-IN-ONE
=====================================================================
Replicate Milinski et al. (2008) "collective-risk social dilemma" với LLM agents,
chạy local trên GPU Kaggle (RTX PRO 6000, 96GB). Một file duy nhất: nạp model
(Cell 1–5) rồi chạy CRSD + lưu kết quả (Cell 6–11).

CÁCH CHẠY:
  1. Tạo notebook Kaggle mới — GPU: ON, Internet: OFF.
  2. + Add Input:
       a) Code: repo public-good-game (read-only), CHỨA src/crsd/ + resources/crsd_*.
       b) Model: Qwen2.5-7B-Instruct (mount ở /kaggle/input/<...>).
  3. Copy file này vào notebook, chia cell theo "# CELL N".
  4. Sửa MODEL_PATH + KAGGLE_CODE_INPUT ở Cell 1/3 cho đúng path của bạn.
  5. Run lần lượt Cell 1 → 11.

Output: /kaggle/working/crsd_results/  (crsd_results.json, crsd_all_games.csv,
crsd_metrics.json, per-treatment CSV) + crsd_results.zip ở Output tab.
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
TEMPERATURE = 0.8         # 0.7–1.0: cần >0 để 6 agent khác nhau
MAX_TOKENS = 512          # đủ cho reasoning ngắn + dòng ">>> CONTRIBUTION = X"
GPU_UTIL = 0.90           # chỉ dùng cho vllm
TP_SIZE = 1               # tensor parallel (vllm); single GPU = 1
FAIRGAME_VERBOSE_LOGS = "0"

# --- CRSD runner ---
import random  # noqa: E402
from pathlib import Path  # noqa: E402

SEED = 20080219                # reproducibility (ngày publish paper)
BATCH_SIZE = 256               # prompts/forward; 0 = cả vòng 1 batch. 7B+96GB: 256 an toàn.
MAX_PARSE_RETRIES = 2          # re-hỏi riêng reply không parse được
RUN_LANGUAGE_BLOCK = True      # Block A+B: 5 langs × neutral × 10
RUN_PERSONALITY_BLOCK = True   # Block C: en × {coop, selfish, risk_averse} × 10
PERSONALITY_LANG = "en"
PERSONALITY_CONDITIONS = ["cooperative", "selfish", "risk_averse"]
TREATMENTS = [90, 50, 10]      # loss probabilities (%)
OUTPUT_DIR = Path("/kaggle/working/crsd_results")

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
    """Import module local_vllm_connector dù repo dùng prefix nào (legacy.FAIRGAME.src / src / FAIRGAME.src)."""
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

# Kiểm tra các file CRSD có mặt trong repo đã upload chưa
_need = [FAIRGAME_ROOT / "src" / "crsd" / "crsd_game.py",
         FAIRGAME_ROOT / "resources" / "crsd_templates" / "crsd_en.txt",
         FAIRGAME_ROOT / "resources" / "crsd_config" / "crsd_p90.json"]
_missing = [str(p) for p in _need if not p.exists()]
print(f"✅ Code ready — project root: {FAIRGAME_ROOT}")
print(f"📁 Model path: {MODEL_PATH} | exists={Path(MODEL_PATH).exists()}")
if _missing:
    print("❌ THIẾU file CRSD trong repo (push & re-add Code input):")
    for m in _missing:
        print("   -", m)
else:
    print("✅ CRSD files present (src/crsd + resources/crsd_*).")

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
# CELL 6: Import crsd modules (anchor theo __file__ của connector)
# =====================================================================
import json

conn = import_local_connector()
send_prompts_global = conn.send_prompts_global

SRC_DIR = Path(conn.__file__).resolve().parent.parent          # .../src
FAIRGAME_BASE = SRC_DIR.parent                                  # repo root chứa src/ + resources/
sys.path.insert(0, str(SRC_DIR / "crsd"))

import crsd_results                                             # noqa: E402
from crsd_game import CRSDGame, run_games_lockstep              # noqa: E402
from crsd_prompt import parse_contribution                      # noqa: E402

TEMPLATE_DIR = FAIRGAME_BASE / "resources" / "crsd_templates"
CONFIG_DIR = FAIRGAME_BASE / "resources" / "crsd_config"
templates = {p.stem.replace("crsd_", ""): p.read_text(encoding="utf-8")
             for p in TEMPLATE_DIR.glob("crsd_*.txt")}
print(f"✅ crsd imported. templates={sorted(templates)} | configs dir={CONFIG_DIR}")

# =====================================================================
# CELL 7: Smoke test — model có tuân thủ token ">>> CONTRIBUTION = X"?
# =====================================================================
_probe = CRSDGame("probe", "en", "neutral", ["none"] * 6, templates["en"],
                  dict(n_players=6, n_rounds=10, endowment=40, target=120,
                       loss_prob=90, contribution_options=(0, 2, 4)))
_resp = send_prompts_global(_probe.build_round_prompts()[:1], batch_size=0)
print("🧪 Sample reply (Player 1, round 1) — 600 ký tự cuối:\n", _resp[0][-600:])
print("🧪 Parsed:", parse_contribution(_resp[0]),
      "(primary_ok=True nghĩa là model tuân thủ token)")

# =====================================================================
# CELL 8: Build games theo run plan
# =====================================================================
def load_config(loss_prob):
    return json.loads((CONFIG_DIR / f"crsd_p{loss_prob}.json").read_text(encoding="utf-8"))


def params_from_config(cfg):
    return dict(n_players=cfg["nPlayers"], n_rounds=cfg["nRounds"], endowment=cfg["endowment"],
                target=cfg["target"], loss_prob=cfg["treatmentLossProb"],
                contribution_options=tuple(cfg["contributionOptions"]))


games = []
for loss_prob in TREATMENTS:
    cfg = load_config(loss_prob)
    params = params_from_config(cfg)
    n_groups = cfg["groupsPerCondition"]
    conds = cfg["personalityConditions"]

    if RUN_LANGUAGE_BLOCK:                       # Block A+B: langs × neutral
        for lang in cfg["languages"]:
            for k in range(n_groups):
                games.append(CRSDGame(f"p{loss_prob}_{lang}_neutral_{k}", lang, "neutral",
                                      conds["neutral"], templates[lang], params))
    if RUN_PERSONALITY_BLOCK:                    # Block C: en × dispositions
        for cond in PERSONALITY_CONDITIONS:
            for k in range(n_groups):
                games.append(CRSDGame(f"p{loss_prob}_{PERSONALITY_LANG}_{cond}_{k}",
                                      PERSONALITY_LANG, cond, conds[cond],
                                      templates[PERSONALITY_LANG], params))

print(f"🎮 Tổng số games: {len(games)}  (= {len(games) * 6 * 10} generations)")

# =====================================================================
# CELL 9: Chạy lockstep (1 batch lớn mỗi vòng cho TẤT CẢ games)
# =====================================================================
def responder(prompts):
    return send_prompts_global(prompts, batch_size=BATCH_SIZE)


def _progress(done, total):
    print(f"   round {done}/{total} xong  ({len(games)} games × 6 prompts/round)")


print("🚀 Bắt đầu chạy CRSD...")
results = run_games_lockstep(games, responder, rng=random.Random(SEED),
                             max_parse_retries=MAX_PARSE_RETRIES, progress=_progress)
print("✅ Hoàn tất tất cả games.")

# =====================================================================
# CELL 10: Lưu kết quả + bảng so với human
# =====================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "crsd_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

df = crsd_results.to_dataframe(results)
df.to_csv(OUTPUT_DIR / "crsd_all_games.csv", index=False)
for (p, lang), sub in df.groupby(["treatment_loss_prob", "language"]):
    sub.to_csv(OUTPUT_DIR / f"crsd_p{p}_{lang}.csv", index=False)


def _ser(summary):
    return {str(k): v for k, v in summary.items()}


baseline = [r for r in results if r["language"] == "en" and r["personality_condition"] == "neutral"]
metrics = {
    "baseline_en_neutral_by_treatment": _ser(crsd_results.summarize(baseline)),
    "by_treatment_language": _ser(crsd_results.summarize(
        [r for r in results if r["personality_condition"] == "neutral"],
        key=lambda r: f"p{r['treatment_loss_prob']}_{r['language']}")),
    "by_treatment_personality": _ser(crsd_results.summarize(
        [r for r in results if r["language"] == PERSONALITY_LANG],
        key=lambda r: f"p{r['treatment_loss_prob']}_{r['personality_condition']}")),
    "human_benchmark": crsd_results.HUMAN_BENCHMARK,
}
(OUTPUT_DIR / "crsd_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

bsum = crsd_results.summarize(baseline)
print("\n=== LLM (en, neutral) vs HUMAN (Milinski 2008) ===")
print(f"{'p':>4} | {'success LLM/HUM':>16} | {'mean total LLM/HUM':>22} | {'fairshare LLM/HUM':>18} | parse_fb")
for p in TREATMENTS:
    s = bsum.get(p)
    if not s:
        continue
    h = crsd_results.HUMAN_BENCHMARK[p]
    print(f"{p:>4} | {s['success_rate']:.2f} / {h['success_rate']:.2f}        | "
          f"{s['final_total']['mean']:6.1f} / {h['mean_final_total']:6.1f}          | "
          f"{s['fair_sharers_per_group']:.2f} / {h['fair_sharers_per_group']:.2f}          | "
          f"{s['parse_fallback_rate']:.1%}")
print(f"\n📦 Output → {OUTPUT_DIR}")
print("ℹ️  Nếu parse_fb cao (>5%): contribution thấp có thể do model cộng sai/không tuân thủ (faithful+7B).")

# =====================================================================
# CELL 11: Zip để download
# =====================================================================
import zipfile

zip_path = Path("/kaggle/working/crsd_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"✅ {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — tải ở Output tab.")
print("➡️  Sau khi tải về: python crsd_analysis.py crsd_results/crsd_results.json  (bảng so-human + Fig 2/3)")
