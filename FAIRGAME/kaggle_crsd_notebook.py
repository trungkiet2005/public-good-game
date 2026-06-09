"""
=====================================================================
FAIRGAME × CRSD — Kaggle OFFLINE notebook (Internet OFF, GPU ON)  ── MULTI-MODEL
=====================================================================
Replicate Milinski et al. (2008) "collective-risk social dilemma" với LLM agents,
chạy local trên GPU Kaggle. Nạp LẦN LƯỢT nhiều model (vd Qwen2.5-7B, Gemma-2-12B),
chạy CRSD cho từng model, và lưu kết quả RIÊNG theo từng model.

CÁCH CHẠY:
  1. Tạo notebook Kaggle mới — GPU: ON, Internet: OFF.
  2. + Add Input:
       a) Code: repo public-good-game (read-only), CHỨA src/crsd/ + resources/crsd_*.
       b) Model(s): add MỖI model làm một input (Qwen2.5-7B, Gemma-2-12B, ...).
  3. Copy file này vào notebook, chia cell theo "# CELL N".
  4. Sửa MODELS[] + KAGGLE_CODE_INPUT ở Cell 1/3 cho đúng path của bạn.
       Chạy "!ls /kaggle/input/" để xem path thực của từng model.
  5. Run lần lượt Cell 1 → 8.

ĐA MODEL:
  * ENGINE="transformers" được KHUYẾN NGHỊ khi chạy nhiều model trong một lần
    (free_local_llm() giải phóng GPU sạch giữa các model). vLLM giữ VRAM qua
    worker state nên có thể cần restart kernel giữa các model.
  * Mỗi model có thể override engine/temperature/max_tokens riêng trong MODELS[].

Output: /kaggle/working/crsd_results/<model_short_name>/  (crsd_results.json,
crsd_all_games.csv, crsd_metrics.json, per-treatment CSV) + một
crsd_all_models.csv gộp + run_manifest.json + crsd_results.zip ở Output tab.
=====================================================================
"""

# =====================================================================
# CELL 1: CẤU HÌNH — SỬA Ở ĐÂY
# =====================================================================

# --- Danh sách model. Mỗi model đã add làm Kaggle input. -------------------- #
# path:        thư mục model trong /kaggle/input/...  (xem bằng "!ls /kaggle/input/")
# short_name:  tên thư mục output + cột "model" trong CSV (phải DUY NHẤT).
# engine:      "transformers" (ổn định, free GPU được) | "vllm".
# (tuỳ chọn)   temperature / max_tokens / max_model_len: override cho riêng model đó.
MODELS = [
    {
        "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
        "short_name": "qwen25-7b-instruct",
        "engine": "transformers",
    },
    {
        "path": "/kaggle/input/gemma-2/transformers/gemma-2-12b-it/1",
        "short_name": "gemma2-12b-it",
        "engine": "transformers",
    },
    # Thêm model khác ở đây, ví dụ:
    # {"path": "/kaggle/input/.../llama-3.1-8b-instruct", "short_name": "llama31-8b-instruct",
    #  "engine": "transformers"},
]

# --- Tham số sinh MẶC ĐỊNH (model có thể override từng cái trong MODELS[]) --- #
DEFAULT_ENGINE = "transformers"   # "transformers" (ổn định offline) | "vllm"
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.8         # 0.7–1.0: cần >0 để 6 agent khác nhau
MAX_TOKENS = 512          # đủ cho reasoning ngắn + dòng ">>> CONTRIBUTION = X"
GPU_UTIL = 0.90           # chỉ dùng cho vllm
TP_SIZE = 1               # tensor parallel (vllm); single GPU = 1
FAIRGAME_VERBOSE_LOGS = "0"

# --- CRSD runner ---
import random  # noqa: E402
from pathlib import Path  # noqa: E402

SEED = 20080219                # reproducibility (ngày publish paper); CHUNG cho mọi model
BATCH_SIZE = 256               # prompts/forward; 0 = cả vòng 1 batch. 7B+96GB: 256 an toàn.
MAX_PARSE_RETRIES = 2          # re-hỏi riêng reply không parse được
RUN_LANGUAGE_BLOCK = True      # Block A+B: 5 langs × neutral × 10
RUN_PERSONALITY_BLOCK = True   # Block C: en × {coop, selfish, risk_averse} × 10
PERSONALITY_LANG = "en"
PERSONALITY_CONDITIONS = ["cooperative", "selfish", "risk_averse"]
TREATMENTS = [90, 50, 10]      # loss probabilities (%)
OUTPUT_DIR = Path("/kaggle/working/crsd_results")
SMOKE_TEST = True              # in 1 reply mẫu + parse khi nạp mỗi model

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
print("📁 Models khai báo:")
for m in MODELS:
    print(f"   - {m['short_name']:<22} exists={Path(m['path']).exists()}  ({m['path']})")
if _missing:
    print("❌ THIẾU file CRSD trong repo (push & re-add Code input):")
    for m in _missing:
        print("   -", m)
else:
    print("✅ CRSD files present (src/crsd + resources/crsd_*).")

# =====================================================================
# CELL 4: Import crsd modules + load templates/configs + build-plan
# =====================================================================
import json

conn = import_local_connector()
send_prompts_global = conn.send_prompts_global

SRC_DIR = Path(conn.__file__).resolve().parent.parent          # .../src
FAIRGAME_BASE = SRC_DIR.parent                                 # repo root chứa src/ + resources/
sys.path.insert(0, str(SRC_DIR / "crsd"))

import crsd_results                                             # noqa: E402
from crsd_game import CRSDGame, run_games_lockstep             # noqa: E402
from crsd_prompt import parse_contribution                     # noqa: E402

TEMPLATE_DIR = FAIRGAME_BASE / "resources" / "crsd_templates"
CONFIG_DIR = FAIRGAME_BASE / "resources" / "crsd_config"
templates = {p.stem.replace("crsd_", ""): p.read_text(encoding="utf-8")
             for p in TEMPLATE_DIR.glob("crsd_*.txt")}
print(f"✅ crsd imported. templates={sorted(templates)} | configs dir={CONFIG_DIR}")


def load_config(loss_prob):
    return json.loads((CONFIG_DIR / f"crsd_p{loss_prob}.json").read_text(encoding="utf-8"))


def params_from_config(cfg):
    return dict(n_players=cfg["nPlayers"], n_rounds=cfg["nRounds"], endowment=cfg["endowment"],
                target=cfg["target"], loss_prob=cfg["treatmentLossProb"],
                contribution_options=tuple(cfg["contributionOptions"]))


def build_games():
    """Tạo MỚI toàn bộ game cho một lần chạy (game giữ state nên phải build lại mỗi model)."""
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
    return games


_n_games = len(build_games())
print(f"🎮 Mỗi model chạy {_n_games} games (= {_n_games * 6 * 10} generations). "
      f"Tổng {len(MODELS)} model → {_n_games * len(MODELS)} games.")

# =====================================================================
# CELL 5: Helpers — load model, chạy 1 model, lưu RIÊNG theo model
# =====================================================================
def load_model(model_cfg):
    """Nạp một model vào GPU (free model trước nếu có)."""
    engine = model_cfg.get("engine", DEFAULT_ENGINE)
    temperature = model_cfg.get("temperature", TEMPERATURE)
    max_tokens = model_cfg.get("max_tokens", MAX_TOKENS)
    max_model_len = model_cfg.get("max_model_len", MAX_MODEL_LEN)
    print(f"🚀 Loading {model_cfg['short_name']} ({engine}) ← {model_cfg['path']}")
    if engine == "vllm":
        conn.init_local_llm(model_cfg["path"], engine="vllm", force=True,
                            max_model_len=max_model_len, temperature=temperature,
                            max_tokens=max_tokens, gpu_memory_utilization=GPU_UTIL,
                            tensor_parallel_size=TP_SIZE)
    else:
        conn.init_local_llm(model_cfg["path"], engine="transformers", force=True,
                            temperature=temperature, max_tokens=max_tokens)
    print(f"✅ {model_cfg['short_name']} loaded.")


def smoke_test(model_cfg):
    probe = CRSDGame("probe", "en", "neutral", ["none"] * 6, templates["en"],
                     dict(n_players=6, n_rounds=10, endowment=40, target=120,
                          loss_prob=90, contribution_options=(0, 2, 4)))
    resp = send_prompts_global(probe.build_round_prompts()[:1], batch_size=0)
    val, ok = parse_contribution(resp[0])
    print(f"🧪 [{model_cfg['short_name']}] sample reply (P1, r1) — 400 ký tự cuối:\n", resp[0][-400:])
    print(f"🧪 Parsed = {val}  (primary_ok={ok}; True = model tuân thủ token)")


def save_model_results(model_cfg, results):
    """Lưu kết quả của MỘT model vào OUTPUT_DIR/<short_name>/, trả về (df, metrics)."""
    short = model_cfg["short_name"]
    model_dir = OUTPUT_DIR / short
    model_dir.mkdir(parents=True, exist_ok=True)

    # gắn metadata model vào từng game (để gộp + truy vết)
    for r in results:
        r["model"] = short
        r["model_path"] = model_cfg["path"]

    (model_dir / "crsd_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    df = crsd_results.to_dataframe(results)
    df.insert(0, "model", short)                       # cột model đứng đầu
    df.to_csv(model_dir / "crsd_all_games.csv", index=False)
    for (p, lang), sub in df.groupby(["treatment_loss_prob", "language"]):
        sub.to_csv(model_dir / f"crsd_p{p}_{lang}.csv", index=False)

    def _ser(summary):
        return {str(k): v for k, v in summary.items()}

    baseline = [r for r in results
                if r["language"] == "en" and r["personality_condition"] == "neutral"]
    metrics = {
        "model": short,
        "baseline_en_neutral_by_treatment": _ser(crsd_results.summarize(baseline)),
        "by_treatment_language": _ser(crsd_results.summarize(
            [r for r in results if r["personality_condition"] == "neutral"],
            key=lambda r: f"p{r['treatment_loss_prob']}_{r['language']}")),
        "by_treatment_personality": _ser(crsd_results.summarize(
            [r for r in results if r["language"] == PERSONALITY_LANG],
            key=lambda r: f"p{r['treatment_loss_prob']}_{r['personality_condition']}")),
        "human_benchmark": crsd_results.HUMAN_BENCHMARK,
    }
    (model_dir / "crsd_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return df, metrics


def print_human_table(short, results):
    baseline = [r for r in results
                if r["language"] == "en" and r["personality_condition"] == "neutral"]
    bsum = crsd_results.summarize(baseline)
    print(f"\n=== [{short}] LLM (en, neutral) vs HUMAN (Milinski 2008) ===")
    print(f"{'p':>4} | {'success LLM/HUM':>16} | {'mean total LLM/HUM':>22} | "
          f"{'fairshare LLM/HUM':>18} | parse_fb")
    for p in TREATMENTS:
        s = bsum.get(p)
        if not s:
            continue
        h = crsd_results.HUMAN_BENCHMARK[p]
        print(f"{p:>4} | {s['success_rate']:.2f} / {h['success_rate']:.2f}        | "
              f"{s['final_total']['mean']:6.1f} / {h['mean_final_total']:6.1f}          | "
              f"{s['fair_sharers_per_group']:.2f} / {h['fair_sharers_per_group']:.2f}          | "
              f"{s['parse_fallback_rate']:.1%}")


def run_one_model(model_cfg):
    """Nạp → smoke test → build games MỚI → chạy lockstep → lưu RIÊNG → free GPU."""
    short = model_cfg["short_name"]
    if not Path(model_cfg["path"]).exists():
        print(f"⏭️  BỎ QUA {short}: path không tồn tại ({model_cfg['path']}).")
        return None
    load_model(model_cfg)
    if SMOKE_TEST:
        smoke_test(model_cfg)

    games = build_games()                              # state-free start cho model này

    def responder(prompts):
        return send_prompts_global(prompts, batch_size=BATCH_SIZE)

    def _progress(done, total):
        print(f"   [{short}] round {done}/{total} xong  ({len(games)} games × 6 prompts/round)")

    print(f"🚀 [{short}] Bắt đầu chạy CRSD...")
    results = run_games_lockstep(games, responder, rng=random.Random(SEED),
                                 max_parse_retries=MAX_PARSE_RETRIES, progress=_progress)
    df, _ = save_model_results(model_cfg, results)
    print_human_table(short, results)
    conn.free_local_llm()                              # giải phóng GPU cho model kế tiếp
    return df

# =====================================================================
# CELL 6: Chạy LẦN LƯỢT tất cả model
# =====================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
all_dfs = []
manifest = {"seed": SEED, "treatments": TREATMENTS, "n_games_per_model": _n_games,
            "run_language_block": RUN_LANGUAGE_BLOCK,
            "run_personality_block": RUN_PERSONALITY_BLOCK, "models": []}

for _cfg in MODELS:
    _df = run_one_model(_cfg)
    if _df is None:
        manifest["models"].append({"short_name": _cfg["short_name"], "status": "skipped"})
        continue
    all_dfs.append(_df)
    manifest["models"].append({
        "short_name": _cfg["short_name"], "path": _cfg["path"],
        "engine": _cfg.get("engine", DEFAULT_ENGINE),
        "temperature": _cfg.get("temperature", TEMPERATURE),
        "n_games": int(len(_df)), "status": "done",
        "output_dir": str(OUTPUT_DIR / _cfg["short_name"]),
    })

(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n✅ Hoàn tất {sum(1 for m in manifest['models'] if m['status'] == 'done')}/{len(MODELS)} model.")

# =====================================================================
# CELL 7: Gộp CSV mọi model (tiện so sánh chéo model)
# =====================================================================
if all_dfs:
    import pandas as pd
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "crsd_all_models.csv", index=False)
    print(f"📊 Gộp {len(all_dfs)} model → {OUTPUT_DIR / 'crsd_all_models.csv'} "
          f"({len(combined)} games).")
    # bảng success-rate theo model × treatment (cell en/neutral)
    base = combined[(combined.language == "en") &
                    (combined.personality_condition == "neutral")]
    piv = base.pivot_table(index="model", columns="treatment_loss_prob",
                           values="reached_target", aggfunc="mean")
    print("\n=== Success rate (en, neutral) theo model × treatment ===")
    print(piv.to_string(float_format=lambda x: f"{x:.2f}"))
else:
    print("⚠️  Không có model nào chạy thành công — kiểm tra path trong MODELS[].")

# =====================================================================
# CELL 8: Zip để download
# =====================================================================
import zipfile

zip_path = Path("/kaggle/working/crsd_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"✅ {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — tải ở Output tab.")
print("➡️  Phân tích từng model (bảng so-human + Fig 2/3):")
for m in manifest["models"]:
    if m.get("status") == "done":
        print(f"    python crsd_analysis.py crsd_results/{m['short_name']}/crsd_results.json")
