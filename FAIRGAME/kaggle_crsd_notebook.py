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
    # {
    #     "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
    #     "short_name": "qwen25-7b-instruct",
    #     "engine": "vllm",
    # },
    {
        "path": "/kaggle/input/models/google/gemma-2/transformers/gemma-2-9b-it/2",
        "short_name": "gemma2-9b-it",
        "engine": "vllm",
    },
    {
        "path": "/kaggle/input/datasets/foundnotkiet/llama-3-1-8b/model_weights",
        "short_name": "llama-3-1-8b",
        "engine": "vllm",
    },
]

# --- Tham số sinh MẶC ĐỊNH (model có thể override từng cái trong MODELS[]) --- #
DEFAULT_ENGINE = "transformers"   # "transformers" (ổn định offline) | "vllm"
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.8         # 0.7–1.0: cần >0 để 6 agent khác nhau
MAX_TOKENS = 512          # đủ cho reasoning ngắn + dòng "CONTRIBUTION = X"
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
# Cover stories ("framings"). "climate" = Milinski's exact climate-protection narrative
# (faithful to the paper, for the human comparison); "neutral" = abstract shared-account
# control. Running both lets you measure the FRAMING EFFECT on the same LLM.
# Set to ["climate"] to replicate the paper only, or ["neutral"] for the abstract control.
FRAMINGS = ["climate"]
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
# CELL 2.5: (TUỲ CHỌN) Cài vLLM OFFLINE từ wheels đã build sẵn
# =====================================================================
# Chỉ cần khi muốn engine="vllm" mà image Kaggle CHƯA có vllm. Yêu cầu:
#   - 1 Dataset chứa toàn bộ .whl (build bằng kaggle_build_vllm_wheels.py,
#     chạy trong notebook Internet ON CÙNG image GPU), đã + Add Input.
#   - Đặt VLLM_WHEELS_DIR cho khớp tên dataset (xem bằng "!ls /kaggle/input/").
# Nếu mọi model dùng engine="transformers" → cell này tự bỏ qua.
import importlib.util
import subprocess

VLLM_WHEELS_DIR = Path("/kaggle/input/datasets/trungkiet/vllm-wheels/vllm_offline_wheels")  # sửa cho đúng tên dataset
VLLM_VERSION = ""   # "" = bản có trong wheels; hoặc ghim "0.6.x" để khớp lúc build

_want_vllm = (DEFAULT_ENGINE == "vllm") or any(
    m.get("engine", DEFAULT_ENGINE) == "vllm" for m in MODELS)
_have_vllm = importlib.util.find_spec("vllm") is not None

if _want_vllm:
    # Tắt sampler FlashInfer: trên GPU mới (Blackwell sm_120) nó JIT-compile và
    # ngã "FlashInfer requires GPUs with sm75 or higher". Sampler PyTorch-native
    # thay thế (không JIT). Set ở kernel → vLLM subprocess thừa hưởng env này.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print("🔧 VLLM_USE_FLASHINFER_SAMPLER=0 (dùng sampler PyTorch-native).")

if not _want_vllm:
    print("ℹ️  Không model nào dùng vllm — bỏ qua cài đặt.")
elif _have_vllm:
    print("✅ vLLM đã có sẵn trên image — không cần cài.")
elif not VLLM_WHEELS_DIR.is_dir():
    raise FileNotFoundError(
        f"Cần engine vllm nhưng không thấy wheels ở {VLLM_WHEELS_DIR}. "
        "Hãy + Add Input dataset wheels (build bằng kaggle_build_vllm_wheels.py), "
        "sửa VLLM_WHEELS_DIR, hoặc đổi engine='transformers'.")
else:
    _spec = "vllm" + (f"=={VLLM_VERSION}" if VLLM_VERSION else "")
    _cmd = [sys.executable, "-m", "pip", "install", "--no-index",
            f"--find-links={VLLM_WHEELS_DIR}", _spec]
    print("📦 Cài vLLM offline:", " ".join(_cmd))
    subprocess.run(_cmd, check=True)
    importlib.invalidate_caches()
    # Sanity-check: torch vừa cài có init được GPU không (bắt lỗi CUDA/driver SỚM,
    # tránh crash sâu lúc vLLM nạp model: "Failed to get device capability ...").
    _probe = ("import torch;"
              "print('torch', torch.__version__, '| cuda', torch.version.cuda,"
              " '| is_available', torch.cuda.is_available());"
              "torch.zeros(1).cuda(); print('GPU_OK')")
    if subprocess.run([sys.executable, "-c", _probe]).returncode != 0:
        raise RuntimeError(
            "torch vừa cài KHÔNG init được GPU — gần như chắc do wheels build SAI "
            "CUDA so với driver Kaggle (vd torch CUDA-13 trên driver cu128). "
            "Hãy build lại wheels bằng kaggle_build_vllm_wheels.py với "
            "PIN_TO_KAGGLE_TORCH=True (giữ torch cu128 của Kaggle), hoặc tạm đổi "
            "engine='transformers'.")
    print("✅ vLLM đã cài từ wheels và torch init GPU OK.")

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
         FAIRGAME_ROOT / "resources" / "crsd_templates" / "crsd_climate_en.txt",
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
# Templates are keyed framing -> language. Filename convention:
#   crsd_<lang>.txt           -> framing "neutral" (default, backward-compatible)
#   crsd_<framing>_<lang>.txt  -> e.g. crsd_climate_en.txt -> framing "climate"
templates = {}
for p in TEMPLATE_DIR.glob("crsd_*.txt"):
    rest = p.stem[len("crsd_"):]                  # "en"  or  "climate_en"
    framing, lang = rest.split("_", 1) if "_" in rest else ("neutral", rest)
    templates.setdefault(framing, {})[lang] = p.read_text(encoding="utf-8")
print(f"✅ crsd imported. framings={sorted(templates)} | "
      f"langs(neutral)={sorted(templates.get('neutral', {}))} | configs dir={CONFIG_DIR}")


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
        # framings to run = FRAMINGS ∩ config ∩ available templates (keeps it robust)
        cfg_framings = FRAMINGS or cfg.get("framings", ["neutral"])
        framings = [f for f in cfg_framings if f in templates]

        for framing in framings:
            tpl = templates[framing]
            if RUN_LANGUAGE_BLOCK:                   # Block A+B: framing × langs × neutral
                for lang in cfg["languages"]:
                    if lang not in tpl:
                        continue
                    for k in range(n_groups):
                        games.append(CRSDGame(f"p{loss_prob}_{framing}_{lang}_neutral_{k}", lang,
                                              "neutral", conds["neutral"], tpl[lang], params,
                                              framing=framing))
            if RUN_PERSONALITY_BLOCK and PERSONALITY_LANG in tpl:   # Block C: framing × en × dispositions
                for cond in PERSONALITY_CONDITIONS:
                    for k in range(n_groups):
                        games.append(CRSDGame(f"p{loss_prob}_{framing}_{PERSONALITY_LANG}_{cond}_{k}",
                                              PERSONALITY_LANG, cond, conds[cond],
                                              tpl[PERSONALITY_LANG], params, framing=framing))
    return games


_n_games = len(build_games())
_active_framings = [f for f in (FRAMINGS or ["neutral"]) if f in templates]
print(f"🎮 Mỗi model chạy {_n_games} games (framings={_active_framings}; = {_n_games * 6 * 10} "
      f"generations). Tổng {len(MODELS)} model → {_n_games * len(MODELS)} games.")

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
    pf = next((f for f in ("climate", "neutral") if f in templates), sorted(templates)[0])
    probe_tpl = templates[pf].get("en") or next(iter(templates[pf].values()))
    probe = CRSDGame("probe", "en", "neutral", ["none"] * 6, probe_tpl,
                     dict(n_players=6, n_rounds=10, endowment=40, target=120,
                          loss_prob=90, contribution_options=(0, 2, 4)), framing=pf)
    resp = send_prompts_global(probe.build_round_prompts()[:1], batch_size=0)
    val, ok = parse_contribution(resp[0])
    print(f"🧪 [{model_cfg['short_name']}] sample reply (P1, r1, framing={pf}) — 400 ký tự cuối:\n", resp[0][-400:])
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
    for (framing, p, lang), sub in df.groupby(["framing", "treatment_loss_prob", "language"]):
        sub.to_csv(model_dir / f"crsd_{framing}_p{p}_{lang}.csv", index=False)

    def _ser(summary):
        return {str(k): v for k, v in summary.items()}

    def _fr(r):                                        # backward-compat for older result dicts
        return r.get("framing", "neutral")

    metrics = {
        "model": short,
        # baseline = English, neutral disposition, split by framing × treatment.
        # The "climate_*" rows are the faithful comparison to Milinski; "neutral_*" is the control.
        "baseline_en_neutral_by_framing_treatment": _ser(crsd_results.summarize(
            [r for r in results if r["language"] == "en" and r["personality_condition"] == "neutral"],
            key=lambda r: f"{_fr(r)}_p{r['treatment_loss_prob']}")),
        "by_framing_treatment_language": _ser(crsd_results.summarize(
            [r for r in results if r["personality_condition"] == "neutral"],
            key=lambda r: f"{_fr(r)}_p{r['treatment_loss_prob']}_{r['language']}")),
        "by_framing_treatment_personality": _ser(crsd_results.summarize(
            [r for r in results if r["language"] == PERSONALITY_LANG],
            key=lambda r: f"{_fr(r)}_p{r['treatment_loss_prob']}_{r['personality_condition']}")),
        "human_benchmark": crsd_results.HUMAN_BENCHMARK,
    }
    (model_dir / "crsd_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return df, metrics


def print_human_table(short, results):
    print(f"\n=== [{short}] LLM (en, neutral) vs HUMAN (Milinski 2008) — by framing ===")
    print(f"{'framing':>8} {'p':>4} | {'success LLM/HUM':>16} | {'mean total LLM/HUM':>22} | "
          f"{'fairshare LLM/HUM':>18} | parse_fb")
    framings = sorted({r.get("framing", "neutral") for r in results
                       if r["language"] == "en" and r["personality_condition"] == "neutral"})
    for framing in framings:
        base = [r for r in results if r["language"] == "en"
                and r["personality_condition"] == "neutral" and r.get("framing", "neutral") == framing]
        bsum = crsd_results.summarize(base)
        for p in TREATMENTS:
            s = bsum.get(p)
            if not s:
                continue
            h = crsd_results.HUMAN_BENCHMARK[p]
            print(f"{framing:>8} {p:>4} | {s['success_rate']:.2f} / {h['success_rate']:.2f}        | "
                  f"{s['final_total']['mean']:6.1f} / {h['mean_final_total']:6.1f}          | "
                  f"{s['fair_sharers_per_group']:.2f} / {h['fair_sharers_per_group']:.2f}          | "
                  f"{s['parse_fallback_rate']:.1%}")
    print("   (HUMAN = Milinski's climate framing → the 'climate' rows are the faithful comparison; "
          "'neutral' rows = abstract control, so climate−neutral = the framing effect on this LLM.)")


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
manifest = {"seed": SEED, "treatments": TREATMENTS, "framings": _active_framings,
            "n_games_per_model": _n_games, "run_language_block": RUN_LANGUAGE_BLOCK,
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
    piv = base.pivot_table(index=["model", "framing"], columns="treatment_loss_prob",
                           values="reached_target", aggfunc="mean")
    print("\n=== Success rate (en, neutral) theo model × framing × treatment ===")
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
