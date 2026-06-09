"""
=====================================================================
FAIRGAME × PGG-Punishment — Kaggle OFFLINE notebook (Internet OFF, GPU ON)  ── MULTI-MODEL
=====================================================================
Replicate Herrmann, Thoeni & Gaechter (2008) "Antisocial Punishment Across
Societies" — public goods game WITH/WITHOUT punishment — bằng LLM agents, chạy
local trên GPU Kaggle. Nạp LẦN LƯỢT nhiều model (vd Gemma-2-9B, Llama-3.1-8B),
chạy PGG±P cho từng model, và lưu kết quả RIÊNG theo từng model.
Song song với CRSD; KHÔNG đụng tới module CRSD.

CÁCH CHẠY:
  1. Tạo notebook Kaggle mới — GPU: ON, Internet: OFF.
  2. + Add Input:
       a) Code: repo public-good-game (read-only), CHỨA src/pgg_punish/ +
          resources/pgg_punish_*.
       b) Model(s): add MỖI model làm một input (Gemma-2-9B, Llama-3.1-8B, ...).
  3. Copy file này vào notebook, chia cell theo "# CELL N".
  4. Sửa MODELS[] + KAGGLE_CODE_INPUT ở Cell 1/3 cho đúng path của bạn.
       Chạy "!ls /kaggle/input/" để xem path thực của từng model.
  5. Run lần lượt Cell 1 → 8.

ĐA MODEL:
  * Mỗi model nạp → smoke test → chạy PGG±P → lưu RIÊNG → free GPU rồi mới sang
    model kế tiếp. free_local_llm() giải phóng GPU sạch giữa các model.
  * vLLM giữ VRAM qua worker state nên nếu free không sạch giữa 2 model vLLM, có
    thể cần restart kernel rồi chạy tiếp model còn lại; "transformers" thì free
    luôn sạch. Mỗi model có thể override engine/temperature/max_tokens riêng trong MODELS[].

Output: /kaggle/working/pgg_punish_results/<model_short_name>/  (pgg_results.json,
pgg_all_games.csv, pgg_metrics.json, per-treatment CSV) + một
pgg_all_models.csv gộp + run_manifest.json + pgg_punish_results.zip ở Output tab.

LƯU Ý TẢI: treatment P gọi LLM 2 lần/kỳ (đóng góp + trừng phạt). Full design có thể
~28k generation/model — dùng các flag RUN_*_BLOCK + cắt SOCIETIES + giảm
groupsPerCondition để vừa quota GPU.
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
TEMPERATURE = 0.8         # 0.7–1.0: cần >0 để 4 agent khác nhau
MAX_TOKENS = 512          # đủ reasoning ngắn + dòng ">>> CONTRIBUTION/DEDUCT"
GPU_UTIL = 0.90           # chỉ dùng cho vllm
TP_SIZE = 1               # tensor parallel (vllm); single GPU = 1
FAIRGAME_VERBOSE_LOGS = "0"

# --- PGG runner ---
import random  # noqa: E402
from pathlib import Path  # noqa: E402

SEED = 20080307                # reproducibility (ngày publish paper: 7 Mar 2008); CHUNG cho mọi model
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
SMOKE_TEST = True              # in 1 reply mẫu đóng góp + trừng phạt + parse khi nạp mỗi model

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

# Kiểm tra các file PGG-Punishment có mặt trong repo đã upload chưa
_need = [FAIRGAME_ROOT / "src" / "pgg_punish" / "pgg_game.py",
         FAIRGAME_ROOT / "resources" / "pgg_punish_templates" / "pgg_contrib_en.txt",
         FAIRGAME_ROOT / "resources" / "pgg_punish_templates" / "pgg_punish_en.txt",
         FAIRGAME_ROOT / "resources" / "pgg_punish_config" / "pgg_P.json"]
_missing = [str(p) for p in _need if not p.exists()]
print(f"✅ Code ready — project root: {FAIRGAME_ROOT}")
print("📁 Models khai báo:")
for m in MODELS:
    print(f"   - {m['short_name']:<22} exists={Path(m['path']).exists()}  ({m['path']})")
if _missing:
    print("❌ THIẾU file PGG-Punishment trong repo (push & re-add Code input):")
    for m in _missing:
        print("   -", m)
else:
    print("✅ pgg_punish files present (src/pgg_punish + resources/pgg_punish_*).")

# =====================================================================
# CELL 4: Import pgg_punish modules + load templates/configs + build-plan
# =====================================================================
import json

conn = import_local_connector()
send_prompts_global = conn.send_prompts_global

SRC_DIR = Path(conn.__file__).resolve().parent.parent          # .../src
FAIRGAME_BASE = SRC_DIR.parent                                 # repo root chứa src/ + resources/
sys.path.insert(0, str(SRC_DIR / "pgg_punish"))

import pgg_results                                             # noqa: E402
from pgg_game import PGGGame, run_games_lockstep              # noqa: E402
from pgg_prompt import parse_contribution, parse_punishment    # noqa: E402

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


def build_games():
    """Tạo MỚI toàn bộ game cho một lần chạy (game giữ state nên phải build lại mỗi model)."""
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
    return games


_games_preview = build_games()
_n_games = len(_games_preview)
_n_p = sum(1 for g in _games_preview if g.treatment == "P")
print(f"🎮 Mỗi model chạy {_n_games} games (P games: {_n_p}).")
print(f"   ≈ {_n_games * 4 * 10} generation đóng góp + {_n_p * 4 * 10} generation trừng phạt / model. "
      f"Tổng {len(MODELS)} model → {_n_games * len(MODELS)} games.")

# =====================================================================
# CELL 5: Helpers — load model, smoke test, chạy 1 model, lưu RIÊNG theo model
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
    """In 1 reply đóng góp + 1 reply trừng phạt + kết quả parse cho model hiện tại."""
    short = model_cfg["short_name"]
    cfgP = load_config("P")
    pp = params_from_config(cfgP)
    probe = PGGGame("probe", "en", "neutral", cfgP["personalityConditions"]["neutral"], "none",
                    templates_contrib["en"], templates_punish["en"], pp)
    probe._rng = random.Random(SEED)
    cresp = send_prompts_global(probe.build_contrib_prompts()[:1], batch_size=0)
    print(f"🧪 [{short}] Contribution reply (Member 1, period 1) — 500 ký tự cuối:\n", cresp[0][-500:])
    print("🧪 Parsed contribution:", parse_contribution(cresp[0], probe.options),
          "(primary_ok=True nghĩa là model tuân thủ token)")
    # đẩy 1 lượt đóng góp giả để vào giai đoạn trừng phạt, rồi test token DEDUCT
    probe.ingest_contributions([10, 10, 10, 10], [""] * 4, [True] * 4)
    presp = send_prompts_global(probe.build_punish_prompts()[:1], batch_size=0)
    print(f"\n🧪 [{short}] Deduction reply (Member 1, period 1) — 500 ký tự cuối:\n", presp[0][-500:])
    print("🧪 Parsed punishment:", parse_punishment(presp[0], probe.n_players - 1, probe.max_punish))


def save_model_results(model_cfg, results):
    """Lưu kết quả của MỘT model vào OUTPUT_DIR/<short_name>/, trả về (df, metrics)."""
    short = model_cfg["short_name"]
    model_dir = OUTPUT_DIR / short
    model_dir.mkdir(parents=True, exist_ok=True)

    # gắn metadata model vào từng game (để gộp + truy vết)
    for r in results:
        r["model"] = short
        r["model_path"] = model_cfg["path"]

    (model_dir / "pgg_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    df = pgg_results.to_dataframe(results)
    df.insert(0, "model", short)                       # cột model đứng đầu
    df.to_csv(model_dir / "pgg_all_games.csv", index=False)
    for (tr, lang), sub in df.groupby(["treatment", "language"]):
        sub.to_csv(model_dir / f"pgg_{tr}_{lang}.csv", index=False)

    def _ser(summary):
        return {str(k): v for k, v in summary.items()}

    baseline = [r for r in results if r["language"] == "en"
                and r["personality_condition"] == "neutral" and r["society"] == "none"]
    society_games = [r for r in results if r["society"] != "none"]
    metrics = {
        "model": short,
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
    (model_dir / "pgg_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return df, metrics


def print_human_table(short, results):
    # --- Bảng: contribution LLM (en, neutral) vs HUMAN grand-mean, theo treatment ---
    baseline = [r for r in results if r["language"] == "en"
                and r["personality_condition"] == "neutral" and r["society"] == "none"]
    bsum = pgg_results.summarize(baseline)
    hum = pgg_results.HUMAN_BENCHMARK
    hum_mean = {"N": sum(hum["N_mean_contribution"].values()) / len(hum["N_mean_contribution"]),
                "P": sum(hum["P_mean_contribution"].values()) / len(hum["P_mean_contribution"])}
    print(f"\n=== [{short}] LLM (en, neutral) vs HUMAN grand-mean (Herrmann 2008) ===")
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
    society_games = [r for r in results if r["society"] != "none"]
    if society_games:
        soc_sum = pgg_results.summarize(
            [r for r in society_games if r["treatment"] == "P"],
            key=lambda r: r["society"])
        anti_vals, coop_vals = [], []
        print(f"\n=== [{short}] Cross-societal (treatment P): mean contribution per society ===")
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
        print(f"   [{short}] period {done}/{total} xong  ({len(games)} games; game P gọi 2 batch/kỳ)")

    print(f"🚀 [{short}] Bắt đầu chạy PGG±Punishment...")
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
            "run_personality_block": RUN_PERSONALITY_BLOCK,
            "run_society_block": RUN_SOCIETY_BLOCK, "models": []}

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
    combined.to_csv(OUTPUT_DIR / "pgg_all_models.csv", index=False)
    print(f"📊 Gộp {len(all_dfs)} model → {OUTPUT_DIR / 'pgg_all_models.csv'} "
          f"({len(combined)} games).")
    # bảng mean-contribution theo model × treatment (cell en/neutral/none)
    base = combined[(combined.language == "en") &
                    (combined.personality_condition == "neutral") &
                    (combined.society == "none")]
    piv = base.pivot_table(index="model", columns="treatment",
                           values="mean_contribution", aggfunc="mean")
    print("\n=== Mean contribution (en, neutral) theo model × treatment ===")
    print(piv.to_string(float_format=lambda x: f"{x:.2f}"))
else:
    print("⚠️  Không có model nào chạy thành công — kiểm tra path trong MODELS[].")

# =====================================================================
# CELL 8: Zip để download
# =====================================================================
import zipfile

zip_path = Path("/kaggle/working/pgg_punish_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"✅ {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — tải ở Output tab.")
print("➡️  Phân tích từng model:")
for m in manifest["models"]:
    if m.get("status") == "done":
        print(f"    python pgg_punish_analysis.py pgg_punish_results/{m['short_name']}/pgg_results.json")
