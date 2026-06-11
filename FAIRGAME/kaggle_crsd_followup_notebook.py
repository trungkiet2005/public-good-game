"""
=====================================================================
FAIRGAME × CRSD — FOLLOW-UP EXPERIMENTS (Kaggle OFFLINE, GPU ON)
=====================================================================
Các thí nghiệm bổ sung cho paper "Do language-model agents take catastrophe
risk seriously?" để trả lời trước các câu hỏi reviewer Q1:

  1. BIG_BASELINE   — model LỚN (70B AWQ / 27B / 32B): risk-insensitivity có
                      biến mất theo scale không? (baseline climate, EN, neutral, n=30)
  2. SMALL_HIGHN    — 3 model gốc 7–9B, baseline n=30 group/cell (tăng power
                      cho TOST/Bayes factor; n=10 cũ hơi mỏng).
  3. TOTAL_CONTROL  — framing "climatetotal": HIỂN THỊ tổng tích lũy → giết
                      confound self-summation (Limitations #2 của paper).
  4. PARAPHRASE     — 2 bản diễn đạt lại của brief (climatepara2/3) → chặn
                      critique "kết quả là artefact của một cách viết prompt".
  5. SEED_SWEEP     — 2 engine seed khác → sampling-stream robustness.
  6. TEMP_SWEEP     — temperature 0.3 / 1.2 (không reload model) → decoding
                      robustness.

YÊU CẦU: repo Code input phải là bản ĐÃ CÓ commit follow-up này
(src/crsd hỗ trợ show_running_total, template crsd_climatetotal_en.txt +
crsd_climatepara{2,3}_en.txt, connector có seed= và set_sampling_temperature).
Cell 3 sẽ tự kiểm tra và báo thiếu file.

CÁCH CHẠY:
  1. Notebook Kaggle mới — GPU ON, Internet OFF.
  2. + Add Input: (a) Code: repo public-good-game; (b) các model cần chạy;
     (c) dataset vllm wheels (nếu image chưa có vllm).
  3. Copy file này vào notebook, chia cell theo "# CELL N", sửa Cell 1 + path.
  4. Run Cell 1 → 8.

MODEL LỚN trên RTX PRO 6000 96GB (1 GPU):
  * Qwen2.5-72B-Instruct-AWQ  (~41GB)  → vllm nhận AWQ tự động từ config.json
  * Llama-3.3-70B-Instruct-AWQ (~40GB) → vd bản casperhansen/llama-3.3-70b-instruct-awq
  * Gemma-2-27B-it (bf16 ~55GB) / Qwen2.5-32B-Instruct (bf16 ~65GB) chạy thẳng.
  Tải bằng download_model.py (máy local, có mạng) → upload làm Kaggle Dataset.

NGÂN SÁCH 12h (ước lượng, vLLM batch 256):
  * 1 model 70B-AWQ baseline n=30: 3 treatment × 30 group × 60 gen = 5 400 gen
    ≈ 30–60 phút. CẢ 3 model lớn + nạp model: thoải mái trong một session.
  * Gói SMALL (HIGHN + TOTAL_CONTROL + PARAPHRASE + sweeps) cho cả 3 model nhỏ:
    ≈ 2–4h. Nếu lo thiếu giờ: chạy gói BIG và gói SMALL ở 2 session riêng
    (tắt bớt flag ở Cell 1).

Output: /kaggle/working/crsd_followup_results/<short>__<tag>/ (crsd_results.json
+ crsd_all_games.csv) + crsd_followup_all.csv gộp + run_manifest.json + zip.
=====================================================================
"""

# =====================================================================
# CELL 1: CẤU HÌNH — SỬA Ở ĐÂY
# =====================================================================
import random  # noqa: E402
from pathlib import Path  # noqa: E402

# --- Model LỚN (sửa path theo dataset của bạn; xem "!ls /kaggle/input/") ----- #
MODELS_BIG = [
    {
        "path": "/kaggle/input/datasets/foundnotkiet/qwen25-72b-instruct-awq/model_weights",
        "short_name": "qwen25-72b-awq",
        "engine": "vllm",
        "gpu_util": 0.92,          # 41GB weights + KV cache: 0.92 an toàn trên 96GB
    },
    {
        "path": "/kaggle/input/datasets/foundnotkiet/llama-33-70b-instruct-awq/model_weights",
        "short_name": "llama33-70b-awq",
        "engine": "vllm",
        "gpu_util": 0.92,
    },
    {
        "path": "/kaggle/input/models/google/gemma-2/transformers/gemma-2-27b-it/1",
        "short_name": "gemma2-27b-it",
        "engine": "vllm",
    },
]

# --- 3 model gốc của paper (giữ nguyên path các lần chạy trước) -------------- #
MODELS_SMALL = [
    {
        "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
        "short_name": "qwen25-7b-instruct",
        "engine": "vllm",
    },
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

# --- BẬT/TẮT từng thí nghiệm (xem docstring đầu file) ------------------------ #
RUN_BIG_BASELINE = True    # 1. model lớn × climate baseline, n=N_GROUPS_BIG
RUN_SMALL_HIGHN = True     # 2. model nhỏ × climate baseline, n=N_GROUPS_HIGHN
RUN_TOTAL_CONTROL = True   # 3. model nhỏ × climatetotal (hiện tổng), n=N_GROUPS_CONTROL
RUN_PARAPHRASE = True      # 4. model nhỏ × climatepara2+3, n=N_GROUPS_PARA
RUN_SEED_SWEEP = True      # 5. model nhỏ × climate, n=N_GROUPS_SWEEP × EXTRA_SEEDS
RUN_TEMP_SWEEP = True      # 6. model nhỏ × climate, n=N_GROUPS_SWEEP × SWEEP_TEMPS

N_GROUPS_BIG = 30
N_GROUPS_HIGHN = 30
N_GROUPS_CONTROL = 15
N_GROUPS_PARA = 10
N_GROUPS_SWEEP = 10
EXTRA_SEEDS = [101, 202]          # engine seed (seed 0 = các lần chạy gốc)
SWEEP_TEMPS = [0.3, 1.2]          # default 0.8 đã có từ lần chạy gốc

# --- Tham số sinh mặc định --------------------------------------------------- #
DEFAULT_ENGINE = "vllm"
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.8
MAX_TOKENS = 512
GPU_UTIL = 0.90
TP_SIZE = 1
FAIRGAME_VERBOSE_LOGS = "0"

SEED = 20080219                # settlement-lottery rng (giữ như paper)
BATCH_SIZE = 256
MAX_PARSE_RETRIES = 2
TREATMENTS = [90, 50, 10]
OUTPUT_DIR = Path("/kaggle/working/crsd_followup_results")
SMOKE_TEST = True

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
    """Import local_vllm_connector dù repo dùng prefix nào."""
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
import importlib.util
import subprocess

VLLM_WHEELS_DIR = Path("/kaggle/input/datasets/trungkiet/vllm-wheels/vllm_offline_wheels")
VLLM_VERSION = ""

_ALL_MODELS = (MODELS_BIG if RUN_BIG_BASELINE else []) + MODELS_SMALL
_want_vllm = (DEFAULT_ENGINE == "vllm") or any(
    m.get("engine", DEFAULT_ENGINE) == "vllm" for m in _ALL_MODELS)
_have_vllm = importlib.util.find_spec("vllm") is not None

if _want_vllm:
    # Blackwell sm_120: sampler FlashInfer JIT ngã → ép sampler PyTorch-native.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print("🔧 VLLM_USE_FLASHINFER_SAMPLER=0 (sampler PyTorch-native).")

if not _want_vllm:
    print("ℹ️  Không model nào dùng vllm — bỏ qua cài đặt.")
elif _have_vllm:
    print("✅ vLLM đã có sẵn trên image — không cần cài.")
elif not VLLM_WHEELS_DIR.is_dir():
    raise FileNotFoundError(
        f"Cần engine vllm nhưng không thấy wheels ở {VLLM_WHEELS_DIR}. "
        "Hãy + Add Input dataset wheels, sửa VLLM_WHEELS_DIR, hoặc đổi engine.")
else:
    _spec = "vllm" + (f"=={VLLM_VERSION}" if VLLM_VERSION else "")
    _cmd = [sys.executable, "-m", "pip", "install", "--no-index",
            f"--find-links={VLLM_WHEELS_DIR}", _spec]
    print("📦 Cài vLLM offline:", " ".join(_cmd))
    subprocess.run(_cmd, check=True)
    importlib.invalidate_caches()
    _probe = ("import torch;"
              "print('torch', torch.__version__, '| cuda', torch.version.cuda,"
              " '| is_available', torch.cuda.is_available());"
              "torch.zeros(1).cuda(); print('GPU_OK')")
    if subprocess.run([sys.executable, "-c", _probe]).returncode != 0:
        raise RuntimeError(
            "torch vừa cài KHÔNG init được GPU — wheels build sai CUDA so với "
            "driver Kaggle. Build lại bằng kaggle_build_vllm_wheels.py với "
            "PIN_TO_KAGGLE_TORCH=True.")
    print("✅ vLLM đã cài từ wheels và torch init GPU OK.")

# =====================================================================
# CELL 3: Setup FAIRGAME source (copy repo + patch game_round + check follow-up files)
# =====================================================================
import shutil

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
    """Ghi đè src/game_round.py bản không cần PyPI ``retry``."""
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
    raise RuntimeError("game_round.py còn import ``retry`` mà thiếu patch asset.")


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

# Check: file CRSD gốc + file FOLLOW-UP (template mới, connector mới)
_need = [FAIRGAME_ROOT / "src" / "crsd" / "crsd_game.py",
         FAIRGAME_ROOT / "resources" / "crsd_templates" / "crsd_climate_en.txt",
         FAIRGAME_ROOT / "resources" / "crsd_templates" / "crsd_climatetotal_en.txt",
         FAIRGAME_ROOT / "resources" / "crsd_templates" / "crsd_climatepara2_en.txt",
         FAIRGAME_ROOT / "resources" / "crsd_templates" / "crsd_climatepara3_en.txt",
         FAIRGAME_ROOT / "resources" / "crsd_config" / "crsd_p90.json"]
_missing = [str(p) for p in _need if not p.exists()]
_prompt_src = (FAIRGAME_ROOT / "src" / "crsd" / "crsd_prompt.py").read_text(encoding="utf-8")
if "show_running_total" not in _prompt_src:
    _missing.append("src/crsd/crsd_prompt.py KHÔNG có show_running_total (repo cũ)")
print(f"✅ Code ready — project root: {FAIRGAME_ROOT}")
if _missing:
    raise RuntimeError("❌ Code input là snapshot CŨ — push commit follow-up rồi "
                       "re-add Code input. Thiếu:\n  - " + "\n  - ".join(_missing))
print("✅ CRSD follow-up files present.")
for m in _ALL_MODELS:
    print(f"   - {m['short_name']:<22} exists={Path(m['path']).exists()}  ({m['path']})")

# =====================================================================
# CELL 4: Import crsd modules + load templates + build helpers
# =====================================================================
import json

conn = import_local_connector()
send_prompts_global = conn.send_prompts_global

SRC_DIR = Path(conn.__file__).resolve().parent.parent
FAIRGAME_BASE = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR / "crsd"))

import crsd_results                                             # noqa: E402
from crsd_game import CRSDGame, run_games_lockstep             # noqa: E402
from crsd_prompt import parse_contribution                     # noqa: E402

TEMPLATE_DIR = FAIRGAME_BASE / "resources" / "crsd_templates"
CONFIG_DIR = FAIRGAME_BASE / "resources" / "crsd_config"
templates = {}
for p in TEMPLATE_DIR.glob("crsd_*.txt"):
    rest = p.stem[len("crsd_"):]
    framing, lang = rest.split("_", 1) if "_" in rest else ("neutral", rest)
    templates.setdefault(framing, {})[lang] = p.read_text(encoding="utf-8")
print(f"✅ crsd imported. framings={sorted(templates)}")


def load_config(loss_prob):
    return json.loads((CONFIG_DIR / f"crsd_p{loss_prob}.json").read_text(encoding="utf-8"))


def params_from_config(cfg, show_running_total=False):
    return dict(n_players=cfg["nPlayers"], n_rounds=cfg["nRounds"], endowment=cfg["endowment"],
                target=cfg["target"], loss_prob=cfg["treatmentLossProb"],
                contribution_options=tuple(cfg["contributionOptions"]),
                show_running_total=show_running_total)


def build_block_games(framing, n_groups, tag, show_total=False):
    """EN / neutral / 3 treatments / n_groups — một block thí nghiệm follow-up."""
    games = []
    for loss_prob in TREATMENTS:
        cfg = load_config(loss_prob)
        params = params_from_config(cfg, show_running_total=show_total)
        pers = cfg["personalityConditions"]["neutral"]
        tpl = templates[framing]["en"]
        for k in range(n_groups):
            games.append(CRSDGame(f"p{loss_prob}_{framing}_en_neutral_{tag}_{k}",
                                  "en", "neutral", pers, tpl, params, framing=framing))
    return games


def set_temperature(temp):
    """Đổi temperature KHÔNG reload model (connector mới có sẵn helper)."""
    fn = getattr(conn, "set_sampling_temperature", None)
    if fn is not None:
        fn(temp)
        return
    # Fallback cho snapshot connector cũ: thay SamplingParams trực tiếp.
    sp = conn._GLOBAL_SAMPLING_PARAMS
    if isinstance(sp, dict):
        sp["temperature"] = temp
    else:
        from vllm import SamplingParams
        conn._GLOBAL_SAMPLING_PARAMS = SamplingParams(
            temperature=temp, max_tokens=sp.max_tokens, logprobs=sp.logprobs)
    print(f"[notebook] sampling temperature -> {temp}")

# =====================================================================
# CELL 5: Kế hoạch chạy — mỗi ENGINE-JOB = 1 lần nạp model (model × engine_seed),
#         bên trong là các SUBRUN (temperature × danh sách block).
# =====================================================================
# block spec: (tag, framing, n_groups, show_total)
ENGINE_JOBS = []

if RUN_BIG_BASELINE:
    for m in MODELS_BIG:
        ENGINE_JOBS.append({
            "model": m, "engine_seed": 0,
            "subruns": [{"temperature": TEMPERATURE, "tag": "big",
                         "blocks": [("big", "climate", N_GROUPS_BIG, False)]}],
        })

for m in MODELS_SMALL:
    main_blocks = []
    if RUN_SMALL_HIGHN:
        main_blocks.append(("n30", "climate", N_GROUPS_HIGHN, False))
    if RUN_TOTAL_CONTROL:
        main_blocks.append(("ctl", "climatetotal", N_GROUPS_CONTROL, True))
    if RUN_PARAPHRASE:
        main_blocks.append(("para2", "climatepara2", N_GROUPS_PARA, False))
        main_blocks.append(("para3", "climatepara3", N_GROUPS_PARA, False))
    subruns = []
    if main_blocks:
        subruns.append({"temperature": TEMPERATURE, "tag": "main", "blocks": main_blocks})
    if RUN_TEMP_SWEEP:
        for t in SWEEP_TEMPS:
            subruns.append({"temperature": t, "tag": f"t{t}",
                            "blocks": [(f"t{t}", "climate", N_GROUPS_SWEEP, False)]})
    if subruns:
        ENGINE_JOBS.append({"model": m, "engine_seed": 0, "subruns": subruns})
    if RUN_SEED_SWEEP:
        for s in EXTRA_SEEDS:
            ENGINE_JOBS.append({
                "model": m, "engine_seed": s,
                "subruns": [{"temperature": TEMPERATURE, "tag": f"seed{s}",
                             "blocks": [(f"seed{s}", "climate", N_GROUPS_SWEEP, False)]}],
            })

_tot_games = sum(n * len(TREATMENTS)
                 for j in ENGINE_JOBS for sr in j["subruns"]
                 for _, _, n, _ in sr["blocks"])
print(f"🗺️  {len(ENGINE_JOBS)} engine-job; ≈ {_tot_games} games "
      f"(~{_tot_games * 60} generations chưa kể retry).")
for j in ENGINE_JOBS:
    tags = [f"{sr['tag']}(T={sr['temperature']})" for sr in j["subruns"]]
    print(f"   - {j['model']['short_name']:<22} seed={j['engine_seed']:<4} subruns: {tags}")

# =====================================================================
# CELL 6: Runner
# =====================================================================
def load_model(model_cfg, engine_seed):
    engine = model_cfg.get("engine", DEFAULT_ENGINE)
    kwargs = dict(temperature=TEMPERATURE,
                  max_tokens=model_cfg.get("max_tokens", MAX_TOKENS))
    print(f"🚀 Loading {model_cfg['short_name']} ({engine}, seed={engine_seed}) "
          f"← {model_cfg['path']}")
    if engine == "vllm":
        kwargs.update(max_model_len=model_cfg.get("max_model_len", MAX_MODEL_LEN),
                      gpu_memory_utilization=model_cfg.get("gpu_util", GPU_UTIL),
                      tensor_parallel_size=TP_SIZE)
        try:
            conn.init_local_llm(model_cfg["path"], engine="vllm", force=True,
                                seed=engine_seed, **kwargs)
        except TypeError:
            if engine_seed != 0:
                raise RuntimeError("Connector cũ không nhận seed= — cần repo snapshot mới "
                                   "cho SEED_SWEEP.") from None
            conn.init_local_llm(model_cfg["path"], engine="vllm", force=True, **kwargs)
    else:
        conn.init_local_llm(model_cfg["path"], engine="transformers", force=True, **kwargs)
    print(f"✅ {model_cfg['short_name']} loaded.")


def smoke_test(model_cfg):
    tpl = templates["climate"]["en"]
    probe = CRSDGame("probe", "en", "neutral", ["none"] * 6, tpl,
                     dict(n_players=6, n_rounds=10, endowment=40, target=120,
                          loss_prob=90, contribution_options=(0, 2, 4)), framing="climate")
    resp = send_prompts_global(probe.build_round_prompts()[:1], batch_size=0)
    val, ok = parse_contribution(resp[0])
    print(f"🧪 [{model_cfg['short_name']}] sample reply — 300 ký tự cuối:\n", resp[0][-300:])
    print(f"🧪 Parsed = {val} (primary_ok={ok})")


def save_subrun(model_cfg, engine_seed, subrun, results):
    short = model_cfg["short_name"]
    run_dir = OUTPUT_DIR / f"{short}__{subrun['tag']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r["model"] = short
        r["model_path"] = model_cfg["path"]
        r["temperature"] = subrun["temperature"]
        r["engine_seed"] = engine_seed
        r["run_tag"] = subrun["tag"]
    (run_dir / "crsd_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    df = crsd_results.to_dataframe(results)
    df.insert(0, "model", short)
    df.insert(1, "run_tag", subrun["tag"])
    df.insert(2, "temperature", subrun["temperature"])
    df.insert(3, "engine_seed", engine_seed)
    df.to_csv(run_dir / "crsd_all_games.csv", index=False)
    return df


def quick_table(short, tag, results):
    """In nhanh success-rate + mean total theo framing × treatment cho subrun."""
    framings = sorted({r["framing"] for r in results})
    print(f"\n=== [{short} | {tag}] success / mean-total theo framing × p ===")
    for fr in framings:
        for p in TREATMENTS:
            sub = [r for r in results if r["framing"] == fr and r["treatment_loss_prob"] == p]
            if not sub:
                continue
            succ = sum(r["reached_target"] for r in sub) / len(sub)
            tot = sum(r["group_total"] for r in sub) / len(sub)
            print(f"   {fr:>14} p={p:>2}: success {succ:5.0%} | total {tot:6.1f} (n={len(sub)})")


all_dfs, manifest = [], {"settlement_seed": SEED, "treatments": TREATMENTS, "jobs": []}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for job in ENGINE_JOBS:
    m = job["model"]
    if not Path(m["path"]).exists():
        print(f"⏭️  BỎ QUA {m['short_name']}: path không tồn tại.")
        manifest["jobs"].append({"model": m["short_name"], "status": "skipped"})
        continue
    load_model(m, job["engine_seed"])
    if SMOKE_TEST:
        smoke_test(m)
    for subrun in job["subruns"]:
        set_temperature(subrun["temperature"])
        games = []
        for tag, framing, n_groups, show_total in subrun["blocks"]:
            games.extend(build_block_games(framing, n_groups, tag, show_total))
        print(f"🚀 [{m['short_name']} | {subrun['tag']}] {len(games)} games "
              f"(T={subrun['temperature']}, engine_seed={job['engine_seed']})...")

        def responder(prompts):
            return send_prompts_global(prompts, batch_size=BATCH_SIZE)

        results = run_games_lockstep(
            games, responder, rng=random.Random(SEED),
            max_parse_retries=MAX_PARSE_RETRIES,
            progress=lambda d, t: print(f"   [{m['short_name']}|{subrun['tag']}] round {d}/{t}"))
        df = save_subrun(m, job["engine_seed"], subrun, results)
        quick_table(m["short_name"], subrun["tag"], results)
        all_dfs.append(df)
        manifest["jobs"].append({"model": m["short_name"], "tag": subrun["tag"],
                                 "temperature": subrun["temperature"],
                                 "engine_seed": job["engine_seed"],
                                 "n_games": int(len(df)), "status": "done"})
    conn.free_local_llm()

(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n✅ Hoàn tất {sum(1 for j in manifest['jobs'] if j.get('status') == 'done')} subrun.")

# =====================================================================
# CELL 7: Gộp CSV
# =====================================================================
if all_dfs:
    import pandas as pd
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "crsd_followup_all.csv", index=False)
    print(f"📊 {OUTPUT_DIR / 'crsd_followup_all.csv'} ({len(combined)} games).")
    piv = combined.pivot_table(index=["model", "run_tag", "framing"],
                               columns="treatment_loss_prob",
                               values="group_total", aggfunc="mean")
    print("\n=== Mean group-total theo model × tag × framing × p ===")
    print(piv.to_string(float_format=lambda x: f"{x:.1f}"))
else:
    print("⚠️  Không có subrun nào chạy thành công.")

# =====================================================================
# CELL 8: Zip để download
# =====================================================================
import zipfile

zip_path = Path("/kaggle/working/crsd_followup_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"✅ {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — tải ở Output tab.")
print("➡️  Sau khi tải về: giải nén vào FAIRGAME/results/crsd_followup_results/ rồi chạy "
      "trace/analysis script trong Paper_CRSD_Climate/analysis/.")
