"""
=====================================================================
FAIRGAME × PGG-Punishment — FOLLOW-UP EXPERIMENTS (Kaggle OFFLINE, GPU ON)
=====================================================================
Các thí nghiệm bổ sung cho paper "Antisocial but inconsequential" để trả lời
trước các câu hỏi reviewer Q1:

  1. BIG_BASELINE   — model LỚN (70B AWQ / 27B): institutional null (P−N ≈ 0,
                      antisocial share cao) có biến mất theo scale không?
                      ({N,P} × EN × neutral × n=30)
  2. SMALL_HIGHN    — 3 model gốc 7–9B, baseline {N,P} n=30 (power cho TOST/BF).
  3. NORM_PERSONA   — POSITIVE CONTROL cho null cross-cultural: 8 persona
                      norm-laden (4 pool civic-norm mạnh: Boston, Copenhagen,
                      St.Gallen, Zurich; 4 yếu: Athens, Istanbul, Riyadh,
                      Muscat). Nếu persona giàu chuẩn mực TÁCH được 2 nhóm
                      → null của neutral label là statement về label;
                      nếu vẫn flat → geography thật sự ngoài tầm với của model.
  4. CONGRUENT_LANG — persona thành phố × NGÔN NGỮ BẢN ĐỊA (Riyadh+ar,
                      Muscat+ar, Chengdu+cn, Boston/Nottingham/Melbourne+en):
                      label văn hóa có mạnh hơn khi prompt cùng ngôn ngữ không?
                      So với cell cùng thành phố bằng tiếng Anh từ run gốc.

YÊU CẦU: repo Code input là bản ĐÃ CÓ commit follow-up (pgg_prompt.py có
NORM_PERSONAS_EN + society "norm:<City>"). Cell 3 tự kiểm tra.

CÁCH CHẠY: như kaggle_pgg_punish_notebook.py (GPU ON, Internet OFF, add Code
input + model inputs + vllm wheels nếu cần; chia cell theo "# CELL N").

NGÂN SÁCH 12h (vLLM batch 256; treatment P gọi LLM 2 lần/kỳ):
  * BIG_BASELINE / model:  {N,P} × 30 group → 30×400 + 30×800 = 36 000 gen?
    KHÔNG — mỗi game 4 player × 10 kỳ: N = 30×40 = 1 200; P = 30×80 = 2 400
    → 3 600 gen/model. 70B-AWQ ≈ 30–60 phút/model.
  * SMALL_HIGHN / model: 3 600 gen. NORM_PERSONA / model: {N,P} × 8 persona
    × 10 group → 9 600 gen. CONGRUENT / model: P × 6 cell × 10 → 4 800 gen.
  Cả gói SMALL (3 model) ≈ 54 000 gen ≈ 2–5h; gói BIG ≈ 1.5–3h. Nếu sợ thiếu
  giờ → 2 session (tắt bớt flag).

Output: /kaggle/working/pgg_followup_results/<short>__<tag>/ + pgg_followup_all.csv
+ run_manifest.json + zip.
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
        "gpu_util": 0.92,
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

# --- 3 model gốc của paper ---------------------------------------------------- #
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

# --- BẬT/TẮT từng thí nghiệm -------------------------------------------------- #
RUN_BIG_BASELINE = True     # 1. model lớn × {N,P} baseline n=N_GROUPS_BIG
RUN_SMALL_HIGHN = True      # 2. model nhỏ × {N,P} baseline n=N_GROUPS_HIGHN
RUN_NORM_PERSONA = True     # 3. model nhỏ × {N,P} × 8 norm personas × N_GROUPS_NORM
RUN_CONGRUENT_LANG = True   # 4. model nhỏ × P × 6 cell city×native-language × N_GROUPS_CONG

N_GROUPS_BIG = 30
N_GROUPS_HIGHN = 30
N_GROUPS_NORM = 10
N_GROUPS_CONG = 10

# 8 persona norm-laden (positive control) — khớp NORM_PERSONAS_EN trong pgg_prompt.py
NORM_SOCIETIES = ["norm:Boston", "norm:Copenhagen", "norm:St.Gallen", "norm:Zurich",
                  "norm:Athens", "norm:Istanbul", "norm:Riyadh", "norm:Muscat"]
# (city, language) đồng nhất văn hóa-ngôn ngữ; baseline so sánh = cell city×en của run gốc
CONGRUENT_CELLS = [("Boston", "en"), ("Nottingham", "en"), ("Melbourne", "en"),
                   ("Riyadh", "ar"), ("Muscat", "ar"), ("Chengdu", "cn")]

# --- Tham số sinh mặc định ----------------------------------------------------- #
DEFAULT_ENGINE = "vllm"
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.8
MAX_TOKENS = 512
GPU_UTIL = 0.90
TP_SIZE = 1
FAIRGAME_VERBOSE_LOGS = "0"

SEED = 20080307
BATCH_SIZE = 256
MAX_PARSE_RETRIES = 2
OUTPUT_DIR = Path("/kaggle/working/pgg_followup_results")
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
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print("🔧 VLLM_USE_FLASHINFER_SAMPLER=0 (sampler PyTorch-native).")

if not _want_vllm:
    print("ℹ️  Không model nào dùng vllm — bỏ qua cài đặt.")
elif _have_vllm:
    print("✅ vLLM đã có sẵn trên image — không cần cài.")
elif not VLLM_WHEELS_DIR.is_dir():
    raise FileNotFoundError(
        f"Cần engine vllm nhưng không thấy wheels ở {VLLM_WHEELS_DIR}.")
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
        raise RuntimeError("torch vừa cài KHÔNG init được GPU — build lại wheels "
                           "với PIN_TO_KAGGLE_TORCH=True.")
    print("✅ vLLM đã cài từ wheels và torch init GPU OK.")

# =====================================================================
# CELL 3: Setup FAIRGAME source (copy repo + patch + check follow-up files)
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

_need = [FAIRGAME_ROOT / "src" / "pgg_punish" / "pgg_game.py",
         FAIRGAME_ROOT / "resources" / "pgg_punish_templates" / "pgg_contrib_en.txt",
         FAIRGAME_ROOT / "resources" / "pgg_punish_config" / "pgg_P.json"]
_missing = [str(p) for p in _need if not p.exists()]
_prompt_src = (FAIRGAME_ROOT / "src" / "pgg_punish" / "pgg_prompt.py").read_text(encoding="utf-8")
if "NORM_PERSONAS_EN" not in _prompt_src:
    _missing.append("src/pgg_punish/pgg_prompt.py KHÔNG có NORM_PERSONAS_EN (repo cũ)")
print(f"✅ Code ready — project root: {FAIRGAME_ROOT}")
if _missing:
    raise RuntimeError("❌ Code input là snapshot CŨ — push commit follow-up rồi "
                       "re-add Code input. Thiếu:\n  - " + "\n  - ".join(_missing))
print("✅ pgg_punish follow-up files present.")
for m in _ALL_MODELS:
    print(f"   - {m['short_name']:<22} exists={Path(m['path']).exists()}  ({m['path']})")

# =====================================================================
# CELL 4: Import pgg_punish modules + load templates + build helpers
# =====================================================================
import json

conn = import_local_connector()
send_prompts_global = conn.send_prompts_global

SRC_DIR = Path(conn.__file__).resolve().parent.parent
FAIRGAME_BASE = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR / "pgg_punish"))

import pgg_results                                             # noqa: E402
from pgg_game import PGGGame, run_games_lockstep              # noqa: E402
from pgg_prompt import (parse_contribution, parse_punishment,  # noqa: E402
                        society_block)

# sanity: norm persona render được (sẽ raise KeyError nếu thiếu)
_probe_norm = society_block("en", "norm:Boston")
assert "Boston" in _probe_norm and len(_probe_norm) > 100, "norm persona quá ngắn?"
print("✅ norm persona OK, ví dụ:", _probe_norm[:90], "...")

TEMPLATE_DIR = FAIRGAME_BASE / "resources" / "pgg_punish_templates"
CONFIG_DIR = FAIRGAME_BASE / "resources" / "pgg_punish_config"
templates_contrib = {p.stem.replace("pgg_contrib_", ""): p.read_text(encoding="utf-8")
                     for p in TEMPLATE_DIR.glob("pgg_contrib_*.txt")}
templates_punish = {p.stem.replace("pgg_punish_", ""): p.read_text(encoding="utf-8")
                    for p in TEMPLATE_DIR.glob("pgg_punish_*.txt")}
print(f"✅ pgg_punish imported. langs={sorted(templates_contrib)}")


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


def build_baseline_games(n_groups, tag, treatments=("N", "P")):
    games = []
    for tr in treatments:
        cfg = load_config(tr)
        params = params_from_config(cfg)
        neutral = cfg["personalityConditions"]["neutral"]
        ct, pt = templates_contrib["en"], templates_punish["en"]
        for k in range(n_groups):
            games.append(PGGGame(f"{tr}_en_neutral_{tag}_{k}", "en", "neutral",
                                 neutral, "none", ct, pt, params))
    return games


def build_norm_games(n_groups, treatments=("N", "P")):
    games = []
    for tr in treatments:
        cfg = load_config(tr)
        params = params_from_config(cfg)
        neutral = cfg["personalityConditions"]["neutral"]
        ct, pt = templates_contrib["en"], templates_punish["en"]
        for soc in NORM_SOCIETIES:
            city = soc.split(":", 1)[1]
            for k in range(n_groups):
                games.append(PGGGame(f"{tr}_en_normsoc-{city}_{k}", "en", "neutral",
                                     neutral, soc, ct, pt, params))
    return games


def build_congruent_games(n_groups):
    """Treatment P; persona thành phố bằng NGÔN NGỮ BẢN ĐỊA của pool đó."""
    games = []
    cfg = load_config("P")
    params = params_from_config(cfg)
    neutral = cfg["personalityConditions"]["neutral"]
    for city, lang in CONGRUENT_CELLS:
        ct, pt = templates_contrib[lang], templates_punish[lang]
        for k in range(n_groups):
            games.append(PGGGame(f"P_{lang}_congsoc-{city}_{k}", lang, "neutral",
                                 neutral, city, ct, pt, params))
    return games

# =====================================================================
# CELL 5: Kế hoạch chạy
# =====================================================================
# job = 1 lần nạp model; blocks = các (tag, games_builder)
ENGINE_JOBS = []

if RUN_BIG_BASELINE:
    for m in MODELS_BIG:
        ENGINE_JOBS.append({"model": m, "blocks": [
            ("big", lambda: build_baseline_games(N_GROUPS_BIG, "big"))]})

for m in MODELS_SMALL:
    blocks = []
    if RUN_SMALL_HIGHN:
        blocks.append(("n30", lambda: build_baseline_games(N_GROUPS_HIGHN, "n30")))
    if RUN_NORM_PERSONA:
        blocks.append(("norm", lambda: build_norm_games(N_GROUPS_NORM)))
    if RUN_CONGRUENT_LANG:
        blocks.append(("cong", lambda: build_congruent_games(N_GROUPS_CONG)))
    if blocks:
        ENGINE_JOBS.append({"model": m, "blocks": blocks})


def _count(builder):
    return len(builder())


print(f"🗺️  {len(ENGINE_JOBS)} engine-job:")
for j in ENGINE_JOBS:
    sizes = {tag: _count(b) for tag, b in j["blocks"]}
    print(f"   - {j['model']['short_name']:<22} blocks: {sizes}")

# =====================================================================
# CELL 6: Runner
# =====================================================================
def load_model(model_cfg):
    engine = model_cfg.get("engine", DEFAULT_ENGINE)
    print(f"🚀 Loading {model_cfg['short_name']} ({engine}) ← {model_cfg['path']}")
    if engine == "vllm":
        conn.init_local_llm(model_cfg["path"], engine="vllm", force=True,
                            max_model_len=model_cfg.get("max_model_len", MAX_MODEL_LEN),
                            temperature=model_cfg.get("temperature", TEMPERATURE),
                            max_tokens=model_cfg.get("max_tokens", MAX_TOKENS),
                            gpu_memory_utilization=model_cfg.get("gpu_util", GPU_UTIL),
                            tensor_parallel_size=TP_SIZE)
    else:
        conn.init_local_llm(model_cfg["path"], engine="transformers", force=True,
                            temperature=model_cfg.get("temperature", TEMPERATURE),
                            max_tokens=model_cfg.get("max_tokens", MAX_TOKENS))
    print(f"✅ {model_cfg['short_name']} loaded.")


def smoke_test(model_cfg):
    short = model_cfg["short_name"]
    cfgP = load_config("P")
    pp = params_from_config(cfgP)
    probe = PGGGame("probe", "en", "neutral", cfgP["personalityConditions"]["neutral"],
                    "norm:Athens", templates_contrib["en"], templates_punish["en"], pp)
    probe._rng = random.Random(SEED)
    cresp = send_prompts_global(probe.build_contrib_prompts()[:1], batch_size=0)
    print(f"🧪 [{short}] Contribution reply (norm:Athens persona) — 400 ký tự cuối:\n",
          cresp[0][-400:])
    print("🧪 Parsed contribution:", parse_contribution(cresp[0], probe.options))
    probe.ingest_contributions([10, 10, 10, 10], [""] * 4, [True] * 4)
    presp = send_prompts_global(probe.build_punish_prompts()[:1], batch_size=0)
    print("🧪 Parsed punishment:", parse_punishment(presp[0], probe.n_players - 1, probe.max_punish))


def save_block(model_cfg, tag, results):
    short = model_cfg["short_name"]
    run_dir = OUTPUT_DIR / f"{short}__{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r["model"] = short
        r["model_path"] = model_cfg["path"]
        r["run_tag"] = tag
    (run_dir / "pgg_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    df = pgg_results.to_dataframe(results)
    df.insert(0, "model", short)
    df.insert(1, "run_tag", tag)
    df.to_csv(run_dir / "pgg_all_games.csv", index=False)
    return df


def quick_table(short, tag, results):
    print(f"\n=== [{short} | {tag}] mean contribution theo cell ===")
    bytag = {}
    for r in results:
        key = (r["treatment"], r["language"], r["society"])
        bytag.setdefault(key, []).append(
            sum(r["mean_contribution_by_period"]) / len(r["mean_contribution_by_period"]))
    for key in sorted(bytag):
        vals = bytag[key]
        print(f"   {key[0]} | {key[1]:>2} | {key[2]:<18}: "
              f"{sum(vals) / len(vals):5.2f}  (n={len(vals)})")


all_dfs, manifest = [], {"seed": SEED, "jobs": []}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for job in ENGINE_JOBS:
    m = job["model"]
    if not Path(m["path"]).exists():
        print(f"⏭️  BỎ QUA {m['short_name']}: path không tồn tại.")
        manifest["jobs"].append({"model": m["short_name"], "status": "skipped"})
        continue
    load_model(m)
    if SMOKE_TEST:
        smoke_test(m)
    for tag, builder in job["blocks"]:
        games = builder()
        print(f"🚀 [{m['short_name']} | {tag}] {len(games)} games...")

        def responder(prompts):
            return send_prompts_global(prompts, batch_size=BATCH_SIZE)

        results = run_games_lockstep(
            games, responder, rng=random.Random(SEED),
            max_parse_retries=MAX_PARSE_RETRIES,
            progress=lambda d, t: print(f"   [{m['short_name']}|{tag}] period {d}/{t}"))
        df = save_block(m, tag, results)
        quick_table(m["short_name"], tag, results)
        all_dfs.append(df)
        manifest["jobs"].append({"model": m["short_name"], "tag": tag,
                                 "n_games": int(len(df)), "status": "done"})
    conn.free_local_llm()

(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n✅ Hoàn tất {sum(1 for j in manifest['jobs'] if j.get('status') == 'done')} block.")

# =====================================================================
# CELL 7: Gộp CSV + bảng nhanh norm-persona (positive control)
# =====================================================================
if all_dfs:
    import pandas as pd
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "pgg_followup_all.csv", index=False)
    print(f"📊 {OUTPUT_DIR / 'pgg_followup_all.csv'} ({len(combined)} games).")

    norm = combined[combined.society.str.startswith("norm:", na=False)
                    & (combined.treatment == "P")]
    if len(norm):
        strong = ["norm:Boston", "norm:Copenhagen", "norm:St.Gallen", "norm:Zurich"]
        norm = norm.assign(civic=norm.society.isin(strong).map(
            {True: "strong-norm", False: "weak-norm"}))
        piv = norm.pivot_table(index=["model", "civic"], values="mean_contribution",
                               aggfunc=["mean", "count"])
        print("\n=== POSITIVE CONTROL (P): contribution theo nhóm civic-norm ===")
        print(piv.to_string(float_format=lambda x: f"{x:.2f}"))
        print("   (Nếu strong ≈ weak → geography thật sự ngoài tầm persona;"
              " nếu tách → null của neutral label là về label.)")
else:
    print("⚠️  Không có block nào chạy thành công.")

# =====================================================================
# CELL 8: Zip để download
# =====================================================================
import zipfile

zip_path = Path("/kaggle/working/pgg_followup_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"✅ {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — tải ở Output tab.")
print("➡️  Sau khi tải về: giải nén vào FAIRGAME/results/pgg_followup_results/ rồi chạy "
      "trace/analysis script trong Paper_PGG_Punishment/analysis/.")
