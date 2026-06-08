"""
=====================================================================
FAIRGAME — CRSD (Milinski 2008) runner cell for the OFFLINE Kaggle notebook
=====================================================================
Dán các cell dưới đây VÀO SAU Cell 5 của kaggle_notebook.py (offline).
Tức là: Cell 1–5 của kaggle_notebook.py vẫn dùng để LOAD MODEL như cũ
(init_local_llm đã được gọi, send_prompts_global sẵn sàng). Cell này thay
cho "Cell 6: Chạy FAIRGAME experiments" của notebook gốc.

Yêu cầu trước khi chạy:
  • Đã chạy Cell 1–4 của kaggle_notebook.py → model nạp xong (Qwen2.5-7B-Instruct).
  • Repo upload có thư mục src/crsd/ và resources/crsd_templates/ + resources/crsd_config/.
  • Internet OFF, GPU ON (RTX PRO 6000 96GB).

Run plan (khớp scope đã chốt):
  • Treatments: p = 90 / 50 / 10
  • Block A+B (language bias): 5 ngôn ngữ × neutral × 10 nhóm
  • Block C (personality bias): English × {cooperative, selfish, risk_averse} × 10 nhóm
  → 80 games/treatment × 3 = 240 games. Mỗi game 6 người × 10 vòng = 60 generations.
=====================================================================
"""

# =====================================================================
# CELL CRSD-1: Cấu hình runner
# =====================================================================
import json
import random
import sys
from pathlib import Path

SEED = 20080219                 # reproducibility (ngày xuất bản paper :)
BATCH_SIZE = 256                # prompts/forward; 0 = cả vòng 1 batch. 7B+96GB: 256 an toàn.
MAX_PARSE_RETRIES = 2           # re-hỏi riêng các reply không parse được
RUN_LANGUAGE_BLOCK = True       # Block A+B: 5 langs × neutral
RUN_PERSONALITY_BLOCK = True    # Block C: en × {coop, selfish, risk_averse}
PERSONALITY_LANG = "en"         # ngôn ngữ cho Block C
PERSONALITY_CONDITIONS = ["cooperative", "selfish", "risk_averse"]
OUTPUT_DIR = Path("/kaggle/working/crsd_results")

# =====================================================================
# CELL CRSD-2: Import connector (cùng path đang chạy được ở Cell 4) + crsd modules
# =====================================================================
# Tìm project root (Cell 3 của notebook gốc đã ghi marker).
MARKER_ROOT = Path("/kaggle/working/.fairgame_project_root")
if MARKER_ROOT.exists():
    FAIRGAME_ROOT = Path(MARKER_ROOT.read_text(encoding="utf-8").strip())
else:
    FAIRGAME_ROOT = Path("/kaggle/working/FAIRGAME")  # fallback
print(f"FAIRGAME_ROOT = {FAIRGAME_ROOT}")

# send_prompts_global: thử các path import giống connector (path nào chạy ở Cell 4 thì dùng path đó).
send_prompts_global = None
for modpath in ("legacy.FAIRGAME.src.llm_connectors.local_vllm_connector",
                "src.llm_connectors.local_vllm_connector",
                "FAIRGAME.src.llm_connectors.local_vllm_connector"):
    try:
        mod = __import__(modpath, fromlist=["send_prompts_global"])
        send_prompts_global = mod.send_prompts_global
        print(f"✅ imported send_prompts_global from {modpath}")
        break
    except Exception:
        continue
if send_prompts_global is None:
    raise RuntimeError("Không import được send_prompts_global — kiểm tra Cell 4 đã load model chưa.")

# crsd modules: import phẳng (relative-import shim sẽ tự xử lý)
CRSD_DIR = FAIRGAME_ROOT / "src" / "crsd"
sys.path.insert(0, str(CRSD_DIR))
import crsd_results                                  # noqa: E402
from crsd_game import CRSDGame, run_games_lockstep   # noqa: E402

TEMPLATE_DIR = FAIRGAME_ROOT / "resources" / "crsd_templates"
CONFIG_DIR = FAIRGAME_ROOT / "resources" / "crsd_config"

# =====================================================================
# CELL CRSD-3: Smoke test — model có tuân thủ token >>> CONTRIBUTION = X không?
# =====================================================================
_probe = (TEMPLATE_DIR / "crsd_en.txt").read_text(encoding="utf-8")
_g = CRSDGame("probe", "en", "neutral", ["none"] * 6, _probe,
              dict(n_players=6, n_rounds=10, endowment=40, target=120,
                   loss_prob=90, contribution_options=(0, 2, 4)))
_resp = send_prompts_global(_g.build_round_prompts()[:1], batch_size=0)
print("🧪 Sample reply (Player 1, round 1):\n", _resp[0][-600:])
from crsd_prompt import parse_contribution  # noqa: E402
print("🧪 Parsed:", parse_contribution(_resp[0]))

# =====================================================================
# CELL CRSD-4: Build games theo run plan
# =====================================================================
def load_config(loss_prob):
    return json.loads((CONFIG_DIR / f"crsd_p{loss_prob}.json").read_text(encoding="utf-8"))


def params_from_config(cfg):
    return dict(
        n_players=cfg["nPlayers"], n_rounds=cfg["nRounds"], endowment=cfg["endowment"],
        target=cfg["target"], loss_prob=cfg["treatmentLossProb"],
        contribution_options=tuple(cfg["contributionOptions"]),
    )


templates = {p.stem.replace("crsd_", ""): p.read_text(encoding="utf-8")
             for p in TEMPLATE_DIR.glob("crsd_*.txt")}

games = []
for loss_prob in (90, 50, 10):
    cfg = load_config(loss_prob)
    params = params_from_config(cfg)
    n_groups = cfg["groupsPerCondition"]
    conds = cfg["personalityConditions"]

    if RUN_LANGUAGE_BLOCK:                       # Block A+B: langs × neutral
        for lang in cfg["languages"]:
            for k in range(n_groups):
                games.append(CRSDGame(
                    f"p{loss_prob}_{lang}_neutral_{k}", lang, "neutral",
                    conds["neutral"], templates[lang], params))

    if RUN_PERSONALITY_BLOCK:                    # Block C: en × dispositions
        for cond in PERSONALITY_CONDITIONS:
            for k in range(n_groups):
                games.append(CRSDGame(
                    f"p{loss_prob}_{PERSONALITY_LANG}_{cond}_{k}", PERSONALITY_LANG, cond,
                    conds[cond], templates[PERSONALITY_LANG], params))

print(f"🎮 Tổng số games: {len(games)}  (= {len(games) * 6 * 10} generations)")

# =====================================================================
# CELL CRSD-5: Chạy lockstep (1 batch lớn mỗi vòng cho toàn bộ games)
# =====================================================================
def responder(prompts):
    return send_prompts_global(prompts, batch_size=BATCH_SIZE)


def _progress(done, total):
    print(f"   round {done}/{total} xong ({len(games)} games × 6 prompts/round)")


print("🚀 Bắt đầu chạy CRSD...")
results = run_games_lockstep(
    games, responder, rng=random.Random(SEED),
    max_parse_retries=MAX_PARSE_RETRIES, progress=_progress,
)
print("✅ Hoàn tất tất cả games.")

# =====================================================================
# CELL CRSD-6: Lưu kết quả (JSON đầy đủ + CSV phẳng + metrics)
# =====================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "crsd_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

df = crsd_results.to_dataframe(results)
df.to_csv(OUTPUT_DIR / "crsd_all_games.csv", index=False)
for (p, lang), sub in df.groupby(["treatment_loss_prob", "language"]):
    sub.to_csv(OUTPUT_DIR / f"crsd_p{p}_{lang}.csv", index=False)

# Metrics: theo treatment (baseline en-neutral), theo (treatment,language), theo (treatment,personality)
def _ser(summary):
    return {str(k): v for k, v in summary.items()}

baseline = [r for r in results if r["language"] == "en" and r["personality_condition"] == "neutral"]
metrics = {
    "baseline_en_neutral_by_treatment": _ser(crsd_results.summarize(baseline)),
    "by_treatment_language": _ser(crsd_results.summarize(
        results, key=lambda r: f"p{r['treatment_loss_prob']}_{r['language']}")),
    "by_treatment_personality": _ser(crsd_results.summarize(
        [r for r in results if r["language"] == PERSONALITY_LANG],
        key=lambda r: f"p{r['treatment_loss_prob']}_{r['personality_condition']}")),
    "human_benchmark": crsd_results.HUMAN_BENCHMARK,
}
(OUTPUT_DIR / "crsd_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# In nhanh bảng so với human (baseline en-neutral)
print("\n=== LLM (en, neutral) vs HUMAN (Milinski 2008) ===")
print(f"{'p':>4} | {'success LLM/HUM':>16} | {'mean total LLM/HUM':>20} | {'fairshare LLM/HUM':>18} | parse_fb")
for p in (90, 50, 10):
    s = crsd_results.summarize(baseline).get(p)
    if not s:
        continue
    h = crsd_results.HUMAN_BENCHMARK[p]
    print(f"{p:>4} | {s['success_rate']:.2f} / {h['success_rate']:.2f}      | "
          f"{s['final_total']['mean']:6.1f} / {h['mean_final_total']:6.1f}       | "
          f"{s['fair_sharers_per_group']:.2f} / {h['fair_sharers_per_group']:.2f}        | "
          f"{s['parse_fallback_rate']:.1%}")

print(f"\n📦 Output → {OUTPUT_DIR}")

# =====================================================================
# CELL CRSD-7: Zip để download
# =====================================================================
import zipfile
zip_path = Path("/kaggle/working/crsd_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"✅ {zip_path} ({zip_path.stat().st_size/1024/1024:.2f} MB) — tải ở Output tab.")
