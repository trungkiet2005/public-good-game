# Follow-up experiments cho 2 paper (CRSD-climate & PGG-punishment)

Mục tiêu: vá các điểm reviewer Q1 sẽ đánh — scale (model lớn), power (n=30),
confound tự cộng dồn (running-total control), prompt-paraphrase robustness,
seed/temperature robustness, positive control cho null cross-cultural, và
society×ngôn ngữ đồng nhất. Reasoning-trace analysis chạy LOCAL trên data đã có
(không cần GPU).

## Files

| File | Vai trò |
|---|---|
| `kaggle_crsd_followup_notebook.py` | Notebook Kaggle: BIG_BASELINE, SMALL_HIGHN, TOTAL_CONTROL, PARAPHRASE, SEED_SWEEP, TEMP_SWEEP |
| `kaggle_pgg_followup_notebook.py` | Notebook Kaggle: BIG_BASELINE, SMALL_HIGHN, NORM_PERSONA (positive control), CONGRUENT_LANG |
| `src/crsd` (`show_running_total`) | Param mới: hiển thị tổng tích lũy trong history (control) |
| `resources/crsd_templates/crsd_climatetotal_en.txt` | Framing "climatetotal" — brief nói rõ là CÓ hiện tổng |
| `resources/crsd_templates/crsd_climatepara{2,3}_en.txt` | 2 bản paraphrase trung thực của brief climate |
| `src/pgg_punish/pgg_prompt.py` (`NORM_PERSONAS_EN`, society `norm:<City>`) | 8 persona norm-laden (4 civic mạnh / 4 yếu), không rò rỉ ngôn ngữ game |
| `src/llm_connectors/local_vllm_connector.py` | `seed=` cho engine vLLM + `set_sampling_temperature()` (đổi temp không reload) |
| `../Paper_CRSD_Climate/analysis/trace_analysis.py` | Manipulation check: model có VERBALISE p / tổng tích lũy không (chạy local) |
| `../Paper_PGG_Punishment/analysis/trace_analysis.py` | Motive check: lời nói vs mục tiêu phạt thực tế (chạy local) |

## Trước khi chạy Kaggle

1. **Commit + push** repo này, rồi re-add Code input trên Kaggle (Cell 3 của cả
   2 notebook tự kiểm tra snapshot có đủ file follow-up; thiếu sẽ raise sớm).
2. **Model lớn** (96GB VRAM, 1 GPU): tải bằng `download_model.py` (máy có mạng)
   rồi upload làm Kaggle Dataset:
   - `Qwen/Qwen2.5-72B-Instruct-AWQ` (~41GB) — vLLM tự nhận AWQ từ config.
   - `casperhansen/llama-3.3-70b-instruct-awq` (~40GB).
   - Gemma-2-27B-it có sẵn trên Kaggle Models (bf16 ~55GB, chạy thẳng).
3. Sửa `MODELS_BIG[].path` + `KAGGLE_CODE_INPUT` + `VLLM_WHEELS_DIR` trong Cell 1/2.5/3.

## Kế hoạch session 12h (gợi ý)

| Session | Notebook | Flags bật | Ước lượng |
|---|---|---|---|
| 1 | CRSD followup | RUN_BIG_BASELINE (3 model lớn) | ~3–5h |
| 2 | CRSD followup | SMALL_HIGHN + TOTAL_CONTROL + PARAPHRASE + SEED + TEMP (3 model nhỏ) | ~4–6h |
| 3 | PGG followup | RUN_BIG_BASELINE (3 model lớn) | ~2–4h |
| 4 | PGG followup | SMALL_HIGHN + NORM_PERSONA + CONGRUENT_LANG (3 model nhỏ) | ~4–7h |

(Nếu session dư giờ: gộp 1+2 hoặc 3+4 — notebook chạy tuần tự theo flag.)

## Sau khi chạy

1. Tải zip từ Output tab, giải nén vào:
   - `FAIRGAME/results/crsd_followup_results/`
   - `FAIRGAME/results/pgg_followup_results/`
2. Reasoning-trace (data gốc, chạy được NGAY không cần data mới):
   ```
   cd Paper_CRSD_Climate    && python analysis/trace_analysis.py
   cd Paper_PGG_Punishment  && python analysis/trace_analysis.py
   ```
   (đã chạy 2026-06-11 trên data gốc — kết quả trong analysis/trace_tables.md)
3. Khi data follow-up về đủ → mở rộng `analysis/analyze.py` của từng paper để
   nạp các CSV follow-up (cột `run_tag`/`temperature`/`engine_seed` đã có sẵn
   trong `*_followup_all.csv`).

## Map thí nghiệm → điểm yếu paper

| Block | Trả lời câu hỏi reviewer |
|---|---|
| BIG_BASELINE (cả 2 paper) | "7–9B kém là hiển nhiên" → risk-insensitivity / institutional-null có scale không |
| SMALL_HIGHN | n=10/cell quá mỏng cho null claims → n=30 nuôi TOST/Bayes factor |
| TOTAL_CONTROL (CRSD) | Limitations #2: flatness có phải do không tự cộng được tổng? |
| PARAPHRASE (CRSD) | "Kết quả là artefact của 1 cách viết prompt" |
| SEED/TEMP_SWEEP (CRSD) | "1 seed, 1 temperature" |
| NORM_PERSONA (PGG) | "Null cross-cultural là trivial vì label nghèo" → positive control |
| CONGRUENT_LANG (PGG) | Persona văn hóa + ngôn ngữ bản địa có khôi phục geography không |
| trace_analysis (cả 2) | Cơ chế: risk được verbalise nhưng không điều khiển hành vi; motive nói free-riding nhưng điểm rơi vào cooperator |
