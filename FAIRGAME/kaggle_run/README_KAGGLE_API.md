# Chạy code FAIRGAME trên Kaggle bằng API (chế độ OFFLINE: GPU ON, Internet OFF)

Quy trình này cho phép bạn **sửa code ở máy → đẩy lên Kaggle chạy bằng GPU → tải kết quả về**,
không cần copy-paste thủ công vào trình duyệt.

---

## Bước 1 — Lấy API token (kaggle.json)

1. Đăng nhập kaggle.com → bấm avatar góc phải → **Settings**.
2. Kéo xuống mục **API** → bấm **Create New Token**. Trình duyệt tải về file `kaggle.json`.
3. Đặt file vào đúng chỗ trên Windows:
   ```
   C:\Users\<tên-của-bạn>\.kaggle\kaggle.json
   ```
   (Nếu chưa có thư mục `.kaggle`, tạo mới. Mở PowerShell và chạy:
   `mkdir $HOME\.kaggle` rồi chép `kaggle.json` đã tải vào đó.)
4. Cài thư viện:
   ```
   pip install kaggle
   ```
5. Kiểm tra hoạt động:
   ```
   kaggle kernels list --mine
   ```
   Nếu in ra danh sách (kể cả rỗng) là OK. Nếu báo lỗi 401 → token sai chỗ.

> ⚠️ `kaggle.json` chứa khoá bí mật — **không** commit lên Git, không chia sẻ.

---

## Bước 2 — Chuẩn bị "đầu vào" (inputs) trên Kaggle

Chế độ OFFLINE không có internet, nên mọi thứ phải được gắn sẵn làm **input**. File
`kaggle_pgg_punish_notebook.py` cần 3 loại input (xem các path nó tham chiếu trong Cell 1/3):

| Loại | Là gì | Khai báo trong metadata | Path trong notebook |
|------|-------|--------------------------|---------------------|
| **Code repo** | Source FAIRGAME (`src/`, `resources/`) | `kernel_sources` | `/kaggle/input/notebooks/...` |
| **Model** | Trọng số LLM (Gemma, Llama…) | `model_sources` hoặc `dataset_sources` | `/kaggle/input/models/...` hoặc `/kaggle/input/datasets/...` |
| **vLLM wheels** | (nếu image Kaggle chưa có vllm) | `dataset_sources` | `/kaggle/input/datasets/.../vllm_offline_wheels` |

- **Model**: thêm từ Kaggle (Models hoặc Dataset bạn đã upload). Lấy slug ở URL trang model/dataset.
- **Code repo**: cách đang dùng là một **Kaggle Notebook liên kết Git** với repo public-good-game →
  khai trong `kernel_sources` (vd `trungkiet/git-public-good-game`).
- Muốn xem path thực: tạm chạy một cell `!ls /kaggle/input/` trên web một lần để lấy đúng chuỗi.

---

## Bước 3 — Sửa `pgg_punish/kernel-metadata.json`

Mở file và thay cho đúng tài khoản + input của bạn:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/fairgame-pgg-punish",   // <- helper tự điền username nếu bạn để nguyên
  "title": "FAIRGAME PGG Punishment (offline GPU)",
  "code_file": "kaggle_pgg_punish_notebook.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",        // GPU ON
  "enable_tpu": "false",
  "enable_internet": "false",  // Internet OFF
  "dataset_sources": ["foundnotkiet/llama-3-1-8b", "trungkiet/vllm-wheels"],
  "kernel_sources":  ["trungkiet/git-public-good-game"],
  "model_sources":   ["google/gemma-2/transformers/gemma-2-9b-it/2"]
}
```

Các slug ở trên chỉ là **ví dụ khớp với mặc định trong notebook** — đổi sang slug thật của bạn.

> Lưu ý slug `id`: chỉ gồm chữ thường, số và dấu gạch ngang. Đổi `id` = đổi sang kernel khác
> (sẽ tạo notebook mới trên Kaggle); giữ nguyên `id` = đẩy version mới đè lên notebook cũ.

---

## Bước 4 — Sửa cấu hình trong notebook (Cell 1)

Trong `FAIRGAME/kaggle_pgg_punish_notebook.py`, sửa:
- `MODELS[]` — path + `short_name` từng model cho khớp `/kaggle/input/...`.
- `KAGGLE_CODE_INPUT` — path tới repo source đã gắn.
- `VLLM_WHEELS_DIR` — nếu dùng vllm offline.

Đây chính là lợi ích: sửa ở máy bằng editor quen thuộc, rồi đẩy lên.

---

## Bước 5 — Đẩy & chạy bằng helper

Từ thư mục `FAIRGAME/kaggle_run/`:

```bash
# Đẩy + chờ chạy xong + tải output về (output/ trong thư mục kernel)
python run_on_kaggle.py --kernel pgg_punish --watch --download

# Hoặc chỉ đẩy (mỗi lần push = một "Save Version" mới)
python run_on_kaggle.py --kernel pgg_punish

# Xem trạng thái
python run_on_kaggle.py --kernel pgg_punish --status
```

Helper sẽ tự copy `kaggle_pgg_punish_notebook.py` vào thư mục kernel rồi `kaggle kernels push`.
Tương đương các lệnh thủ công:
```bash
kaggle kernels push   -p pgg_punish
kaggle kernels status YOUR_KAGGLE_USERNAME/fairgame-pgg-punish
kaggle kernels output YOUR_KAGGLE_USERNAME/fairgame-pgg-punish -p pgg_punish/output -o
```

---

## Giới hạn cần nhớ

- Kaggle giới hạn **số phiên chạy đồng thời** (CPU/GPU, tương tác/commit) và **quota giờ GPU/tuần**.
  Vượt thì version mới sẽ xếp hàng chờ. Quản lý/Stop phiên trong editor (mục *Active Events*).
- **Không** dùng nhiều tài khoản để lách giới hạn — vi phạm điều khoản Kaggle, có thể bị khoá.
  Cách hợp lệ: đẩy các version **tuần tự** (helper có `--watch` chờ xong mới chạy cái tiếp).

---

## Mở rộng sang file khác

`run_on_kaggle.py` đã khai sẵn 3 kernel trong dict `KERNELS`:
`pgg_punish`, `crsd_followup`, `online`. Mỗi cái cần một thư mục con chứa `kernel-metadata.json`
riêng (hiện mới tạo `pgg_punish/`). Tạo thêm `crsd_followup/` hoặc `online/` tương tự khi cần.
Lưu ý `online` thì đặt `enable_internet: "true"`.
