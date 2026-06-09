"""
=====================================================================
Build vLLM OFFLINE wheels for Kaggle  (Internet ON → tạo Dataset wheels)
=====================================================================
MỤC ĐÍCH
  Notebook CRSD chạy OFFLINE (Internet OFF) muốn dùng engine vLLM, nhưng image
  Kaggle có thể CHƯA cài vllm và không pip được khi offline. Script này TẢI
  TRƯỚC toàn bộ wheel (.whl) của vllm + dependency để bạn lưu thành một Kaggle
  Dataset, rồi cài offline ở notebook chính bằng:

      pip install --no-index --find-links=<wheels_dir> vllm

PHIÊN BẢN CUDA (đọc kỹ):
  Trên image Kaggle Blackwell (RTX PRO 6000, SM 12.0) driver 580 / CUDA 13.0:
  bản "lấy mới nhất" (vllm + torch 2.11 + CUDA 13) CHẠY TỐT — driver hỗ trợ
  CUDA 13. KHÔNG cần ghim cu128. (Lỗi "FlashInfer requires sm75 or higher" mà
  bạn gặp KHÔNG do CUDA — đó là sampler FlashInfer JIT; sửa bằng env
  VLLM_USE_FLASHINFER_SAMPLER=0 trong notebook, không liên quan build wheels.)
  => Mặc định PIN_TO_KAGGLE_TORCH=False (lấy bản mới nhất).
     CHỈ bật True nếu driver CŨ không hỗ trợ CUDA mà vLLM kéo về (xem nvidia-smi
     field "CUDA Version"): khi đó ghim torch theo Kaggle để pip lùi vLLM cho hợp.

CÁCH DÙNG (KHUYẾN NGHỊ — khớp môi trường 100%):
  1. Tạo notebook Kaggle MỚI, **Internet: ON**, GPU: ON
     (chọn ĐÚNG image GPU sẽ chạy notebook offline → Python + CUDA + torch khớp).
  2. Dán nguyên file này vào 1 cell → Run. Script tự đọc torch của Kaggle, ghim,
     và tải đúng cu<XXX> channel.
  3. Wheels nằm ở OUT_DIR (= /kaggle/working/vllm_offline_wheels).
  4. "Save Version" (Save & Run All) → mở Output → "Create Dataset" từ thư mục
     wheels (đặt tên ví dụ: "vllm-offline-wheels").
  5. Sang notebook OFFLINE: + Add Input dataset đó, set VLLM_WHEELS_DIR ở CELL 2.5
     cho khớp tên, đặt engine="vllm".

LƯU Ý
  * BẮT BUỘC chạy trong Kaggle Internet-ON CÙNG image GPU (Linux x86_64, đúng
    Python + driver). Tải trên Windows sẽ ra wheel sai nền tảng.
  * Nếu resolver báo KHÔNG có vLLM nào hợp torch của Kaggle (vd torch quá mới,
    chưa vLLM nào support cu128): xem CẢNH BÁO cuối script — lúc đó vLLM chưa
    chạy được trên image này, nên dùng engine="transformers".
=====================================================================
"""
import subprocess
import sys
from pathlib import Path

# --- CẤU HÌNH --------------------------------------------------------------- #
VLLM_VERSION = ""          # "" = để pip chọn bản mới nhất HỢP với torch đã ghim.
OUT_DIR = Path("/kaggle/working/vllm_offline_wheels")
EXTRA_PACKAGES = []        # gói phụ muốn kèm, vd ["flashinfer-python"]
PIN_TO_KAGGLE_TORCH = False  # True CHỈ khi driver cũ không hỗ trợ CUDA vLLM kéo về.
# --------------------------------------------------------------------------- #

print(f"🐍 Python {sys.version.split()[0]} | nền tảng {sys.platform}")
if sys.platform.startswith("win"):
    print("⚠️  Đang chạy trên Windows — wheel tải về SẼ SAI nền tảng cho Kaggle. "
          "Hãy chạy file này trong notebook Kaggle Internet-ON.")

OUT_DIR.mkdir(parents=True, exist_ok=True)
spec = "vllm" + (f"=={VLLM_VERSION}" if VLLM_VERSION else "")
targets = [spec, *EXTRA_PACKAGES]

# --- Ghim torch theo Kaggle + xác định CUDA channel ------------------------- #
constraints_file = OUT_DIR.parent / "vllm_torch_constraints.txt"
extra_index = []
if PIN_TO_KAGGLE_TORCH:
    try:
        import torch
        pins = []
        for name in ("torch", "torchvision", "torchaudio"):
            try:
                pins.append(f"{name}=={__import__(name).__version__}")
            except Exception:  # noqa: BLE001
                pass  # gói không có thì bỏ qua
        constraints_file.write_text("\n".join(pins) + "\n", encoding="utf-8")
        cuda = (getattr(torch.version, "cuda", "") or "").replace(".", "")
        if cuda:
            extra_index = ["--extra-index-url",
                           f"https://download.pytorch.org/whl/cu{cuda}"]
        print(f"📌 Ghim theo torch Kaggle: {pins}")
        print(f"📌 CUDA channel: cu{cuda or '?'}  (giữ nguyên driver hiện tại)")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Không đọc được torch ({e}); build sẽ để vLLM tự kéo torch "
              "(RỦI RO sai CUDA). Cài lại torch hoặc tự set PIN thủ công.")
        PIN_TO_KAGGLE_TORCH = False

constraint_args = (["-c", str(constraints_file)]
                   if PIN_TO_KAGGLE_TORCH else [])

# 1) Nâng pip để resolver tải đúng wheel.
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)

# 2) Tải TOÀN BỘ closure dạng wheel.
#    -c constraints  → giữ torch của Kaggle, pip tự lùi vLLM cho hợp.
#    --extra-index-url cu<XXX> → lấy torch đúng biến thể CUDA (nếu phải tải).
cmd = [sys.executable, "-m", "pip", "download",
       "--dest", str(OUT_DIR),
       "--prefer-binary",
       *constraint_args, *extra_index,
       *targets]
print("\n⬇️   ", " ".join(cmd))
ret_dl = subprocess.run(cmd).returncode
if ret_dl != 0:
    print("\n❌ pip download lỗi. Nếu lý do là 'ResolutionImpossible' / không có "
          "vLLM nào hợp torch đã ghim → image Kaggle quá mới, CHƯA có vLLM cho "
          "torch này. Tạm dùng engine='transformers', hoặc thử "
          "PIN_TO_KAGGLE_TORCH=False rồi KIỂM TRA `nvidia-smi` xem driver có hỗ "
          "trợ CUDA của torch mà vLLM kéo về không.")
    sys.exit(1)

wheels = sorted(p for p in OUT_DIR.iterdir() if p.is_file())
total_mb = sum(f.stat().st_size for f in wheels) / 1024 / 1024
n_sdist = sum(1 for f in wheels if f.suffix in (".gz", ".zip"))
torch_whls = [f.name for f in wheels if f.name.lower().startswith("torch-")]
print(f"\n✅ Tải {len(wheels)} file → {OUT_DIR}  ({total_mb:.1f} MB)")
print(f"🔦 torch wheel kéo về: {torch_whls or '(không có → dùng torch sẵn của Kaggle ✅)'}")
if n_sdist:
    print(f"⚠️  Có {n_sdist} sdist (.tar.gz/.zip) — phải BUILD khi cài offline (dễ fail). "
          "Cân nhắc đổi VLLM_VERSION hoặc thêm bản có wheel vào EXTRA_PACKAGES.")

# 3) Kiểm tra cụm wheel đủ để cài offline (resolve từ local, không động mạng).
check_dir = Path("/kaggle/working/_vllm_install_check")
verify = [sys.executable, "-m", "pip", "install",
          "--no-index", f"--find-links={OUT_DIR}",
          *constraint_args,
          "--target", str(check_dir), "--dry-run", spec]
print("\n🔎 Dry-run kiểm tra đầy đủ:", " ".join(verify))
ret = subprocess.run(verify).returncode
if ret == 0:
    print("\n✅ Wheels ĐỦ để cài offline. → Save Version → Create Dataset từ thư mục:")
    print(f"     {OUT_DIR}")
    if torch_whls:
        print("ℹ️  Có torch wheel trong bộ này — đúng bản cu của Kaggle nên khi cài "
              "offline pip sẽ thấy 'already satisfied' và KHÔNG thay torch. Tốt.")
else:
    print("\n⚠️  Dry-run lỗi — thiếu wheel cho dependency nào đó. Xem log phía trên; "
          "thêm gói thiếu vào EXTRA_PACKAGES hoặc đổi VLLM_VERSION rồi chạy lại.")
