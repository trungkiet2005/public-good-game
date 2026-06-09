"""
=====================================================================
Build vLLM OFFLINE wheels for Kaggle  (Internet ON → tạo Dataset wheels)
=====================================================================
MỤC ĐÍCH
  Notebook CRSD chạy OFFLINE (Internet OFF) muốn dùng engine vLLM, nhưng image
  Kaggle có thể CHƯA cài vllm và không pip được khi offline. Script này TẢI
  TRƯỚC toàn bộ wheel (.whl) của vllm + MỌI dependency (kể cả torch) để bạn lưu
  thành một Kaggle Dataset, rồi cài offline ở notebook chính bằng:

      pip install --no-index --find-links=<wheels_dir> vllm

CÁCH DÙNG (KHUYẾN NGHỊ — khớp môi trường 100%):
  1. Tạo notebook Kaggle MỚI, **Internet: ON**, GPU: ON
     (chọn ĐÚNG image GPU bạn sẽ chạy notebook offline → Python + CUDA + torch
      khớp tuyệt đối). Bật GPU vì vài wheel của vLLM chỉ build cho biến thể cuda.
  2. Dán nguyên file này vào 1 cell. Chỉnh VLLM_VERSION nếu muốn ghim. Run.
  3. Wheels nằm ở OUT_DIR (= /kaggle/working/vllm_offline_wheels).
  4. "Save Version" (Save & Run All) → mở Output → "Create Dataset" từ thư mục
     wheels (đặt tên ví dụ: "vllm-offline-wheels").
  5. Sang notebook OFFLINE (kaggle_crsd_notebook.py): + Add Input dataset đó,
     set VLLM_WHEELS_DIR ở CELL 2.5 cho khớp tên, và đặt engine="vllm".

LƯU Ý QUAN TRỌNG
  * BẮT BUỘC tải trên Linux x86_64 + ĐÚNG phiên bản Python của Kaggle. Chạy ngay
    trong Kaggle Internet-ON là cách chắc nhất. Tải trên Windows sẽ ra wheel sai
    nền tảng (vLLM không có wheel cho Windows).
  * vLLM ghim torch cụ thể — cứ để pip kéo nguyên cụm, ĐỪNG tự ghim torch riêng.
  * Dùng CÙNG image cho notebook download và notebook offline. Nếu Kaggle đổi
    image (Python mới), phải build lại wheels.
=====================================================================
"""
import subprocess
import sys
from pathlib import Path

# --- CẤU HÌNH --------------------------------------------------------------- #
VLLM_VERSION = ""          # "" = bản mới nhất tương thích; hoặc ghim, vd "0.6.3.post1"
OUT_DIR = Path("/kaggle/working/vllm_offline_wheels")
EXTRA_PACKAGES = []        # gói phụ muốn kèm, vd ["flashinfer-python"]
# --------------------------------------------------------------------------- #

print(f"🐍 Python {sys.version.split()[0]} | nền tảng {sys.platform}")
if sys.platform.startswith("win"):
    print("⚠️  Đang chạy trên Windows — wheel tải về SẼ SAI nền tảng cho Kaggle. "
          "Hãy chạy file này trong notebook Kaggle Internet-ON.")

OUT_DIR.mkdir(parents=True, exist_ok=True)
spec = "vllm" + (f"=={VLLM_VERSION}" if VLLM_VERSION else "")
targets = [spec, *EXTRA_PACKAGES]

# 1) Nâng pip để resolver tải đúng wheel mới nhất.
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)

# 2) Tải TOÀN BỘ closure dạng wheel (ưu tiên binary để tránh sdist phải build offline).
cmd = [sys.executable, "-m", "pip", "download",
       "--dest", str(OUT_DIR),
       "--prefer-binary",
       *targets]
print("\n⬇️   ", " ".join(cmd))
subprocess.run(cmd, check=True)

wheels = sorted(p for p in OUT_DIR.iterdir() if p.is_file())
total_mb = sum(f.stat().st_size for f in wheels) / 1024 / 1024
n_sdist = sum(1 for f in wheels if f.suffix in (".gz", ".zip"))
print(f"\n✅ Tải {len(wheels)} file → {OUT_DIR}  ({total_mb:.1f} MB)")
if n_sdist:
    print(f"⚠️  Có {n_sdist} sdist (.tar.gz/.zip) — các gói này sẽ phải BUILD khi cài "
          "offline (cần internet/compiler nên dễ fail). Cân nhắc ghim version khác "
          "hoặc thêm vào EXTRA_PACKAGES bản có wheel.")

# 3) Kiểm tra cụm wheel đã ĐỦ để cài offline (resolve từ local, không động mạng).
check_dir = Path("/kaggle/working/_vllm_install_check")
verify = [sys.executable, "-m", "pip", "install",
          "--no-index", f"--find-links={OUT_DIR}",
          "--target", str(check_dir), "--dry-run", spec]
print("\n🔎 Dry-run kiểm tra đầy đủ:", " ".join(verify))
ret = subprocess.run(verify).returncode
if ret == 0:
    print("\n✅ Wheels ĐỦ để cài offline. → Save Version → Create Dataset từ thư mục:")
    print(f"     {OUT_DIR}")
else:
    print("\n⚠️  Dry-run lỗi — thiếu wheel cho dependency nào đó. Xem log phía trên; "
          "thêm gói thiếu vào EXTRA_PACKAGES hoặc đổi VLLM_VERSION rồi chạy lại.")
