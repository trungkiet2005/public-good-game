#!/usr/bin/env python3
"""
run_on_kaggle.py — Day mot file code FAIRGAME len Kaggle, chay (GPU) va tai ket qua ve.
Dung Kaggle API (CLI `kaggle`) — KHONG can mo trinh duyet.

 Vi du dung:
    # Day + chay pgg_punish, roi tu cho xong va tai output ve:
    python run_on_kaggle.py --kernel pgg_punish --watch --download

    # Chi day (push) thoi:
    python run_on_kaggle.py --kernel pgg_punish

    # Kiem tra trang thai:
    python run_on_kaggle.py --kernel pgg_punish --status

Yeu cau:
  * Da cai:  pip install kaggle
  * Da co file token:  ~/.kaggle/kaggle.json  (Windows: C:\\Users\\<ten>\\.kaggle\\kaggle.json)

Cau truc thu muc (mac dinh):
  kaggle_run/
    run_on_kaggle.py          <- file nay
    pgg_punish/
        kernel-metadata.json  <- cau hinh kernel
    (file code goc nam o ../kaggle_pgg_punish_notebook.py — script tu copy vao)
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../FAIRGAME/kaggle_run
FAIRGAME = HERE.parent                            # .../FAIRGAME

# Map ten kernel -> (thu muc metadata, file code goc trong FAIRGAME/)
KERNELS = {
    "pgg_punish":   ("pgg_punish",   "kaggle_pgg_punish_notebook.py"),
    "crsd_followup":("crsd_followup","kaggle_crsd_followup_notebook.py"),
    "online":       ("online",       "kaggle_notebook_online.py"),
}


def kaggle_username() -> str | None:
    """Doc username tu kaggle.json de dien vao 'id' neu dang la placeholder."""
    for p in [Path.home()/".kaggle"/"kaggle.json",
              Path(os.environ.get("KAGGLE_CONFIG_DIR", ""))/"kaggle.json"]:
        try:
            if p.is_file():
                return json.loads(p.read_text()).get("username")
        except Exception:
            pass
    return os.environ.get("KAGGLE_USERNAME")


def run(cmd: list[str]) -> int:
    print("»", " ".join(cmd))
    return subprocess.run(cmd).returncode


def capture(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def prepare(folder: Path, src_name: str) -> dict:
    meta_path = folder / "kernel-metadata.json"
    if not meta_path.is_file():
        sys.exit(f"❌ Khong thay {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Tu dien username that vao id neu con placeholder
    if meta["id"].startswith("YOUR_KAGGLE_USERNAME/"):
        user = kaggle_username()
        if not user:
            sys.exit("❌ Chua doc duoc username. Sua truong 'id' trong kernel-metadata.json, "
                     "hoac dat bien moi truong KAGGLE_USERNAME.")
        meta["id"] = meta["id"].replace("YOUR_KAGGLE_USERNAME", user)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Da dien id = {meta['id']}")

    # Copy file code goc (single source of truth) vao thu muc kernel
    src = FAIRGAME / src_name
    if not src.is_file():
        sys.exit(f"❌ Khong thay file code goc: {src}")
    shutil.copy2(src, folder / meta["code_file"])
    print(f"✓ Da copy {src.name} -> {folder/meta['code_file']}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(KERNELS),
                    help="Ten cau hinh kernel (xem dict KERNELS).")
    ap.add_argument("--watch", action="store_true", help="Cho cho den khi chay xong.")
    ap.add_argument("--download", action="store_true", help="Tai output ve sau khi xong.")
    ap.add_argument("--status", action="store_true", help="Chi xem trang thai roi thoat.")
    ap.add_argument("--poll", type=int, default=30, help="Giay giua moi lan kiem tra (mac dinh 30).")
    args = ap.parse_args()

    sub, src_name = KERNELS[args.kernel]
    folder = HERE / sub
    meta = prepare(folder, src_name) if not args.status else json.loads(
        (folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    kid = meta["id"]

    if args.status:
        run(["kaggle", "kernels", "status", kid]); return

    # Push (= tao 1 version moi = "Save Version")
    if run(["kaggle", "kernels", "push", "-p", str(folder)]) != 0:
        sys.exit("❌ Push that bai. Kiem tra metadata / sources / token.")

    if not (args.watch or args.download):
        print(f"✓ Da push. Theo doi: kaggle kernels status {kid}")
        return

    # Cho cho den khi xong
    print(f"⏳ Theo doi {kid} (moi {args.poll}s)…")
    while True:
        out = capture(["kaggle", "kernels", "status", kid]).lower()
        print("   ", out.strip().replace("\n", " ") or "(khong co output)")
        if any(s in out for s in ["complete", "error", "cancel"]):
            break
        time.sleep(args.poll)

    if args.download:
        dest = folder / "output"
        dest.mkdir(exist_ok=True)
        run(["kaggle", "kernels", "output", kid, "-p", str(dest), "-o"])
        print(f"✓ Output tai ve: {dest}")


if __name__ == "__main__":
    main()
