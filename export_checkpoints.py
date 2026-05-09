"""Скрипт для копирования чекпоинтов из Google Drive в репо.

Запустите в Colab:
    !python export_checkpoints.py

Копирует GAN и ablation чекпоинты + JSON результаты из Drive в repo,
показывает размеры, и предлагает закоммитить.
"""
import os
import shutil
from pathlib import Path

DRIVE_CKPT = "/content/drive/MyDrive/concrete_project/experiments/checkpoints"
REPO_CKPT = "/content/concrete-strength/checkpoints"

def main():
    if not os.path.exists(DRIVE_CKPT):
        print(f"ERROR: {DRIVE_CKPT} not found. Mount Google Drive first.")
        return

    os.makedirs(REPO_CKPT, exist_ok=True)

    # Copy GAN checkpoints
    total_size = 0
    copied = 0

    print("Copying checkpoints from Drive to repo...\n")

    for f in sorted(os.listdir(DRIVE_CKPT)):
        src = os.path.join(DRIVE_CKPT, f)
        dst = os.path.join(REPO_CKPT, f)

        # Copy only essential files
        keep = (
            f.startswith("gan_sup_")      # GAN models
            or f.endswith(".json")          # Result JSONs
            or f.startswith("ablation_")   # Ablation checkpoints
        )

        if not keep:
            continue

        size = os.path.getsize(src)
        total_size += size
        shutil.copy2(src, dst)
        copied += 1
        print(f"  {f:55s} {size/1024:8.1f} KB")

    print(f"\nTotal: {copied} files, {total_size/1024/1024:.1f} MB")

    if total_size > 100 * 1024 * 1024:
        print("\n⚠️  WARNING: > 100 MB — consider using Git LFS or GitHub Releases")
    else:
        print("\n✅ Small enough to commit to repo directly")

    print(f"\nTo commit and push:")
    print(f"  cd /content/concrete-strength")
    print(f"  git add checkpoints/")
    print(f"  git commit -m 'Add pre-trained checkpoints for inference'")
    print(f"  git push")


if __name__ == "__main__":
    main()
