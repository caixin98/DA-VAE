import shutil
import sys
from pathlib import Path

from PIL import Image
from cleanfid import fid


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}

# Directories to ignore when scanning for corrupted images.
# This prevents counting cached symlink trees (e.g. __fid_cache__) as extra "images".
EXCLUDE_DIR_NAMES = {
    "__fid_cache__",
    "__staging__",
    "_slurm_logs",
    "__corrupted__",
}


def filter_corrupted_images(image_dir: Path) -> int:
    """Move truncated/corrupted images out of the folder tree before FID."""

    image_dir = image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    corrupted_root = image_dir.parent / f"{image_dir.name}_corrupted"
    corrupted_root.mkdir(parents=True, exist_ok=True)

    def is_image(path: Path) -> bool:
        # Skip known cache/log/corrupted directories.
        parts = set(path.parts)
        if parts & EXCLUDE_DIR_NAMES:
            return False
        # Also skip any "<name>_corrupted" directories created by this function.
        if any(p.endswith("_corrupted") for p in path.parts):
            return False
        return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

    image_files = [path for path in image_dir.rglob("*") if is_image(path)]
    moved = 0

    total = len(image_files)
    # This scan can take a while for 30k+ images; print periodic progress.
    print(f"Scanning {total} images under {image_dir} for corruption...", file=sys.stderr)
    for idx, img_path in enumerate(image_files, start=1):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            relative_path = img_path.relative_to(image_dir)
            destination = corrupted_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_path), str(destination))
            moved += 1

        if idx % 1000 == 0 or idx == total:
            print(
                f"[corrupted-scan] checked {idx}/{total}, moved {moved}",
                file=sys.stderr,
            )

    if moved:
        print(
            f"Moved {moved} corrupted images from {image_dir} to {corrupted_root}.",
            file=sys.stderr,
        )
    else:
        print(f"No corrupted images found in {image_dir}.", file=sys.stderr)

    return moved


if __name__ == "__main__":
    reference_dir = Path("./evaluation/mjhq30k_imgs")
    # generated_dir = Path(
    #     "./evaluation_outputs/token_text_image_da_vae_with_lora_diff_sd35-01_steps30_cfg3.5_shift5"
    # )

    generated_dir = Path(
        "./evaluation_outputs/base_original_steps30_cfg3.5_shift5"
    )
    filter_corrupted_images(generated_dir)

    score = fid.compute_fid(str(reference_dir), str(generated_dir))
    print(score)