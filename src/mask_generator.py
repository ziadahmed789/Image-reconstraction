"""
Mask Generation Module for Face Image Inpainting
=================================================

Provides three mask types for training inpainting models:
  1. Center mask       — rectangular hole at the center of the image
  2. Random square mask — randomly positioned rectangular hole
  3. Irregular mask     — free-form brush strokes using OpenCV

All masks guarantee:
  - Shape: (H, W) with values 0 (masked) and 1 (visible)
  - Masked area ratio between 10% and 40% of total pixels
  - Diversity through randomness
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default image dimensions (CelebA preprocessed to 64×64)
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 64
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"
MASK_OUTPUT_BASE = PROJECT_ROOT / "data" / "masks"
DATA_SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Masked area ratio constraints
# ---------------------------------------------------------------------------
MIN_MASK_RATIO: float = 0.10   # at least 10% of pixels masked
MAX_MASK_RATIO: float = 0.40   # at most 40% of pixels masked


# ============================= Helper ======================================

def _mask_ratio(mask: np.ndarray) -> float:
    """Return the fraction of pixels that are masked (value == 0)."""
    return 1.0 - mask.mean()


def _validate_mask(mask: np.ndarray) -> bool:
    """Check whether a mask meets the area constraints."""
    ratio = _mask_ratio(mask)
    return MIN_MASK_RATIO <= ratio <= MAX_MASK_RATIO


# ========================= 1. Center Mask ==================================

def generate_center_mask(
    h: int = IMAGE_SIZE,
    w: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Generate a centered rectangular mask.

    The side length of the square hole is sampled so that
    the masked area falls within [MIN_MASK_RATIO, MAX_MASK_RATIO].

    Parameters
    ----------
    h, w : int
        Height and width of the output mask.

    Returns
    -------
    mask : np.ndarray, shape (h, w), dtype float32
        Binary mask: 0 = masked, 1 = visible.
    """
    total_pixels = h * w

    # Compute side-length bounds from area constraints
    min_side = int(np.ceil(np.sqrt(MIN_MASK_RATIO * total_pixels)))
    max_side = int(np.floor(np.sqrt(MAX_MASK_RATIO * total_pixels)))

    # Clamp to image dimensions
    min_side = max(min_side, 1)
    max_side = min(max_side, min(h, w))

    # Sample a side length
    side = np.random.randint(min_side, max_side + 1)

    # Center the hole
    y1 = (h - side) // 2
    x1 = (w - side) // 2

    mask = np.ones((h, w), dtype=np.float32)
    mask[y1 : y1 + side, x1 : x1 + side] = 0.0

    return mask


# ====================== 2. Random Square Mask ==============================

def generate_random_square_mask(
    h: int = IMAGE_SIZE,
    w: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Generate a randomly positioned rectangular mask.

    The rectangle aspect ratio is slightly varied for diversity.

    Parameters
    ----------
    h, w : int
        Height and width of the output mask.

    Returns
    -------
    mask : np.ndarray, shape (h, w), dtype float32
        Binary mask: 0 = masked, 1 = visible.
    """
    total_pixels = h * w

    # Target masked area
    target_ratio = np.random.uniform(MIN_MASK_RATIO, MAX_MASK_RATIO)
    target_area = int(target_ratio * total_pixels)

    # Choose width and height of the rectangle (allow slight aspect-ratio variation)
    aspect = np.random.uniform(0.7, 1.4)
    rect_h = int(np.clip(np.sqrt(target_area / aspect), 1, h))
    rect_w = int(np.clip(np.sqrt(target_area * aspect), 1, w))

    # Clamp dimensions so the area ratio stays valid
    rect_h = min(rect_h, h)
    rect_w = min(rect_w, w)

    # Random top-left position
    y1 = np.random.randint(0, h - rect_h + 1)
    x1 = np.random.randint(0, w - rect_w + 1)

    mask = np.ones((h, w), dtype=np.float32)
    mask[y1 : y1 + rect_h, x1 : x1 + rect_w] = 0.0

    return mask


# ====================== 3. Irregular Mask ==================================

def generate_irregular_mask(
    h: int = IMAGE_SIZE,
    w: int = IMAGE_SIZE,
    max_attempts: int = 20,
) -> np.ndarray:
    """
    Generate a free-form irregular mask using random brush strokes.

    Multiple polylines with varying thickness are drawn on a blank canvas.
    The function retries until the masked area satisfies the constraints.

    Parameters
    ----------
    h, w : int
        Height and width of the output mask.
    max_attempts : int
        Maximum number of generation attempts before falling back
        to a random square mask.

    Returns
    -------
    mask : np.ndarray, shape (h, w), dtype float32
        Binary mask: 0 = masked, 1 = visible.
    """
    for _ in range(max_attempts):
        canvas = np.zeros((h, w), dtype=np.uint8)

        # Number of independent strokes
        num_strokes = np.random.randint(3, 10)

        for _ in range(num_strokes):
            # Random starting point
            start_x = np.random.randint(0, w)
            start_y = np.random.randint(0, h)

            # Build a polyline with random vertices
            num_vertices = np.random.randint(4, 12)
            points = [(start_x, start_y)]

            for _ in range(num_vertices):
                # Walk randomly from the previous vertex
                dx = np.random.randint(-20, 21)
                dy = np.random.randint(-20, 21)
                next_x = np.clip(points[-1][0] + dx, 0, w - 1)
                next_y = np.clip(points[-1][1] + dy, 0, h - 1)
                points.append((int(next_x), int(next_y)))

            # Random brush thickness
            thickness = np.random.randint(2, 8)

            # Draw the polyline
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=1, thickness=thickness)

            # Optionally draw circles at vertices for rounder strokes
            if np.random.random() < 0.5:
                for pt in points:
                    radius = thickness // 2 + np.random.randint(0, 3)
                    cv2.circle(canvas, pt, radius, 1, -1)

        # Convert canvas to mask (canvas==1 means masked region)
        mask = 1.0 - canvas.astype(np.float32)

        if _validate_mask(mask):
            return mask

    # Fallback: if irregular mask couldn't meet constraints, use random square
    return generate_random_square_mask(h, w)


# ========================= Public API =====================================

# Mask type names → generator functions
_MASK_GENERATORS = {
    "center": generate_center_mask,
    "random_square": generate_random_square_mask,
    "irregular": generate_irregular_mask,
}

MASK_TYPES = list(_MASK_GENERATORS.keys())


def generate_mask(
    img: np.ndarray,
    mask_type: str = "random",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a binary mask to an image.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, 3), dtype float32
        Input image normalised to [0, 1].
    mask_type : str
        One of  ``"center"``, ``"random_square"``, ``"irregular"``,
        or ``"random"`` (picks one at random).

    Returns
    -------
    masked_img : np.ndarray, shape (H, W, 3)
        Image with masked pixels zeroed out.
    mask : np.ndarray, shape (H, W)
        Binary mask — 0 = masked, 1 = visible.
    """
    h, w = img.shape[:2]

    if mask_type == "random":
        mask_type = np.random.choice(MASK_TYPES)

    if mask_type not in _MASK_GENERATORS:
        raise ValueError(
            f"Unknown mask_type '{mask_type}'. "
            f"Choose from {MASK_TYPES + ['random']}"
        )

    mask = _MASK_GENERATORS[mask_type](h, w)

    # Apply mask: zero out masked regions
    masked_img = img * mask[:, :, np.newaxis]

    return masked_img, mask


def ensure_output_dirs() -> None:
    for split in DATA_SPLITS:
        (MASK_OUTPUT_BASE / split / "masked").mkdir(parents=True, exist_ok=True)
        (MASK_OUTPUT_BASE / split / "mask").mkdir(parents=True, exist_ok=True)


def save_masked_dataset(mask_type: str = "random") -> None:
    ensure_output_dirs()

    total = 0
    processed = 0
    skipped_missing = 0
    skipped_corrupted = 0

    for split in DATA_SPLITS:
        input_dir = PROCESSED_BASE / split
        masked_dir = MASK_OUTPUT_BASE / split / "masked"
        mask_dir = MASK_OUTPUT_BASE / split / "mask"

        if not input_dir.exists():
            continue

        image_paths = sorted(
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        for image_path in tqdm(image_paths, desc=f"Generating masks for {split}"):
            total += 1

            if not image_path.exists():
                skipped_missing += 1
                continue

            try:
                with Image.open(image_path) as img:
                    rgb_img = img.convert("RGB")
                    img_arr = np.asarray(rgb_img, dtype=np.float32) / 255.0
            except (UnidentifiedImageError, OSError, ValueError):
                skipped_corrupted += 1
                continue

            masked_img, mask = generate_mask(img_arr, mask_type)

            masked_uint8 = (masked_img * 255).clip(0, 255).astype(np.uint8)
            mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)

            output_name = image_path.stem + ".png"
            Image.fromarray(masked_uint8).save(masked_dir / output_name)
            Image.fromarray(mask_uint8, mode="L").save(mask_dir / output_name)
            processed += 1

    print("\nDone.")
    print(f"Total images scanned: {total}")
    print(f"Masked images saved: {processed}")
    print(f"Missing files skipped: {skipped_missing}")
    print(f"Corrupted files skipped: {skipped_corrupted}")
    print(f"Saved to: {MASK_OUTPUT_BASE.resolve()}")

if __name__ == "__main__":
    save_masked_dataset(mask_type="random")
