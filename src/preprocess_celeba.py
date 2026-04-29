from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMAGES_DIR = PROJECT_ROOT / "archive" / "img_align_celeba" / "img_align_celeba"
PARTITION_FILE = PROJECT_ROOT / "archive" / "list_eval_partition.csv"
OUTPUT_BASE = PROJECT_ROOT / "data" / "processed"

IMAGE_SIZE = (64, 64)

TRAIN_DIR = OUTPUT_BASE / "train"
VAL_DIR = OUTPUT_BASE / "val"
TEST_DIR = OUTPUT_BASE / "test"

def ensure_dirs():
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

def get_split_folder(partition_value: int) -> Path:
    if partition_value == 0:
        return TRAIN_DIR
    elif partition_value == 1:
        return VAL_DIR
    elif partition_value == 2:
        return TEST_DIR
    else:
        raise ValueError(f"Unexpected partition value: {partition_value}")

def process_image(src_path: Path, dst_path: Path):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(dst_path)

def main():
    ensure_dirs()

    df = pd.read_csv(PARTITION_FILE)

    required_cols = {"image_id", "partition"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Partition file must contain columns {required_cols}, found {set(df.columns)}"
        )

    total = 0
    skipped_missing = 0
    skipped_corrupted = 0
    processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CelebA"):
        total += 1
        image_name = row["image_id"]
        partition = int(row["partition"])

        src_path = RAW_IMAGES_DIR / image_name
        dst_folder = get_split_folder(partition)
        dst_path = dst_folder / image_name

        if not src_path.exists():
            skipped_missing += 1
            continue

        try:
            process_image(src_path, dst_path)
            processed += 1
        except (UnidentifiedImageError, OSError):
            skipped_corrupted += 1
            continue

    print("\nDone.")
    print(f"Total rows in partition file: {total}")
    print(f"Processed images: {processed}")
    print(f"Missing files skipped: {skipped_missing}")
    print(f"Corrupted files skipped: {skipped_corrupted}")
    print(f"Saved to: {OUTPUT_BASE.resolve()}")

if __name__ == "__main__":
    main()