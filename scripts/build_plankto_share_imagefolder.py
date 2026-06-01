"""
Convert PlanktoShare from a flat .tif dump + CSV labels into an ImageFolder
layout of PNGs, so it can be uploaded by upload_to_hf.py.

Source layout:
    <SRC>/PlanktoShare/<imageID>.tif      # 53,325 TIFs (flat dir)
    <SRC>/PlanktoShare_labels.csv         # 52,882 labeled rows

Output layout:
    <DST>/<class_name>/<imageID>.png      # 52,882 PNGs across 132 customName dirs

Class column: `customName` (132 classes, fully populated in CSV).
Unlabeled TIFs (in zip but not in CSV) are skipped.
Class-name sanitisation: spaces -> '_', '/' -> '_' (no other transforms).
"""
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

SRC = Path("/scratch/daniela/DEAL/Plankton/plankto-share")
SRC_IMG_DIR = SRC / "PlanktoShare"
SRC_CSV = SRC / "PlanktoShare_labels.csv"


def sanitize_class_name(name: str) -> str:
    # Keep it conservative: replace whitespace and slashes only.
    s = re.sub(r"\s+", "_", str(name).strip())
    s = s.replace("/", "_")
    return s


def convert_one(args):
    src_tif, dst_png = args
    try:
        with Image.open(src_tif) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.save(dst_png, format="PNG", optimize=False)
        return None
    except Exception as e:
        return f"FAILED {src_tif}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="Output ImageFolder root")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only convert this many images (debug)")
    args = ap.parse_args()

    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC_CSV)
    print(f"CSV rows: {len(df)}")
    print(f"Unique customName: {df['customName'].nunique()}")

    # Build (src_tif, dst_png) tuples.
    tasks: list[tuple[Path, Path]] = []
    missing_tif = 0
    for _, row in df.iterrows():
        image_id = row["imageID"]
        class_name = sanitize_class_name(row["customName"])
        src_tif = SRC_IMG_DIR / image_id
        if not src_tif.is_file():
            missing_tif += 1
            continue
        class_dir = dst_root / class_name
        class_dir.mkdir(exist_ok=True)
        dst_png = class_dir / (Path(image_id).stem + ".png")
        tasks.append((src_tif, dst_png))

    if missing_tif:
        print(f"  warning: {missing_tif} CSV rows had no matching .tif on disk")

    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"Converting {len(tasks)} images with {args.workers} workers ...")

    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures)):
            err = f.result()
            if err:
                n_fail += 1
                if n_fail < 10:
                    print(err, file=sys.stderr)

    print(f"Done. Converted={len(tasks) - n_fail}  Failed={n_fail}")
    print(f"Class dirs in {dst_root}:")
    n_dirs = sum(1 for p in dst_root.iterdir() if p.is_dir())
    print(f"  {n_dirs} class directories")


if __name__ == "__main__":
    main()
