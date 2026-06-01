#!/usr/bin/env python3
"""
Download, extract, and merge the WHOI-Plankton dataset (2006-2014) into a
single ImageFolder ready for upload via upload_to_hf.py.

The dataset is hosted at https://darchive.mblwhoilibrary.org/ (handle 1912/7341,
DOI 10.1575/1912/7341). It contains ~3.5M annotated IFCB plankton images
across 103 classes, split into 9 per-year archives.

Output layout (matches project-oceania/whoi-plankton):
  <out>/<class_name>/IFCB*_<year>_*.png   (one big merged tree, year in filename)

Per-year archives are downloaded, extracted, merged via symlinks, then the zip
files are removed (--keep-zips to disable). Each year already ships as an
ImageFolder so extraction is straightforward.

Usage:
  python prepare_whoi_plankton.py \\
      --root /scratch/daniela/DEAL/Plankton/whoi-plankton \\
      [--years 2006,2007,...] [--keep-zips] [--skip-existing-2014]

Re-runs are idempotent: existing zips are kept (unless --redownload), existing
extracted year dirs are reused, and existing symlinks are not recreated.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

# DSpace bitstream UUIDs scraped from the handle pages (see chat history).
WHOI_DOWNLOADS = {
    2006: "6968c380-3713-57b1-bdca-5b21e514a996",
    2007: "ff635112-6337-5b34-9354-4035847dae24",
    2008: "c9bf8e43-fa1c-5fd7-9328-04598db52c2e",
    2009: "18b14b5a-2a68-5f85-845f-1c1591a8f1a6",
    2010: "7d6bd792-3fad-59aa-8906-af1fb377115e",
    2011: "c1b63530-b104-5a1f-a8b3-f55898f788e7",
    2012: "67fd1c6a-9268-58f6-808d-a757bf49a345",
    2013: "e1fd23e9-1b79-51a8-a165-1f9bb30177d8",
    2014: "5bf89ef0-0155-5ac2-923b-f2a8578c963a",
}

BASE_URL = "https://darchive.mblwhoilibrary.org/bitstreams/{uuid}/download"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) prepare_whoi_plankton/1.0"

# Files that may appear inside the archives but are not images we want to keep.
JUNK_FILENAMES = {"Thumbs.db", ".DS_Store"}


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download(url: str, dst: Path, redownload: bool = False) -> Path:
    """Stream a URL to disk with a progress bar."""
    if dst.exists() and not redownload:
        print(f"  [skip] {dst.name} already exists ({human_bytes(dst.stat().st_size)})")
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        print(f"  downloading {dst.name} ({human_bytes(total) if total else 'unknown size'}) ...")
        downloaded = 0
        chunk = 1024 * 1024
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"\r    {human_bytes(downloaded)} / {human_bytes(total)} ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\r    {human_bytes(downloaded)}", end="", flush=True)
        print()
    tmp.rename(dst)
    return dst


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    """Extract a ZIP archive into out_dir/<archive_stem>/. Idempotent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"  [skip] {zip_path.name} already extracted at {out_dir}")
        return out_dir
    print(f"  extracting {zip_path.name} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text(zip_path.name + "\n")
    return out_dir


def find_year_imagefolder(extracted_root: Path, year: int) -> Path:
    """
    Locate the per-class subdirectory layout inside an extracted year.

    Each year's zip has a top-level <year>/ directory containing
    <class_name>/*.png. Sometimes the top-level is the year, sometimes it's
    directly the class names. We auto-detect.
    """
    # Common case: a single <year>/ subdir
    year_subdir = extracted_root / str(year)
    if year_subdir.is_dir() and any(p.is_dir() for p in year_subdir.iterdir()):
        return year_subdir
    # Alternative: classes directly under extracted_root
    class_like = [p for p in extracted_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if class_like:
        return extracted_root
    raise RuntimeError(f"Could not find imagefolder layout in {extracted_root}")


def link_year_into_merged(year_root: Path, merged_root: Path, year: int) -> tuple[int, int]:
    """
    Symlink every <year_root>/<class>/<file>.png into <merged_root>/<class>/.
    Returns (n_linked, n_skipped).
    """
    n_linked = 0
    n_skipped = 0
    for class_dir in sorted(year_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        target_class = merged_root / class_dir.name
        target_class.mkdir(parents=True, exist_ok=True)
        for img in class_dir.iterdir():
            if img.is_dir():
                continue
            if img.name in JUNK_FILENAMES:
                continue
            if not img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                continue
            link_name = target_class / img.name
            if link_name.exists() or link_name.is_symlink():
                n_skipped += 1
                continue
            os.symlink(img.resolve(), link_name)
            n_linked += 1
    return n_linked, n_skipped


def process_year(year: int, root: Path, redownload: bool, existing_2014: Path | None) -> Path:
    """
    Returns the per-year imagefolder path (with <class>/<file>.png inside).
    Uses the existing /scratch/datasets/DEAL/Plankton/WHO/2014 directory if
    available to skip a re-download for 2014.
    """
    if year == 2014 and existing_2014 is not None and existing_2014.is_dir():
        print(f"[2014] reusing existing extracted dir: {existing_2014}")
        return existing_2014

    uuid = WHOI_DOWNLOADS[year]
    url = BASE_URL.format(uuid=uuid)
    zips_dir = root / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / f"{year}.zip"
    extracted_dir = root / "extracted" / str(year)

    print(f"[{year}] downloading + extracting")
    download(url, zip_path, redownload=redownload)
    extract_zip(zip_path, extracted_dir)
    return find_year_imagefolder(extracted_dir, year)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="Working dir for downloads + merged ImageFolder")
    ap.add_argument("--years", default="2006,2007,2008,2009,2010,2011,2012,2013,2014",
                    help="Comma-separated years to process")
    ap.add_argument("--keep-zips", action="store_true",
                    help="Keep per-year zip archives after extraction")
    ap.add_argument("--redownload", action="store_true",
                    help="Force re-download even if zip exists")
    ap.add_argument("--existing-2014", default="/scratch/datasets/DEAL/Plankton/WHO/2014",
                    help="Reuse this pre-extracted 2014 directory instead of re-downloading. "
                         "Empty string disables.")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="Concurrent downloads (keep low to be polite to the server)")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",") if y.strip()]
    for y in years:
        if y not in WHOI_DOWNLOADS:
            sys.exit(f"Unknown year {y}; valid: {sorted(WHOI_DOWNLOADS)}")

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    merged_root = root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)

    existing_2014 = Path(args.existing_2014) if args.existing_2014 else None

    # 1. Download + extract per year. Parallelize downloads but keep concurrency low.
    year_imagefolders: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process_year, y, root, args.redownload, existing_2014): y for y in years}
        for f in as_completed(futs):
            y = futs[f]
            try:
                year_imagefolders[y] = f.result()
            except Exception as e:
                print(f"[{y}] FAILED: {e}")
                raise

    # 2. Merge into one ImageFolder via symlinks
    print()
    print(f"Merging {len(year_imagefolders)} years -> {merged_root}")
    total_linked = 0
    total_skipped = 0
    for y in sorted(year_imagefolders):
        year_root = year_imagefolders[y]
        n_linked, n_skipped = link_year_into_merged(year_root, merged_root, y)
        total_linked += n_linked
        total_skipped += n_skipped
        print(f"  {y}: linked {n_linked:>8d}  skipped {n_skipped:>6d}  from {year_root}")
    print(f"Total: linked {total_linked}, skipped {total_skipped}")

    # 3. Optionally remove zips
    if not args.keep_zips:
        zips_dir = root / "zips"
        if zips_dir.exists():
            for z in sorted(zips_dir.glob("*.zip")):
                # only remove zips whose extracted/<year>/.extracted marker exists
                year_marker = root / "extracted" / z.stem / ".extracted"
                if year_marker.exists():
                    print(f"removing {z}")
                    z.unlink()
            try:
                zips_dir.rmdir()
            except OSError:
                pass

    # 4. Final summary
    classes = sorted([p.name for p in merged_root.iterdir() if p.is_dir()])
    n_imgs = sum(1 for p in merged_root.rglob("*") if p.is_file() or p.is_symlink())
    print()
    print(f"WHOI-Plankton merged at: {merged_root}")
    print(f"  Classes: {len(classes)}")
    print(f"  Images (symlinks): {n_imgs}")
    print()
    print("Next:")
    print(f"  Set `data_dir: {merged_root}` in the YAML config, then:")
    print(f"  python upload_to_hf.py hf_upload_configs/whoi_plankton.yaml --dry-run")


if __name__ == "__main__":
    main()
