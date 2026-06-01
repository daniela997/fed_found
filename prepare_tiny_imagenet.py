#!/usr/bin/env python3
"""
Build an ImageFolder-compatible view of Tiny-ImageNet-200 via symlinks.

Source layout (read-only friendly):
  <src>/train/<wnid>/images/*.JPEG
  <src>/val/images/val_<i>.JPEG  +  <src>/val/val_annotations.txt

Output view:
  <dst>/train/<wnid>/*.JPEG     (symlinks)
  <dst>/val/<wnid>/*.JPEG       (symlinks, grouped by class from annotations)
"""
import argparse, os, sys

def link(src, dst):
    if os.path.lexists(dst):
        return
    os.symlink(src, dst)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/scratch/datasets/other/tiny-imagenet-200")
    p.add_argument("--dst", default="/home/daniela/mine/fedDINO/data/tiny_in_view")
    args = p.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if not os.path.isdir(os.path.join(src, "train")):
        sys.exit(f"missing {src}/train")

    # ---- train ----
    train_src = os.path.join(src, "train")
    train_dst = os.path.join(dst, "train")
    n_train = 0
    for wnid in sorted(os.listdir(train_src)):
        cls_imgs = os.path.join(train_src, wnid, "images")
        if not os.path.isdir(cls_imgs):
            continue
        out_cls = os.path.join(train_dst, wnid)
        os.makedirs(out_cls, exist_ok=True)
        for fn in os.listdir(cls_imgs):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                link(os.path.join(cls_imgs, fn), os.path.join(out_cls, fn))
                n_train += 1
    print(f"train: {n_train} symlinks across {len(os.listdir(train_dst))} classes")

    # ---- val ----
    val_anno = os.path.join(src, "val", "val_annotations.txt")
    val_imgs = os.path.join(src, "val", "images")
    val_dst = os.path.join(dst, "val")
    if not os.path.isfile(val_anno):
        sys.exit(f"missing {val_anno}")
    n_val = 0
    with open(val_anno) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fn, wnid = parts[0], parts[1]
            out_cls = os.path.join(val_dst, wnid)
            os.makedirs(out_cls, exist_ok=True)
            link(os.path.join(val_imgs, fn), os.path.join(out_cls, fn))
            n_val += 1
    print(f"val:   {n_val} symlinks across {len(os.listdir(val_dst))} classes")
    print(f"View ready at: {dst}")

if __name__ == "__main__":
    main()
