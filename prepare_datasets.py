"""
Step 3 — Build assignment subsets from official archives.
Creates:
  ANN_DL_Assignment/flowers_subset/flower_class_XXX/*.jpg   (5 x 40 = 200)
  ANN_DL_Assignment/pets_subset/images/*.jpg
  ANN_DL_Assignment/pets_subset/masks/*.png                 (150 pairs)
"""
from __future__ import annotations

import random
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import scipy.io as sio

SEED = 42
ROOT = Path(__file__).resolve().parent
FLOWERS_TGZ = ROOT / "Visual Geometery Group" / "102flowers.tgz"
LABELS_MAT = ROOT / "Visual Geometery Group" / "imagelabels.mat"
PETS_IMG_TGZ = ROOT / "oxford-iiit-pet" / "images.tar.gz"
PETS_ANN_TGZ = ROOT / "oxford-iiit-pet" / "annotations.tar.gz"

BASE = ROOT / "ANN_DL_Assignment"
FLOWERS_OUT = BASE / "flowers_subset"
PETS_IMG_OUT = BASE / "pets_subset" / "images"
PETS_MASK_OUT = BASE / "pets_subset" / "masks"

N_FLOWER_CLASSES = 5
N_PER_CLASS = 40
N_PET_PAIRS = 150


def build_flowers() -> dict:
    labels = sio.loadmat(str(LABELS_MAT))["labels"].flatten()  # length 8189, values 1..102
    by_class: dict[int, list[int]] = defaultdict(list)
    # image_00001.jpg <-> labels[0]
    for i, lab in enumerate(labels):
        by_class[int(lab)].append(i + 1)  # 1-based image id

    eligible = sorted(c for c, ids in by_class.items() if len(ids) >= N_PER_CLASS)
    rng = random.Random(SEED)
    chosen_classes = sorted(rng.sample(eligible, N_FLOWER_CLASSES))

    if FLOWERS_OUT.exists():
        shutil.rmtree(FLOWERS_OUT)
    FLOWERS_OUT.mkdir(parents=True)

    # Map: archive member name -> destination path
    to_extract: dict[str, Path] = {}
    summary = {"chosen_oxford_class_ids": chosen_classes, "per_folder": {}}

    for folder_i, class_id in enumerate(chosen_classes):
        folder_name = f"flower_class_{folder_i:03d}"
        dest_dir = FLOWERS_OUT / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        ids = sorted(by_class[class_id])
        picked = sorted(rng.sample(ids, N_PER_CLASS))
        summary["per_folder"][folder_name] = {
            "oxford_label": class_id,
            "count": len(picked),
            "image_ids": picked,
        }
        for img_id in picked:
            member = f"jpg/image_{img_id:05d}.jpg"
            to_extract[member] = dest_dir / f"image_{img_id:05d}.jpg"

    print(f"Extracting {len(to_extract)} flower images...")
    with tarfile.open(FLOWERS_TGZ, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name in to_extract:
                f = tar.extractfile(member)
                if f is None:
                    raise RuntimeError(f"Cannot read {member.name}")
                dest = to_extract[member.name]
                dest.write_bytes(f.read())
                del to_extract[member.name]
    if to_extract:
        raise RuntimeError(f"Missing flower members: {list(to_extract)[:5]}")

    return summary


def build_pets() -> dict:
    # Collect stems from both archives
    with tarfile.open(PETS_IMG_TGZ, "r:gz") as tar:
        img_members = {}
        for m in tar.getmembers():
            name = m.name.replace("\\", "/")
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue
            stem = Path(name).stem
            if stem.startswith("."):
                continue
            img_members[stem] = name

    with tarfile.open(PETS_ANN_TGZ, "r:gz") as tar:
        mask_members = {}
        for m in tar.getmembers():
            name = m.name.replace("\\", "/")
            if "trimaps/" not in name or not name.lower().endswith(".png"):
                continue
            fname = Path(name).name
            if fname.startswith(".") or fname.startswith("_"):
                continue
            stem = Path(fname).stem
            mask_members[stem] = name

    common = sorted(set(img_members) & set(mask_members))
    rng = random.Random(SEED)
    picked = sorted(rng.sample(common, N_PET_PAIRS))

    if (BASE / "pets_subset").exists():
        shutil.rmtree(BASE / "pets_subset")
    PETS_IMG_OUT.mkdir(parents=True)
    PETS_MASK_OUT.mkdir(parents=True)

    need_img = {img_members[s]: s for s in picked}
    need_mask = {mask_members[s]: s for s in picked}

    print(f"Extracting {len(picked)} pet images...")
    with tarfile.open(PETS_IMG_TGZ, "r:gz") as tar:
        for m in tar.getmembers():
            key = m.name.replace("\\", "/")
            if key in need_img:
                stem = need_img[key]
                f = tar.extractfile(m)
                if f is None:
                    raise RuntimeError(key)
                data = f.read()
                # keep original extension
                ext = Path(key).suffix.lower() or ".jpg"
                (PETS_IMG_OUT / f"{stem}{ext}").write_bytes(data)
                del need_img[key]
    if need_img:
        raise RuntimeError(f"Missing pet images: {list(need_img)[:5]}")

    print(f"Extracting {len(picked)} pet masks...")
    with tarfile.open(PETS_ANN_TGZ, "r:gz") as tar:
        for m in tar.getmembers():
            key = m.name.replace("\\", "/")
            if key in need_mask:
                stem = need_mask[key]
                f = tar.extractfile(m)
                if f is None:
                    raise RuntimeError(key)
                (PETS_MASK_OUT / f"{stem}.png").write_bytes(f.read())
                del need_mask[key]
    if need_mask:
        raise RuntimeError(f"Missing pet masks: {list(need_mask)[:5]}")

    return {
        "pairs": len(picked),
        "available_common_stems": len(common),
        "sample_stems": picked[:10],
    }


def sanity() -> None:
    class_dirs = sorted(p for p in FLOWERS_OUT.iterdir() if p.is_dir())
    print("\n=== SANITY ===")
    print("Flower class folders:", len(class_dirs))
    total = 0
    for d in class_dirs:
        n = len(list(d.glob("*.jpg")))
        total += n
        print(f"  {d.name}: {n}")
    print("Flower total images:", total)

    imgs = sorted(PETS_IMG_OUT.glob("*.jpg")) + sorted(PETS_IMG_OUT.glob("*.jpeg"))
    masks = sorted(PETS_MASK_OUT.glob("*.png"))
    img_stems = {p.stem for p in imgs}
    mask_stems = {p.stem for p in masks}
    print("Pet images:", len(imgs))
    print("Pet masks :", len(masks))
    print("Matched pairs:", len(img_stems & mask_stems))
    print("Orphans:", len(img_stems ^ mask_stems))


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    print("Building flowers_subset...")
    flower_info = build_flowers()
    print("Chosen Oxford class IDs:", flower_info["chosen_oxford_class_ids"])
    for folder, info in flower_info["per_folder"].items():
        print(f"  {folder} <- oxford label {info['oxford_label']} ({info['count']} imgs)")

    print("\nBuilding pets_subset...")
    pet_info = build_pets()
    print("Pet pairs:", pet_info["pairs"], "/", pet_info["available_common_stems"], "available")

    sanity()
    print("\nDONE. Base folder:", BASE)


if __name__ == "__main__":
    main()
