import shutil
import random
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.yolo_dataset import write_yolo_data_yaml


# ─────────────────────────────────────────
# ADMS — copy with class IDs unchanged
# ─────────────────────────────────────────

def collect_adms(cfg):
    """
    ADMS class IDs already match the unified classes.yaml ordering.
    No remapping needed, only file collection with a source prefix.
    """
    train_images = Path(cfg["dataset"]["train_images"])
    train_labels = Path(cfg["dataset"]["train_labels"])
    test_images  = Path(cfg["dataset"]["test_images"])
    test_labels  = Path(cfg["dataset"]["test_labels"])

    train_pool = _pair_images_labels(train_images, train_labels, prefix="adms")
    test_pool  = _pair_images_labels(test_images, test_labels, prefix="adms")

    # Combine all ADMS to perform a clean unified split later
    full_pool = train_pool + test_pool
    print(f"  ADMS: Collected a total of {len(full_pool)} image-label pairs")
    return full_pool


# ─────────────────────────────────────────
# DMS — remap class IDs, drop unmapped classes
# ─────────────────────────────────────────

def build_dms_id_map(cfg, unified_classes):
    """
    Builds a mapping from DMS's raw class_id (int) to the unified class_id (int).
    Classes mapped to null (e.g. Open Eye) are excluded — their id maps to None.
    """
    source_classes = cfg["dataset"]["source_classes"]
    class_mapping  = cfg["dataset"]["class_mapping"]

    name_to_unified_id = {name: cid for cid, name in unified_classes.items()}

    raw_id_to_unified_id = {}
    for raw_id, source_name in enumerate(source_classes):
        mapped_name = class_mapping.get(source_name)
        if mapped_name is None:
            raw_id_to_unified_id[raw_id] = None
        else:
            raw_id_to_unified_id[raw_id] = name_to_unified_id[mapped_name]

    return raw_id_to_unified_id


def remap_dms_label_file(src_label_path, dst_label_path, id_map):
    """
    Reads a DMS YOLO label file, remaps class ids using id_map,
    drops lines whose class maps to None, writes the result.
    """
    lines_out = []

    with open(src_label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            raw_id = int(parts[0])
            new_id = id_map.get(raw_id)

            if new_id is None:
                continue  # dropped class (e.g. Open Eye)

            new_line = " ".join([str(new_id)] + parts[1:])
            lines_out.append(new_line)

    if not lines_out:
        return False

    with open(dst_label_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    return True


def collect_dms(cfg, unified_classes):
    """
    Collects DMS train/valid/test pools and builds the remapping ID map.
    """
    id_map = build_dms_id_map(cfg, unified_classes)

    train_images = Path(cfg["dataset"]["train_images"])
    train_labels = Path(cfg["dataset"]["train_labels"])
    valid_images = Path(cfg["dataset"]["valid_images"])
    valid_labels = Path(cfg["dataset"]["valid_labels"])
    test_images  = Path(cfg["dataset"]["test_images"])
    test_labels  = Path(cfg["dataset"]["test_labels"])

    train_pool = _pair_images_labels(train_images, train_labels, prefix="dms")
    val_pool   = _pair_images_labels(valid_images, valid_labels, prefix="dms")
    test_pool  = _pair_images_labels(test_images, test_labels, prefix="dms")

    full_pool = train_pool + val_pool + test_pool
    print(f"  DMS: Collected a total of {len(full_pool)} image-label pairs")
    return full_pool, id_map


# ─────────────────────────────────────────
# YawDD — collect preprocessed yawning subset
# ─────────────────────────────────────────

def collect_yawdd(yawdd_root_path):
    """
    Collects the preprocessed YawDD images and labels (yawning class 0).
    """
    yawdd_root = Path(yawdd_root_path)
    images_dir = yawdd_root / "images"
    labels_dir = yawdd_root / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"  WARNING: YawDD directories not found at {yawdd_root}")
        return []

    yawdd_pool = _pair_images_labels(images_dir, labels_dir, prefix="yawdd")
    print(f"  YawDD: Collected a total of {len(yawdd_pool)} image-label pairs")
    return yawdd_pool


# ─────────────────────────────────────────
# Shared helper — pair image/label files
# ─────────────────────────────────────────

def _pair_images_labels(images_dir, labels_dir, prefix):
    """
    Matches each image file to its label file by stem.
    """
    pairs = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"    WARNING: no label for {image_path.name}, skipping")
            continue

        new_stem = f"{prefix}_{image_path.stem}"
        pairs.append((image_path, label_path, new_stem))

    return pairs


# ─────────────────────────────────────────
# Copy a pool of (image, label, new_stem) into an output split
# ─────────────────────────────────────────

def copy_pool(pool, out_images_dir, out_labels_dir, remap_id_map=None):
    """
    Copies images and labels. Applies remapping for DMS if remap_id_map is present.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped_empty = 0

    for image_path, label_path, new_stem in pool:
        dst_image = out_images_dir / f"{new_stem}{image_path.suffix}"
        dst_label = out_labels_dir / f"{new_stem}.txt"

        if remap_id_map is None:
            shutil.copy2(label_path, dst_label)
        else:
            ok = remap_dms_label_file(label_path, dst_label, remap_id_map)
            if not ok:
                dropped_empty += 1
                continue

        shutil.copy2(image_path, dst_image)
        kept += 1

    print(f"    copied {kept} pairs"
          + (f", dropped {dropped_empty} (empty after remap)" if dropped_empty else ""))


# ─────────────────────────────────────────
# Unified Splitter: 3-Way Split (70/15/15)
# ─────────────────────────────────────────

def split_pool_three_way(pool, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Splits a given pool into exactly three parts: Train, Val, and Test
    based on the configured ratios and a fixed seed.
    """
    rng = random.Random(seed)
    shuffled = pool.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_part = shuffled[:n_train]
    val_part   = shuffled[n_train:n_train + n_val]
    test_part  = shuffled[n_train + n_val:]

    return train_part, val_part, test_part


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(
    adms_config_path="configs/data/adms.yaml",
    dms_config_path="configs/data/dms.yaml",
    classes_config_path="configs/data/classes.yaml",
    yawdd_raw_path="data/raw/YawDD",
    output_dir="data/processed",
):
    adms_cfg    = load_config(adms_config_path)
    dms_cfg     = load_config(dms_config_path)
    classes_cfg = load_config(classes_config_path)

    unified_classes = classes_cfg["classes"]
    output_dir = Path(output_dir)

    # Clean existing processed directory if it exists to prevent leftovers
    if output_dir.exists():
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    print("=" * 60)
    print("Vision preprocessing — unifying ADMS + DMS + YawDD")
    print(f"Unified classes: {unified_classes}")
    print("=" * 60)

    # 1. Collect all pools
    print("\nCollecting ADMS...")
    adms_pool = collect_adms(adms_cfg)

    print("\nCollecting DMS...")
    dms_pool, dms_id_map = collect_dms(dms_cfg, unified_classes)

    print("\nCollecting YawDD...")
    yawdd_pool = collect_yawdd(yawdd_raw_path)

    # 2. Split each pool separately to maintain balanced classes and avoid leaks
    seed = adms_cfg["preprocessing"].get("seed", 42)
    
    print("\nSplitting pools (70% Train / 15% Val / 15% Test)...")
    adms_train, adms_val, adms_test = split_pool_three_way(adms_pool, seed=seed)
    dms_train, dms_val, dms_test = split_pool_three_way(dms_pool, seed=seed)
    yawdd_train, yawdd_val, yawdd_test = split_pool_three_way(yawdd_pool, seed=seed)

    # 3. Copy and process everything into the final destination
    print("\nWriting merged dataset...")

    # --- Train Split ---
    print("  [train] ADMS portion:")
    copy_pool(adms_train, output_dir / "train" / "images", output_dir / "train" / "labels")
    print("  [train] DMS portion:")
    copy_pool(dms_train, output_dir / "train" / "images", output_dir / "train" / "labels", remap_id_map=dms_id_map)
    print("  [train] YawDD portion:")
    copy_pool(yawdd_train, output_dir / "train" / "images", output_dir / "train" / "labels")

    # --- Val Split ---
    print("  [val] ADMS portion:")
    copy_pool(adms_val, output_dir / "val" / "images", output_dir / "val" / "labels")
    print("  [val] DMS portion:")
    copy_pool(dms_val, output_dir / "val" / "images", output_dir / "val" / "labels", remap_id_map=dms_id_map)
    print("  [val] YawDD portion:")
    copy_pool(yawdd_val, output_dir / "val" / "images", output_dir / "val" / "labels")

    # --- Test Split ---
    print("  [test] ADMS portion:")
    copy_pool(adms_test, output_dir / "test" / "images", output_dir / "test" / "labels")
    print("  [test] DMS portion:")
    copy_pool(dms_test, output_dir / "test" / "images", output_dir / "test" / "labels", remap_id_map=dms_id_map)
    print("  [test] YawDD portion:")
    copy_pool(yawdd_test, output_dir / "test" / "images", output_dir / "test" / "labels")

    print()
    print("=" * 60)
    print("PREPROCESSING & SPLITTING COMPLETE")
    print(f"Output: {output_dir}")
    print(f"YOLO data config: {write_yolo_data_yaml(output_dir, classes_cfg)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
