import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import wfdb


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]


@dataclass(frozen=True)
class PtbxlRecord:
    ecg_id: int
    filename_hr: str
    filename_lr: str | None = None


def parse_ecg_id_from_image_name(image_path: Path) -> int | None:
    """
    Examples:
      21836_clean.png -> 21836
      21836.png       -> 21836
      IMG_5785.JPG    -> 5785, но для raw_images ожидаем именно PTB-XL id в названии.
    """
    match = re.search(r"(\d+)", image_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def load_ptbxl_metadata(metadata_csv: Path) -> dict[int, PtbxlRecord]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"PTB-XL metadata file not found: {metadata_csv}")

    records: dict[int, PtbxlRecord] = {}

    with metadata_csv.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {"ecg_id", "filename_hr"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing columns in {metadata_csv}: {sorted(missing_columns)}"
            )

        for row in reader:
            ecg_id = int(row["ecg_id"])
            filename_hr = str(row["filename_hr"]).strip()
            filename_lr = str(row.get("filename_lr", "")).strip() or None

            records[ecg_id] = PtbxlRecord(
                ecg_id=ecg_id,
                filename_hr=filename_hr,
                filename_lr=filename_lr,
            )

    return records


def iter_images_by_class(
    raw_images_root: Path,
    class_names: list[str],
) -> Iterable[tuple[str, Path]]:
    if not raw_images_root.exists():
        raise FileNotFoundError(f"raw_images root not found: {raw_images_root}")

    for class_name in class_names:
        class_dir = raw_images_root / class_name
        if not class_dir.exists():
            print(f"[WARN] Class directory not found: {class_dir}")
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield class_name, image_path


def build_raw_image_series_manifest(
    *,
    raw_images_root: Path,
    ptbxl_root: Path,
    metadata_csv: Path,
    series_dir: Path,
    output_manifest_csv: Path,
    class_names: list[str],
) -> Path:
    metadata = load_ptbxl_metadata(metadata_csv)

    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    rows: list[dict[str, str | int | float]] = []

    for class_name, image_path in iter_images_by_class(
        raw_images_root=raw_images_root,
        class_names=class_names,
    ):
        stats["images_seen"] += 1

        ecg_id = parse_ecg_id_from_image_name(image_path)
        if ecg_id is None:
            stats["missing_ecg_id_in_filename"] += 1
            continue

        record = metadata.get(ecg_id)
        if record is None:
            stats["ecg_id_not_found_in_ptbxl_metadata"] += 1
            continue

        ptbxl_record_path = ptbxl_root / record.filename_hr
        hea_path = ptbxl_record_path.with_suffix(".hea")
        dat_path = ptbxl_record_path.with_suffix(".dat")

        if not hea_path.exists() or not dat_path.exists():
            stats["missing_wfdb_files"] += 1
            continue

        signal_path = series_dir / f"{ecg_id}.npy"

        rows.append(
            {
                "image_path": str(image_path),
                "relative_path": str(image_path.relative_to(raw_images_root)),
                "class_name": class_name,
                "ecg_id": ecg_id,
                "ptbxl_record_path": str(ptbxl_record_path),
                "signal_path": str(signal_path),
                "source_set": "raw",
                "match_method": "filename",
                "match_score": 1.0,
            }
        )
        stats["matched"] += 1

    fieldnames = [
        "image_path",
        "relative_path",
        "class_name",
        "ecg_id",
        "ptbxl_record_path",
        "signal_path",
        "source_set",
        "match_method",
        "match_score",
    ]

    with output_manifest_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[build_raw_image_series_manifest] Done")
    print(f"  output_manifest_csv={output_manifest_csv}")
    print(f"  images_seen={stats['images_seen']}")
    print(f"  matched={stats['matched']}")
    print(f"  missing_ecg_id_in_filename={stats['missing_ecg_id_in_filename']}")
    print(f"  ecg_id_not_found_in_ptbxl_metadata={stats['ecg_id_not_found_in_ptbxl_metadata']}")
    print(f"  missing_wfdb_files={stats['missing_wfdb_files']}")

    class_counts = Counter(row["class_name"] for row in rows)
    print("  class_counts:")
    for class_name in class_names:
        print(f"    {class_name}: {class_counts[class_name]}")

    return output_manifest_csv


def read_manifest_rows(manifest_csv: Path) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    with manifest_csv.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def export_ptbxl_signals_from_manifest(
    *,
    manifest_csv: Path,
    force: bool = False,
) -> None:
    rows = read_manifest_rows(manifest_csv)

    by_ecg_id: dict[int, dict[str, str]] = {}
    for row in rows:
        ecg_id = int(row["ecg_id"])
        by_ecg_id[ecg_id] = row

    exported = 0
    skipped = 0
    failed = 0

    for ecg_id, row in sorted(by_ecg_id.items()):
        ptbxl_record_path = Path(row["ptbxl_record_path"])
        signal_path = Path(row["signal_path"])
        signal_path.parent.mkdir(parents=True, exist_ok=True)

        if signal_path.exists() and not force:
            skipped += 1
            continue

        try:
            signal, _meta = wfdb.rdsamp(str(ptbxl_record_path))
            # wfdb returns [time, leads], сохраняем [leads, time]
            signal = signal.T.astype(np.float32)

            np.save(signal_path, signal)
            exported += 1
        except Exception as exc:
            failed += 1
            print(
                f"[WARN] Failed to export ecg_id={ecg_id}, "
                f"record={ptbxl_record_path}: {exc}"
            )

    print("[export_ptbxl_signals_from_manifest] Done")
    print(f"  manifest_csv={manifest_csv}")
    print(f"  unique_ecg_ids={len(by_ecg_id)}")
    print(f"  exported={exported}")
    print(f"  skipped={skipped}")
    print(f"  failed={failed}")


def split_manifest_grouped_by_ecg_id(
    *,
    manifest_csv: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Path]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must be equal to 1.0"
        )

    rows = read_manifest_rows(manifest_csv)

    groups_by_ecg_id: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups_by_ecg_id[int(row["ecg_id"])].append(row)

    # Если один ecg_id случайно попал в несколько классов,
    # вся группа всё равно попадёт только в один split.
    grouped_items: list[tuple[str, int, list[dict[str, str]]]] = []
    for ecg_id, group_rows in groups_by_ecg_id.items():
        class_counter = Counter(row["class_name"] for row in group_rows)
        dominant_class = class_counter.most_common(1)[0][0]
        grouped_items.append((dominant_class, ecg_id, group_rows))

    grouped_by_class: dict[str, list[tuple[int, list[dict[str, str]]]]] = defaultdict(list)
    for class_name, ecg_id, group_rows in grouped_items:
        grouped_by_class[class_name].append((ecg_id, group_rows))

    rng = random.Random(seed)

    split_rows = {
        "train": [],
        "val": [],
        "test": [],
    }

    for class_name, groups in grouped_by_class.items():
        rng.shuffle(groups)

        n = len(groups)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_groups = groups[:n_train]
        val_groups = groups[n_train : n_train + n_val]
        test_groups = groups[n_train + n_val :]

        for _ecg_id, group_rows in train_groups:
            split_rows["train"].extend(group_rows)
        for _ecg_id, group_rows in val_groups:
            split_rows["val"].extend(group_rows)
        for _ecg_id, group_rows in test_groups:
            split_rows["test"].extend(group_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else [
        "image_path",
        "relative_path",
        "class_name",
        "ecg_id",
        "ptbxl_record_path",
        "signal_path",
        "source_set",
        "match_method",
        "match_score",
    ]

    output_paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }

    for split_name, path in output_paths.items():
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows[split_name])

    print("[split_manifest_grouped_by_ecg_id] Done")
    print(f"  output_dir={output_dir}")
    for split_name, path in output_paths.items():
        class_counts = Counter(row["class_name"] for row in split_rows[split_name])
        ecg_count = len({int(row["ecg_id"]) for row in split_rows[split_name]})
        print(f"  {split_name}: rows={len(split_rows[split_name])}, ecg_ids={ecg_count}, path={path}")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}")

    return output_paths


def validate_image_series_split(
    *,
    split_dir: Path,
) -> None:
    split_paths = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
    }

    all_ecg_ids: dict[str, set[int]] = {}

    for split_name, path in split_paths.items():
        rows = read_manifest_rows(path)

        missing_images = 0
        missing_signals = 0
        class_counts = Counter()

        ecg_ids = set()
        for row in rows:
            image_path = Path(row["image_path"])
            signal_path = Path(row["signal_path"])

            if not image_path.exists():
                missing_images += 1
            if not signal_path.exists():
                missing_signals += 1

            class_counts[row["class_name"]] += 1
            ecg_ids.add(int(row["ecg_id"]))

        all_ecg_ids[split_name] = ecg_ids

        print(f"[validate] {split_name}")
        print(f"  rows={len(rows)}")
        print(f"  unique_ecg_ids={len(ecg_ids)}")
        print(f"  missing_images={missing_images}")
        print(f"  missing_signals={missing_signals}")
        print("  class_counts:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}")

        if missing_images > 0 or missing_signals > 0:
            raise RuntimeError(
                f"Split {split_name} has missing files: "
                f"missing_images={missing_images}, missing_signals={missing_signals}"
            )

    train_val_overlap = all_ecg_ids["train"] & all_ecg_ids["val"]
    train_test_overlap = all_ecg_ids["train"] & all_ecg_ids["test"]
    val_test_overlap = all_ecg_ids["val"] & all_ecg_ids["test"]

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "ECG id leakage between splits detected: "
            f"train_val={len(train_val_overlap)}, "
            f"train_test={len(train_test_overlap)}, "
            f"val_test={len(val_test_overlap)}"
        )

    print("[validate] OK: no missing files and no ECG id leakage")