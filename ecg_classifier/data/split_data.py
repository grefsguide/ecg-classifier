from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from random import Random

from ecg_classifier.utils.io_utils import ensure_dir


@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    val_csv: Path
    test_csv: Path


def list_images_by_class(data_root: Path, class_names: list[str]) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for class_name in class_names:
        class_dir = data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            relative_path = image_path.relative_to(data_root).as_posix()
            records.append((relative_path, class_name))
    return records


def stratified_split(
    records: list[tuple[str, str]],
    class_names: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    random_state = Random(seed)

    class_to_records: dict[str, list[tuple[str, str]]] = {name: [] for name in class_names}
    for relative_path, class_name in records:
        class_to_records[class_name].append((relative_path, class_name))

    train_records: list[tuple[str, str]] = []
    val_records: list[tuple[str, str]] = []
    test_records: list[tuple[str, str]] = []

    for class_name in class_names:
        class_records = class_to_records[class_name]
        random_state.shuffle(class_records)

        total_count = len(class_records)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)

        train_records.extend(class_records[:train_count])
        val_records.extend(class_records[train_count : train_count + val_count])
        test_records.extend(class_records[train_count + val_count :])

    random_state.shuffle(train_records)
    random_state.shuffle(val_records)
    random_state.shuffle(test_records)

    return train_records, val_records, test_records


def write_csv(csv_path: Path, rows: list[tuple[str, str]]) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["relative_path", "class_name"])
        for relative_path, class_name in rows:
            writer.writerow([relative_path, class_name])


def make_split_paths(splits_dir: Path, output_name: str) -> SplitPaths:
    return SplitPaths(
        train_csv=splits_dir / output_name / "train.csv",
        val_csv=splits_dir / output_name / "val.csv",
        test_csv=splits_dir / output_name / "test.csv",
    )


def split_and_save(
    data_root: Path,
    splits_dir: Path,
    class_names: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    output_name: str,
) -> SplitPaths:
    records = list_images_by_class(data_root=data_root, class_names=class_names)
    train_records, val_records, test_records = stratified_split(
        records=records,
        class_names=class_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_paths = make_split_paths(splits_dir=splits_dir, output_name=output_name)
    write_csv(split_paths.train_csv, train_records)
    write_csv(split_paths.val_csv, val_records)
    write_csv(split_paths.test_csv, test_records)
    return split_paths