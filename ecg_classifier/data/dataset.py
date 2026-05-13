import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    class_index: int
    signal_path: Path | None = None
    ecg_id: int | None = None


def normalize_path_value(path_value: str) -> str:
    return path_value.strip().replace("\\", "/")


def resolve_path(path_value: str, root_dir: Path | None = None) -> Path:
    normalized = normalize_path_value(path_value)
    path = Path(normalized)

    if path.is_absolute():
        return path

    if root_dir is not None:
        return root_dir / path

    return path


class EcgImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_csv: Path,
        class_names: list[str],
        transform,
        signal_length: int = 5000,
    ) -> None:
        self.data_root = data_root
        self.transform = transform
        self.class_names = class_names
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.signal_length = signal_length
        self.items = self._read_split(split_csv=split_csv)

    def _read_split(self, split_csv: Path) -> list[DatasetItem]:
        items: list[DatasetItem] = []

        with split_csv.open("r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or [])

            has_signal_path = "signal_path" in fieldnames

            for row in reader:
                class_name = row["class_name"]
                class_index = self.class_to_index[class_name]

                if row.get("relative_path"):
                    image_path = resolve_path(row["relative_path"], self.data_root)
                elif row.get("image_path"):
                    image_path = resolve_path(row["image_path"])
                else:
                    raise ValueError(
                        f"Split row has no relative_path/image_path: {row}"
                    )

                signal_path = None
                if has_signal_path and row.get("signal_path"):
                    signal_path = resolve_path(row["signal_path"])

                ecg_id = int(row["ecg_id"]) if row.get("ecg_id") else None

                items.append(
                    DatasetItem(
                        image_path=image_path,
                        class_index=class_index,
                        signal_path=signal_path,
                        ecg_id=ecg_id,
                    )
                )

        return items

    def __len__(self) -> int:
        return len(self.items)

    def _load_signal(self, signal_path: Path) -> torch.Tensor:
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_path}")

        signal = np.load(signal_path).astype(np.float32)

        if signal.ndim != 2:
            raise ValueError(
                f"Expected signal shape [leads, time], got {signal.shape}"
            )

        if signal.shape[1] > self.signal_length:
            signal = signal[:, : self.signal_length]
        elif signal.shape[1] < self.signal_length:
            pad_width = self.signal_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)))

        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-6
        signal = (signal - mean) / std

        signal = np.clip(signal, -3.0, 3.0) / 3.0

        return torch.from_numpy(signal)

    def __getitem__(self, index: int):
        dataset_item = self.items[index]

        if not dataset_item.image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {dataset_item.image_path}. "
                "Check data.root_dir and relative_path in split CSV."
            )

        image = Image.open(dataset_item.image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        batch_item = {
            "image": image,
            "target": dataset_item.class_index,
        }

        if dataset_item.ecg_id is not None:
            batch_item["ecg_id"] = dataset_item.ecg_id

        if dataset_item.signal_path is not None:
            batch_item["signal"] = self._load_signal(dataset_item.signal_path)

        return batch_item