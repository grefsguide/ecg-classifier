from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    class_index: int


class EcgImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_csv: Path,
        class_names: list[str],
        transform,
    ) -> None:
        self.data_root = data_root
        self.transform = transform
        self.class_names = class_names
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.items = self._read_split(split_csv=split_csv)

    def _read_split(self, split_csv: Path) -> list[DatasetItem]:
        items: list[DatasetItem] = []
        with split_csv.open("r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                relative_path = row["relative_path"]
                class_name = row["class_name"]
                class_index = self.class_to_index[class_name]
                image_path = self.data_root / Path(relative_path)
                items.append(DatasetItem(image_path=image_path, class_index=class_index))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        dataset_item = self.items[index]
        image = Image.open(dataset_item.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "target": dataset_item.class_index}