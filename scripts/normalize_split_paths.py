import csv
from pathlib import Path


def normalize_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError(f"No CSV header found: {path}")

    changed = 0

    for row in rows:
        for column in ["relative_path", "image_path", "signal_path", "ptbxl_record_path"]:
            if column in row and row[column]:
                old_value = row[column]
                new_value = old_value.replace("\\", "/")
                row[column] = new_value

                if old_value != new_value:
                    changed += 1

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{path}: changed={changed}")


def main() -> None:
    split_dir = Path("artifacts/splits/image_series_raw")

    for filename in ["train.csv", "val.csv", "test.csv"]:
        normalize_csv(split_dir / filename)


if __name__ == "__main__":
    main()