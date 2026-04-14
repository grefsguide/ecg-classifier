from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
from hydra import compose, initialize_config_module

from ecg_classifier.data.download_yadisk import DownloadConfig, download_and_extract_ecg_archive
from ecg_classifier.utils.dvc_utils import try_dvc_pull
from ecg_classifier.data.split_data import split_and_save
from ecg_classifier.training.eval import evaluate
from ecg_classifier.training.train import train
from ecg_classifier.utils.seed import seed_everything

def _compose_cfg(overrides: list[str]) -> Any:
    with initialize_config_module(config_module="ecg_classifier.conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg

def _has_images_in_directory(directory_path: Path) -> bool:
    if not directory_path.exists():
        return False

    for file_path in directory_path.rglob("*"):
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return True
    return False

def _is_dataset_ready(data_root: Path, class_names: list[str]) -> bool:
    if not data_root.exists():
        return False

    for class_name in class_names:
        class_dir = data_root / class_name
        if not _has_images_in_directory(class_dir):
            return False
    return True

def _ensure_dataset_available(cfg: Any) -> None:
    data_root = Path(cfg.data.root_dir)
    class_names = list(cfg.data.class_names)

    if _is_dataset_ready(data_root=data_root, class_names=class_names):
        return

    dvc_descriptor_path = Path("ecg_img.dvc")
    if dvc_descriptor_path.exists():
        print("Dataset not found. Trying to restore via DVC pull...")
        dvc_ok = try_dvc_pull(target_path=dvc_descriptor_path, max_retries=3)
        if dvc_ok and _is_dataset_ready(data_root=data_root, class_names=class_names):
            print("Dataset restored via DVC.")
            return
        print("DVC pull did not restore dataset. Falling back to Yandex.Disk download...")

    public_url = str(cfg.download.public_url).strip()
    if not public_url:
        raise ValueError(
            "Dataset directory is missing or incomplete and download.public_url is empty.\n"
            "Provide Yandex.Disk public folder link, for example:\n"
            "  python -m ecg_classifier.commands train download.public_url=\"https://disk.360.yandex.ru/d/XXXX\""
        )

    download_cfg = DownloadConfig(
        public_url=public_url,
        file_names=list(cfg.download.file_names),
        download_dir=Path(cfg.download.download_dir),
        extract_dir=data_root,
        seven_zip_path=str(cfg.download.seven_zip_path),
        timeout_sec=int(cfg.download.timeout_sec),
        chunk_size_bytes=int(cfg.download.chunk_size_bytes),
        max_retries=int(cfg.download.max_retries),
    )

    print("Dataset not found. Downloading from Yandex.Disk...")
    extracted_dir = download_and_extract_ecg_archive(download_cfg=download_cfg)
    print(f"Dataset extracted to: {extracted_dir}")

    if not _is_dataset_ready(data_root=data_root, class_names=class_names):
        raise RuntimeError(
            "Dataset download/extraction finished, but dataset structure is still incomplete.\n"
            "Expected folders with images:\n"
            f"  {data_root}/" + ", ".join(class_names)
        )

    print("Hint (optional, for reproducibility): track downloaded data with DVC:")
    print(f"  dvc add {download_cfg.download_dir}")
    print(f"  dvc add {data_root}")
    print('  git add . && git commit -m "Add ECG dataset" && dvc push')

def _splits_exist(splits_dir: Path, split_name: str) -> bool:
    split_root = splits_dir / split_name
    return (split_root / "train.csv").exists() and (split_root / "val.csv").exists() and (
        split_root / "test.csv"
    ).exists()


def _ensure_splits_available(cfg: Any) -> None:
    splits_dir = Path(cfg.data.splits_dir)
    split_name = str(cfg.split.output_name)

    if _splits_exist(splits_dir=splits_dir, split_name=split_name):
        return

    print("Splits not found. Creating train/val/test splits...")

    split_paths = split_and_save(
        data_root=Path(cfg.data.root_dir),
        splits_dir=splits_dir,
        class_names=list(cfg.data.class_names),
        train_ratio=float(cfg.split.train_ratio),
        val_ratio=float(cfg.split.val_ratio),
        test_ratio=float(cfg.split.test_ratio),
        seed=int(cfg.seed),
        output_name=split_name,
    )

    print(
        "Splits created:\n"
        f"  train={split_paths.train_csv}\n"
        f"  val={split_paths.val_csv}\n"
        f"  test={split_paths.test_csv}\n"
    )
    print("Hint (optional, for reproducibility): track splits with DVC:")
    print(f"  dvc add {splits_dir}")
    print('  git add . && git commit -m "Add splits" && dvc push')

class Commands:
    def split(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        seed_everything(int(cfg.seed))

        split_paths = split_and_save(
            data_root=Path(cfg.data.root_dir),
            splits_dir=Path(cfg.data.splits_dir),
            class_names=list(cfg.data.class_names),
            train_ratio=float(cfg.split.train_ratio),
            val_ratio=float(cfg.split.val_ratio),
            test_ratio=float(cfg.split.test_ratio),
            seed=int(cfg.seed),
            output_name=str(cfg.split.output_name),
        )

        print(f"Saved splits:\n  train={split_paths.train_csv}\n  val={split_paths.val_csv}\n  test={split_paths.test_csv}")
        print("Don't forget to track splits with DVC:")
        print(f"  dvc add {Path(cfg.data.splits_dir)}")
        print("  git add . && git commit -m \"Add data splits\" && dvc push")

    def train(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        seed_everything(int(cfg.seed))

        _ensure_dataset_available(cfg)
        _ensure_splits_available(cfg)

        best_checkpoint_path = train(cfg=cfg)
        print(f"Best checkpoint: {best_checkpoint_path}")
        print("Don't forget to track checkpoints with DVC:")
        print("  dvc add artifacts/checkpoints")
        print("  git add . && git commit -m \"Add checkpoints\" && dvc push")

    def evaluate(self, checkpoint_path: str | None = None, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        seed_everything(int(cfg.seed))

        if checkpoint_path is None:
            raise ValueError("Please pass checkpoint_path=...")

        metrics_path = evaluate(cfg=cfg, checkpoint_path=Path(checkpoint_path))
        print(f"Saved metrics: {metrics_path}")
        print("Don't forget to track metrics with DVC:")
        print("  dvc add artifacts/metrics")
        print("  git add . && git commit -m \"Add metrics\" && dvc push")

    def download_data(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))

        download_cfg = DownloadConfig(
            public_url=str(cfg.download.public_url),
            file_names=list(cfg.download.file_names),
            download_dir=Path(cfg.download.download_dir),
            extract_dir=Path(cfg.download.extract_dir),
            seven_zip_path=str(cfg.download.seven_zip_path),
            timeout_sec=int(cfg.download.timeout_sec),
            chunk_size_bytes=int(cfg.download.chunk_size_bytes),
            max_retries=int(cfg.download.max_retries),
        )

        extracted_dir = download_and_extract_ecg_archive(download_cfg=download_cfg)

        print(f"Extracted dataset to: {extracted_dir}")
        print("Next steps (DVC):")
        print(f"  dvc add {Path(cfg.download.download_dir)}")
        print(f"  dvc add {Path(cfg.download.extract_dir)}")
        print('  git add . && git commit -m "Add ECG dataset (archive + extracted)"')
        print("  dvc push")

def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()