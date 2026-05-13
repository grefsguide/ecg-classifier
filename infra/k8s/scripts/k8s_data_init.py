import os
import shutil
import subprocess
from pathlib import Path


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def run(command: list[str]) -> None:
    print(f"[data-init] run: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def download_with_gdown(url_or_id: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[data-init] archive already exists: {output_path}", flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            "gdown",
            "--fuzzy",
            url_or_id,
            "-O",
            str(output_path),
        ]
    )


def extract_7z(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_path = output_dir / ".extracted"
    if marker_path.exists():
        print(f"[data-init] already extracted: {output_dir}", flush=True)
        return

    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            "7z",
            "x",
            str(archive_path),
            f"-o{tmp_dir}",
            "-y",
        ]
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    tmp_dir.rename(output_dir)
    marker_path.write_text("ok", encoding="utf-8")

    print(f"[data-init] extracted {archive_path} -> {output_dir}", flush=True)


def main() -> None:
    shared_data_dir = Path(os.getenv("SHARED_DATA_DIR", "/shared-data"))
    archives_dir = shared_data_dir / "archives"

    ptbxl_url = require_env("GDRIVE_PTBXL_ARCHIVE_URL")
    ecg_img_url = require_env("GDRIVE_ECG_IMG_ARCHIVE_URL")

    ptbxl_archive = archives_dir / "PTB-XL.7z"
    ecg_img_archive = archives_dir / "ecg_img.7z"

    ptbxl_output_dir = shared_data_dir / "PTB-XL"
    ecg_img_output_dir = shared_data_dir / "ecg_img"

    print("[data-init] starting", flush=True)
    print(f"[data-init] shared_data_dir={shared_data_dir}", flush=True)

    download_with_gdown(ptbxl_url, ptbxl_archive)
    download_with_gdown(ecg_img_url, ecg_img_archive)

    extract_7z(ptbxl_archive, ptbxl_output_dir)
    extract_7z(ecg_img_archive, ecg_img_output_dir)

    print("[data-init] done", flush=True)


if __name__ == "__main__":
    main()