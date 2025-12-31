from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ecg_classifier.utils.io_utils import ensure_dir


_YADISK_DOWNLOAD_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


@dataclass(frozen=True)
class DownloadConfig:
    public_url: str
    file_names: list[str]
    download_dir: Path
    extract_dir: Path
    seven_zip_path: str
    timeout_sec: int
    chunk_size_bytes: int
    max_retries: int


def _get_download_href(public_url: str, file_name: str, timeout_sec: int) -> str:
    response = requests.get(
        _YADISK_DOWNLOAD_API,
        params={"public_key": public_url, "path": file_name},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()

    href = payload.get("href")
    if not href:
        raise RuntimeError(
            "Yandex Disk API response does not contain 'href'. "
            f"Response payload: {json.dumps(payload, ensure_ascii=False)}"
        )
    return str(href)


def _download_with_retries(
    url: str,
    output_path: Path,
    timeout_sec: int,
    chunk_size_bytes: int,
    max_retries: int,
) -> None:
    ensure_dir(output_path.parent)

    for attempt_index in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout_sec) as response:
                response.raise_for_status()
                with output_path.open("wb") as file_handle:
                    for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                        if chunk:
                            file_handle.write(chunk)
            return
        except Exception as exc:
            if attempt_index == max_retries - 1:
                raise RuntimeError(
                    f"Failed to download after {max_retries} attempts: {output_path}"
                ) from exc
            sleep_seconds = 2 * (attempt_index + 1)
            time.sleep(sleep_seconds)


def _resolve_7zip(seven_zip_path: str) -> Path:
    if seven_zip_path:
        candidate = Path(seven_zip_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"seven_zip_path does not exist: {candidate}")

    from_path = shutil.which("7z") or shutil.which("7za")
    if from_path:
        return Path(from_path)

    windows_candidates = [
        Path("C:/Program Files/7-Zip/7z.exe"),
        Path("C:/Program Files (x86)/7-Zip/7z.exe"),
    ]
    for candidate in windows_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "7-Zip executable not found. Install 7-Zip and add '7z' to PATH, "
        "or pass download.seven_zip_path='C:/Program Files/7-Zip/7z.exe'."
    )


def _extract_multipart_zip(zip_path: Path, extract_dir: Path, seven_zip_exe: Path) -> None:
    ensure_dir(extract_dir)

    # важно: 7z должен видеть рядом .z01/.z02/.z03, поэтому cwd = zip_path.parent
    command = [
        str(seven_zip_exe),
        "x",
        "-y",
        f"-o{str(extract_dir)}",
        str(zip_path),
    ]
    process = subprocess.run(
        command,
        cwd=str(zip_path.parent),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "7-Zip extraction failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}\n"
        )


def download_and_extract_ecg_archive(download_cfg: DownloadConfig) -> Path:
    if not download_cfg.public_url:
        raise ValueError("download.public_url is empty. Provide a public Yandex.Disk folder URL.")

    ensure_dir(download_cfg.download_dir)
    ensure_dir(download_cfg.extract_dir)

    print(f"Downloading to: {download_cfg.download_dir}")
    for file_name in download_cfg.file_names:
        target_path = download_cfg.download_dir / file_name
        if target_path.exists() and target_path.stat().st_size > 0:
            print(f"Skip (already exists): {target_path.name}")
            continue

        download_href = _get_download_href(
            public_url=download_cfg.public_url,
            file_name=file_name,
            timeout_sec=download_cfg.timeout_sec,
        )
        print(f"Downloading: {file_name}")
        _download_with_retries(
            url=download_href,
            output_path=target_path,
            timeout_sec=download_cfg.timeout_sec,
            chunk_size_bytes=download_cfg.chunk_size_bytes,
            max_retries=download_cfg.max_retries,
        )

    main_zip = download_cfg.download_dir / "ecg_img.zip"
    if not main_zip.exists():
        raise FileNotFoundError(f"Main zip not found after download: {main_zip}")

    seven_zip_exe = _resolve_7zip(download_cfg.seven_zip_path)

    print(f"Extracting with 7z: {seven_zip_exe}")
    _extract_multipart_zip(zip_path=main_zip, extract_dir=download_cfg.extract_dir, seven_zip_exe=seven_zip_exe)

    return download_cfg.extract_dir