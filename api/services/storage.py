import os
import uuid
from pathlib import Path

SHARED_DIR = "/shared-data"

def save_upload_to_shared_dir(filename: str, content: bytes, subdir: str | None = None) -> str:
    target_dir = SHARED_DIR
    if subdir:
        target_dir = target_dir / Path(subdir)

    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = filename or "upload.bin"
    path = target_dir / f"{uuid.uuid4()}_{safe_name}"

    with path.open("wb") as f:
        f.write(content)

    return str(path)


def read_uploaded_file_bytes(path: str) -> bytes:
    file_path = Path(path)
    with file_path.open("rb") as f:
        return f.read()