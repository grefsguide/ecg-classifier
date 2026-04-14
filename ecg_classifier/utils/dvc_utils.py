from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def try_dvc_pull(
    target_path: Path,
    max_retries: int = 3,
) -> bool:
    """
    Tries to pull DVC-tracked data to the workspace.
    Returns True if succeeded, False otherwise.
    """
    if not target_path.exists():
        return False

    for attempt_index in range(max_retries):
        try:
            process = subprocess.run(
                [sys.executable, "-m", "dvc", "pull", str(target_path)],
                capture_output=True,
                text=True,
            )
            if process.returncode == 0:
                return True

            if attempt_index == max_retries - 1:
                print("DVC pull failed.")
                print("STDOUT:\n", process.stdout)
                print("STDERR:\n", process.stderr)
                return False

            sleep_seconds = 2 * (attempt_index + 1)
            time.sleep(sleep_seconds)

        except Exception as exc:
            if attempt_index == max_retries - 1:
                print(f"DVC pull crashed: {exc}")
                return False
            time.sleep(2 * (attempt_index + 1))

    return False