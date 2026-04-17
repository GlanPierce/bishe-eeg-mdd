from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

REQUIRED = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "mne",
    "pyedflib",
    "sklearn",
    "torch",
    "torch_geometric",
]


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Workspace: {Path.cwd()}")

    failed = []
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {name:<16} {version}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, str(exc)))
            print(f"[FAIL] {name:<16} {exc}")

    # Quick runtime checks for ML stack
    if not failed:
        import torch

        print(f"Torch CUDA available: {torch.cuda.is_available()}")
        print("Environment check passed.")
        return 0

    print("\nEnvironment check failed for:")
    for name, msg in failed:
        print(f" - {name}: {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
