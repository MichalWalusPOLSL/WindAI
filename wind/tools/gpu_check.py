"""Report what this machine can actually train on.

Run after `pipenv install` on a new box: it prints the resolved versions, the
NVIDIA driver, and the device each optional booster will really use, which is
the device registry.build_model puts into the pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import registry


def _versions() -> None:
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    for name in ("numpy", "pandas", "sklearn", "scipy", "pyarrow", "xgboost", "lightgbm"):
        try:
            module = __import__(name)
        except ImportError:
            print(f"{name:<11} not installed")
        else:
            print(f"{name:<11} {getattr(module, '__version__', 'unknown')}")


def _driver() -> None:
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi  not found: no NVIDIA driver visible")
        return
    query = "--query-gpu=name,driver_version,memory.total --format=csv,noheader"
    output = subprocess.run(
        ["nvidia-smi", *query.split()], capture_output=True, text=True, check=False
    )
    print("nvidia-smi  " + (output.stdout.strip() or output.stderr.strip()))


def _devices() -> None:
    try:
        import xgboost
    except ImportError:
        print("xgb         not installed")
    else:
        build = xgboost.build_info()
        cuda = f"CUDA {'.'.join(str(v) for v in build['CUDA_VERSION'])}" if build.get("USE_CUDA") else "CPU-only build"
        print(f"xgb         device={registry._gpu_device('xgb', 'cuda', 'cpu')}  ({cuda})")

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("lgbm        not installed")
    else:
        device = registry._gpu_device("lgbm", "gpu", "cpu")
        note = "" if device != "cpu" else "  (PyPI wheels have no GPU tree learner)"
        print(f"lgbm        device={device}{note}")

    print("rf/gbt/knn/logreg/svm/nb/lda  CPU: scikit-learn has no CUDA backend")


if __name__ == "__main__":
    _versions()
    print()
    _driver()
    print()
    _devices()
