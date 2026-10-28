"""Prints the PyTorch / CUDA setup of the current node and runs a small GPU operation.

Called at the start of the PyTorch GPU jobs (e.g. ``workflows/borzoi/jobs/*.slurm``)
so that a broken CUDA setup shows up in the log before data staging starts.

Usage:
    python workflows/common/check_torch_cuda.py
"""
import os
from ctypes.util import find_library


def main() -> None:
    """Prints versions, visible devices, CUDA libraries and environment variables."""
    import torch

    print("Checking PyTorch and CUDA setup...")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"Device {i}:",
              torch.cuda.get_device_name(i),
              "| capability:",
              torch.cuda.get_device_capability(i))

    print("\nPyTorch build CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

    print("\nLoaded CUDA libs:")
    for lib in ["cudart", "cublas", "cufft", "cusparse", "cudnn"]:
        print(f"{lib}:", find_library(lib))

    print("\nCUDA-related environment variables:")
    for var in ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"]:
        print(f"{var}:", os.environ.get(var, "Not set"))

    if torch.cuda.is_available():
        x = torch.randn(10, device="cuda")
        x @ x.T
        print("\nCUDA tensor op successful")
    else:
        print("\nCUDA not available — skipped test")


if __name__ == "__main__":
    main()
