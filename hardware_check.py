"""Detect the best available PyTorch device (CUDA, MPS, or CPU)."""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """Return the best available torch device and optionally print details.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        details = f"CUDA ({torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} device(s))"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        details = "MPS (Apple Silicon GPU)"
    else:
        device = torch.device("cpu")
        details = "CPU"

    if verbose:
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: {details}")

    return device


if __name__ == "__main__":
    get_device()
