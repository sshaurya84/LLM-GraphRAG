"""
Device selection helpers for Apple Metal (MPS) with CUDA portability.
"""

from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_dtype():
    """
    Use float32 by default because half precision can be unstable on MPS.
    """
    return torch.float32


__all__ = ["get_torch_device", "torch_dtype"]

