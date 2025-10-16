"""Reproducibility utilities for consistent training and evaluation.

This module provides functions for setting random seeds and managing device
configurations to ensure reproducible results across different runs.
"""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for all libraries to ensure reproducibility.

    This function sets the random seed for:
    - Python's built-in random module
    - NumPy
    - PyTorch (both CPU and CUDA)
    - CUDA operations (for deterministic behavior)

    Args:
        seed: Integer seed value to use across all libraries

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set deterministic behavior for CUDA operations
    # Note: This may impact performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the torch device to use for computations.

    Args:
        device: Device string ('cuda', 'cpu', 'mps', or 'auto').
               If 'auto' or None, automatically selects the best available device.

    Returns:
        torch.device: The device to use for tensor operations

    Raises:
        ValueError: If the specified device is not available

    Example:
        >>> device = get_device('auto')
        >>> tensor = torch.tensor([1, 2, 3]).to(device)
    """
    if device is None or device == "auto":
        # Automatically select the best available device
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        else:
            return torch.device("cpu")

    # Validate and return the specified device
    device_lower = device.lower()

    if device_lower == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but CUDA is not available. "
                "Please check your PyTorch installation and GPU drivers."
            )
        return torch.device("cuda")

    elif device_lower == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(
                "MPS device requested but MPS is not available. "
                "MPS is only available on Apple Silicon Macs with macOS 12.3+."
            )
        return torch.device("mps")

    elif device_lower == "cpu":
        return torch.device("cpu")

    else:
        raise ValueError(
            f"Invalid device: {device}. "
            "Valid options are: 'cuda', 'cpu', 'mps', or 'auto'"
        )


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about available compute devices.

    Returns:
        Dictionary containing device information:
            - cuda_available: Whether CUDA is available
            - cuda_version: CUDA version (if available)
            - cudnn_version: cuDNN version (if available)
            - cuda_device_count: Number of CUDA devices
            - cuda_devices: List of CUDA device names
            - mps_available: Whether MPS is available (Apple Silicon)
            - cpu_count: Number of CPU cores
            - pytorch_version: PyTorch version

    Example:
        >>> info = get_device_info()
        >>> print(f"CUDA available: {info['cuda_available']}")
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": torch.get_num_threads(),
    }

    # CUDA information
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["cuda_current_device"] = torch.cuda.current_device()
    else:
        info["cuda_version"] = None
        info["cudnn_version"] = None
        info["cuda_device_count"] = 0
        info["cuda_devices"] = []
        info["cuda_current_device"] = None

    # MPS (Apple Silicon) information
    info["mps_available"] = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    return info


def print_device_info() -> None:
    """Print detailed information about available compute devices.

    This function prints a formatted summary of all available compute devices
    and their properties, useful for debugging and system verification.

    Example:
        >>> print_device_info()
        Device Information:
        ==================
        PyTorch Version: 2.0.0
        CUDA Available: True
        CUDA Version: 11.8
        ...
    """
    info = get_device_info()

    print("Device Information:")
    print("=" * 50)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CPU Threads: {info['cpu_count']}")
    print()

    print(f"CUDA Available: {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        print(f"Current CUDA Device: {info['cuda_current_device']}")
        print("CUDA Devices:")
        for i, device_name in enumerate(info["cuda_devices"]):
            # Get additional device properties
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            device_name_str: str = device_name  # Type annotation for mypy
            print(f"  [{i}] {device_name_str}")
            print(f"      Total Memory: {memory_gb:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
    print()

    print(f"MPS (Apple Silicon) Available: {info['mps_available']}")
    print("=" * 50)


def get_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get memory information for the specified device.

    Args:
        device: Device to get memory info for. If None, uses current device.

    Returns:
        Dictionary with memory information (in GB):
            - allocated: Currently allocated memory
            - reserved: Reserved memory
            - free: Free memory (CUDA only)
            - total: Total memory (CUDA only)

    Example:
        >>> device = torch.device('cuda')
        >>> mem_info = get_memory_info(device)
        >>> print(f"Allocated: {mem_info['allocated']:.2f} GB")
    """
    if device is None:
        device = get_device("auto")

    memory_info = {}

    if device.type == "cuda":
        # Get CUDA memory statistics
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)

        # Get device properties for total memory
        device_id = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory / (1024**3)

        memory_info = {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved,
        }
    else:
        # For CPU/MPS, we don't have detailed memory info
        memory_info = {
            "allocated": 0.0,
            "reserved": 0.0,
            "total": 0.0,
            "free": 0.0,
        }

    return memory_info


def clear_cuda_cache() -> None:
    """Clear the CUDA cache to free up GPU memory.

    This function releases all unoccupied cached memory from CUDA allocator
    so that it can be used in other GPU applications.

    Example:
        >>> clear_cuda_cache()
        >>> # GPU memory cache is now cleared
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
