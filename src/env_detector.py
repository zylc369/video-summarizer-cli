"""
Hardware environment detection module.

Detects OS, GPU capabilities (CUDA/MPS), CPU count, and total RAM
to determine the runtime environment for pipeline configuration.
"""

import logging
import platform

import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def detect_environment() -> str:
    """
    Detect the runtime environment based on OS and GPU availability.

    Returns:
        "mac" for macOS or any system without CUDA,
        "windows_gpu" for Windows/Linux with CUDA available.
    """
    system = platform.system()

    if system == "Darwin":
        logger.info("Detected macOS environment")
        return "mac"

    if not TORCH_AVAILABLE:
        logger.info("No CUDA GPU detected, falling back to CPU mode (mac profile)")
        return "mac"

    import torch as _torch

    if _torch.cuda.is_available():
        logger.info("Detected Windows/Linux environment with CUDA GPU")
        return "windows_gpu"

    logger.info("No CUDA GPU detected, falling back to CPU mode (mac profile)")
    return "mac"


def detect_gpu_info() -> tuple[str | None, int | None]:
    """
    Detect GPU name and VRAM size.

    Returns:
        Tuple of (gpu_name, gpu_vram_mb), or (None, None) if
        torch is not available or CUDA is not available.
    """
    if not TORCH_AVAILABLE:
        return (None, None)

    import torch as _torch

    if not _torch.cuda.is_available():
        return (None, None)

    gpu_name = _torch.cuda.get_device_name(0)
    vram_bytes = _torch.cuda.get_device_properties(0).total_mem
    vram_mb = int(vram_bytes / (1024 * 1024))

    logger.info("Detected GPU: %s with %d MB VRAM", gpu_name, vram_mb)
    return (gpu_name, vram_mb)


def detect_mps_available() -> bool:
    """
    Check if Apple Metal Performance Shaders (MPS) is available.

    Returns:
        True if MPS is available, False if torch is not installed
        or MPS is not supported.
    """
    if not TORCH_AVAILABLE:
        return False

    import torch as _torch

    available = _torch.backends.mps.is_available()
    if available:
        logger.info("Apple MPS is available")
    else:
        logger.info("Apple MPS is not available")
    return available


def detect_system_info() -> tuple[str, str, bool, bool, str | None, int | None, int, float]:
    """
    Perform full system detection and return all environment information.

    Returns:
        Tuple containing:
        - env: "mac" or "windows_gpu"
        - os_name: lowercase OS name ("darwin" or "windows" or "linux")
        - has_cuda: whether CUDA GPU is available
        - has_mps: whether Apple MPS is available
        - gpu_name: GPU device name or None
        - gpu_vram_mb: GPU VRAM in MB or None
        - cpu_count: number of logical CPU cores
        - total_ram_gb: total system RAM in GB
    """
    env = detect_environment()
    os_name = platform.system().lower()

    has_cuda = False
    if TORCH_AVAILABLE:
        import torch as _torch

        has_cuda = _torch.cuda.is_available()

    has_mps = detect_mps_available()
    gpu_name, gpu_vram_mb = detect_gpu_info()
    cpu_count = psutil.cpu_count(logical=True) or 1
    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    logger.info(
        "System info: env=%s, os=%s, cuda=%s, mps=%s, gpu=%s, vram=%s MB, "
        "cpu_count=%d, ram=%.1f GB",
        env,
        os_name,
        has_cuda,
        has_mps,
        gpu_name,
        gpu_vram_mb,
        cpu_count,
        total_ram_gb,
    )

    return (env, os_name, has_cuda, has_mps, gpu_name, gpu_vram_mb, cpu_count, total_ram_gb)
