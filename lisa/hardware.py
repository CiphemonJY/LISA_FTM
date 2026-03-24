#!/usr/bin/env python3
"""
Hardware detection for LISA + AutoResearch (Windows/Linux/Mac compatible).

Detects available hardware and recommends optimal settings:
- CPU cores and type
- GPU availability (CUDA, MPS/Metal)
- RAM amount and availability
- Disk space for offloading
- Optimal model size selection

Usage:
    python hardware.py              # Full report
    python -c "from hardware import detect; print(detect())"  # Programmatic
"""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("hardware")


@dataclass
class HardwareInfo:
    """Hardware information container."""
    os_name: str
    os_version: str
    architecture: str
    cpu_brand: str
    cpu_cores: int
    cpu_threads: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    gpu_type: Optional[str]  # "cuda", "mps", "metal", None
    total_disk_gb: float
    available_disk_gb: float
    max_model_size: str
    use_disk_offload: bool
    recommended_layer_groups: int
    estimated_training_speed: str
    recommended_framework: str  # "pytorch", "mlx", "auto"


def get_system_info() -> Tuple[str, str, str]:
    """Get OS information."""
    os_name = platform.system()
    os_version = platform.version()
    architecture = platform.machine()
    return os_name, os_version, architecture


def get_cpu_info() -> Tuple[str, int, int]:
    """Get CPU information cross-platform."""
    cpu_brand = "Unknown"
    cpu_cores = os.cpu_count() or 1
    cpu_threads = cpu_cores

    system = platform.system()

    if system == "Darwin":
        # Try /usr/sbin/sysctl first (not always in PATH), then fallback to psutil
        for cmd in ["/usr/sbin/sysctl", "sysctl"]:
            try:
                brand = subprocess.check_output(
                    [cmd, "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL, timeout=5
                ).decode().strip()
                if brand:
                    cpu_brand = brand
                    break
            except Exception:
                continue
        # Fallback to psutil
        if cpu_brand == "Unknown":
            try:
                import psutil
                cpu_brand = psutil.cpu_freq().current if psutil.cpu_freq() else "Apple Silicon"
                if psutil.cpu_freq():
                    cpu_brand = f"Apple Silicon @ {psutil.cpu_freq().current:.0f}MHz"
                else:
                    cpu_brand = "Apple Silicon"
            except Exception:
                pass

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_brand = line.split(":")[1].strip()
                        break
        except:
            pass

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                cpu_brand = result.stdout.strip()
        except:
            pass

    return cpu_brand, cpu_cores, cpu_threads


def get_memory_info() -> Tuple[float, float]:
    """Get RAM information cross-platform."""
    total_gb = 0.0
    available_gb = 0.0

    system = platform.system()

    if system == "Darwin":
        # Primary: use psutil (most reliable on macOS)
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024 ** 3)
            available_gb = vm.available / (1024 ** 3)
            return total_gb, available_gb
        except Exception as e:
            logger.warning(f"psutil memory detection failed: {e}")

        # Fallback: use sysctl + vm_stat
        try:
            for cmd in ["/usr/sbin/sysctl", "sysctl"]:
                try:
                    total_b = int(subprocess.check_output(
                        [cmd, "-n", "hw.memsize"], stderr=subprocess.DEVNULL, timeout=5
                    ).decode().strip())
                    total_gb = total_b / (1024 ** 3)
                    break
                except Exception:
                    continue

            vm = subprocess.check_output(["vm_stat"], stderr=subprocess.DEVNULL, timeout=5).decode()

            page_size = 4096
            for line in vm.split("\n"):
                if "page size" in line.lower():
                    try:
                        page_size = int(line.split("(")[1].split()[0])
                    except Exception:
                        pass

            free_pages = inactive_pages = 0
            for line in vm.split("\n"):
                if "Pages free" in line:
                    try:
                        free_pages = int(line.split(":")[1].strip().rstrip("."))
                    except Exception:
                        pass
                elif "Pages inactive" in line:
                    try:
                        inactive_pages = int(line.split(":")[1].strip().rstrip("."))
                    except Exception:
                        pass

            available_gb = (free_pages + inactive_pages) * page_size / (1024 ** 3)

        except Exception as e:
            logger.warning(f"Memory detection error (macOS): {e}")

    elif system == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        total_kb = int(line.split()[1])
                        total_gb = total_kb / (1024 ** 2)
                    elif line.startswith("MemAvailable"):
                        available_kb = int(line.split()[1])
                        available_gb = available_kb / (1024 ** 2)
        except Exception as e:
            logger.warning(f"Memory detection error (Linux): {e}")

    elif system == "Windows":
        try:
            result = subprocess.run([
                "powershell", "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB"
            ], capture_output=True, text=True, timeout=10)
            if result.stdout.strip():
                total_gb = float(result.stdout.strip())

            result2 = subprocess.run([
                "powershell", "-Command",
                "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024 / 1GB"
            ], capture_output=True, text=True, timeout=10)
            if result2.stdout.strip():
                available_gb = float(result2.stdout.strip())
        except Exception as e:
            logger.warning(f"Memory detection error (Windows): {e}")

    # Sanity check
    if available_gb > total_gb or total_gb == 0:
        available_gb = total_gb * 0.5 if total_gb > 0 else 0

    return total_gb, available_gb


def get_gpu_info() -> Tuple[bool, Optional[str], Optional[float], Optional[str]]:
    """Get GPU information cross-platform."""
    gpu_available = False
    gpu_name = None
    gpu_memory_gb = None
    gpu_type = None

    system = platform.system()

    # Check for CUDA first (works in any Python env with torch)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_type = "cuda"
            return gpu_available, gpu_name, gpu_memory_gb, gpu_type
    except Exception:
        pass

    # Check for Apple Silicon / MPS (PyTorch on Mac)
    if system == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                gpu_available = True
                gpu_type = "mps"
                # MPS uses unified memory - get from system
                total_ram, _ = get_memory_info()
                gpu_memory_gb = total_ram if total_ram > 0 else None

                # Try to get chip name
                for cmd in ["/usr/sbin/sysctl", "sysctl"]:
                    try:
                        brand = subprocess.check_output(
                            [cmd, "-n", "machdep.cpu.brand_string"],
                            stderr=subprocess.DEVNULL, timeout=5
                        ).decode().strip()
                        if brand and "Apple" in brand:
                            gpu_name = f"{brand} (MPS)"
                            break
                    except Exception:
                        continue
                if not gpu_name:
                    gpu_name = "Apple GPU (MPS)"

                return gpu_available, gpu_name, gpu_memory_gb, gpu_type
        except Exception:
            pass

        # Fallback: check chip name via sysctl
        try:
            for cmd in ["/usr/sbin/sysctl", "sysctl"]:
                try:
                    brand = subprocess.check_output(
                        [cmd, "-n", "machdep.cpu.brand_string"],
                        stderr=subprocess.DEVNULL, timeout=5
                    ).decode().strip()
                    if brand and "Apple" in brand:
                        gpu_available = True
                        gpu_name = f"{brand} (Unified Memory)"
                        gpu_type = "mps"
                        total_ram, _ = get_memory_info()
                        gpu_memory_gb = total_ram if total_ram > 0 else None
                        return gpu_available, gpu_name, gpu_memory_gb, gpu_type
                except Exception:
                    continue
        except Exception:
            pass

    return gpu_available, gpu_name, gpu_memory_gb, gpu_type


def get_disk_info() -> Tuple[float, float]:
    """Get disk space information cross-platform."""
    total_gb = 0.0
    available_gb = 0.0

    system = platform.system()

    try:
        if system == "Darwin":
            # Use df without -g (plain df returns 512-byte blocks)
            result = subprocess.run(
                ["df", str(Path.home())],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        # parts[1]=512-byte blocks total, parts[3]=available
                        total_blocks = float(parts[1])
                        avail_blocks = float(parts[3])
                        total_gb = total_blocks * 512 / (1024 ** 3)
                        available_gb = avail_blocks * 512 / (1024 ** 3)

        elif system == "Linux":
            result = subprocess.run(
                ["df", "--block-size=G", str(Path.home())],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        total_gb = float(parts[1].rstrip("G"))
                        available_gb = float(parts[3].rstrip("G"))

        elif system == "Windows":
            try:
                drive_letter = str(Path.home())[0]
                result = subprocess.run([
                    "powershell", "-Command",
                    f"(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='{drive_letter}:'\").Size / 1GB; "
                    f"(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='{drive_letter}:'\").FreeSpace / 1GB"
                ], capture_output=True, text=True, timeout=10)
                if result.stdout.strip():
                    lines = result.stdout.strip().split()
                    if len(lines) >= 2:
                        total_gb = float(lines[0])
                        available_gb = float(lines[1])
            except Exception as e:
                logger.warning(f"Disk detection error (Windows): {e}")

    except Exception as e:
        logger.warning(f"Disk detection error: {e}")

    return total_gb, available_gb


def recommend_settings(hw: HardwareInfo) -> Tuple[str, bool, int, str, str]:
    """Recommend optimal settings based on hardware."""

    available_gb = hw.available_ram_gb
    framework = "pytorch"  # Default for Windows/Linux

    # Check for MLX (Apple Silicon)
    if hw.os_name == "Darwin" and "Apple" in hw.cpu_brand:
        try:
            import mlx
            framework = "mlx"
        except ImportError:
            framework = "pytorch"  # PyTorch fallback

    # CUDA GPU available
    if hw.gpu_type == "cuda" and hw.gpu_memory_gb:
        available_gb = hw.gpu_memory_gb + (hw.available_ram_gb * 0.3)

    # Model size recommendations (4-bit quantization, approximate)
    if available_gb >= 24:
        max_model = "32B"
    elif available_gb >= 16:
        max_model = "14B"
    elif available_gb >= 10:
        max_model = "7B"
    elif available_gb >= 6:
        max_model = "3B"
    elif available_gb >= 4:
        max_model = "1.5B"
    else:
        max_model = "0.5B"

    # Disk offload for larger models on small RAM
    use_offload = False
    if hw.available_ram_gb >= 4 and hw.available_disk_gb >= 20:
        if available_gb < 16:
            use_offload = True

    # Layer groups
    layer_groups = max(4, int(32 / max(1, hw.available_ram_gb / 4)))

    # Training speed
    if hw.gpu_type in ("cuda", "mps"):
        speed = "fast"
    elif hw.cpu_cores >= 8:
        speed = "medium"
    else:
        speed = "slow"

    return max_model, use_offload, layer_groups, speed, framework


def detect_hardware() -> HardwareInfo:
    """Detect all hardware information."""
    os_name, os_version, architecture = get_system_info()
    cpu_brand, cpu_cores, cpu_threads = get_cpu_info()
    total_ram, available_ram = get_memory_info()
    gpu_available, gpu_name, gpu_memory, gpu_type = get_gpu_info()
    total_disk, available_disk = get_disk_info()

    hw = HardwareInfo(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        cpu_brand=cpu_brand,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
        gpu_type=gpu_type,
        total_disk_gb=total_disk,
        available_disk_gb=available_disk,
        max_model_size="",
        use_disk_offload=False,
        recommended_layer_groups=6,
        estimated_training_speed="",
        recommended_framework="",
    )

    max_model, use_offload, layer_groups, speed, framework = recommend_settings(hw)
    hw.max_model_size = max_model
    hw.use_disk_offload = use_offload
    hw.recommended_layer_groups = layer_groups
    hw.estimated_training_speed = speed
    hw.recommended_framework = framework

    return hw


def print_report(hw: HardwareInfo):
    """Print formatted hardware report."""
    print("="*70)
    print("  HARDWARE DETECTION REPORT")
    print("="*70)
    print()
    print(f"  OS:          {hw.os_name} {hw.os_version[:40]}")
    print(f"  Arch:        {hw.architecture}")
    print()
    print(f"  CPU:         {hw.cpu_brand}")
    print(f"  Cores:       {hw.cpu_cores} cores, {hw.cpu_threads} threads")
    print()
    print(f"  RAM:         {hw.total_ram_gb:.1f} GB total, {hw.available_ram_gb:.1f} GB available")
    print()
    if hw.gpu_available:
        print(f"  GPU:         {hw.gpu_name}")
        print(f"  GPU Memory:  {hw.gpu_memory_gb:.1f} GB" if hw.gpu_memory_gb else "  GPU Memory:  N/A")
        print(f"  GPU Type:    {hw.gpu_type}")
    else:
        print("  GPU:         None (CPU only)")
    print()
    print(f"  Disk:        {hw.available_disk_gb:.0f} GB available")
    print()
    print("-"*70)
    print("  RECOMMENDATIONS")
    print("-"*70)
    print(f"  Framework:   {hw.recommended_framework.upper()}")
    print(f"  Max Model:   {hw.max_model_size}")
    print(f"  Offloading:  {'Yes' if hw.use_disk_offload else 'No'}")
    print(f"  Layer Groups:{hw.recommended_layer_groups}")
    print(f"  Speed:       {hw.estimated_training_speed}")
    print()

    # Suggested commands
    print("-"*70)
    print("  SUGGESTED COMMANDS")
    print("-"*70)

    if hw.recommended_framework == "mlx":
        print("  # Apple Silicon - MLX (fast)")
        print(f"  python -m lisa.train_lisa --model Qwen/Qwen2.5-3B-Instruct --iters 100")
    elif hw.recommended_framework == "pytorch":
        if hw.gpu_type == "cuda":
            print("  # NVIDIA GPU - PyTorch + CUDA")
            print(f"  python -m lisa.train_torch --model microsoft/phi-2 --device cuda --iters 100")
        else:
            print("  # CPU - PyTorch (federated learning ready)")
            print(f"  python run_client.py --client-id my-client --server http://localhost:8000")
            print(f"  python run_server.py --model distilbert/distilgpt2 --rounds 5")

    if hw.use_disk_offload:
        print()
        print("  # Disk offload for larger models")
        print(f"  python -m lisa.offload_torch --model Qwen/Qwen2.5-7B-Instruct --groups {hw.recommended_layer_groups} --max-mem 5")

    print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hardware Detection")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    hw = detect_hardware()

    if args.json:
        import json
        print(json.dumps(asdict(hw), indent=2))
    else:
        print_report(hw)


if __name__ == "__main__":
    main()
