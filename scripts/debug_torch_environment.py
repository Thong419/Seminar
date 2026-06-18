"""Debug Torch import and DLL loading issues on Windows.

Usage:
    python scripts/debug_torch_environment.py
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Torch DLL loading issues.")
    parser.add_argument(
        "--preimport",
        action="append",
        default=[],
        help=(
            "Module to import before probing torch (can be passed multiple times or as comma-separated values)."
        ),
    )
    return parser.parse_args()


def _normalize_preimport_modules(raw_values: list[str]) -> list[str]:
    modules: list[str] = []
    for raw in raw_values:
        parts = [part.strip() for part in raw.split(",")]
        modules.extend([part for part in parts if part])
    return modules


def _run_preimports(modules: list[str]) -> None:
    _print_header("Pre-Import Chain")
    if not modules:
        print("No pre-import modules requested.")
        return

    for name in modules:
        started = time.perf_counter()
        try:
            importlib.import_module(name)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(f"IMPORT OK: {name} ({elapsed_ms:.2f} ms)")
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(f"IMPORT FAIL: {name} ({elapsed_ms:.2f} ms)")
            print(f"  error: {exc}")
            traceback.print_exc()
            raise


def _print_interpreter_info() -> None:
    _print_header("Interpreter")
    print(f"sys.executable: {sys.executable}")
    print(f"sys.version: {sys.version}")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {Path.cwd()}")


def _print_environment_info() -> list[str]:
    _print_header("Environment")
    path_value = os.environ.get("PATH", "")
    path_entries = path_value.split(os.pathsep) if path_value else []
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '<not set>')}")
    print(f"PATH entries ({len(path_entries)}):")
    for index, entry in enumerate(path_entries, start=1):
        print(f"  [{index:02d}] {entry}")
    return path_entries


def _find_torch_package() -> tuple[Path | None, Path | None]:
    _print_header("Torch Package Discovery")
    spec = importlib.util.find_spec("torch")
    if spec is None:
        print("torch spec: not found")
        return None, None

    origin = Path(spec.origin).resolve() if spec.origin else None
    print(f"torch spec origin: {origin}")

    torch_dir = origin.parent if origin else None
    if torch_dir is None:
        print("torch package directory: unavailable")
        return None, None

    lib_dir = torch_dir / "lib"
    print(f"torch package directory: {torch_dir}")
    print(f"torch lib directory: {lib_dir}")
    print(f"torch lib exists: {lib_dir.exists()}")
    return torch_dir, lib_dir


def _attempt_torch_import() -> bool:
    _print_header("Torch Import")
    try:
        import torch

        print("torch import: success")
        print(f"torch.__version__: {torch.__version__}")
        print(f"torch.__file__: {torch.__file__}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        return True
    except Exception as exc:
        print("torch import: failed")
        print(f"error: {exc}")
        traceback.print_exc()
        return False


def _locate_expected_dlls(lib_dir: Path | None) -> list[Path]:
    _print_header("Torch DLL Discovery")
    dll_names = ["c10.dll", "torch_cpu.dll", "fbgemm.dll"]
    dll_paths: list[Path] = []

    if lib_dir is None:
        print("lib directory unavailable; cannot locate torch DLLs")
        return dll_paths

    for name in dll_names:
        dll_path = lib_dir / name
        exists = dll_path.exists()
        print(f"{name}: {'FOUND' if exists else 'MISSING'} ({dll_path})")
        if exists:
            dll_paths.append(dll_path)

    return dll_paths


def _attempt_dll_loads(lib_dir: Path | None, dll_paths: list[Path]) -> None:
    _print_header("ctypes WinDLL Load Checks")

    if lib_dir is None:
        print("Skipped: no torch lib directory")
        return

    if not dll_paths:
        print("Skipped: no located DLLs")
        return

    dll_dir_handle = None
    if hasattr(os, "add_dll_directory"):
        try:
            dll_dir_handle = os.add_dll_directory(str(lib_dir))
            print(f"Added DLL search directory: {lib_dir}")
        except OSError as exc:
            print(f"Failed to add DLL directory {lib_dir}: {exc}")

    first_failure: tuple[Path, OSError] | None = None
    for dll_path in dll_paths:
        try:
            ctypes.WinDLL(str(dll_path))
            print(f"LOAD OK: {dll_path.name}")
        except OSError as exc:
            print(f"LOAD FAIL: {dll_path.name}")
            print(f"  winerror: {getattr(exc, 'winerror', None)}")
            print(f"  message: {exc}")
            if first_failure is None:
                first_failure = (dll_path, exc)

    if first_failure is not None:
        print("First failing DLL:")
        print(f"  {first_failure[0]}")
        print(f"  {first_failure[1]}")
    else:
        print("All located DLLs loaded successfully with ctypes.WinDLL().")

    if dll_dir_handle is not None:
        dll_dir_handle.close()


def main() -> None:
    args = _parse_args()
    preimport_modules = _normalize_preimport_modules(args.preimport)

    _print_interpreter_info()
    _print_environment_info()
    _run_preimports(preimport_modules)
    _attempt_torch_import()
    _torch_dir, lib_dir = _find_torch_package()
    dll_paths = _locate_expected_dlls(lib_dir)
    _attempt_dll_loads(lib_dir, dll_paths)


if __name__ == "__main__":
    main()
