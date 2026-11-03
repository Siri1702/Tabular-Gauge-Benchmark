"""
Memory and timing profiling utilities for benchmark runs.
"""

import os
import time
import psutil
import numpy as np
from typing import Optional
from contextlib import contextmanager


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.
    Uses psutil for cross-platform compatibility.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_cpu_percent() -> float:
    """
    Get current CPU usage percentage for this process.
    """
    process = psutil.Process(os.getpid())
    return process.cpu_percent(interval=0.1)


def get_gpu_memory_mb() -> Optional[float]:
    """
    Get GPU memory usage in MB if CUDA is available.
    Returns None if no GPU is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return None


class MemoryTimer:
    """
    Context manager for tracking memory and time usage.

    Usage:
        with MemoryTimer() as mt:
            # your code here
            results = model.fit(X, y)

        print(f"Peak memory: {mt.peak_memory_mb:.1f} MB")
        print(f"Time elapsed: {mt.elapsed_sec:.2f} seconds")
    """

    def __init__(self):
        self.start_memory_mb = None
        self.peak_memory_mb = None
        self.start_time = None
        self.elapsed_sec = None
        self.cpu_percent = None

    def __enter__(self):
        self.start_memory_mb = get_memory_usage_mb()
        self.peak_memory_mb = self.start_memory_mb
        self.start_time = time.perf_counter()

        # Start tracking memory in background
        self._tracking = True
        return self

    def __exit__(self, *args):
        self.elapsed_sec = time.perf_counter() - self.start_time
        current_mem = get_memory_usage_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current_mem)
        self.cpu_percent = get_cpu_percent()

    def update_peak(self):
        """Update peak memory usage."""
        current_mem = get_memory_usage_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current_mem)


class ProfilingResult:
    """Store profiling results for a single run."""

    def __init__(self):
        self.start_memory_mb: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.memory_delta_mb: float = 0.0
        self.fit_time_sec: float = 0.0
        self.predict_time_sec: float = 0.0
        self.cpu_percent: float = 0.0
        self.gpu_memory_mb: Optional[float] = None


def profile_model_fit(model, X_train, y_train, X_test=None, y_test=None):
    """
    Profile a model fit operation.

    Returns a ProfilingResult with timing and memory metrics.
    """
    result = ProfilingResult()

    # Record start state
    result.start_memory_mb = get_memory_usage_mb()

    # Time fit
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    result.fit_time_sec = time.perf_counter() - t0

    # Update memory after fit
    result.peak_memory_mb = get_memory_usage_mb()
    result.memory_delta_mb = result.peak_memory_mb - result.start_memory_mb

    # CPU during fit (approximate)
    result.cpu_percent = get_cpu_percent()

    # GPU memory if available
    result.gpu_memory_mb = get_gpu_memory_mb()

    return result


def profile_model_predict(model, X_test):
    """
    Profile a model predict operation.

    Returns timing metrics.
    """
    t0 = time.perf_counter()
    y_proba = model.predict_proba(X_test)
    predict_time = time.perf_counter() - t0

    return y_proba, predict_time


def get_system_info():
    """
    Get system information for reproducibility.
    """
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
        "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
    }

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_device_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        else:
            info["gpu_available"] = False
    except ImportError:
        info["gpu_available"] = False

    return info