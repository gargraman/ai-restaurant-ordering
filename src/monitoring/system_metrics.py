"""System metrics collector for the hybrid search application."""

import asyncio
from datetime import datetime
import time
import psutil
from threading import Thread
from typing import Dict, Any

from src.metrics import CPU_PERCENT, MEMORY_PERCENT, PROCESS_MEMORY_MB, FILE_DESCRIPTOR_COUNT


def collect_system_metrics() -> Dict[str, Any]:
    """Collect system-level metrics like CPU, memory, disk, etc.

    Returns:
        Dictionary containing system metrics
    """
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_percent = psutil.virtual_memory().percent
    memory_total = psutil.virtual_memory().total
    memory_available = psutil.virtual_memory().available
    memory_used = psutil.virtual_memory().used
    
    # Disk usage
    disk_usage = psutil.disk_usage('/')
    
    # Network I/O
    net_io = psutil.net_io_counters()
    
    # Process metrics
    process = psutil.Process()
    process_memory_mb = process.memory_info().rss / 1024 / 1024
    file_descriptor_count = process.num_fds()
    
    # Calculate disk usage percentage
    disk_usage_percent = (disk_usage.used / disk_usage.total) * 100 if disk_usage.total > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "memory_total_bytes": memory_total,
        "memory_available_bytes": memory_available,
        "memory_used_bytes": memory_used,
        "disk_total_bytes": disk_usage.total,
        "disk_used_bytes": disk_usage.used,
        "disk_free_bytes": disk_usage.free,
        "disk_usage_percent": disk_usage_percent,
        "network_bytes_sent": net_io.bytes_sent,
        "network_bytes_recv": net_io.bytes_recv,
        "process_memory_mb": process_memory_mb,
        "file_descriptor_count": file_descriptor_count,
        "timestamp": datetime.now().isoformat()
    }
    
    return metrics


class SystemMetricsCollector:
    """Collects system-level metrics like CPU, memory, etc."""

    def __init__(self):
        self._running = False
        self._thread = None

    def start(self):
        """Start collecting system metrics in a background thread."""
        if not self._running:
            self._running = True
            self._thread = Thread(target=self._collect_metrics, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop collecting system metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _collect_metrics(self):
        """Internal method to collect metrics periodically."""
        while self._running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent

                # Update metrics
                CPU_PERCENT.set(cpu_percent)
                MEMORY_PERCENT.set(memory_percent)

                # Process memory
                process = psutil.Process()
                PROCESS_MEMORY_MB.set(process.memory_info().rss / 1024 / 1024)

                # File descriptor count
                FILE_DESCRIPTOR_COUNT.set(process.num_fds())

                # Sleep for 10 seconds between collections
                time.sleep(10)

            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                time.sleep(10)  # Still sleep to avoid busy loop


# Global system metrics collector instance
system_metrics_collector = SystemMetricsCollector()