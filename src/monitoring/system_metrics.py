"""System metrics collector for the hybrid search application."""

import asyncio
import time
import psutil
from threading import Thread

from src.metrics import CPU_PERCENT, MEMORY_PERCENT, PROCESS_MEMORY_MB, FILE_DESCRIPTOR_COUNT


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