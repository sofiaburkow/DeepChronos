import time
import os
import csv
import threading
import psutil
from datetime import datetime, timezone


def _now_ts():
    return datetime.now().astimezone(timezone.utc).isoformat() + "Z"


class SystemMonitor:
    def __init__(self, out_csv, interval=1.0, include_gpu=False, gpu_getter=None):
        """
        Background system/process monitor that writes periodic samples to a CSV.

        Parameters
        ----------
        out_csv: str or Path
            Path to CSV file to write. Header written on first sample.
        interval: float
            Sampling interval in seconds (0.5-2.0 recommended).
        include_gpu: bool
            If True, call gpu_getter() each sample and include returned keys.
        gpu_getter: callable or None
            Function returning dict of GPU metrics, e.g. from torch.cuda.
        """
        self.out_csv = str(out_csv)
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread = None
        self.include_gpu = bool(include_gpu)
        self.gpu_getter = gpu_getter
        self.process = psutil.Process(os.getpid())

    def _sample_once(self):
        ts = _now_ts()
        rss = self.process.memory_info().rss
        rss_mb = rss / (1024**2)
        cpu = self.process.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        row = {
            "ts": ts,
            "rss_bytes": int(rss),
            "rss_mb": float(rss_mb),
            "proc_cpu_percent": float(cpu),
            "system_mem_percent": float(vm.percent),
            "system_mem_available_mb": float(vm.available / (1024**2)),
        }
        if self.include_gpu and self.gpu_getter is not None:
            try:
                gpu_row = self.gpu_getter()
                if isinstance(gpu_row, dict):
                    row.update(gpu_row)
            except Exception as e:
                row.update({"gpu_error": str(e)})
        return row

    def _thread_fn(self):
        first = True
        while not self._stop.wait(self.interval):
            row = self._sample_once()
            if first:
                header = list(row.keys())
                with open(self.out_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerow(row)
                first = False
            else:
                with open(self.out_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writerow(row)

    def start(self):
        # warm-up CPU percent internal counter
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_fn, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
