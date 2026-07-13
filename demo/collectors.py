"""Netis SMLR Tech Preview — real host-data collectors.

Two collectors turn the machine the demo runs on into a live event stream for SMLR:

- HostMetrics   : real CPU / memory / load / disk / net via `psutil`, projected into
                  the network-link metrics schema SMLR was trained on (see mapping note).
- HostLogs      : the system's real-time logs (journalctl / `log stream` / file tail).

Mapping note (honest): SMLR's metrics tier was trained on network-link telemetry
(`latency_ms / loss_pct / jitter_ms / throughput_mbps`). Real host resources have no
such fields, so we *project* utilization into that schema — high utilization reads as
"degradation." This lets the discriminate-then-alert machinery engage on genuine load
while staying quiet on a healthy machine. It is a preview projection, not a claim that
CPU% "is" packet loss.
"""

from __future__ import annotations

import collections
import platform
import shutil
import subprocess
import time
from typing import Optional

import psutil


class HostMetrics:
    """Sample real host resources and project them into SMLR's link-metrics schema."""

    def __init__(self, window: int = 8):
        self._hist = collections.defaultdict(lambda: collections.deque(maxlen=window))
        self._last_net = psutil.net_io_counters()
        self._last_disk = psutil.disk_io_counters()
        self._last_t = time.time()
        psutil.cpu_percent(interval=None)  # prime

    def raw(self) -> dict:
        """Real, human-readable host numbers (shown in the UI as-is)."""
        vm = psutil.virtual_memory()
        try:
            load1 = psutil.getloadavg()[0]
        except (AttributeError, OSError):
            load1 = 0.0
        cores = psutil.cpu_count() or 1
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "mem_pct": vm.percent,
            "mem_used_gb": round(vm.used / 1e9, 2),
            "mem_total_gb": round(vm.total / 1e9, 2),
            "load1": round(load1, 2),
            "load_per_core": round(load1 / cores, 2),
            "swap_pct": psutil.swap_memory().percent,
        }

    def _rates(self) -> dict:
        now = time.time()
        dt = max(1e-3, now - self._last_t)
        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        net_mbps = (net.bytes_sent + net.bytes_recv
                    - self._last_net.bytes_sent - self._last_net.bytes_recv) * 8 / 1e6 / dt
        disk_mbps = 0.0
        if disk and self._last_disk:
            disk_mbps = (disk.read_bytes + disk.write_bytes
                         - self._last_disk.read_bytes - self._last_disk.write_bytes) / 1e6 / dt
        self._last_net, self._last_disk, self._last_t = net, disk, now
        return {"net_mbps": round(net_mbps, 1), "disk_mbps": round(disk_mbps, 1)}

    def frame(self, seq: int, t: float) -> tuple[dict, dict]:
        """Return (model_event, raw) — model_event is the projected metrics frame."""
        raw = self.raw()
        rates = self._rates()
        raw.update(rates)

        def jitter(key, val):
            self._hist[key].append(val)
            h = self._hist[key]
            if len(h) < 2:
                return 0.0
            mean = sum(h) / len(h)
            return round((sum((x - mean) ** 2 for x in h) / len(h)) ** 0.5, 1)

        def project(util, tput):
            # utilization (0-100) -> degradation-shaped link metrics
            return {
                "latency_ms": round(1.0 + util * 3.0, 1),   # 1ms idle .. ~300ms saturated
                "loss_pct": round(max(0.0, util - 60.0) * 2.5, 1),  # starts "dropping" past 60%
                "jitter_ms": 0.0,  # filled below
                "throughput_mbps": round(tput, 1),
            }

        links = {}
        cpu = project(raw["cpu_pct"], max(0.0, (100 - raw["cpu_pct"]) * 10))
        cpu["jitter_ms"] = jitter("cpu", raw["cpu_pct"])
        links["cpu"] = cpu

        mem = project(raw["mem_pct"], max(0.0, (100 - raw["mem_pct"]) * 10))
        mem["jitter_ms"] = jitter("mem", raw["mem_pct"])
        links["mem"] = mem

        load_util = min(100.0, raw["load_per_core"] * 100.0)
        load = project(load_util, 0.0)
        load["jitter_ms"] = jitter("load", load_util)
        links["load"] = load

        event = {"seq": seq, "t": round(t, 1), "type": "metrics_frame", "links": links}
        return event, raw


class HostLogs:
    """Tail the system's real-time logs, cross-platform, best-effort, no sudo assumed."""

    def __init__(self, log_file: Optional[str] = None):
        self._proc = None
        self._buf: "collections.deque[str]" = collections.deque(maxlen=400)
        self._cmd, self._how = self._pick(log_file)
        self._start()

    def _pick(self, log_file):
        if log_file:
            return ["tail", "-n", "0", "-F", log_file], f"tail {log_file}"
        sysname = platform.system()
        if sysname == "Linux":
            if shutil.which("journalctl"):
                return (["journalctl", "-f", "-n", "0", "--no-pager", "-o", "short-iso"],
                        "journalctl -f")
            for p in ("/var/log/syslog", "/var/log/messages"):
                if shutil.which("tail"):
                    return ["tail", "-n", "0", "-F", p], f"tail {p}"
        if sysname == "Darwin" and shutil.which("log"):
            return (["log", "stream", "--style", "syslog", "--level", "default"],
                    "log stream")
        return None, "unavailable"

    def _start(self):
        if not self._cmd:
            return
        try:
            self._proc = subprocess.Popen(
                self._cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
            import threading
            threading.Thread(target=self._pump, daemon=True).start()
        except Exception:
            self._proc = None

    def _pump(self):
        for line in self._proc.stdout:  # type: ignore
            line = line.rstrip("\n")
            if line.strip():
                self._buf.append(line)

    @property
    def source(self) -> str:
        return self._how

    def drain(self, limit: int = 40) -> list[str]:
        """Pop up to `limit` new log lines seen since the last drain."""
        out = []
        while self._buf and len(out) < limit:
            out.append(self._buf.popleft())
        return out

    def frame(self, seq: int, t: float, lines: list[str]) -> dict:
        # crude source tag + severity hint the model can read
        parsed = []
        for ln in lines:
            low = ln.lower()
            sev = ("error" if any(w in low for w in ("error", "fail", "panic", "fatal", "oom"))
                   else "warn" if any(w in low for w in ("warn", "denied", "timeout", "refused"))
                   else "info")
            parsed.append({"severity": sev, "line": ln[:400]})
        return {"seq": seq, "t": round(t, 1), "type": "log_frame",
                "source": self.source, "lines": parsed}
