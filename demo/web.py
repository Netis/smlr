"""Netis SMLR Tech Preview — custom HTML interface with a real server-push stream.

A stdlib HTTP server that serves index.html and a Server-Sent-Events endpoint
`/stream`. A background producer continuously samples this machine's real telemetry,
runs one closed-loop frame on the SGLang server, and PUSHES each frame to every
connected browser — a continuous frame stream, not client polling.

    SMLR_SERVER_URL=http://localhost:8100 python web.py     # open http://localhost:8130

The producer only runs while at least one browser is connected (no GPU work when
nobody is watching). SMLR is a frame-level event reasoner, so the stream is a
continuous flow of frames (~1/s, paced by inference), pushed as they are produced.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from collectors import HostMetrics
from smlr_client import SmlrClient

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("SMLR_WEB_PORT", "8130"))
FLOOR = float(os.environ.get("SMLR_STREAM_FLOOR", "0.4"))   # min seconds between frames

print("[Netis SMLR Tech Preview] connecting to SGLang server...")
_client = SmlrClient()
_metrics = HostMetrics()
_t0 = time.time()
print(f"[Netis SMLR Tech Preview] connected to {_client.url} (model={_client.model_name})")


class Broadcaster:
    """Fan out each produced frame to all connected SSE clients."""

    def __init__(self):
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._subs.discard(q)

    def count(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish(self, item: dict):
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(item)
            except queue.Full:          # slow client: drop this frame for it
                pass


bus = Broadcaster()


def _one_frame() -> dict:
    mev, raw = _metrics.frame(_client.seq + 1, time.time() - _t0)
    pending = _client.take_pending()
    t = time.perf_counter()
    out = _client.step(pending if pending else mev)
    latency_ms = round((time.perf_counter() - t) * 1000)
    return {"metrics": raw, "model_out": out, "model": _client.model_name,
            "latency_ms": latency_ms, "subs": bus.count()}


def _producer():
    """Continuously produce frames while someone is watching."""
    while True:
        if bus.count() == 0:
            time.sleep(0.3)
            continue
        t = time.perf_counter()
        try:
            frame = _one_frame()
        except Exception as e:
            frame = {"error": str(e)}
        bus.publish(frame)
        dt = time.perf_counter() - t
        if dt < FLOOR:
            time.sleep(FLOOR - dt)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = bus.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        item = q.get(timeout=15)
                        payload = f"data: {json.dumps(item)}\n\n".encode()
                    except queue.Empty:
                        payload = b": keepalive\n\n"    # keep the connection warm
                    self.wfile.write(payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                bus.unsubscribe(q)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=_producer, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[Netis SMLR Tech Preview] HTML interface (SSE) on http://localhost:{PORT}")
    srv.serve_forever()
