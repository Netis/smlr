"""Netis SMLR Tech Preview — custom HTML interface (no Gradio).

A tiny stdlib HTTP server: serves index.html (left = live host metrics + charts,
right = the model's streaming output) and a /tick JSON endpoint that samples this
machine's real telemetry and runs one closed-loop frame on the SGLang server.

    SMLR_SERVER_URL=http://localhost:8100 python web.py     # then open http://localhost:8130

Same backend as app.py: real psutil metrics + smlr_client -> the SGLang inference
server on the GPU host. No model runs locally.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from collectors import HostMetrics
from smlr_client import SmlrClient

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("SMLR_WEB_PORT", "8130"))

print("[Netis SMLR Tech Preview] connecting to SGLang server...")
_client = SmlrClient()
_metrics = HostMetrics()
_t0 = time.time()
_lock = threading.Lock()
print(f"[Netis SMLR Tech Preview] connected to {_client.url} (model={_client.model_name})")


def _one_tick() -> dict:
    with _lock:                       # serialize: the closed loop has mutable state
        mev, raw = _metrics.frame(_client.seq + 1, time.time() - _t0)
        pending = _client.take_pending()
        t = time.perf_counter()
        out = _client.step(pending if pending else mev)
        latency_ms = round((time.perf_counter() - t) * 1000)
    return {"metrics": raw, "model_out": out, "model": _client.model_name,
            "latency_ms": latency_ms}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # quiet
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/tick":
            try:
                body = json.dumps(_one_tick()).encode()
                self._send(200, body, "application/json")
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[Netis SMLR Tech Preview] HTML interface on http://localhost:{PORT}")
    srv.serve_forever()
