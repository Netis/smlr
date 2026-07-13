"""Netis SMLR Tech Preview — custom HTML interface with a real token stream.

Serves index.html and an SSE endpoint `/stream`. A background producer continuously
samples this machine's real telemetry and, per frame, opens a token stream to the
GPU-host stream server (`/feed`): the policy DECISION arrives the instant prefill
finishes, then `reasoning` streams token-by-token (ASR partial -> final), then a `done`
carries the remaining lanes. The producer relays those to every connected browser and
does the closed-loop bookkeeping (carry state_patch, synthesize tool_result on query).

    SMLR_STREAM_URL=http://localhost:8140 python web.py     # open http://localhost:8130

Inference runs on the GPU host; nothing runs locally. SMLR is a frame-level reasoner, so
this is a continuous stream of frames with token-level output streaming within each frame.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from collectors import HostMetrics

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("SMLR_WEB_PORT", "8130"))
STREAM_URL = os.environ.get("SMLR_STREAM_URL", "http://localhost:8140").rstrip("/")
FLOOR = float(os.environ.get("SMLR_STREAM_FLOOR", "0.3"))

with open(os.path.join(HERE, "system_prompt.md")) as f:
    SYSTEM = f.read().strip()
_http = httpx.Client(timeout=120.0)
_metrics = HostMetrics()
_t0 = time.time()

# closed-loop session state
_working = {"link_state": {}}
_recent = collections.deque(maxlen=6)
_pending = None
_seq = 0


class Broadcaster:
    def __init__(self):
        self._subs: set[queue.Queue] = set(); self._lock = threading.Lock()
    def subscribe(self):
        q = queue.Queue(maxsize=64)
        with self._lock: self._subs.add(q)
        return q
    def unsubscribe(self, q):
        with self._lock: self._subs.discard(q)
    def count(self):
        with self._lock: return len(self._subs)
    def publish(self, item):
        with self._lock: subs = list(self._subs)
        for q in subs:
            try: q.put_nowait(item)
            except queue.Full: pass


bus = Broadcaster()


def _compact(ev):
    e = {k: ev.get(k) for k in ("seq", "t", "type") if k in ev}
    for k in ("links", "result"):
        if k in ev: e[k] = ev[k]
    if "lines" in ev: e["lines"] = ev["lines"][-6:]
    return e


def _as_dict(x):
    if isinstance(x, dict): return x
    try: return json.loads(x)
    except Exception: return x


def _run_frame():
    """One frame: stream tokens from the GPU host, relay to browsers, close the loop."""
    global _pending, _seq
    _seq += 1
    mev, raw = _metrics.frame(_seq, time.time() - _t0)
    event = _pending if _pending else mev
    _pending = None
    event.setdefault("seq", _seq)
    user_obj = {"recent_window": list(_recent), "working_state": _working,
                "retrieved_memory": [], "new_event": event}
    body = {"system": SYSTEM, "user_obj": user_obj, "stream_lane": "reasoning"}

    lanes_final = {}
    with _http.stream("POST", STREAM_URL + "/feed", json=body) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "decision":
                bus.publish({"type": "frame_start", "seq": _seq, "metrics": raw,
                             "event_type": event.get("type"), "next_action": ev["next_action"]})
            elif ev["type"] == "token":
                bus.publish({"type": "token", "seq": _seq, "text": ev["text"]})
            elif ev["type"] == "done":
                lanes_final = ev.get("lanes", {})
                break                       # stop reading; close the upstream stream

    # closed-loop bookkeeping on the final lanes
    pub = _as_dict(lanes_final.get("public_output", {})) or {}
    pub_mode = pub.get("mode", "SILENT") if isinstance(pub, dict) else "SILENT"
    pub_text = pub.get("text", "") if isinstance(pub, dict) else str(pub)
    acts = lanes_final.get("actions", [])
    acts = acts if isinstance(acts, list) else []
    tool_calls = [a for a in acts if isinstance(a, dict)]
    sp = _as_dict(lanes_final.get("state_patch", {}))
    if isinstance(sp, dict):
        for k, v in sp.items():
            if k == "link_state" and isinstance(v, dict):
                _working.setdefault("link_state", {}).update(v)
            else:
                _working[k] = v
    _recent.append(_compact(event))
    if tool_calls and event.get("type") != "tool_result":
        _pending = {"type": "tool_result", "call_id": tool_calls[0].get("call_id", "t1"),
                    "result": {"tool": tool_calls[0].get("tool", "probe"),
                               "finding": "sustained resource saturation confirmed on the flagged target",
                               "confirmed": True}}
    MODE_OK = {"NOTE", "SUMMARY", "QUESTION", "WARN", "ALERT", "RESOLVE"}
    status = pub_mode if (pub_mode in MODE_OK and pub_text) else ("QUERY_TOOL" if tool_calls else lanes_final.get("next_action", "WAIT"))
    obs = lanes_final.get("observation", "")
    bus.publish({"type": "frame_done", "seq": _seq, "status": status,
                 "public_mode": pub_mode, "public_text": pub_text,
                 "observation": obs, "tool_calls": tool_calls,
                 "event_type": event.get("type")})


def _producer():
    while True:
        if bus.count() == 0:
            time.sleep(0.3); continue
        t = time.perf_counter()
        try:
            _run_frame()
        except Exception as e:
            bus.publish({"type": "error", "message": str(e)})
            time.sleep(1.0)
        dt = time.perf_counter() - t
        if dt < FLOOR:
            time.sleep(FLOOR - dt)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = bus.subscribe()
            try:
                self.wfile.write(b": connected\n\n"); self.wfile.flush()
                while True:
                    try:
                        item = q.get(timeout=15)
                        payload = f"data: {json.dumps(item)}\n\n".encode()
                    except queue.Empty:
                        payload = b": keepalive\n\n"
                    self.wfile.write(payload); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                bus.unsubscribe(q)
        else:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()


if __name__ == "__main__":
    print(f"[Netis SMLR Tech Preview] stream server {STREAM_URL} | HTML (SSE) on http://localhost:{PORT}")
    threading.Thread(target=_producer, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
