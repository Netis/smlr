"""SMLR SGLang token-streaming server — concurrency (continuous batching) + streaming.

Speaks the same SSE `/feed` protocol as the demo's transformers stream server, but is
backed by the SGLang engine, so many concurrent sessions batch together on one card
(the M18 serving win) while each still streams token-by-token:

  decision  -> the policy head on the prompt-end hidden, emitted at prefill latency
  token     -> the `reasoning` lane streamed via engine.async_generate(stream=True)
  done      -> the remaining lanes (observation/public_output/notes/state_patch/actions)

The reasoning stream and the other 5 lanes are issued as separate requests to the shared
engine, so the scheduler batches them with every other session's requests.

  CKPT=$HOME/models/smlr-1b-ml6 SGL_GPU=1 PORT=8141 \
    ~/miniconda3/envs/sglang/bin/python smlr_sgl_stream_server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from smlr_sglang_client import SglangVLAClient, LANES
from runtime.vla_client import _render_chatml, _coerce, MAX_NEW
from data.build_sft_dataset import SFT_SYSTEM

PORT = int(os.environ.get("PORT", "8141"))
CKPT = os.environ.get("CKPT", os.path.expanduser("~/models/smlr-1b-ml6"))
_STREAM_CAP = int(os.environ.get("REASON_CAP", str(MAX_NEW.get("reasoning", 160))))

client = None            # engine is created under __main__ (SGLang spawns worker subprocesses)
_SENT = object()


def stream_frame(system: str, user_obj: dict):
    """Sync generator of {decision|token|done} events; bridges the engine's async loop."""
    prompt = _render_chatml(system or SFT_SYSTEM, user_obj)

    # 1) decision at prefill latency (policy head on the prompt-end hidden)
    na, probs, conf = client._policy(prompt)
    yield {"type": "decision", "next_action": na,
           "policy_probs": {k: round(v, 3) for k, v in probs.items()}}

    # 2) stream reasoning + decode the other lanes concurrently on the shared engine
    q: "queue.Queue" = queue.Queue()

    async def _run():
        try:
            others = [l for l in LANES if l != "reasoning"]
            sps_o = [{"temperature": 0.0, "max_new_tokens": MAX_NEW.get(l, 128),
                      "custom_params": {"lane": l}} for l in others]
            others_task = asyncio.ensure_future(client.engine.async_generate(
                [prompt] * len(others), sps_o,
                custom_logit_processor=[client._clp] * len(others)))

            sp_r = {"temperature": 0.0, "max_new_tokens": _STREAM_CAP,
                    "custom_params": {"lane": "reasoning"}}
            prev = ""
            gen = client.engine.async_generate(
                prompt, sp_r, custom_logit_processor=client._clp, stream=True)
            if asyncio.iscoroutine(gen):
                gen = await gen
            async for out in gen:
                txt = out.get("text", "")
                if len(txt) > len(prev):
                    q.put({"type": "token", "lane": "reasoning", "text": txt[len(prev):]})
                    prev = txt

            others_out = await others_task
            lanes = {"reasoning": _coerce("reasoning", prev)}
            for l, o in zip(others, others_out):
                lanes[l] = _coerce(l, o["text"])
            lanes["next_action"] = na
            q.put({"type": "done", "lanes": lanes})
        except Exception as e:
            q.put({"type": "error", "message": repr(e)})
        finally:
            q.put(_SENT)

    asyncio.run_coroutine_threadsafe(_run(), client._loop)
    while True:
        item = q.get()
        if item is _SENT:
            break
        yield item


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            body = json.dumps({"ok": True, "model": client.model_name,
                               "lanes": client.lanes, "mode": "sglang-token-stream"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()

    def do_POST(self):
        if self.path.split("?")[0] != "/feed":
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        t0 = time.perf_counter()
        try:
            for ev in stream_frame(req.get("system", ""), req.get("user_obj", {})):
                ev["ms"] = round((time.perf_counter() - t0) * 1000)
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


if __name__ == "__main__":
    print(f"[sgl-stream] loading engine {CKPT} ...", flush=True)
    client = SglangVLAClient(CKPT, gpu=os.environ.get("SGL_GPU", "1"))
    print(f"[sgl-stream] ready lanes={client.lanes}", flush=True)
    print(f"[sgl-stream] token-stream SSE on :{PORT} (continuous batching)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
