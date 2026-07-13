"""Netis SMLR Tech Preview — token-streaming inference server (runs on the GPU host).

Unlike the frame server (which returns a whole frame at once), this serves each frame
as a token stream: the policy DECISION is emitted the instant the prefill finishes
(the policy head reads the prompt-end hidden before any lane token), then the
`reasoning` lane is streamed token-by-token (ASR-style partial -> final), and a final
`done` event carries the remaining lanes (public_output / state_patch / actions) for the
closed loop.

    MODEL_DIR=~/smlr_hf_test GPU=1 python stream_server.py     # SSE on :8140

In-distribution: the prompt is still the per-frame render the model was trained on; only
the OUTPUT is streamed. (True input-incremental — persistent KV + delta feed + a
StreamingLLM sliding window — needs a streaming-format retrain and is out of scope here.)
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "1"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.expanduser(os.environ.get("MODEL_DIR", "netis-ai/smlr-metrics-1b"))
PORT = int(os.environ.get("PORT", "8140"))
CAPS = {"observation": 56, "reasoning": 96, "public_output": 64,
        "notes": 64, "state_patch": 96, "actions": 64}

print(f"[stream-server] loading {MODEL_DIR} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
LABELS = ["WAIT", "NOTE", "SUMMARY", "QUESTION", "VERIFY",
          "QUERY_TOOL", "WARN", "ALERT", "RESOLVE", "REVISE"]
_EOS = set(model.config.eos_token_id if isinstance(model.config.eos_token_id, (list, tuple))
           else [model.config.eos_token_id])
print(f"[stream-server] ready lanes={model.lanes} on {next(model.parameters()).device}", flush=True)


@torch.no_grad()
def stream_frame(prompt: str, stream_lane: str = "reasoning"):
    """Yield {type:decision}, then {type:token} for the streamed lane, then {type:done}."""
    lanes = model.lanes
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    B = len(lanes)
    out = model.model(input_ids=ids.repeat(B, 1), use_cache=True)
    last_h = out.last_hidden_state[:, -1, :]
    past = out.past_key_values

    plog = model.policy_head(last_h[:1])[0]
    probs = torch.softmax(plog.float(), -1)
    yield {"type": "decision", "next_action": LABELS[int(plog.argmax())],
           "policy_probs": {LABELS[i]: round(float(probs[i]), 3) for i in range(len(LABELS))}}

    si = lanes.index(stream_lane)
    heads = [model.head_for(l) for l in lanes]
    caps = CAPS
    gens = {l: [] for l in lanes}
    done = [False] * B
    cur = torch.empty(B, 1, dtype=torch.long, device=last_h.device)
    prev_text = ""

    def emit_stream():
        nonlocal prev_text
        txt = tok.decode(gens[stream_lane], skip_special_tokens=True)
        if len(txt) > len(prev_text):
            piece = txt[len(prev_text):]
            prev_text = txt
            return {"type": "token", "lane": stream_lane, "text": piece}
        return None

    for i, lane in enumerate(lanes):
        t = int(heads[i](last_h[i:i + 1]).argmax(-1))
        cur[i, 0] = t
        if t in _EOS:
            done[i] = True
        else:
            gens[lane].append(t)
    ev = emit_stream()
    if ev:
        yield ev

    for _ in range(max(caps.values())):
        if all(done):
            break
        o = model.model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        h = o.last_hidden_state[:, -1, :]
        for i, lane in enumerate(lanes):
            if done[i] or len(gens[lane]) >= caps.get(lane, 96):
                done[i] = True
                cur[i, 0] = next(iter(_EOS))
                continue
            t = int(heads[i](h[i:i + 1]).argmax(-1))
            cur[i, 0] = t
            if t in _EOS:
                done[i] = True
            else:
                gens[lane].append(t)
        ev = emit_stream()
        if ev:
            yield ev

    def _coerce(lane, s):
        s = tok.decode(gens[lane], skip_special_tokens=True).strip()
        try:
            return json.loads(s)
        except Exception:
            return s
    yield {"type": "done", "lanes": {l: _coerce(l, gens[l]) for l in lanes}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            body = json.dumps({"ok": True, "model": os.path.basename(MODEL_DIR),
                               "lanes": model.lanes, "mode": "token-stream"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()

    def do_POST(self):
        if self.path.split("?")[0] != "/feed":
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = (f"<|im_start|>system\n{req.get('system','')}<|im_end|>\n"
                  f"<|im_start|>user\n{json.dumps(req.get('user_obj', {}))}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        t0 = time.perf_counter()
        try:
            for ev in stream_frame(prompt, req.get("stream_lane", "reasoning")):
                ev["ms"] = round((time.perf_counter() - t0) * 1000)
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[stream-server] token-stream SSE on :{PORT}", flush=True)
    srv.serve_forever()
