# Netis SMLR Tech Preview — live demo app

Point SMLR at **this machine's real telemetry** and watch it monitor in real time: live
CPU / memory / load, and the system's real-time logs. Two interfaces:

- **HTML dashboard** (`web.py`) — left: live metrics with rolling charts; right: the model's
  output, **streamed token-by-token** (decision the instant prefill finishes, reasoning types out,
  then the final `WARN` / `ALERT`).
- **Gradio app** (`app.py`) — tabbed **Metric Monitoring** / **Log Monitoring**.

Both are **thin clients**: they read real host telemetry and stream frames to an inference server on
the GPU host. No model, no torch, no GPU on the client side.

<p align="center"><i>Netis SMLR Tech Preview — research preview, not a production monitor.
Keep a human in the loop for any real action.</i></p>

## 1. Start an inference server (GPU host)

Two server flavors — pick the one for the interface you want:

**Token-stream server** (for the HTML dashboard, option A) — streams each frame token-by-token
(the policy DECISION the instant prefill finishes, then `reasoning` token-by-token). Two flavors,
same SSE `/feed` protocol:

- **SGLang** — *recommended for concurrency.* Continuous batching serves many sessions on one card
  (see [`../inference/`](../inference/) to set up the SGLang port + patches):
  ```bash
  CKPT=$HOME/models/smlr-1b-ml6 SGL_GPU=<idle> PORT=8141 \
    ~/miniconda3/envs/sglang/bin/python ../inference/smlr_sgl_stream_server.py
  ```
- **transformers** — simple, no SGLang needed, but **single-session** (no batching):
  ```bash
  MODEL_DIR=netis-ai/smlr-metrics-1b GPU=<idle> PORT=8140 python stream_server.py
  ```

Concurrency (per-frame latency at K concurrent sessions, one card): SGLang **K=1 0.7s → K=8 3.1s**
(batched); transformers **K=1 2.5s → K=8 32s** (serialized). Point `SMLR_STREAM_URL` at whichever.

**SGLang frame server** (for the Gradio app, option B) — returns a whole frame at once:
```bash
PORT=8100 SGL_GPU=<idle> CKPT=$HOME/models/smlr-1b-ml6 ./start_sgl_server.sh
```

## 2. Reach the server from where the demo runs

If the demo runs on another machine (e.g. your laptop), tunnel the port over SSH:

```bash
ssh -N -L 8141:localhost:8141 <gpu-host>       # SGLang token-stream server (dashboard, recommended)
ssh -N -L 8140:localhost:8140 <gpu-host>       # transformers token-stream server (dashboard, simple)
ssh -N -L 8100:localhost:8100 <gpu-host>       # SGLang frame server (Gradio app)
```

## 3. Run the demo (client)

**A. Custom HTML dashboard** (left = live metrics + charts, right = model output) — a real
**token stream over Server-Sent Events**: a background loop continuously samples telemetry and,
per frame, opens a token stream to the GPU host — the decision badge appears immediately, then
`reasoning` types out token-by-token in the browser, then the final status lands. No client
polling; inference only runs while a browser is connected.
```bash
cd demo
pip install -r requirements.txt                # httpx + psutil (+ gradio for option B)
SMLR_STREAM_URL=http://localhost:8141 python web.py     # SGLang backend; open http://localhost:8130
```

> **What "stream" means here (honest).** The *transport* is a real push stream, and *output* is
> token-level (ASR-style partial→final). But SMLR is a **frame-level** event reasoner — it consumes
> discrete frames (a metrics snapshot / a batch of log lines) one at a time, ~1/s, not a continuous
> signal. True input-incremental streaming (persistent KV + delta feed + a StreamingLLM sliding
> window) needs a streaming-format retrain and is out of scope for this preview.

**B. Gradio app** (tabbed: Metric + Log monitoring):
```bash
SMLR_SERVER_URL=http://localhost:8100 python app.py
```

- Click **▶ Start** on a tab to begin streaming; **⏸ Stop** to pause. One frame every ~3 s.
- **To see it escalate:** create real, sustained load — e.g. `yes > /dev/null` (CPU), a
  compile, or a memory hog — and watch the Metric tab move `WAIT → WARN → ALERT`. On a
  healthy machine it correctly stays `WAIT` (not crying wolf is the point).

## How it works

- **`collectors.py`** (client) — real host data. `HostMetrics` samples CPU/mem/load/disk/net via
  `psutil`; `HostLogs` tails the system's real-time logs (`journalctl -f` on Linux, `log stream` on
  macOS, or a file via `SMLR_LOG_FILE=/path`).
- **`web.py`** (client) — the HTML dashboard backend: a background loop samples telemetry and, per
  frame, opens a token stream to the GPU host, relaying `frame_start → token → frame_done` to the
  browser over SSE; runs the closed loop (**carry `state_patch` forward**, synthesize a `tool_result`
  on a tool query so the query→confirm→alert loop completes).
- **`stream_server.py`** / **[`../inference/smlr_sgl_stream_server.py`](../inference/smlr_sgl_stream_server.py)**
  (GPU host) — the token-stream servers (transformers / SGLang): emit the decision at prefill latency,
  stream `reasoning` token-by-token, then a `done` with the remaining lanes.
- **`smlr_client.py`** (client) — the frame-protocol client for the Gradio app (`POST /v1/frame`, whole
  frame at once, carry state forward).
- **`app.py`** — the Gradio UI (two tabs, "Netis SMLR Tech Preview" throughout).

## Honest note on domain

The served metrics tier was trained on a specific synthetic domain — **network-link**
telemetry (`latency_ms / loss_pct / jitter_ms / throughput_mbps`) and specific host-log
incident types. Real host CPU/memory and general syslog are **out-of-distribution**. The demo
*projects* host utilization into the trained schema (high utilization reads as "degradation"),
which lets the discriminate-then-alert machinery engage on genuine load — but detections here
are a **preview**, not calibrated production monitoring. Full methodology, results, and
limitations: [`../TECHNICAL_REPORT.md`](../TECHNICAL_REPORT.md).

## Config

| Env | Default | Meaning |
|---|---|---|
| `SMLR_STREAM_URL` | `http://localhost:8140` | token-stream server for the HTML dashboard (`web.py`) — point at `:8141` for SGLang |
| `SMLR_SERVER_URL` | `http://localhost:8100` | SGLang frame server for the Gradio app (`app.py`) |
| `SMLR_LOG_FILE` | *(auto)* | tail this file instead of journalctl/`log stream` |

Cadence / stream floor are constants at the top of `web.py` and `app.py`.
