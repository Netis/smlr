# Netis SMLR Tech Preview — live demo app

Point SMLR at **this machine's real telemetry** and watch it monitor in real time: live
CPU / memory / load, and the system's real-time logs. Two tabs — **Metric Monitoring** and
**Log Monitoring** — each streams frames to the **SMLR SGLang inference server on the GPU
host** and shows the model's per-frame cognition: the policy action (`WAIT → WARN → ALERT`),
its observation, its reasoning, and how escalation builds as an anomaly sustains.

Inference runs on the GPU host (SGLang, continuous batching). The demo app itself is a **thin
client** — no model, no torch, no GPU on the client side.

<p align="center"><i>Netis SMLR Tech Preview — research preview, not a production monitor.
Keep a human in the loop for any real action.</i></p>

## 1. Start the inference server (GPU host)

Set up the SGLang serving port (see [`../inference/REPRODUCE.md`](../inference/REPRODUCE.md) —
apply the two patches, build the merged checkpoint), then launch the server:

```bash
# on the GPU host
PORT=8100 SGL_GPU=<idle-gpu> CKPT=$HOME/models/smlr-1b-ml6 ./start_sgl_server.sh
curl -s http://localhost:8100/health          # {"ok": true, "model": ..., "lanes": [...]}
```

## 2. Reach the server from where the demo runs

If the demo runs on a different machine (e.g. your laptop), tunnel the port over SSH:

```bash
ssh -N -L 8100:localhost:8100 <gpu-host>       # leave running in a second terminal
```

## 3. Run the demo (client)

```bash
cd demo
pip install -r requirements.txt                # gradio + httpx + psutil only
SMLR_SERVER_URL=http://localhost:8100 python app.py
```

- Click **▶ Start** on a tab to begin streaming; **⏸ Stop** to pause. One frame every ~3 s.
- **To see it escalate:** create real, sustained load — e.g. `yes > /dev/null` (CPU), a
  compile, or a memory hog — and watch the Metric tab move `WAIT → WARN → ALERT`. On a
  healthy machine it correctly stays `WAIT` (not crying wolf is the point).

## How it works

- **`collectors.py`** (runs on the client) — real host data. `HostMetrics` samples
  CPU/mem/load/disk/net via `psutil`; `HostLogs` tails the system's real-time logs
  (`journalctl -f` on Linux, `log stream` on macOS, or a file via `SMLR_LOG_FILE=/path`).
- **`smlr_client.py`** (runs on the client) — the closed loop, no model: render frame →
  `POST /v1/frame` to the SGLang server → **carry `state_patch` forward** into the next frame,
  and synthesize a `tool_result` when the model queries a tool so the query→confirm→alert
  loop completes.
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
| `SMLR_SERVER_URL` | `http://localhost:8100` | SMLR SGLang server (tunnel to the GPU host) |
| `SMLR_LOG_FILE` | *(auto)* | tail this file instead of journalctl/`log stream` |

Cadence is a constant at the top of `app.py`.
