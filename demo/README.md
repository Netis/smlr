# Netis SMLR Tech Preview — live demo app

Point SMLR at **this machine's real telemetry** and watch it monitor in real time: live
CPU / memory / load, and the system's real-time logs. Two tabs — **Metric Monitoring** and
**Log Monitoring** — each streams frames into `netis-ai/smlr-metrics-1b` and shows the
model's per-frame cognition: the policy action (`WAIT → WARN → ALERT`), its observation,
its reasoning, and how escalation builds as an anomaly sustains.

<p align="center"><i>Netis SMLR Tech Preview — research preview, not a production monitor.
Keep a human in the loop for any real action.</i></p>

## Run

```bash
cd demo
pip install -r requirements.txt
python app.py            # opens a local Gradio UI; first run downloads the model (~4 GB)
```

- **GPU** recommended. Also runs on Apple-Silicon (MPS) and CPU (slower — each frame is a
  prefill + 6 short lane decodes).
- Click **▶ Start** on a tab to begin streaming; **⏸ Stop** to pause. Frames run every ~3 s.
- **To see it escalate:** create real, sustained load — e.g. `yes > /dev/null` (CPU), a
  compile, or a memory hog — and watch the Metric tab move `WAIT → WARN → ALERT`. On a
  healthy machine it correctly stays `WAIT` (not crying wolf is the point).

## How it works

- **`collectors.py`** — real host data. `HostMetrics` samples CPU/mem/load/disk/net via
  `psutil`; `HostLogs` tails the system's real-time logs (`journalctl -f` on Linux,
  `log stream` on macOS, or a file via `SMLR_LOG_FILE=/path`).
- **`smlr_engine.py`** — loads the model (`trust_remote_code`) and runs the closed loop:
  render frame → shared-prefill decode (policy + 6 lanes) → parse lanes → **carry
  `state_patch` forward** into the next frame. A light K-of-M soft-escalation over the
  policy softmax (the shipped mechanism) turns sustained anomaly into a stable WARN/ALERT.
- **`app.py`** — the Gradio UI (two tabs, "Netis SMLR Tech Preview" throughout).

## Honest note on domain

`smlr-metrics-1b` was trained on a specific synthetic domain — **network-link** telemetry
(`latency_ms / loss_pct / jitter_ms / throughput_mbps`) and specific host-log incident
types. Real host CPU/memory and general syslog are **out-of-distribution**. The demo
*projects* host utilization into the trained schema (high utilization reads as
"degradation"), which lets the discriminate-then-alert machinery engage on genuine load —
but detections here are a **preview**, not calibrated production monitoring. Full
methodology, results, and limitations: [`../TECHNICAL_REPORT.md`](../TECHNICAL_REPORT.md).

## Config

| Env | Default | Meaning |
|---|---|---|
| `SMLR_REPO_ID` | `netis-ai/smlr-metrics-1b` | model to load |
| `SMLR_LOG_FILE` | *(auto)* | tail this file instead of journalctl/`log stream` |

Cadence and soft-escalation thresholds are constants at the top of `app.py` / `smlr_engine.py`.
