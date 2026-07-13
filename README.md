# SMLR — a Streaming Multi-Lane Reasoner

**A small language model that watches instead of answering.** SMLR runs as a continuous cognitive process
over a live event stream — metrics, logs — deciding at every frame whether to stay silent or escalate,
while carrying hypotheses and state forward across time. It is trained to *discriminate before it alerts*:
notice an anomaly, form a hypothesis, query a diagnostic tool, read the decisive field, and only then cry
wolf.

> **Research release.** This repository is the model's architecture, training recipe, evaluation
> methodology, measured results (including the negative ones), and a reproducible inference path. It is not
> a turnkey product. Every headline number is stated with its sample size and its caveat.
>
> 📄 **[Technical Report](TECHNICAL_REPORT.md)** · 🗂 **[Model Card](MODEL_CARD.md)** ·
> 🤗 **[Model on HF](https://huggingface.co/netis-ai/smlr-metrics-1b)** · 🖥️ **[Live demo](demo/)**

## 🛰️ Live demo — Netis SMLR Tech Preview

Point SMLR at **your own machine's real telemetry** and watch it monitor live. A custom web dashboard —
**left:** real CPU / memory / load with rolling charts; **right:** the model's output — streams
frame-by-frame: the **decision appears the instant prefill finishes**, then the **reasoning types out
token-by-token** (ASR-style), then the final `WARN` / `ALERT`. It stays quiet on a healthy box and
escalates on genuine sustained load. (There's also a Gradio tabbed app for Metric + Log monitoring.)

Inference runs on the GPU host (no local model). Start a token-stream server, then run the dashboard:

```bash
# GPU host — SGLang backend (continuous batching → many concurrent sessions on one card)
CKPT=$HOME/models/smlr-1b-ml6 SGL_GPU=<idle> PORT=8141 \
  ~/miniconda3/envs/sglang/bin/python inference/smlr_sgl_stream_server.py
# client (tunnel first if remote:  ssh -N -L 8141:localhost:8141 <gpu-host>)
cd demo && pip install -r requirements.txt && SMLR_STREAM_URL=http://localhost:8141 python web.py
# open http://localhost:8130
```

Two backends (SGLang for concurrency, transformers for a simple single-session setup) — see
[`demo/README.md`](demo/README.md). Concurrency (per-frame latency, one card): SGLang
**K=1 0.7 s → K=8 3.1 s** (batched) vs a non-batching server **K=8 32 s**.

*Tech Preview — the metrics model is trained on a specific network-monitoring domain, so real host signals
are projected into its schema; and it streams a continuous flow of frames (~1/s) with token-level output —
SMLR is a frame-level reasoner, not a continuous-signal model. Detections are a preview, not calibrated
production monitoring.*

---

## The idea

Most LLMs answer a complete question and reset. Monitoring isn't like that: signals never stop, the
"question" is never complete, the right action depends on everything seen so far, and the correct output
*most of the time* is nothing at all. SMLR models this as a recurrence:

```
S_t = Model(S_{t-1}, E_t, R_t)
```

state ← f(previous state, new event, returning tool result). It sits on an event-sourced spine (an
append-only log as the single source of truth, state as a pure reduction, deterministic replay at any
speed), so every trajectory is reproducible.

## What makes it hard (and interesting)

- **Discriminate-then-alert.** Every incident type has a deliberately-built *look-alike* separable only by
  one field you must fetch with a tool. Surface pattern-matching fires on the wrong cause.
- **Closed-loop compounding.** The model's own state estimate feeds the next frame. Small errors compound —
  which is why static and single-frame validation lied throughout this project, and only closed-loop
  evaluation told the truth (see the report's *Lessons*).

## Architecture in one picture

SMLR is a **fused multi-head VLA**: one shared backbone, one policy head, and one decode head per output
*lane*.

```
frame → tokenize → prefill once (shared KV, batched over lanes) → hidden H
                          ├─ policy_head(H[-1]) → next_action  (1 of 10, at prefill latency)
                          └─ head_{lane}(H) → per-lane autoregressive decode (own head each)
```

- **policy head** — `Linear(H, 10)`, predicts the action (`WAIT · NOTE · SUMMARY · QUESTION · VERIFY ·
  QUERY_TOOL · WARN · ALERT · RESOLVE · REVISE`) from the prompt-end hidden, **before any lane is decoded**.
  The model knows immediately whether it will stay silent.
- **6 lanes** — `observation · reasoning · public_output · notes · state_patch · actions`, each decoded in
  parallel through its own head sharing one prefill (wall-clock = `prefill + slowest lane`, not the sum).
- **closed loop** — `state_patch` merges into the working state and re-enters the next frame; `actions`
  drives the tool loop.

Full detail — heads, caps, the decode loop, what actually carries forward — in the
[Technical Report §3](TECHNICAL_REPORT.md#3-architecture).

## Results (headline)

Two shipped tiers, evaluated on **deliberately zero-shot** held-out incidents (look-alike types never seen
in training). The real-time bar is **recall → 1.0 and false-alerts = 0**.

| Tier | Backbone | Metrics detect / alert / false-alerts | Notes |
|---|---|---|---|
| **smlr-metrics-1b** | MiniCPM5-1B | **1.0 / 1.0 / 0** | clean pass; root-cause 0.75 |
| **smlr-logs-4b** | Qwen3-4B | 1.0 / 1.0 / 0 | mid-trained base; logs root-cause 0.875 |

Sample sizes are small (n=16 metrics, n=12 logs) — read directional signals only. Serving the 1B tier on
SGLang reaches **detect/alert parity with the reference path, ~2.9× single-card latency, and ~16×
concurrent-session density** ([report §7–8](TECHNICAL_REPORT.md#7-results)).

## What failed (documented honestly)

- **Root cause** stays weak (~0.44 on the hard cascade metric) — deliberately outside the real-time bar,
  not solved.
- **Speculative decoding** is a negative result (kept off): ~2.2× on paper, ~1.2–1.7× with a P99
  regression in reality.
- **Real-time lag** missed the bar on a shared GPU — needs an exclusive-GPU retest.

The reasoning behind each of these — and the [banked lessons](TECHNICAL_REPORT.md#9-research-history-and-lessons)
— is the real content of the [report](TECHNICAL_REPORT.md).

## Repository layout

| Path | What |
|---|---|
| [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) | The full report: task · architecture · training · evaluation · results · lessons · limits |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Concise model card (tiers, intended use, metrics, limitations) |
| [`demo/`](demo/) | **Netis SMLR Tech Preview** live demo — HTML dashboard (`web.py`) + Gradio app (`app.py`), monitoring your machine's real CPU/mem + logs |
| [`inference/`](inference/) | The SGLang serving port — production inference backends for the model |
| [`inference/smlr_multilane.py`](inference/smlr_multilane.py) | Custom multi-head model: CUDA-graph-safe per-lane routing |
| [`inference/smlr_sgl_stream_server.py`](inference/smlr_sgl_stream_server.py) | Token-streaming server: continuous batching (concurrency) + per-token output |
| [`inference/SGLANG_PORT.md`](inference/SGLANG_PORT.md) | Serving-port deep-dive (incl. the RoPE-base finding) |
| [`inference/REPRODUCE.md`](inference/REPRODUCE.md) | Environment, patches, checkpoint build, every reproduction command |

The eval/bench harnesses under `inference/` reference the (separate) SMLR training repository via
`$SMLR_REPO` for scenarios and scoring; they are included as methodology. Model weights are not shipped —
the checkpoint is rebuilt from base + adapter (see `inference/REPRODUCE.md`).

## License

[MIT](LICENSE).
