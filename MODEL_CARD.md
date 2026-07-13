# Model Card — SMLR (Streaming Multi-Lane Reasoner)

## Overview

SMLR is a fused multi-head VLA for **real-time streaming monitoring**: it processes a continuous event
stream (metrics or logs) frame by frame, decides an action per frame, escalates only after
tool-confirmed discrimination, and carries state forward across the trajectory. See the
[Technical Report](TECHNICAL_REPORT.md) for full detail.

- **Model type:** decoder LM backbone + one policy classification head + 6 per-lane autoregressive decode
  heads (LoRA/QLoRA adapters over a frozen base).
- **Input:** one JSON frame (`recent_window`, `working_state`, `retrieved_memory`, `new_event`), rendered
  as chatml.
- **Output:** one JSON update — `next_action` (policy) + only the lanes that changed
  (`observation · reasoning · public_output · notes · state_patch · actions`).
- **Languages:** English (task I/O is structured JSON; a speech demo exercises cross-lingual recall).

## Tiers

| Tier | Base | Adapter | Escalation thresholds | Target stream |
|---|---|---|---|---|
| `smlr-metrics-1b` | MiniCPM5-1B-SFT (llama arch) | metrics VLA adapter | ON (τ_warn 0.2, τ_alert 0.3, K1/M2) | network metrics |
| `smlr-logs-4b` | Qwen3-4B (mid-trained) | logs VLA adapter | OFF | host logs |

## Intended use

- **In scope:** real-time anomaly detection and alerting on streaming metrics/logs, in the
  discriminate-then-alert regime; research on streaming/closed-loop LLM agents, multi-head serving, and
  small-model monitoring.
- **Out of scope:** a certified production incident-response system; safety-critical automated remediation
  without a human in the loop; root-cause attribution as a sole source of truth (rc is weak — see below).

## Evaluation

Offline scoring over recorded event logs + authored ground truth. Held-out incidents are **deliberately
zero-shot** (look-alike types never in training). Real-time bar: **recall → 1.0 and false-alerts = 0**.

| Metric | `smlr-metrics-1b` | `smlr-logs-4b` |
|---|---|---|
| Metrics detect / alert / false-alerts | **1.0 / 1.0 / 0** | 1.0 / 1.0 / 0 |
| Logs detect / alert / false-alerts | 0.75 / 0.75 / 0 | 0.25–0.42 / … / 0 |
| Root-cause (metrics held-out) | 0.75 | 0.875 |

Sample sizes: n=16 metrics, n=12 logs — a noise band; trust directional signals only.

**Serving (1B metrics tier, SGLang port):** detect/alert parity with the HF reference; ~2.9× single-card
latency (clears a 2 s cadence); ~16× concurrent-session density.

## Limitations and known failure modes

- **Root cause is weak** (~0.44 on the hard cascade metric) and is deliberately outside the real-time bar.
- **`dns_failure` is a stable blind spot** on both tiers; `thermal_throttle` shows small run-to-run flips.
- **Real-time lag missed the SLA on a shared GPU** — pending an exclusive-GPU retest.
- **Mid-training is model-dependent** — it helped the 4B and hurt the 1B; do not assume it transfers.
- **The 4B logs tier is not ported to the serving engine** (SGLang GO is 1B-metrics-only).
- **Speculative decoding is off** (a documented negative result).

## Training

Base pretraining → optional domain-corpus mid-training → fused multi-lane SFT (joint lane + policy loss)
→ optional offline cascade. SFT data is built by event-sourced replay of recorded teacher logs with
ground-truth curation. Full recipe and hyperparameters in the [Technical Report §4](TECHNICAL_REPORT.md#4-training).

## Ethical / responsible-use notes

SMLR emits alerts and remedies that could drive operational action. It is a research artifact with
documented false-negative blind spots (`dns_failure`) and a weak root-cause channel; **keep a human in the
loop** for any real remediation. Its judgments should not be treated as authoritative root-cause
attribution.

## Weights

The **metrics-1B** tier is published as a Tech Preview on the Hub:
[**`netis-ai/smlr-metrics-1b`**](https://huggingface.co/netis-ai/smlr-metrics-1b) (merged, transformers-loadable
via `trust_remote_code`). The logs-4B tier is not published; it can be rebuilt from base + adapter using the
build script in [`inference/REPRODUCE.md`](inference/REPRODUCE.md). Base models (MiniCPM5-1B, Qwen3-4B) are
governed by their own upstream licenses.

## License

Code and documentation in this repository: [MIT](LICENSE). Base-model weights and any published adapters
remain under their respective upstream licenses.
