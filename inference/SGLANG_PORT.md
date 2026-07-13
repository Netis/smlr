# Breaking the SMLR real-time wall: speculative decode (negative) → SGLang serving port (GO)

**Goal.** SMLR is a fused multi-head VLA (one shared LM backbone + a `policy_head` + 6 per-lane decoder
heads) that monitors metrics/logs streams in real time. The shipped hand-rolled serving path did all-WAIT
K=64 in ~3.1–3.4s and could not reach the **2s cadence** target, nor scale to the **32/64 concurrent
sessions** mandate. The question: can we break that wall without changing the model?

Two lines were tried. One failed honestly; the other succeeded and is a GO.

---

## Line 1 — Speculative decode: NEGATIVE (banked)

Trained per-lane EAGLE draft heads and integrated greedy speculative decode into `call_batch`
(masked-KV ragged rollback, lossless by construction).

- **Correctness held**: greedy spec == greedy AR within the batch floor; state_patch draft reached
  held-out mean_accept 3.57.
- **But it does not reach 2s.** The initially-reported "2.81×" was **self-deception** — a strawman
  baseline (an un-optimized probe AR lacking the shipped T0 host-surgery) plus a decode-only number
  passed off as end-to-end. The real integrated result was **~1.2–1.7×, unreliable** (slower than
  baseline on 2 of 3 decode runs, high variance → a P99 *regression* for a real-time SLA). Root cause:
  the spec accept loop reintroduces the per-row Python bookkeeping that T0 had removed, and prefill is a
  fixed floor spec can't touch.
- **Disposition**: kept **env-gated OFF** (`SMLR_VLA_SPEC`), documented as a negative result.

**Lesson banked**: don't map a decode-only ratio onto end-to-end, and don't measure against a strawman
baseline — compare against the *shipped* path.

---

## Line 2 — SGLang serving port: GO (correctness parity + 2s + ~16× concurrency)

SMLR's multi-head architecture doesn't drop into a serving engine (both vLLM and SGLang assume one
`lm_head`). We designed and built the port; SGLang was chosen over vLLM (its `logits_processor` takes the
head as an argument; RadixAttention; per-request `CustomLogitProcessor`; vLLM's clean path duplicates the
backbone 6×).

### What was built
- **Custom `SmlrMultiLaneForCausalLM`** (subclass of SGLang's Llama): one shared backbone forward, then
  **per-row lane routing** via `forward_batch.sampling_info.custom_params["lane"]` → each request decodes
  through its own head (grouped matmul per lane). Registered via `EntryClass` package-scan.
- **CUDA-graph-safe routing**: compute all lane heads `[L,N,V]` + `gather` by a device lane-id tensor
  threaded through SGLang's `cuda_graph_runner` replay. Without graphs the port LOSES; with them it wins —
  graphs were essential, not upside.
- **Policy read**: `policy_head` (+bias, with full `policy_probs` for soft-escalation) applied to the
  prompt-end hidden via a hidden-state capture.
- A Reasoner-compatible `SglangVLAClient` (`.call`/`.usage`/`CallMeta`).

### The one bug that mattered — and the process that caught it
The full held-out eval first came back **NO-GO**: SGLang missed 8/15 metrics alerts (recall 0.467) that
the shipped model catches. The failure looked like "inherent numeric drift" and was nearly accepted as
such. Chasing *"does SGLang have a bug?"* instead found the real cause:

> **RoPE base misconfiguration.** transformers 5.x nests the trained `rope_theta=5,000,000` under
> `config.rope_parameters` and drops the top-level attribute; SGLang 0.5.3 reads `getattr(config,
> "rope_theta", 10000)` → silently uses **θ=10000, a 500× error**. RoPE error grows with position, so the
> backbone hidden diverged *more as the closed-loop prompt accumulated* — cos(SGLang,HF) 0.90 @278tok →
> **0.41 @714tok**. A one-line shim (read θ from `rope_parameters`) → **0.997**.

**Verified independently** (config object has `rope_theta` MISSING, `getattr`→10000, real value in
`rope_parameters`) and confirmed on a recovered case (`held_bufferbloat_01`: recall 0→1.0, BAR PASS).

### Results (rope-fixed, full held-out — the same rigorous eval that gave the NO-GO)

| dimension | result |
|---|---|
| **Correctness** | detect/alert **parity with shipped HF**: metrics **15/15** (recall 1.0), logs 7/12 = HF, **false_alerts 0** |
| **Single-card latency** | K=64 all-WAIT **~1.26s vs shipped 3.7s = ~2.9×** — clears the 2s cadence spec couldn't |
| **Concurrency (the mandate)** | max sessions/card within 2s: **SGLang ~128 vs hand-rolled ~8 ≈ 16× fleet density**; ~9× throughput at K=128 |

The concurrency win is architectural: hand-rolled fuses all frames into one **synchronized batch** →
straggler-bound (any long-`state_patch` frame ≈ 3.7s regardless of K); SGLang's continuous batching +
compaction **decouple** frames (P50 855ms ≪ P99 1632ms at K=128).

---

## Lessons banked (both cost real detours)

1. **Validate closed-loop models closed-loop.** Every static/single-frame check looked fine and was
   misleading — the spec floor, the "0.88 framework floor" (itself partly the RoPE bug at short prompts),
   the 30-frame lane validation, the 2-scenario gate. Only the full closed-loop held-out eval revealed
   both the spec regression and the RoPE bug. Treat any non-closed-loop "pass" on this model as provisional.
2. **Don't accept "inherent drift" for a numeric anomaly — check the config first.** A length-dependent
   divergence is a RoPE/precision signature, not kernel noise.
3. **Be adversarial with favorable numbers.** The two biggest errors this arc (spec 2.81×; the "0.88
   floor") were over-trusted positives; the biggest save (RoPE) came from distrusting a convenient
   negative ("inherent drift").

---

## Status & scope (honest)

**Verdict: GO — scoped to the 1B metrics tier.** On that tier the port reaches correctness parity with
the shipped model (full held-out), clears 2s single-card (~2.9×), scales ~16× in concurrency, and works
live as a real HTTP service that holds over a 557s multi-incident stream with no latency drift. As a
**research release, this is sufficient**: the port pattern is proven end-to-end and the results are
reproducible.

What is **NOT** covered (scope boundaries, not unknowns):
- **1B metrics tier only.** The shipped **logs tier is a 4B Qwen3 model** — never ported/validated; it
  needs its own port + gate (and a RoPE-class config check).
- **WARN-vs-ALERT dampening open**: deep in the sustained stream, 6/9 incidents reached WARN (detected)
  rather than full ALERT. Not a bar failure (recall counts detection), but HF wasn't run on the concat to
  attribute it — the one live correctness question left.
- **Concurrency was synchronized arrival**, not staggered; the ~16× is directionally sound and
  conservative-for-SGLang, but the true production-load number is unmeasured.
- Root cause (rc) stays weak (~0.44) — the harder cascade metric, out of the realtime bar, on both backends.

Productionization debt (real, all understood): fold the RoPE read into the model class (not a patched
site-packages file); decide packaging for the custom model + `cuda_graph_runner` patch (fork vs upstream);
**unpin the CUDA-13-vs-driver-12.8 stack** (SGLang 0.5.3 / torch 2.8 pin is forced by gpu-host's CUDA-12.8
driver — a driver upgrade removes most of this debt); the fragile 0.5.3 policy hidden-capture path.

Reproducible artifacts: this repository — see `REPRODUCE.md` for the model, both patches, the checkpoint
build, and every harness that produced the numbers above.
