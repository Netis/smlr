# SMLR: a Streaming Multi-Lane Reasoner for real-time monitoring

**Technical report — research release**

SMLR is a small language model that does not answer questions — it *watches*. Instead of the usual
"complete input → one answer" contract, SMLR runs as a continuous cognitive process over a live event
stream, deciding at every frame whether to stay silent or to escalate, while carrying hypotheses and
state forward across time. This report describes the task, the fused multi-head architecture, the
training recipe, the evaluation methodology, the results per model tier, and — at equal length, because
they were the actual product of the work — the negative results and the lessons that cost the most to
learn.

> **What this release is.** A research artifact: the architecture, the methodology, the measured results
> (including the ones that failed), and a reproducible inference path. It is *not* a turnkey product. Every
> headline number is stated with its sample size and its caveat. Where a claim is unconfirmed, it says so.

---

## 1. Motivation

Most LLM serving assumes a request/response shape: a complete prompt arrives, the model emits one
answer, the state resets. Real-time monitoring breaks that shape. Signals arrive continuously; the
"question" is never complete; the right action at time *t* depends on everything seen before *t*; and
the correct output most of the time is **nothing at all** (don't cry wolf).

SMLR models monitoring as a recurrence over an event stream:

```
S_t = Model(S_{t-1}, E_t, R_t)
```

where `S_t` is the evolving cognitive state, `E_t` the newest event (a metrics frame, a log burst), and
`R_t` a returning tool result. The model's job each frame: update its internal picture, decide an
action, and — only when warranted — speak.

Two properties make this hard for a small model, and are the spine of the whole project:

1. **Discriminate-then-alert.** Every incident type in the benchmark has a deliberately-constructed
   *look-alike* separable only by one discriminative field that must be fetched with a tool. A model that
   pattern-matches surface symptoms will fire on the wrong cause. Correct behavior is: notice → hypothesize
   → *query a tool* → read the discriminative field → *then* escalate.
2. **Closed-loop compounding.** The model's own state estimate feeds the next frame's input. Small
   per-frame errors compound over a trajectory. This is why (as §9 documents) every static or single-frame
   validation in this project was misleading, and only closed-loop evaluation told the truth.

---

## 2. The task

### 2.1 The streams

The benchmark centers on two monitoring modalities, each a recorded, replayable event log:

- **Metrics monitoring.** ~10 network links sampled at ~1 Hz; each frame carries per-link
  `latency_ms / loss_pct / jitter_ms / throughput_mbps`. The model must detect degradation, form an
  incident hypothesis, query a diagnostic tool for the root cause, escalate (WARN/ALERT), and later
  RESOLVE — keeping *concurrent* incidents on different links separate.
- **Logs analysis.** A multi-source host log stream (kernel `dmesg`, inference server, web server, GPU
  fabric manager, NVSwitch/NVLink, `sshd`, `cron`, `mail`). The model must correlate across sources to
  catch a cascading incident early, name the root cause, and prescribe a remedy. The hero case is a real
  multi-GPU-host failure: an HBM ECC error escalating through NVSwitch fabric isolation to a host freeze —
  where the value of the model is the *lead time before freeze*, the window in which draining/failover
  could still prevent the outage.

Two further streams exist to prove the protocol generalizes (a Kubernetes event stream, scored the same
way as logs; and a live-speech/transcript demo exercising note-taking, verification, and
minutes-later recall + revision), but the real-time delivery scope of this work is **monitoring only**.

### 2.2 The event-sourced spine

The runtime is built on a single append-only event log as the source of truth. State is a **pure
reduction** over that log; stream-time (`sim_ts`) is decoupled from wall-time (`wall_ts`), so any
trajectory replays deterministically at any speed. Tool results return as *typed events* referencing a
`call_id` — never as natural-language concatenation — which keeps the discriminate-then-alert loop
auditable. The model's memory commits feed back as future input (self-prompting), tiered into public /
ephemeral / durable.

### 2.3 Cadence and concurrency

The production target (milestone M18) is a **2-second cadence** scaling to **32–64 concurrent sessions,
each kept real-time** (per-frame latency below the cadence). Because steady-state monitoring is
mostly-silent, "64 real-time sessions" reduces in practice to "can 64 mostly-`WAIT` frames finish inside
one cadence window." *The 2 s value is itself provisional* — it is the cadence M17 picked to align the
detection numbers; if the true operational beat is 3–4 s, the single-card path already meets it (§7).

---

## 3. Architecture

SMLR is a **fused multi-head VLA**: one shared language-model backbone, one policy classification head,
and one autoregressive decode head per output *lane*. It is built by subclassing whatever causal-LM class
the base's `model_type` resolves to, then attaching the extra heads.

```
                       ┌─────────────────────────── shared backbone (self.bb) ───────────────────────────┐
   frame JSON  ──►  tokenize (chatml)  ──►  prefill once (batched B = #lanes, shared KV)  ──►  hidden H
                       └───────────────────────────────────────────────────────────────────────────────┘
                                                   │
              ┌────────────────────────────────────┼───────────────────────────────────────────┐
              ▼                                     ▼                                            ▼
      policy_head: Linear(H, 10)          head_observation: Linear(H, V)   ...   head_actions: Linear(H, V)
      (reads prompt-end hidden)           (per-lane AR decode, own head)         (drives the tool loop)
              │                                     │                                            │
        next_action (1 of 10)              lane token stream                             lane token stream
     available at prefill latency
```

### 3.1 Heads

- **Backbone** `self.bb` — the transformer stack of the base model (without its `lm_head`). Resolved by
  `model_type`: MiniCPM5-1B loads as `llama`, Qwen3-4B as `qwen3`. Training-time construction mirrors the
  runtime exactly; a mismatch here fails *silently* (see §9, lesson 8).
- **`policy_head`** — `Linear(hidden, 10)`, bias enabled, **randomly initialized** (a fresh classifier).
  It reads the single prompt-end hidden state and predicts `next_action` as one of 10 labels. Crucially,
  this decision is available at **prefill latency, before any lane token is decoded** — the model knows
  whether it is going to stay silent immediately.
- **Per-lane decode heads** `head_{lane}` — `Linear(hidden, vocab)`, no bias, each **warm-started from
  the base `lm_head`**. Each lane decodes its own autoregressive token stream through its own head.

### 3.2 The policy vocabulary

The 10 labels, in a load-bearing order (argmax index → label):

```
WAIT · NOTE · SUMMARY · QUESTION · VERIFY · QUERY_TOOL · WARN · ALERT · RESOLVE · REVISE
```

`WAIT` = stay silent (the correct default). `NOTE/SUMMARY/QUESTION` = low-severity public speech.
`VERIFY` = a needed signal is missing. `QUERY_TOOL` = fire a diagnostic. `WARN/ALERT` = escalate.
`RESOLVE` = incident over. `REVISE` = contradict an earlier belief (must name its target). Note that
`next_action` is produced by the **policy head**, not by a decode lane.

### 3.3 The lanes

The lane set is read at load time from the adapter's manifest. The canonical set (M7 onward) is **6
lanes**, in order, each with a decode-length cap (`MAX_NEW`) chosen so the slowest lane sets the frame's
parallel wall-clock:

| Lane | Role | cap |
|---|---|---|
| `observation` | terse, literal "what I see now" | 64 |
| `reasoning` | 1–2 sentence internal monologue, always filled | 160 |
| `public_output` | `{mode, text}` the user actually sees (may be SILENT) | 96 |
| `notes` | keep-worthy points with importance | 96 |
| `state_patch` | delta merged into working state; carries active hypotheses | 160 |
| `actions` | tool-call JSON; its decoded calls drive the tool loop | 256 |

(Baseline adapters from earlier milestones omit `actions` — 5 lanes — and simply never call tools.)

### 3.4 The per-frame decode loop

1. Render the frame as chatml with the training system prompt; tokenize.
2. **One shared prefill**, batched to `B = #lanes` identical rows — all lanes share the same prompt and
   KV cache, so lane heads decode in parallel and the wall-clock is `prefill + max_lane`, not
   `Σ_lane`.
3. **Policy read** at the prompt-end hidden: `policy_head(H[-1])` → argmax → `next_action`. The full
   softmax distribution (`policy_probs`) is also carried, and is what the M17 soft-escalation consumes.
4. **Per-lane autoregressive decode**: each lane applies *its own* head to *its own* batch row, feeds the
   token back through the shared backbone with `past_key_values`, and stops on EOS or its cap.
5. Assemble `{next_action, <lane>: parsed_lane_output}` and validate.

### 3.5 The closed loop

Which lane outputs re-enter the next frame's prompt is a subtle and load-bearing detail:

- **`state_patch` → confirmed carry.** The harness reducer applies the patch to the working state
  (`active_hypotheses` merged by id, link state, current topic, working memory); the next frame
  re-serializes this as `working_state` in the input. **The model never mutates state itself** — the
  harness owns state, which keeps replay deterministic.
- **`reasoning` → weak / unconfirmed carry.** `reasoning` is appended to a log and is used in the
  speech-mode end-of-talk synthesis, but we **could not confirm** it re-entering the per-frame monitoring
  prompt in the current input builder. A code comment claims it feeds forward; the input builder does not
  appear to include it. Reported here as an honest discrepancy rather than a feature.
- **`actions` → tool loop.** Decoded action JSON is executed as a mock tool; the result is scheduled as a
  typed event at `sim_ts + latency` and arrives on a later frame — closing the discriminate-then-alert
  loop.

### 3.6 Two tiers

The runtime is base-agnostic; tier selection is a deployment concern, not a code branch. The shipped
system runs two tiers:

- **`smlr-metrics-1b`** — a MiniCPM5-1B (llama-arch) backbone for the metrics stream.
- **`smlr-logs-4b`** — a Qwen3-4B backbone for the logs stream.

The same code loads both (`model_type` dispatch). As §7 and §9 explain, these two tiers did *not* respond
the same way to the training interventions — a central finding of the project.

---

## 4. Training

### 4.1 The pipeline

```
general pretraining (base)  →  [mid-training: domain corpus]  →  task SFT (fused multi-lane)  →  cascade
```

The fused SFT stage is the core; mid-training was added later (M16) to attack a root-cause ceiling; the
cascade is an optional offline oracle.

### 4.2 Base tiers, and the bases that were rejected

Four backbones were evaluated as single-variable swaps on identical data and recipe:

| Base | Arch | Outcome |
|---|---|---|
| **Qwen3-4B** | qwen3 | **Shipped** — logs tier |
| **MiniCPM5-1B-SFT** | llama | **Shipped** — metrics tier (4× smaller) |
| HRM-Text-1B (recursive-reasoning) | hrm_text | **Rejected** — never learned to call tools |
| RWKV-7-1.5B (linear attention) | rwkv7 | **Rejected** as the capstone base — long-range recall too weak |
| nanoai gp1b (self-built, 1.2B, ~30% trained) | custom | **Rejected** — root-cause ceiling not moved |

The rejected bases were not wasted effort — each falsified a specific hypothesis:

- **HRM** learned the *discipline* ("don't alert metrics without tool evidence") but never learned to
  *execute* the tool calls, so it permanently waits in any tool-closed-loop domain — while winning on
  self-describing k8s events that need no tool loop. Cost 5–6× decode latency.
- **RWKV-7** learned the protocol *form* (99%+ valid JSON) but its policy collapsed to always-`WAIT`; the
  decisive negative was a direct long-range-recall probe (recall at ~2048-token distance: 0.28 vs 1.0 for
  a same-size attention model). For a task whose whole point is remembering an incident across a
  trajectory, linear-attention recall is a moat, not a scaffold. Kept as a complementary O(1) streaming
  option, not the base.
- **nanoai gp1b** integrated end-to-end (a non-HF base behind the same client) and, after an
  action-upsampling retrain, *cleanly cured under-alerting* — but its root-cause number would not move.
  Three further base-swaps all lost to the original checkpoint. This is what closed the "swap the base"
  line entirely (§9, lesson 3).

### 4.3 Mid-training (M16): the domain-knowledge stage

**Why.** Milestones M13–M15 established that the root-cause (rc) ceiling was **not** an SFT quantity or
behavior problem — it was the base lacking *domain knowledge*. The held-out set is deliberately
zero-shot: training incidents cover one set of causes; held-out tests *different* look-alike causes the
model has never seen (e.g. `dns_failure`, `thermal_throttle`, `bufferbloat`, `silent_drop`). A 4B model
with broad pretraining diagnoses these zero-shot; a small model with no such knowledge cannot.

**What.** A domain-adaptation continued-pretraining stage inserted *before* SFT, leaving the rest of the
pipeline untouched. The corpus is synthetic: a two-stage generator produces a broad incident taxonomy
(296 types across 16 domains, each with signature + look-alikes + root cause + remedy), then per type
writes narratives in four formats (raw multi-source log stream + RCA, postmortem, runbook, on-call
triage dialogue), each carrying explicit symptom→cause discriminative reasoning. Recipe: mostly
next-token continued pretraining, ~70% domain / ~30% general replay (anti-forgetting), LR below the
pretraining peak, tokenizer unchanged. *(This report describes the corpus construction as methodology;
the generated content is not part of this release.)*

**Zero-shot integrity discipline.** The taxonomy is broad and uniform; held-out types are **never**
special-cased, weighted, or seeded with held-out configs/keywords — so held-out rc remains a genuine
generalization test.

**What it changed — and the finding that reframed the project.** Mid-training moved **detection and
alerting** (logs detect/alert rose to full) but did **not** move **root cause** — the knowledge entered
the representation (probe-confirmed) but the fused policy/lane heads did not decode it into a correct
root-cause statement. This drove a scope re-definition: **real-time delivery is judged by detect+alert
only; root cause moves out of the real-time hot path** to the offline cascade. The rc wall was *bypassed*,
not broken. And critically, the *same corpus and recipe* helped the 4B tier and **hurt** the 1B tier
(§7) — mid-training gain is model-dependent.

### 4.4 The fused multi-lane SFT trainer

- **Joint loss.** `L = Σ_lane token_CE(head_lane, lane_targets) + policy_alpha · CE(policy_logits,
  policy_label)`, normalized by total supervised token count. Dataset is one row per **(frame, lane)**,
  each row also carrying the frame's policy label; because the prompt-end hidden is identical across a
  frame's lane rows, supervising the policy on every row is just a denser estimate of the same signal.
- **Head init.** Lane heads warm-start from `lm_head`; the policy head is fresh. LoRA/QLoRA targets the
  attention/MLP projections; the policy head and all lane heads are `modules_to_save` (trained fully).
- **Architecture dispatch (critical).** The trainer resolves the concrete `*ForCausalLM` base class from
  `model_type`, exactly as the runtime does. An earlier trainer hardcoded the Qwen3 class; training a
  llama-arch base inside it randomly initialized qwen3-only `q_norm`/`k_norm` modules that the runtime
  (loading llama) does not have — producing a train/infer hidden-state mismatch that made **training
  metrics perfect while inference silently collapsed to all-`WAIT`**. This burned two multi-hour runs and
  is the origin of lesson 8.
- **Hyperparameters (fused trainer defaults).** 3 epochs, batch 8, grad-accum 2, LR 2e-4, cosine, warmup
  0.03, max length 2048; LoRA r=32 α=64 dropout 0.05; paged 8-bit AdamW; bf16; save every 1500 steps.
  `policy_alpha` recommended 2.0 (recovers the 10-way policy head and lifts ALERT recall, at a ~2-point
  cost on the notes lane). Scaled SFT data ≈ 26k rows.
- **Known hazards.** (a) A custom `compute_loss` that ignores `num_items_in_batch` silently scales the
  effective LR by grad-accum under recent transformers — flagged, impact on earlier runs unverified
  (they converged regardless). (b) Every new base/trainer must pass a **checkpoint-1500 smoke gate**
  (detect ≥ 1 on a few held-out scenarios, ~27 min) before committing to a full run — because the
  convergence signature (`policy_acc = 1.0`, `lane_loss ≈ 0.1`) is *identical* whether the adapter works
  or has silently collapsed.

### 4.5 SFT data construction (methodology)

Training trajectories are built by **event-sourced replay of recorded teacher logs — no new teacher
calls**. For each recorded teacher decision, the event log is reduced up to its trigger and the exact
context the teacher saw is re-derived; the teacher's output is the target. Ground-truth curation makes
targets *cleaner* than raw teacher output: force restraint on benign frames (teaching "noise must not
alert"), suppress premature RESOLVE, and — if the teacher missed an alert-expected incident — inject a
single ground-truth alert at the earliest detectable frame, but only claim tool confirmation when the
tool result is actually visible (to avoid training tool-result hallucination). Frames are balanced
(plain-`WAIT` frames downsampled; tool/REVISE/recall frames upweighted). *Scenario generation and the
generated content are proprietary and not part of this release; only the process is described.*

---

## 5. Evaluation

Scoring is **offline**, over a recorded event log plus authored ground truth — fair and reproducible,
with baselines being different reducers over the same input log.

### 5.1 Metrics scoring

Per incident (ground-truth `onset/resolve` window, link, type, expected tools):

- **detect / recall** — first WARN-or-ALERT naming the incident's link at/after onset; `recall = tp/(tp+fn)`.
- **mttd** — time from onset to that first detection, averaged over detected incidents.
- **alert** — any full ALERT (not just WARN) for the link in-window; **alert_precision** =
  `tp/(tp+false_alerts)`.
- **false_alerts** — any WARN/ALERT matching no incident window (benign/out-of-window escalations).
- **root_cause_accuracy (rc)** — the cause parsed from the *decisive public statement* must equal the
  incident type; an asserted cause ("caused by / root cause:") is preferred so a ruled-out cause cited as
  evidence doesn't score.
- Plus **resolve_latency**, **tool_recall**, **multi_incident_separation**.

### 5.2 Logs scoring

Single-incident ground truth: **detect/mttd**, **alert**, **lead_time_before_freeze** (the headline for
the fabric-freeze case), **root_cause_correct** (scored only from the public statement + remedy, *not* the
private scratchpad), **remedy_present**, **cross_source_correlation**, and **false_alerts** (any
escalation before onset).

### 5.3 Held-out design and the bar

The held-out sets are deliberately zero-shot: metrics held-out uses `bufferbloat` / `silent_drop`
look-alikes (never in the training corpus); logs held-out uses `dns_failure` / `thermal_throttle` (100%
zero-shot). The real-time pass bar (M17) is **detect recall → 1.0 AND false_alerts = 0**. Root cause is
explicitly outside the real-time bar. Sample sizes are small (n=16 metrics, n=12 logs) — a noise band —
so only directional signals are trusted.

---

## 6. The root-cause "wall" was mostly a broken evaluator

For four milestones an apparent root-cause ceiling of **rc ≈ 0.19** on metrics held-out shaped the
narrative ("small models can't diagnose"). In M12-B it was traced — while building something else — to
**the scorer, not the model**. The `_CAUSE_KEYS` table in the metrics evaluator (a) recognized only 5
legacy cause types and omitted every newer look-alike type, so a *correct* diagnosis scored False; and
(b) mismatched evidence words (a look-alike's evidence phrase hit a generic cause's keyword). The fix —
add the look-alike types and order them before the generic ones, same scoring rule — moved the numbers:

| Model | rc before | rc after |
|---|---|---|
| MiniCPM-1B (metrics held-out) | 0.19 | **0.75** |
| Qwen-4B (metrics held-out) | 0.12 | **0.875** |
| HRM-1B | 0.19 | 0.562 |

Recomputed across earlier runs, rc was **never** actually stuck at 0.19 anywhere; a fixed control at
0.125 confirmed the fix corrected scoring rather than handing out free points. A further nuance fell out:
the base-size advantage on rc is **data-dependent and can flip sign** — at 15.7k rows the 1B *beat* the
4B; the 4B only pulls ahead on larger balanced data. Capacity is not the dominant factor. (The logs and
k8s scorers use per-scenario keyword ground truth and were structurally immune; they were audited and
hardened anyway, with zero score change.)

This is lesson 2, and it is the most important methodological result in the project: **measure the noise
floor and audit the scorer before drawing conclusions about the model.**

---

## 7. Results

### 7.1 Shipped configuration (M17)

| Tier | Base + adapter | Thresholds | Metrics det/alert/fa | Logs det/alert/fa |
|---|---|---|---|---|
| **smlr-metrics-1b** | MiniCPM5-1B + metrics adapter | ON (τ_warn 0.2, τ_alert 0.3, K1/M2) | **1.0 / 1.0 / 0** | 0.75 / 0.75 / 0 |
| **smlr-logs-4b** | Qwen3-4B (mid-trained) + logs adapter | OFF | 1.0 / 1.0 / 0 | 0.25–0.42 / … / 0 |

- The 1B metrics tier is a clean pass: **recall 1.0, alert 1.0, false-alerts 0**.
- Mid-training helped the 4B (metrics alert 12→15/16, false-alerts 2→0) but **regressed the 1B** (logs
  detect 6→0/12) — so the 1B ships on its *original* base and the 4B on the *mid-trained* base. Same
  corpus, opposite sign.
- The Qwen soft-threshold mechanism was **disabled** because it introduced metrics false-alerts on
  held-out that more-conservative settings could not remove — and the `false_alerts = 0` bar is
  non-negotiable. (This exposed a trap: the offline calibration sweep is *blind* to metrics false-alerts,
  because its scoring needs the incident link string in the frame — so threshold candidates must be
  validated on real held-out double-runs, not offline sweeps.)

### 7.2 Escalation policy (soft escalation)

A K-of-M sustain mechanism over the policy-head softmax converts run-to-run argmax jitter on anomalous
frames into a stable WARN/ALERT without retraining: keep a sliding window of the last M frames; if ≥ K
carry `P(ALERT) ≥ τ_alert`, upgrade to ALERT, else if ≥ K carry `P(WARN)+P(ALERT) ≥ τ_warn`, WARN.
Upgrade-only; single-frame spikes don't trip it; benign traffic (escalation mass ≈ 0) can't mint false
alerts. It cannot help "stably blind" frames (e.g. `dns_failure`, where the mass is ≈ 0) — that gap
belongs to retraining.

### 7.3 Real-time latency (the honest gap)

Detection numbers reproduce cleanly, but on the M17 SLA run **both tiers missed the real-time lag bar**
(1B: +70–110 s max lag; 4B: +180–310 s) at 2 s cadence. This is attributed to the test running on a
**shared multi-user GPU** and contradicts an earlier exclusive-GPU result (1B at 3 s cadence → lag ≈ 0).
It is flagged as **needing an exclusive-GPU retest** and does not block the correctness verdict — but it
is unresolved. Note also that cadence is a real recall/latency tradeoff: raising cadence to "fix" lag
(8–20 s) crashed detection (4B metrics detect 16→5/16). You cannot tune them independently.

### 7.4 Serving and concurrency (M18)

The hand-rolled serving path was optimized to a `call_batch` that does prefill-once + compaction +
WAIT-skip (5.4× over naive) and then host-surgery + `torch.compile` (dynamic, no-CUDA-graph) for a
further 2.2× (≈ 19 frames/s single-card) — but still fell short of 2 s single-card. The decisive move was
a **serving-engine port to SGLang** (§8), which on the 1B metrics tier reaches:

- **detect/alert parity with the shipped HF path** — metrics 15/15 recall 1.0, logs 7/12 = HF, false
  alerts 0;
- **~2.9× single-card latency** (K=64 all-`WAIT` ~1.26 s vs 3.7 s) — clearing the 2 s cadence;
- **~16× concurrent-session density** (SGLang ~128 vs hand-rolled ~8 sessions/card within 2 s), because
  continuous batching + compaction decouple frames instead of stalling on the slowest one in a
  synchronized batch;
- a live HTTP service holding steady over a 557 s multi-incident stream.

This result is **scoped to the 1B metrics tier**; the 4B logs tier was never ported. Concurrency was
measured under *synchronized* arrival (a conservative bound for SGLang; the true staggered-load number is
unmeasured).

---

## 8. Inference and serving

The reference implementation runs on HuggingFace transformers (the `call`/`call_batch` path described in
§3). For production-scale serving there is a full **SGLang port**, which is where the most interesting
systems finding of the project lives.

Neither vLLM nor SGLang expects a model with more than one `lm_head`. The port
(`inference/smlr_multilane.py`) makes SMLR fit: a custom model with **per-request lane routing** (each
request decodes through its own head, selected by a per-row parameter) that is **CUDA-graph-safe**
(compute all lane heads as one stack and gather by a device lane-id tensor threaded through the graph
runner). CUDA graphs were *essential*, not a bonus — without them the port loses to the hand-rolled path.

**The bug that mattered — a RoPE-base misconfiguration.** The port first failed its held-out gate
(metrics recall 0.47), and the failure looked like "inherent numeric drift." It wasn't. The checkpoint
stores its trained RoPE base (`rope_theta = 5,000,000`) in the transformers-5.x nested `rope_parameters`
block; the serving engine read only the top-level attribute, which is absent in that format, and silently
defaulted to `10000` — a **500× error**. RoPE position error grows with sequence position, so the
backbone hidden diverged *more as the closed-loop prompt accumulated*: on identical inputs,
`cos(port, reference)` fell from 0.90 at 278 tokens to 0.41 at 714. A one-line shim that reads the base
from `rope_parameters` restores fidelity to 0.997 and recall to 15/15.

Full details, both patches, the checkpoint build, and every reproduction command are in
[`inference/`](inference/) — see [`inference/REPRODUCE.md`](inference/REPRODUCE.md) and the serving-port
deep-dive [`inference/SGLANG_PORT.md`](inference/SGLANG_PORT.md).

---

## 9. Research history and lessons

The value of this project is as much in what failed as in what shipped. The arc, briefly:

- **M8–M9** — isolate the base as a single variable; discover that adding pure-`WAIT` frames is not
  "more data" and regresses alerting.
- **M11** — HRM recursive base: NO-GO (tool-calling collapse).
- **M12-B** — the rc "wall" is ¾ a broken evaluator (§6). The pivot of the whole project.
- **M13–M15** — a self-built base and three swaps, all NO-GO → **the wall is domain knowledge, not the
  checkpoint**; base-swap line closed.
- **M16** — mid-training: moves detect/alert but not root cause → scope re-defined to detect+alert.
- **M17** — detect/alert made real and shipped as two tiers; mid-training helps 4B, hurts 1B; soft
  thresholds; the honest real-time-lag gap.
- **M18** — serving and concurrency: a KV-traffic misdiagnosis corrected by profiling; a speculative-decode
  negative result; the SGLang port and its RoPE finding.

The banked lessons, in priority order:

1. **Validate closed-loop models closed-loop.** Every static/single-frame check in this project was
   misleading; only full closed-loop held-out evaluation revealed the real behavior. Numeric error
   compounds frame-to-frame because state feeds forward.
2. **Audit the scorer and measure the noise floor before concluding anything about the model.** A stale
   keyword table biased four milestones.
3. **Changing the base doesn't fix a domain-knowledge gap.** Low validation loss ≠ good task
   representation; the pretraining domain must match the task.
4. **Don't accept "inherent drift" for a numeric anomaly — check the config first.** A length-dependent
   divergence is a RoPE/precision signature, not kernel noise.
5. **Be adversarial with favorable numbers.** The two biggest errors of the serving arc were over-trusted
   *positives* (a self-deceiving "2.81×" speculative-decode number against a strawman baseline; a "0.88
   framework floor" that was really the RoPE bug); the biggest save came from *distrusting* a convenient
   negative. Never benchmark against a strawman — compare against the shipped path.
6. **Every delivery target needs its own A/B.** Mid-training helped the 4B and hurt the 1B, same recipe.
7. **Train/infer architecture mismatch fails silently** — perfect training metrics, all-`WAIT`
   inference. Dispatch the base class by `model_type`; gate the first training run on a smoke eval.
8. **Offline calibration can be blind to the very failure it must catch** (metrics false-alerts) →
   validate on real held-out double-runs.
9. **Cadence is a real recall/latency tradeoff**, not an independently-tunable knob.
10. **Profile before optimizing.** A whole family of optimizations chased a 2% attention cost; the real
    bottleneck was cache-copy overhead and small-kernel launch gaps.

---

## 10. Limitations and open problems

- **Root cause stays weak (~0.44)** on the harder cascade metric, on both backends. It is deliberately
  outside the real-time bar, but it is not solved.
- **`dns_failure` is a stable blind spot** on both shipped models; only a denser mid-training corpus is a
  known fix. `thermal_throttle` still shows small run-to-run flips.
- **Real-time lag misses the bar on a shared GPU** and needs an exclusive-GPU retest.
- **The 4B logs tier was never ported to the serving engine** — the SGLang GO is scoped to the 1B metrics
  tier only, and the 4B port needs its own gate plus a RoPE-class config check.
- **The `reasoning` lane's forward-carry is unconfirmed** in the current input builder (§3.5).
- **Speculative decoding is a negative result**, kept env-gated off: a ~2.2×-on-paper draft did not
  survive end-to-end integration (~1.2–1.7×, with a P99 regression that is disqualifying for a real-time
  SLA).
- **Productionization debt** on the serving port: fold the RoPE read into the model class, package the
  graph-runner patch, and unpin the pinned CUDA stack.

---

## 11. Reproduction

The serving port and every measurement harness are in [`inference/`](inference/), with a complete
environment and command list in [`inference/REPRODUCE.md`](inference/REPRODUCE.md). Model weights are not
shipped in this repository; the checkpoint is rebuilt from base + adapter by the build script described
there. Eval and bench harnesses reference the SMLR training repository (via `SMLR_REPO`) for scenarios and
scoring and are provided as methodology.

---

*This report states each result with its sample size and caveat. Where something is unconfirmed or
failed, it is labeled as such — that honesty is the point of a research release.*
