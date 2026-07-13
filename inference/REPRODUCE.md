# M18 — SGLang port of the SMLR multi-head VLA (research release)

Reproduces the whole result: a custom SGLang model that serves the fused 6-lane SMLR VLA
(MiniCPM5-1B backbone + policy_head + 6 per-lane decode heads) at **detect/alert parity with the
shipped hand-rolled HF path**, with CUDA graphs, ~16× the concurrent-session density per card, and a
live HTTP serving path validated on sustained multi-incident streams.

**The headline finding:** the port initially failed the held-out gate (metrics 7/15) due to a
**RoPE-base misconfiguration** — a one-line serving shim fixes it to full parity (metrics 15/15).

---

## 1. Environment (the reproduction footgun — read this first)

Two conda envs on `gpu-host`:

- **`sglang`** (serving + all SGLang benches). Driver is **CUDA 12.8**, so everything is pinned to
  cu128 — SGLang's default wheels target CUDA 13 and will not load. Exact pins:
  - python 3.12.13, `torch==2.8.0+cu128`, `sglang==0.5.3`
  - `sgl-kernel==0.3.14.post1` (from `https://docs.sglang.ai/whl/cu128`)
  - `torchao==0.12.0`, `flashinfer-python==0.4.0rc3`, `transformers==5.8.1`
  - Runtime env for every SGLang invocation:
    ```bash
    export PATH=~/miniconda3/envs/sglang/bin:$PATH   # puts ninja on PATH (flashinfer JIT)
    export CUDA_HOME=/usr/local/cuda                  # flashinfer JIT needs it
    ```
  - **asyncio loop shim** — the offline `sgl.Engine` on py3.12 needs a set event loop; at the top of
    any offline script: `import asyncio; loop=asyncio.new_event_loop(); asyncio.set_event_loop(loop)`
    and call generation via `loop.run_until_complete(engine.async_generate(...))`. (The server/client
    use a dedicated background loop thread instead — see `smlr_sglang_client.py`.)
  - **WHY:** the box's NVIDIA driver is CUDA 12.8; SGLang 0.5.3's stock wheels assume CUDA 13. The
    cu128 pin set above is the only combination that both loads and runs the custom model + CUDA graphs.

- **`vllm`** (HF reference, PEFT merge / checkpoint build). `~/miniconda3/envs/vllm/bin/python`:
  `transformers==5.9.0`, `peft==0.19.1`, `torch==2.10.0+cu128`. Reads the transformers-5.x
  `rope_parameters` format natively (so HF is the *correct* reference; only SGLang needs the shim).

- GPUs: use a single idle card (`CUDA_VISIBLE_DEVICES=<n>`). Latency/throughput numbers below were
  taken on an otherwise-idle GPU.

---

## 2. The RoPE bug (the key finding, one paragraph)

The checkpoint stores the trained RoPE base (`rope_theta = 5,000,000`) in the **transformers-5.x
nested `rope_parameters` / `rope_scaling` block**. SGLang 0.5.3's `llama.py` reads only the top-level
attribute: `rope_theta = getattr(config, "rope_theta", 10000)` — which is **absent** in this format,
so it silently defaulted to **10000 (a 500× error)**. RoPE position error grows with position, so the
backbone hidden diverged more as the closed-loop prompt accumulated: on identical input_ids,
`cos(SGLang_hidden, HF_hidden)` fell from **0.90 @278 tok → 0.41 @714 tok** (norms 69.6 vs 114.1).
This corrupted the carry lanes over a scenario and collapsed recall (7/15). The **one-line shim**
(read `rope_theta` from `rope_parameters`/`rope_scaling`; null out `rope_type="default"` scaling)
restores fidelity to **cos 0.997–0.9999** and recall to **15/15**.

---

## 3. Apply the two patches to a fresh sglang install

```bash
SGL=~/miniconda3/envs/sglang/lib/python3.12/site-packages/sglang/srt
# (1) RoPE shim — the fix that makes the port a GO
patch $SGL/models/llama.py            < rope_theta_llama.patch
# (2) CUDA-graph lane-id buffer — makes per-request lane routing graph-capturable
patch $SGL/model_executor/cuda_graph_runner.py < cuda_graph_runner.patch
# (3) install the custom model so the registry auto-discovers it (EntryClass scan)
cp smlr_multilane.py $SGL/models/
```
`smlr_multilane.py` registers via the package scan (`EntryClass = SmlrMultiLaneForCausalLM`) — works
in every worker process (avoids the cross-process registration bug #11578). The checkpoint config sets
`architectures=["SmlrMultiLaneForCausalLM"]`.

---

## 4. Build the checkpoint (don't ship the 4 GB weights)

Merged 6-lane checkpoint lives at `/home/user/models/smlr-1b-ml6/` (`model.safetensors` 4.16 GB:
Llama backbone `model.*` + `head_{6 lanes}.weight` + `policy_head.{weight,bias}`; config
`smlr_lanes=[observation,reasoning,public_output,notes,state_patch,actions]`). Rebuild from
base + adapter (verified bit-identical to the adapter's `modules_to_save` heads):

```bash
# vllm env (needs peft). Base=MiniCPM5-1B-SFT + adapter=smlr-1b-minicpm-vla-repro
~/miniconda3/envs/vllm/bin/python smlr_build_ml_ckpt.py \
  # env: KEEP_LANES="observation,reasoning,public_output,notes,state_patch,actions" \
  #      OUT=/home/user/models/smlr-1b-ml6
```

---

## 5. Reproduce each headline result

All SGLang commands assume the env exports from §1 and a free GPU (`CUDA_VISIBLE_DEVICES=3`).
1b reasoner escalation config for every gate/stream run:
`SMLR_POLICY_TAU_WARN=0.2 SMLR_POLICY_TAU_ALERT=0.3 SMLR_POLICY_SUSTAIN_K=1 SMLR_POLICY_SUSTAIN_M=2`.

**A. Full held-out gate (metrics 15/15, logs 7/12, false_alerts 0) — the GO decision**
```bash
# SGLang (rope-fixed):  writes /home/user/heldout_sglang.json
BACKEND=sglang CKPT=/home/user/models/smlr-1b-ml6 <tau env> python smlr_heldout_gate.py
# HF (shipped, vllm env): writes /home/user/heldout_hf.json
BACKEND=hf BASE=/home/user/models/MiniCPM5-1B-SFT ADAPTER=/home/user/models/smlr-1b-minicpm-vla-repro \
  <tau env> ~/miniconda3/envs/vllm/bin/python smlr_heldout_gate.py
# head-to-head compare:
python smlr_heldout_compare.py
```

**B. Per-frame validation vs HF (30 WAIT frames)**
```bash
~/miniconda3/envs/vllm/bin/python smlr_ml6_ref_hf.py     # HF reference -> /home/user/ml6_ref.json
CKPT=/home/user/models/smlr-1b-ml6 python smlr_ml6_validate.py   # policy 28/30, state_patch sim ~0.89
```

**C. RoPE bug diagnosis + fix proof**
```bash
SCEN=held_bufferbloat_01 SGL_CHUNK=32768 python smlr_bug_diag.py   # recall 0.0, max prompt 714 << chunk (rules out chunking)
python smlr_ctrl_sgl.py   # capture SGLang hidden on identical input_ids -> /home/user/ctrl_sgl.json
~/miniconda3/envs/vllm/bin/python smlr_ctrl_hf.py   # cos vs HF: 0.41 pre-fix -> 0.997 post-shim
```

**D. Single-card speed + concurrency scaling (~16× sessions/card @ 2s cadence)**
```bash
python smlr_conc_sgl.py                       # SGLang graphs-on: K∈{1..128}, MAXK@2s(P99)=128
SMLR_VLA_COMPILE=0 ~/miniconda3/envs/vllm/bin/python smlr_conc_hf.py   # hand-rolled: MAXK@2s=8
```

**E. Live HTTP server + sustained streaming**
```bash
PORT=8137 SGL_GPU=3 CKPT=/home/user/models/smlr-1b-ml6 nohup ./start_sgl_server.sh > sgl_server.log 2>&1 &
curl -s http://localhost:8137/health                       # {ok, model, lanes}
URL=http://localhost:8137 <tau env> python smlr_live_remote.py     # single scenario, mttd 4.0, recall 1.0
URL=http://localhost:8137 <tau env> python smlr_stream_concat.py   # 264 frames/9 incidents/557s, recall 1.0, fa 0
```

---

## 6. Results table (number ← script)

| Result | Number | Script(s) |
|---|---|---|
| Held-out gate, metrics (SGLang == HF) | **15/15 pass**, recall 1.0, false_alerts 0, mttd 3.6s | `smlr_heldout_gate.py` + `smlr_heldout_compare.py` |
| Held-out gate, logs | **7/12 pass** (== HF), detect 0.75, rc_correct 0.583, fa 0 | same |
| Per-frame vs HF (30 WAIT) | policy **28/30**, state_patch sim **0.89** | `smlr_ml6_validate.py` / `smlr_ml6_ref_hf.py` |
| RoPE bug (length-dependent divergence) | cos **0.90@278→0.41@714** → **0.997** after shim | `smlr_bug_diag.py` + `smlr_ctrl_{sgl,hf}.py` |
| Single-card 6-lane, graphs on, K=64 | **~1.4 s** vs hand-rolled ~3.7 s (compiled) ≈ **2.6–3×** | `smlr_conc_sgl.py` / `smlr_conc_hf.py` |
| Concurrency, max sessions @ 2s cadence | SGLang **128** vs hand-rolled **8** ≈ **16×**; ~9× tok/s @ K=128 | `smlr_conc_sgl.py` / `smlr_conc_hf.py` |
| Live server, single scenario | recall 1.0, fa 0, **mttd 4.0**, P99 900 ms (incl HTTP) | `smlr_sgl_server.py` + `smlr_live_remote.py` |
| Sustained stream (557 s, 264 frames, 9 incidents) | recall 1.0, **fa 0**, P99 ~1.3–1.5 s **stable early→late**, empty_hidden 0 | `smlr_sgl_server.py` + `smlr_stream_concat.py` |

---

## 7. Artifact inventory

| File | What it is |
|---|---|
| `smlr_multilane.py` | Custom SGLang model: 6-lane graph-safe head routing (`[L,N,V]` stack + device-index gather) + prompt-end hidden capture for the policy read. Auto-registers via `EntryClass`. |
| `rope_theta_llama.patch` | The **fix**: `sglang/srt/models/llama.py` reads `rope_theta` from `rope_parameters`/`rope_scaling`. |
| `cuda_graph_runner.patch` | Adds a `smlr_lane_ids` device buffer to the CUDA-graph runner (capture-time attach + replay-time fill) so per-request lane routing is graph-capturable. |
| `smlr_sglang_client.py` | `SglangVLAClient` — ModelClient-compatible; engine on a background asyncio thread; policy read (bias + full `policy_probs`); radix off + large chunk for reliable hidden capture. |
| `smlr_sgl_server.py`, `start_sgl_server.sh` | FastAPI server on the runtime's `/v1/frame` + `/health` protocol, backed by `SglangVLAClient`; pairs with `runtime.remote_vla_client.RemoteVLAClient`. |
| `smlr_build_ml_ckpt.py` | Builds the 6-lane merged checkpoint from base+adapter (`KEEP_LANES`, `OUT` env). |
| `smlr_heldout_gate.py`, `smlr_heldout_compare.py` | Full held-out detect/alert gate (warm client, both backends) + head-to-head aggregator. |
| `smlr_ml6_validate.py`, `smlr_ml6_ref_hf.py` | 30-frame per-lane + policy validation vs the HF reference. |
| `smlr_bug_diag.py`, `smlr_ctrl_sgl.py`, `smlr_ctrl_hf.py`, `smlr_bug_hf.py` | RoPE bug diagnosis: closed-loop trace + identical-input_ids backbone-fidelity (cos) comparison. |
| `smlr_conc_sgl.py`, `smlr_conc_hf.py` | Concurrency scaling sweep K∈{1,8,16,32,64,128}, both backends. |
| `smlr_live_remote.py`, `smlr_stream_concat.py` | Drive live single / sustained multi-incident streams through the HTTP server. |

---

## 8. Scope & limitations (honest)

- **1B MiniCPM, metrics tier only.** The parity established is detect/alert on the metrics tier
  (recall 1.0, fa 0). The **4B backbone is not ported**, and the **logs-tier root-cause** is only at
  parity (7/12, both backends) — logs `rc` and `dns` are noisy/hard for HF too.
- **WARN/ALERT dampening (open):** deep in a sustained stream, 6/9 incidents reached WARN (detected,
  recall counts it) rather than full ALERT. Bar still holds (recall 1.0, fa 0), but the full-ALERT
  escalation dampens as carried context deepens — unresolved.
- **Concurrency is synchronized-arrival** (all K submitted at t0), not staggered/continuous arrival —
  SGLang's advantage would be larger under real arrival. Hand-rolled was measured **eager** (torch
  .compile warmup was pathologically slow in the harness); compiled is ~2.2× faster (K=64 ≈ 3.7 s) and
  annotated — the ~16× headline holds either way.
- **Policy hidden-capture fragility:** SGLang 0.5.3's FULL hidden capture requires `disable_radix_cache`
  + `chunked_prefill_size ≥ prompt tokens`; fine at one-frame-per-request (the live path), but a known
  0.5.3 sharp edge. `empty_hidden` fallback stayed 0 across the 341-frame live run.
- **Two SGLang-core edits** (the patches) are required; not upstreamed. The RoPE shim is the load-
  bearing one; the graph-runner edit only enables CUDA graphs (correctness holds without it, at ~4×
  slower decode via `--disable-cuda-graph`).
