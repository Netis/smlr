# Reproduce — SGLang serving port

Practical steps to reproduce the SGLang port results. The *why* (design, the RoPE finding, the numbers
and their caveats) is in [`SGLANG_PORT.md`](SGLANG_PORT.md); this file is just the checklist.

## 1. Environment

SGLang's stock wheels target CUDA 13; on a CUDA-12.8 driver everything is pinned to cu128 (the only
combination that loads the custom model + CUDA graphs):

```
python 3.12 · torch==2.8.0+cu128 · sglang==0.5.3 · sgl-kernel==0.3.14.post1 (from https://docs.sglang.ai/whl/cu128)
torchao==0.12.0 · flashinfer-python==0.4.0rc3 · transformers==5.8.1
export PATH=<sglang-env>/bin:$PATH      # ninja on PATH (flashinfer JIT)
export CUDA_HOME=/usr/local/cuda        # flashinfer JIT
```
A second env (`transformers==5.9.0`, `peft==0.19.1`) is used for the HF reference / checkpoint build.
Offline `sgl.Engine` on py3.12 needs a set event loop — see the top of `smlr_sglang_client.py`.

## 2. Apply the two patches + install the model

```bash
SGL=<sglang-env>/lib/python3.12/site-packages/sglang/srt
patch $SGL/models/llama.py                     < rope_theta_llama.patch   # the load-bearing RoPE fix
patch $SGL/model_executor/cuda_graph_runner.py < cuda_graph_runner.patch  # graph-capturable lane routing
cp smlr_multilane.py $SGL/models/                                          # auto-registers via EntryClass
```

## 3. Build the checkpoint (weights are not shipped)

```bash
# base (MiniCPM5-1B-SFT) + adapter -> merged 6-lane checkpoint
KEEP_LANES="observation,reasoning,public_output,notes,state_patch,actions" \
OUT=$HOME/models/smlr-1b-ml6 python smlr_build_ml_ckpt.py
```

## 4. Reproduce the headline results

All SGLang commands assume §1's env exports, a free GPU, and the escalation config
`SMLR_POLICY_TAU_WARN=0.2 SMLR_POLICY_TAU_ALERT=0.3 SMLR_POLICY_SUSTAIN_K=1 SMLR_POLICY_SUSTAIN_M=2`.

```bash
# A. held-out gate (SGLang == HF: metrics 15/15, logs 7/12, false_alerts 0)
BACKEND=sglang CKPT=$HOME/models/smlr-1b-ml6 python smlr_heldout_gate.py
BACKEND=hf     <hf-env>/python           smlr_heldout_gate.py
python smlr_heldout_compare.py

# B. RoPE bug diagnosis + fix proof (cos 0.90@278 -> 0.41@714 -> 0.997 after shim)
python smlr_bug_diag.py && python smlr_ctrl_sgl.py && <hf-env>/python smlr_ctrl_hf.py

# C. single-card speed + concurrency (SGLang ~128 vs hand-rolled ~8 sessions @ 2s)
python smlr_conc_sgl.py ;  SMLR_VLA_COMPILE=0 <hf-env>/python smlr_conc_hf.py

# D. live HTTP server + sustained stream
PORT=8137 SGL_GPU=3 CKPT=$HOME/models/smlr-1b-ml6 ./start_sgl_server.sh &
URL=http://localhost:8137 python smlr_stream_concat.py
```

## Files

`smlr_multilane.py` (custom model) · `rope_theta_llama.patch` + `cuda_graph_runner.patch` (the two
edits) · `smlr_sgl_server.py` / `smlr_sglang_client.py` / `start_sgl_server.sh` (serving) ·
`smlr_build_ml_ckpt.py` (checkpoint) · `smlr_heldout_*`, `smlr_conc_*`, `smlr_bug_*`, `smlr_ctrl_*`,
`smlr_ml6_*`, `smlr_live_remote.py`, `smlr_stream_concat.py` (eval / bench / diag). The eval and bench
harnesses import the SMLR training repo via `$SMLR_REPO` and are provided as methodology.
