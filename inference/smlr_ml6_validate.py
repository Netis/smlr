"""Validate the SGLang 6-lane + policy serve path vs the HF reference (ml6_ref.json) on N WAIT frames.
Per frame: 6 lane requests (greedy, MAX_NEW caps) + 1 policy request (temp1, 1 token, token_ids_logprob
[0..9]) all sharing the frame prompt. Reports per-lane exact/sim, policy exact-match + prob MAE, and
graphs-on wall. This module also exposes decode_frames() reused by the SglangVLAClient."""
import os, sys, json, difflib, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "4"))
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
import asyncio
loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

CKPT = os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6")
REF = os.environ.get("REF", "/home/user/ml6_ref.json")
N = int(os.environ.get("N", "30"))
NOCG = os.environ.get("NOCG", "0") == "1"

from runtime.vla_client import MAX_NEW, LABELS
from safetensors.torch import load_file
LANES = ["observation", "reasoning", "public_output", "notes", "state_patch", "actions"]

# policy_head loaded once from the checkpoint (applied offline to the returned prompt-end hidden)
_sd = load_file(os.path.join(CKPT, "model.safetensors"))
_PW = _sd["policy_head.weight"].float()      # [10,H]
_PB = _sd["policy_head.bias"].float()         # [10]


def policy_from_hidden(h):
    """h: per-token hidden for the prompt ([L,H] flattened). Take the LAST row = prompt-end hidden,
    apply policy_head offline. Returns (next_action, probs, conf) exactly."""
    hv = torch.tensor(h, dtype=torch.float32).view(-1, _PW.shape[1])[-1]   # [H] prompt-end
    logit = hv @ _PW.T + _PB                   # [10]
    soft = torch.softmax(logit, -1)
    probs = {LABELS[i]: float(soft[i]) for i in range(len(LABELS))}
    na = LABELS[int(logit.argmax())]
    return na, probs, float(soft.max())


class IdentityLP(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        return logits


def wait_user_objs(limit):
    objs = []
    for line in open(os.path.join(ROOT, "data/m5_eval_frames.jsonl")):
        r = json.loads(line.strip()) if line.strip() else None
        if not r or str(r.get("true_next_action")) != "WAIT":
            continue
        um = next((m for m in reversed(r["messages"]) if m.get("role") == "user"), None)
        objs.append(json.loads(um["content"]))
        if len(objs) >= limit:
            break
    return objs


def make_engine():
    import sglang as sgl
    return sgl.Engine(model_path=CKPT, dtype="bfloat16", tp_size=1,
                      mem_fraction_static=float(os.environ.get("MEMFRAC", "0.6")),
                      attention_backend="triton", sampling_backend="pytorch",
                      enable_custom_logit_processor=True, enable_return_hidden_states=True,
                      disable_radix_cache=(os.environ.get("NORADIX", "1") == "1"),
                      chunked_prefill_size=int(os.environ.get("CHUNK", "32768")),
                      disable_cuda_graph=NOCG, cuda_graph_max_bs=int(os.environ.get("CGMAXBS", "256")),
                      log_level="warning")


def decode_frames(engine, prompts):
    """Two SGLang batches sharing the radix prompt cache: (1) the 6 lane decodes, (2) the policy
    reads. Mixing logprob + non-logprob requests in ONE batch trips the sampler, so they are split.
    Returns a list of ModelUpdate dicts."""
    clp = IdentityLP.to_str()
    upds = [{} for _ in prompts]

    # --- batch 1: policy read via prompt-end hidden (1-token probe, on an empty KV pool) ---
    Pp = list(prompts)
    spp = [{"temperature": 0.0, "max_new_tokens": 1, "custom_params": {"lane": "observation"}}
           for _ in prompts]
    clpp = [clp] * len(prompts)
    outp = loop.run_until_complete(engine.async_generate(
        Pp, spp, custom_logit_processor=clpp, return_hidden_states=True))
    for pi, o in enumerate(outp):
        hs = o["meta_info"].get("hidden_states")
        h = hs[0] if hs else None                            # per-token hidden for the prompt
        na, probs, conf = policy_from_hidden(h) if h else (None, None, None)
        upds[pi]["next_action"] = na
        upds[pi]["_policy_probs"] = probs
        upds[pi]["_policy_conf"] = conf

    # --- batch 2: the 6 lane decodes (greedy, no logprob) ---
    P, sps, clps, meta = [], [], [], []
    for pi, pr in enumerate(prompts):
        for ln in LANES:
            P.append(pr); clps.append(clp)
            sps.append({"temperature": 0.0, "max_new_tokens": MAX_NEW.get(ln, 128),
                        "custom_params": {"lane": ln}})
            meta.append((pi, ln))
    outs = loop.run_until_complete(engine.async_generate(P, sps, custom_logit_processor=clps))
    for (pi, ln), o in zip(meta, outs):
        upds[pi][ln] = o["text"]
    return upds


def main():
    from data.build_sft_dataset import SFT_SYSTEM
    from runtime.vla_client import _render_chatml
    import time
    objs = wait_user_objs(N)
    prompts = [_render_chatml(SFT_SYSTEM, uo) for uo in objs]
    engine = make_engine()
    got = decode_frames(engine, prompts)                     # warmup + result
    t0 = time.perf_counter(); got = decode_frames(engine, prompts)
    wall = (time.perf_counter() - t0) * 1000

    ref = json.load(open(REF))
    print(f"\n===== 6-lane + policy validation (cuda_graph={'OFF' if NOCG else 'ON'}) =====")
    for ln in LANES:
        exact = 0; sims = []
        for i in range(N):
            g = got[i].get(ln, ""); h = ref[i][ln]
            exact += (g == h)
            sims.append(difflib.SequenceMatcher(None, g, h).ratio())
        print(f"lane={ln:14s} exact={exact:2d}/{N}  mean_sim={sum(sims)/len(sims):.3f}")
    # policy
    pol_exact = sum(1 for i in range(N) if got[i]["next_action"] == ref[i]["next_action"])
    maes = []
    for i in range(N):
        gp = got[i].get("_policy_probs") or {}; hp = ref[i]["_policy_probs"]
        maes.append(sum(abs(gp.get(k, 0.0) - hp.get(k, 0.0)) for k in LABELS) / len(LABELS))
    print(f"policy next_action exact={pol_exact}/{N}  mean prob MAE={sum(maes)/len(maes):.5f}")
    print("policy disagreements:", [(i, got[i]['next_action'], ref[i]['next_action'])
                                    for i in range(N) if got[i]['next_action'] != ref[i]['next_action']])
    print(f"wall(30 frames x 7 reqs, best-effort)={wall:.0f}ms  per_frame={wall/N:.1f}ms")
    engine.shutdown()


if __name__ == "__main__":
    main()
