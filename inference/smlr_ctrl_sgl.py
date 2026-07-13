"""Controlled backbone-fidelity test (SGLang side). Feed IDENTICAL input_ids (tokenized once) to the
SGLang engine with return_hidden_states, dump {input_ids, sgl_last_hidden} for several trace prompts of
varying length. The HF side (smlr_ctrl_hf.py) replays the SAME input_ids -> isolates backbone divergence
from tokenization. Run in sglang env."""
import os, sys, json, asyncio, threading
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "3"))
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
CKPT = os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6")
TRACE = os.environ.get("TRACE", "/home/user/bugdiag_bb01_c32k.json")
OUT = os.environ.get("OUT", "/home/user/ctrl_sgl.json")
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class IdentityLP(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        return logits


def main():
    import sglang as sgl
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    tr = json.load(open(TRACE))["trace"]
    idxs = [0, 8, 14, 19, 22, 31]                    # short -> long -> short
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    eng = sgl.Engine(model_path=CKPT, dtype="bfloat16", tp_size=1, mem_fraction_static=0.6,
                     attention_backend="triton", sampling_backend="pytorch",
                     enable_custom_logit_processor=True, enable_return_hidden_states=True,
                     disable_radix_cache=True, chunked_prefill_size=131072, log_level="warning")
    clp = IdentityLP.to_str()
    out = []
    for i in idxs:
        ids = tok(tr[i]["prompt"], add_special_tokens=True)["input_ids"]
        o = loop.run_until_complete(eng.async_generate(
            input_ids=[ids], sampling_params=[{"temperature": 0.0, "max_new_tokens": 1,
                                               "custom_params": {"lane": "observation"}}],
            custom_logit_processor=[clp], return_hidden_states=True))[0]
        hs = o["meta_info"].get("hidden_states")
        arr = torch.tensor(hs[0], dtype=torch.float32).view(-1, 1536)
        out.append({"frame": i, "n_ids": len(ids), "hrows": arr.shape[0],
                    "input_ids": ids, "sgl_hidden": arr[-1].tolist()})
        print(f"frame {i}: n_ids={len(ids)} hrows={arr.shape[0]}", flush=True)
    json.dump(out, open(OUT, "w"))
    print("WROTE", OUT, flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
