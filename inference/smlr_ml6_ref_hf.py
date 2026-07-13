"""HF reference for the 6-lane + policy checkpoint. For each WAIT frame: policy (next_action +
full 10-way softmax off the prompt-end hidden, WITH bias) and greedy decode of all 6 lanes through
their heads off ONE shared backbone. Uses the production MAX_NEW caps. Dumps per-frame ModelUpdate
+ policy_probs for the sglang gate comparison. Run in vllm env."""
import os, sys, json
import torch

ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "4"))
CKPT = os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6")
DATA = os.path.join(ROOT, "data/m5_eval_frames.jsonl")
N = int(os.environ.get("N", "30"))
OUT = os.environ.get("OUT", "/home/user/ml6_ref.json")

from data.build_sft_dataset import SFT_SYSTEM
from runtime.vla_client import _render_chatml, MAX_NEW, LABELS
from transformers import AutoTokenizer, AutoConfig, LlamaModel
from safetensors.torch import load_file

def wait_user_objs(limit):
    objs = []
    for line in open(DATA):
        r = json.loads(line.strip()) if line.strip() else None
        if not r or str(r.get("true_next_action")) != "WAIT":
            continue
        um = next((m for m in reversed(r["messages"]) if m.get("role") == "user"), None)
        objs.append(json.loads(um["content"]))
        if len(objs) >= limit:
            break
    return objs

dev = "cuda"; dt = torch.bfloat16
cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
lanes = cfg.smlr_lanes
sd = load_file(os.path.join(CKPT, "model.safetensors"))
bb = LlamaModel(cfg).to(dev, dt).eval()
bb.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
heads = {ln: sd[f"head_{ln}.weight"].to(dev, dt) for ln in lanes}
pw = sd["policy_head.weight"].to(dev, torch.float32); pb = sd["policy_head.bias"].to(dev, torch.float32)
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
eos_ids = set(cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id])

@torch.no_grad()
def frame(ids):
    out = bb(input_ids=ids, use_cache=True)
    h0 = out.last_hidden_state[:, -1]                       # [1,H] prompt-end hidden
    plog = (h0.float() @ pw.T + pb)[0]                      # [10]
    psoft = plog.softmax(-1)
    next_action = LABELS[int(plog.argmax(-1))]
    probs = {LABELS[i]: round(float(psoft[i]), 6) for i in range(len(LABELS))}
    upd = {"next_action": next_action, "_policy_probs": probs,
           "_policy_conf": float(psoft.max())}
    for ln in lanes:
        W = heads[ln]
        o = bb(input_ids=ids, use_cache=True)               # fresh cache per lane
        pkv = o.past_key_values
        nxt = int((o.last_hidden_state[:, -1] @ W.T).argmax(-1))
        gen = []
        for _ in range(MAX_NEW.get(ln, 128)):
            if nxt in eos_ids:
                break
            gen.append(nxt)
            oo = bb(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=pkv, use_cache=True)
            pkv = oo.past_key_values
            nxt = int((oo.last_hidden_state[:, -1] @ W.T).argmax(-1))
        upd[ln] = tok.decode(gen, skip_special_tokens=True)
    return upd

objs = wait_user_objs(N)
ref = []
for i, uo in enumerate(objs):
    ids = tok(_render_chatml(SFT_SYSTEM, uo), add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    u = frame(ids)
    ref.append(u)
    print(f"[{i}] act={u['next_action']:7s} conf={u['_policy_conf']:.3f} "
          f"reason={u['reasoning'][:40]!r} sp={u['state_patch'][:40]!r}", flush=True)
json.dump(ref, open(OUT, "w"), ensure_ascii=False)
print("WROTE", OUT, "n=", len(ref))
