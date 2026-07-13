"""Test 2 (decisive): replay the SGLang trajectory's EXACT per-frame prompts through the HF backbone +
policy_head, and compare to SGLang's captured hidden/probs. If HF hidden ~= SGLang hidden (cos~1) and
HF mass is ALSO ~0 -> capture is faithful, the miss is upstream compounding drift (the prompts already
lack escalation evidence). If HF mass FIRES on the same prompts -> SGLang's captured hidden is WRONG
(serving bug). Run in vllm env."""
import os, sys, json
import torch

ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "3"))
CKPT = os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6")
TRACE = os.environ.get("TRACE", "/home/user/bugdiag_bb01_c32k.json")

from runtime.vla_client import LABELS
from transformers import AutoTokenizer, AutoConfig, LlamaModel
from safetensors.torch import load_file

dev = "cuda"; dt = torch.bfloat16
cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
sd = load_file(os.path.join(CKPT, "model.safetensors"))
bb = LlamaModel(cfg).to(dev, dt).eval()
bb.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
PW = sd["policy_head.weight"].to(dev, torch.float32); PB = sd["policy_head.bias"].to(dev, torch.float32)
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)

data = json.load(open(TRACE))
tr = data["trace"]
print(f"=== HF replay of {data['scenario']} (SGLang recall={data['recall']}) frames={len(tr)} ===")
print(f"{'f':>3} {'plen':>5} {'cos':>7} {'probsMAE':>9} {'hf_mass':>8} {'sgl_mass':>8}  {'hf_act':>8} {'sgl_act':>8}")
coss, maes, hf_masses = [], [], []
for i, t in enumerate(tr):
    ids = tok(t["prompt"], add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        h = bb(input_ids=ids, use_cache=False).last_hidden_state[0, -1]      # [H] prompt-end
    logit = h.float() @ PW.T + PB
    soft = torch.softmax(logit, -1)
    hf_probs = {LABELS[j]: float(soft[j]) for j in range(len(LABELS))}
    hf_mass = hf_probs["WARN"] + hf_probs["ALERT"]
    hf_act = LABELS[int(logit.argmax())]
    sgl_h = torch.tensor(t["hidden"], dtype=torch.float32, device=dev)
    cos = float(torch.nn.functional.cosine_similarity(h.float(), sgl_h, dim=0))
    mae = sum(abs(hf_probs[k] - t["probs"].get(k, 0.0)) for k in LABELS) / len(LABELS)
    coss.append(cos); maes.append(mae); hf_masses.append(hf_mass)
    fire = " <HF_FIRES>" if hf_mass >= 0.2 else ""
    print(f"{i:>3} {t['plen']:>5} {cos:>7.4f} {mae:>9.5f} {hf_mass:>8.4f} {t['mass']:>8.4f}  "
          f"{hf_act:>8} {t['next_action']:>8}{fire}")
print(f"\nAGG cos(min={min(coss):.4f} mean={sum(coss)/len(coss):.4f}) "
      f"probsMAE(mean={sum(maes)/len(maes):.5f} max={max(maes):.5f})")
print(f"HF max escalation mass over trajectory: {max(hf_masses):.4f}  "
      f"SGL max: {max(t['mass'] for t in tr):.4f}")
print("VERDICT_HINT:",
      "HF ALSO flat (<0.2) -> upstream compounding drift" if max(hf_masses) < 0.2
      else "HF FIRES where SGLang flat -> SGLang capture/serving BUG")
