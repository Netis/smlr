"""Controlled backbone-fidelity test (HF side). Replay the SAME input_ids captured by smlr_ctrl_sgl.py
through the HF backbone; report cosine(sgl_last_hidden, hf_last_hidden) + norms. High cosine (~0.999)
=> tokenization artifact earlier, backbone faithful. Low cosine => genuine backbone divergence (bug).
Run in vllm env."""
import os, sys, json
import torch
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "3"))
CKPT = os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6")
CTRL = os.environ.get("CTRL", "/home/user/ctrl_sgl.json")
from transformers import AutoConfig, LlamaModel
from safetensors.torch import load_file

dev = "cuda"; dt = torch.bfloat16
cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
sd = load_file(os.path.join(CKPT, "model.safetensors"))
bb = LlamaModel(cfg).to(dev, dt).eval()
bb.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)

rows = json.load(open(CTRL))
print(f"{'frame':>5} {'n_ids':>5} {'cos':>8} {'sgl_norm':>9} {'hf_norm':>9} {'rel_l2':>8}")
for r in rows:
    ids = torch.tensor([r["input_ids"]], device=dev)
    with torch.no_grad():
        h = bb(input_ids=ids, use_cache=False).last_hidden_state[0, -1].float()
    sh = torch.tensor(r["sgl_hidden"], dtype=torch.float32, device=dev)
    cos = float(torch.nn.functional.cosine_similarity(h, sh, dim=0))
    rel = float((h - sh).norm() / h.norm())
    print(f"{r['frame']:>5} {r['n_ids']:>5} {cos:>8.4f} {float(sh.norm()):>9.2f} "
          f"{float(h.norm()):>9.2f} {rel:>8.4f}")
