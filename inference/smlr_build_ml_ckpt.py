"""Build a 2-lane (reasoning + state_patch) + policy SGLang checkpoint from base+adapter.
Merges LoRA into the backbone, keeps head_reasoning / head_state_patch / policy_head, and writes a
self-contained safetensors dir whose config.architectures = ['SmlrMultiLaneForCausalLM'].
Run in the HF env (~/miniconda3/envs/vllm/bin/python)."""
import os, sys, json, shutil
import torch

ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm"))
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "4"))

BASE = "/home/user/models/MiniCPM5-1B-SFT"
ADAPTER = "/home/user/models/smlr-1b-minicpm-vla-repro"
OUT = os.environ.get("OUT", "/home/user/models/smlr-1b-ml2-reasoning-statepatch")
KEEP_LANES = os.environ.get("KEEP_LANES", "reasoning,state_patch").split(",")

from runtime.vla_client import _model_class
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel

lanes = json.loads(open(os.path.join(ADAPTER, "lanes.json")).read())["lanes"]
print("adapter lanes:", lanes)
Cls = _model_class(lanes, BASE)
m = Cls.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map={"": 0},
                        trust_remote_code=True, attn_implementation="sdpa")
wrapped = PeftModel.from_pretrained(m, ADAPTER).eval()
core = wrapped.merge_and_unload()   # fold LoRA into backbone; modules_to_save (heads) preserved
print("merged. core type:", type(core).__name__)

# Assemble the state dict: backbone (model.*) + the 2 lane heads + policy head.
sd = {}
for k, v in core.model.state_dict().items():
    sd[f"model.{k}"] = v.to(torch.bfloat16).contiguous()
for lane in KEEP_LANES:
    w = getattr(core, f"head_{lane}").weight
    sd[f"head_{lane}.weight"] = w.detach().to(torch.bfloat16).contiguous()
sd["policy_head.weight"] = core.policy_head.weight.detach().to(torch.bfloat16).contiguous()
sd["policy_head.bias"] = core.policy_head.bias.detach().to(torch.bfloat16).contiguous()
print("state dict tensors:", len(sd))
print("head/policy keys:", [k for k in sd if k.startswith(("head_", "policy_"))])

os.makedirs(OUT, exist_ok=True)
from safetensors.torch import save_file
save_file(sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})

# config: reuse the merged single-lane config as a template (correct llama backbone params),
# just swap the architecture and record the lane->head order.
cfg = json.load(open("/home/user/models/smlr-1b-statepatch-merged/config.json"))
cfg["architectures"] = ["SmlrMultiLaneForCausalLM"]
cfg["smlr_lanes"] = KEEP_LANES
cfg["smlr_num_labels"] = 10
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

# tokenizer + templates
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.save_pretrained(OUT)
for f in ("generation_config.json", "chat_template.jinja"):
    src = os.path.join("/home/user/models/smlr-1b-statepatch-merged", f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, f))
print("WROTE", OUT)
print("files:", sorted(os.listdir(OUT)))
