"""SglangVLAClient: a ModelClient-compatible client backed by the SGLang 6-lane + policy serve path.
Drop-in for Reasoner.client (.call / .usage / .backend). Per frame: 1 policy probe (prompt-end hidden
-> policy_head offline, exact 10-way softmax + bias) + 6 greedy lane decodes, assembled into a
ModelUpdate dict + CallMeta(policy_conf, policy_probs) for the M17 soft-escalation.

Engine runs on a background asyncio loop/thread so the synchronous .call() works from inside the
Player's own event loop (no nested-loop conflict). Radix cache is disabled so the policy probe's FULL
hidden capture is reliable (a cached prompt prefix returns an empty/mis-sliced hidden in 0.5.3)."""
import os, sys, json, time, threading, asyncio
from types import SimpleNamespace
import torch

ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
from data.build_sft_dataset import SFT_SYSTEM
from runtime.vla_client import _render_chatml, _coerce, MAX_NEW, LABELS
from runtime.model_client import CallMeta, Usage
from safetensors.torch import load_file
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

LANES = ["observation", "reasoning", "public_output", "notes", "state_patch", "actions"]


class IdentityLP(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        return logits


class SglangVLAClient:
    def __init__(self, ckpt, gpu="4"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
        self.ckpt = ckpt
        self.model_name = os.path.basename(ckpt.rstrip("/"))
        self.lanes = LANES
        self.usage = Usage()
        self.backend = SimpleNamespace(model_for=lambda m: self.model_name,
                                       is_local=True, supports_tools=True)
        sd = load_file(os.path.join(ckpt, "model.safetensors"))
        self._PW = sd["policy_head.weight"].float()
        self._PB = sd["policy_head.bias"].float()
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        self._clp = IdentityLP.to_str()
        # background loop/thread for the engine
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        import sglang as sgl
        self.engine = sgl.Engine(
            model_path=ckpt, dtype="bfloat16", tp_size=1,
            mem_fraction_static=float(os.environ.get("MEMFRAC", "0.6")),
            attention_backend="triton", sampling_backend="pytorch",
            enable_custom_logit_processor=True, enable_return_hidden_states=True,
            disable_radix_cache=True,
            chunked_prefill_size=int(os.environ.get("SGL_CHUNK", "32768")),
            disable_cuda_graph=(os.environ.get("NOCG", "0") == "1"),
            cuda_graph_max_bs=int(os.environ.get("CGMAXBS", "16")), log_level="warning")
        self._chunk = int(os.environ.get("SGL_CHUNK", "32768"))
        self.trace = []            # per-frame diagnostics when SMLR_SGL_RECORD is set
        self._last_hidden = None
        self._last_plen = None

    def tok_ids(self, prompt):
        return self._tok(prompt, add_special_tokens=False)["input_ids"]

    def _gen(self, prompts, sps, **kw):
        fut = asyncio.run_coroutine_threadsafe(
            self.engine.async_generate(prompts, sps, custom_logit_processor=[self._clp] * len(prompts), **kw),
            self._loop)
        return fut.result()

    def _policy(self, prompt):
        o = self._gen([prompt], [{"temperature": 0.0, "max_new_tokens": 1,
                                  "custom_params": {"lane": "observation"}}],
                      return_hidden_states=True)[0]
        hs = o["meta_info"].get("hidden_states")
        if not hs or not hs[0]:
            self._empty_hidden = getattr(self, "_empty_hidden", 0) + 1
            if os.environ.get("SMLR_SGL_DEBUG"):
                print(f"[sgl-client] EMPTY HIDDEN (#{self._empty_hidden}) -> WAIT/zero probs",
                      file=sys.stderr, flush=True)
            return "WAIT", {l: 0.0 for l in LABELS}, 0.0
        arr = torch.tensor(hs[0], dtype=torch.float32).view(-1, self._PW.shape[1])
        hv = arr[-1]                               # prompt-end hidden
        self._last_hidden = hv.tolist()
        self._last_hrows = arr.shape[0]            # #hidden rows captured (should == prompt token count)
        logit = hv @ self._PW.T + self._PB
        soft = torch.softmax(logit, -1)
        probs = {LABELS[i]: float(soft[i]) for i in range(len(LABELS))}
        return LABELS[int(logit.argmax())], probs, float(soft.max())

    @torch.no_grad()
    def call(self, tier, system, user_obj, max_tokens=1500, extra_instruction=None):
        prompt = _render_chatml(SFT_SYSTEM, user_obj)
        plen = len(self.tok_ids(prompt))
        t0 = time.perf_counter()
        next_action, probs, conf = self._policy(prompt)
        if os.environ.get("SMLR_SGL_RECORD"):
            self.trace.append({
                "plen": plen, "hrows": getattr(self, "_last_hrows", None),
                "chunk": self._chunk, "chunk_crossed": plen > self._chunk,
                "next_action": next_action, "probs": {k: round(v, 6) for k, v in probs.items()},
                "mass": round(probs.get("WARN", 0) + probs.get("ALERT", 0), 6),
                "hidden": self._last_hidden, "prompt": prompt})
        sps = [{"temperature": 0.0, "max_new_tokens": MAX_NEW.get(l, 128),
                "custom_params": {"lane": l}} for l in LANES]
        outs = self._gen([prompt] * len(LANES), sps)
        upd = {"next_action": next_action}
        ntok = 0
        for l, o in zip(LANES, outs):
            upd[l] = _coerce(l, o["text"]); ntok += o["meta_info"]["completion_tokens"]
        dt = (time.perf_counter() - t0) * 1000
        ntok_in = int(outs[0]["meta_info"].get("prompt_tokens", 0))
        meta = CallMeta(tier=tier or "", model=self.model_name, input_tokens=ntok_in,
                        output_tokens=ntok, policy_conf=conf, policy_probs=probs)
        meta.latency_ms = dt
        self.usage.add(meta)
        return upd, meta

    def shutdown(self):
        try:
            self.engine.shutdown()
        except Exception:
            pass
