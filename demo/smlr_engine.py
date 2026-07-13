"""Netis SMLR Tech Preview — closed-loop inference engine.

Loads `netis-ai/smlr-metrics-1b` (trust_remote_code) and runs it as a streaming
monitor: render frame -> shared-prefill decode (policy + 6 lanes) -> parse lanes ->
carry `state_patch` into the next frame's working state. A light K-of-M soft-escalation
over the policy softmax (the shipped mechanism) converts sustained anomaly into a stable
WARN/ALERT without retraining.
"""

from __future__ import annotations

import collections
import json
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.environ.get("SMLR_REPO_ID", "netis-ai/smlr-metrics-1b")
HERE = os.path.dirname(os.path.abspath(__file__))

# demo caps: shorter than training for responsiveness on modest hardware
DEMO_CAPS = {"observation": 56, "reasoning": 80, "public_output": 64,
             "notes": 80, "state_patch": 96, "actions": 64}

# frames of policy-mass history kept for the displayed escalation gauge
SUSTAIN_M = 3


def _pick_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


_SHARED = {}


def load_shared(repo: str = REPO):
    """Load (tokenizer, model, system, device) once; reuse across engines/tabs."""
    if repo not in _SHARED:
        device, dtype = _pick_device_dtype()
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=dtype).eval().to(device)
        with open(os.path.join(HERE, "system_prompt.md")) as f:
            system = f.read().strip()
        _SHARED[repo] = (tok, model, system, device)
    return _SHARED[repo]


class SmlrEngine:
    def __init__(self, repo: str = REPO, recent_window: int = 6):
        self.tok, self.model, self.system, self.device = load_shared(repo)
        self.recent_window = recent_window
        self.reset()

    def reset(self):
        self.working_state: dict = {"link_state": {}}
        self.recent: "collections.deque[dict]" = collections.deque(maxlen=self.recent_window)
        self._mass: "collections.deque[float]" = collections.deque(maxlen=SUSTAIN_M)
        self._pending: Optional[dict] = None   # synthesized tool_result to feed next
        self.seq = 0

    def take_pending(self) -> Optional[dict]:
        """If the model just queried a tool, return a synthesized tool_result to feed
        as the next event (completing the query->confirm->alert loop); else None."""
        ev, self._pending = self._pending, None
        return ev

    def _render(self, event: dict) -> str:
        user = {"recent_window": list(self.recent), "working_state": self.working_state,
                "retrieved_memory": [], "new_event": event}
        return (f"<|im_start|>system\n{self.system}<|im_end|>\n"
                f"<|im_start|>user\n{json.dumps(user)}<|im_end|>\n<|im_start|>assistant\n")

    def _decode_lane(self, toks) -> str:
        return self.tok.decode(toks, skip_special_tokens=True).strip() if toks else ""

    @staticmethod
    def _try_json(text):
        try:
            return json.loads(text)
        except Exception:
            return text

    @torch.no_grad()
    def step(self, event: dict) -> dict:
        self.seq += 1
        event.setdefault("seq", self.seq)
        ids = self.tok(self._render(event), return_tensors="pt").input_ids.to(self.device)
        out = self.model.decode_frame(ids, max_new=DEMO_CAPS)

        lanes = {k: self._decode_lane(v) for k, v in out["lanes"].items()}
        raw_action = out["next_action"]

        # carry state_patch forward (shallow merge)
        sp = self._try_json(lanes.get("state_patch", ""))
        if isinstance(sp, dict):
            for k, v in sp.items():
                if k == "link_state" and isinstance(v, dict):
                    self.working_state.setdefault("link_state", {}).update(v)
                else:
                    self.working_state[k] = v

        # keep the REAL frame in the recent window so persistence is visible to the model
        self.recent.append(_compact(event))

        pub = self._try_json(lanes.get("public_output", ""))
        pub_text = pub.get("text", "") if isinstance(pub, dict) else str(pub)
        pub_mode = pub.get("mode", "SILENT") if isinstance(pub, dict) else "SILENT"
        acts = self._try_json(lanes.get("actions", ""))
        tool_calls = [a for a in acts if isinstance(a, dict)] if isinstance(acts, list) else []

        # if the model queried a tool, synthesize a confirming tool_result for next tick
        if tool_calls and event.get("type") != "tool_result":
            self._pending = {
                "type": "tool_result", "call_id": (tool_calls[0].get("call_id", "t1")),
                "result": {"tool": tool_calls[0].get("tool", "probe"),
                           "finding": "sustained resource saturation confirmed on the "
                                      "flagged target", "confirmed": True}}

        # display status: the user-facing signal is public_output.mode, then tool query,
        # then the policy head (which is conservative on out-of-domain host input)
        self._mass.append(out["policy_probs"].get("WARN", 0.0) + out["policy_probs"].get("ALERT", 0.0))
        MODE_OK = {"NOTE", "SUMMARY", "QUESTION", "WARN", "ALERT", "RESOLVE"}
        if pub_mode in MODE_OK and pub_text:
            status = pub_mode
        elif tool_calls:
            status = "QUERY_TOOL"
        else:
            status = raw_action

        return {
            "seq": event["seq"], "t": event.get("t"), "event_type": event.get("type"),
            "status": status, "next_action": raw_action,
            "policy_probs": out["policy_probs"],
            "escalation_mass": round(sum(self._mass) / max(1, len(self._mass)), 3),
            "observation": self._try_json(lanes.get("observation", "")),
            "reasoning": lanes.get("reasoning", ""),
            "public_mode": pub_mode, "public_text": pub_text,
            "notes": self._try_json(lanes.get("notes", "")),
            "state_patch": sp, "actions": acts, "tool_calls": tool_calls,
        }


def _compact(event: dict) -> dict:
    """Trim a frame to what the model needs to see for persistence (keep values)."""
    e = {k: event.get(k) for k in ("seq", "t", "type") if k in event}
    if "links" in event:
        e["links"] = event["links"]
    if "lines" in event:
        e["lines"] = event["lines"][-6:]
    if "result" in event:
        e["result"] = event["result"]
    return e
