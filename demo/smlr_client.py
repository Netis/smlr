"""Netis SMLR Tech Preview — thin client to the SMLR SGLang server.

Inference runs on the GPU host (SGLang serving the 6-lane VLA); this client only does
the closed-loop bookkeeping locally (no torch, no model): render the frame, POST it to
`/v1/frame`, carry `state_patch` forward, and synthesize a `tool_result` when the model
queries a tool so the query->confirm->alert loop completes.

Point it at the server with `SMLR_SERVER_URL` (e.g. an SSH tunnel to the GPU host:
`ssh -N -L 8100:localhost:8100 <gpu-host>` then `SMLR_SERVER_URL=http://localhost:8100`).
"""

from __future__ import annotations

import collections
import json
import os
from typing import Optional

import httpx

SERVER_URL = os.environ.get("SMLR_SERVER_URL", "http://localhost:8100").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
SUSTAIN_M = 3


class SmlrClient:
    """Closed-loop monitor backed by the remote SGLang inference server."""

    def __init__(self, url: str = SERVER_URL, recent_window: int = 6, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)
        with open(os.path.join(HERE, "system_prompt.md")) as f:
            self.system = f.read().strip()
        # fail loud if the server is unreachable — a misrouted client is a silent killer
        try:
            h = self._http.get(self.url + "/health").json()
        except Exception as e:
            raise RuntimeError(f"cannot reach SMLR SGLang server at {self.url}: {e}\n"
                               f"Start it on the GPU host and set SMLR_SERVER_URL "
                               f"(e.g. via `ssh -N -L 8100:localhost:8100 <gpu-host>`).")
        self.model_name = h.get("model") or "smlr-sglang"
        self.lanes = h.get("lanes")
        self.recent_window = recent_window
        self.reset()

    def reset(self):
        self.working_state: dict = {"link_state": {}}
        self.recent: "collections.deque[dict]" = collections.deque(maxlen=self.recent_window)
        self._pending: Optional[dict] = None
        self.seq = 0

    def take_pending(self) -> Optional[dict]:
        ev, self._pending = self._pending, None
        return ev

    @staticmethod
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        try:
            return json.loads(x)
        except Exception:
            return x

    def step(self, event: dict) -> dict:
        self.seq += 1
        event.setdefault("seq", self.seq)
        user_obj = {"recent_window": list(self.recent), "working_state": self.working_state,
                    "retrieved_memory": [], "new_event": event}
        r = self._http.post(self.url + "/v1/frame", json={
            "tier": "", "system": self.system, "user_obj": user_obj, "max_tokens": 512})
        r.raise_for_status()
        d = r.json()
        upd, meta = d["upd"], d.get("meta", {})

        raw_action = upd.get("next_action", "WAIT")
        pub = self._as_dict(upd.get("public_output", {})) or {}
        pub_mode = pub.get("mode", "SILENT") if isinstance(pub, dict) else "SILENT"
        pub_text = pub.get("text", "") if isinstance(pub, dict) else str(pub)
        acts = upd.get("actions", [])
        acts = acts if isinstance(acts, list) else []
        tool_calls = [a for a in acts if isinstance(a, dict)]

        # carry state_patch forward
        sp = self._as_dict(upd.get("state_patch", {}))
        if isinstance(sp, dict):
            for k, v in sp.items():
                if k == "link_state" and isinstance(v, dict):
                    self.working_state.setdefault("link_state", {}).update(v)
                else:
                    self.working_state[k] = v

        self.recent.append(_compact(event))

        # synthesize a confirming tool_result for next tick when the model queries a tool
        if tool_calls and event.get("type") != "tool_result":
            self._pending = {
                "type": "tool_result", "call_id": tool_calls[0].get("call_id", "t1"),
                "result": {"tool": tool_calls[0].get("tool", "probe"),
                           "finding": "sustained resource saturation confirmed on the flagged target",
                           "confirmed": True}}

        MODE_OK = {"NOTE", "SUMMARY", "QUESTION", "WARN", "ALERT", "RESOLVE"}
        if pub_mode in MODE_OK and pub_text:
            status = pub_mode
        elif tool_calls:
            status = "QUERY_TOOL"
        else:
            status = raw_action

        obs = upd.get("observation", "")
        return {
            "seq": event["seq"], "t": event.get("t"), "event_type": event.get("type"),
            "status": status, "next_action": raw_action,
            "server_p50_ms": meta.get("latency_ms") or meta.get("ms"),
            "observation": obs, "reasoning": upd.get("reasoning", ""),
            "public_mode": pub_mode, "public_text": pub_text,
            "notes": upd.get("notes", []), "state_patch": sp,
            "actions": acts, "tool_calls": tool_calls,
        }


def _compact(event: dict) -> dict:
    e = {k: event.get(k) for k in ("seq", "t", "type") if k in event}
    if "links" in event:
        e["links"] = event["links"]
    if "lines" in event:
        e["lines"] = event["lines"][-6:]
    if "result" in event:
        e["result"] = event["result"]
    return e
