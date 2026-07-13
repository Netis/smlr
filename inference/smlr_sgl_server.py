"""SMLR SGLang inference server — the rope-fixed 6-lane VLA served over the SAME HTTP protocol the
runtime uses (POST /v1/frame, GET /health), backed by SglangVLAClient (SGLang engine loaded ONCE at
startup on a background asyncio thread; policy read with bias + policy_probs). SGLang does continuous
batching internally, so each /v1/frame just runs client.call() in a worker thread — concurrent
requests overlap on the engine loop and the scheduler batches them.

  CKPT=/home/user/models/smlr-1b-ml6 uvicorn smlr_sgl_server:app --host 0.0.0.0 --port 8100

Pairs with runtime.remote_vla_client.RemoteVLAClient (drop-in for Reasoner.client).
"""
from __future__ import annotations
import os, sys, time, dataclasses, asyncio, statistics
sys.path.insert(0, os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")))
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SMLR-SGLang-Server")
_client = None
_lat = []                                # per-frame HTTP-loop latency (ms), server-side
_stats = {"frames": 0, "wait": 0, "started": None, "load_ms": None}


class FrameReq(BaseModel):
    tier: str | None = None
    system: str = ""
    user_obj: dict
    max_tokens: int = 1500
    extra: str | None = None


@app.on_event("startup")
async def _load() -> None:
    global _client
    t0 = time.perf_counter()
    from smlr_sglang_client import SglangVLAClient
    _client = SglangVLAClient(os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6"),
                              gpu=os.environ.get("SGL_GPU", "3"))
    _stats["load_ms"] = round((time.perf_counter() - t0) * 1000)
    _stats["started"] = time.time()
    print(f"[server] SglangVLAClient ready in {_stats['load_ms']}ms lanes={_client.lanes}", flush=True)


@app.get("/health")
async def health() -> dict:
    p50 = round(statistics.median(_lat)) if _lat else None
    p99 = round(sorted(_lat)[max(0, int(0.99 * len(_lat)) - 1)]) if _lat else None
    return {"ok": _client is not None, "model": getattr(_client, "model_name", None),
            "lanes": getattr(_client, "lanes", None), "frames": _stats["frames"],
            "wait_frames": _stats["wait"], "load_ms": _stats["load_ms"],
            "p50_ms": p50, "p99_ms": p99,
            "empty_hidden": getattr(_client, "_empty_hidden", 0)}


@app.post("/v1/frame")
async def frame(req: FrameReq) -> dict:
    t0 = time.perf_counter()
    upd, meta = await asyncio.to_thread(
        _client.call, req.tier or "", req.system, req.user_obj, req.max_tokens, req.extra)
    dt = (time.perf_counter() - t0) * 1000
    _lat.append(dt)
    _stats["frames"] += 1
    if upd.get("next_action") == "WAIT":
        _stats["wait"] += 1
    return {"upd": upd, "meta": dataclasses.asdict(meta)}
