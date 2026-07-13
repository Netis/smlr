"""Concurrency scaling — SGLang (rope-fixed 6-lane ml6, graphs ON, radix ON). For each K in the sweep,
K sessions each submit the WAIT carry lanes (reasoning cap48 + state_patch cap256) as CONCURRENT
coroutines (asyncio.gather) so the scheduler continuously batches. Reports aggregate frames/s, tok/s,
and per-frame P50/P99 (frame latency = max of its 2 lane latencies). Synchronized arrival (all K at t0)
-- a batched-K test, not staggered arrival (noted)."""
import os, sys, json, time, asyncio
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "3"))
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

CKPT = os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6")
KS = [int(x) for x in os.environ.get("KS", "1,8,16,32,64,128").split(",")]
CAPS = {"reasoning": 48, "state_patch": 256}
LANES = ["reasoning", "state_patch"]


class IdentityLP(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        return logits


def wait_prompts(n):
    from data.build_sft_dataset import SFT_SYSTEM
    from runtime.vla_client import _render_chatml
    pool = []
    for line in open(os.path.join(ROOT, "data/m5_eval_frames.jsonl")):
        r = json.loads(line.strip()) if line.strip() else None
        if not r or str(r.get("true_next_action")) != "WAIT":
            continue
        um = next((m for m in reversed(r["messages"]) if m.get("role") == "user"), None)
        pool.append(_render_chatml(SFT_SYSTEM, json.loads(um["content"])))
    out = [pool[i % len(pool)] for i in range(n)]     # cycle if K > distinct WAIT frames
    return out, len(pool)


def main():
    import sglang as sgl
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    clp = IdentityLP.to_str()
    maxK = max(KS)
    allp, ndistinct = wait_prompts(maxK)
    eng = sgl.Engine(model_path=CKPT, dtype="bfloat16", tp_size=1,
                     mem_fraction_static=float(os.environ.get("MEMFRAC", "0.85")),
                     attention_backend="triton", sampling_backend="pytorch",
                     enable_custom_logit_processor=True,
                     cuda_graph_max_bs=int(os.environ.get("CGMAXBS", "256")),
                     max_running_requests=int(os.environ.get("MAXRUN", "512")),
                     log_level="warning")
    print(f"### SGLANG concurrency  distinct_WAIT_frames={ndistinct}  (K>that cycles) ###", flush=True)

    async def one(prompt, lane):
        sp = {"temperature": 0.0, "max_new_tokens": CAPS[lane], "custom_params": {"lane": lane}}
        t0 = time.perf_counter()
        r = await eng.async_generate(prompt, sp, custom_logit_processor=clp)
        return (time.perf_counter() - t0) * 1000, r["meta_info"]["completion_tokens"]

    async def run_K(K):
        reqs = []                                     # (frame_idx, lane)
        for i in range(K):
            for ln in LANES:
                reqs.append((i, ln))
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(allp[i], ln) for (i, ln) in reqs])
        wall = (time.perf_counter() - t0) * 1000
        lat = {}                                      # frame -> [lane latencies]
        toks = 0
        for (i, ln), (ms, tk) in zip(reqs, res):
            lat.setdefault(i, []).append(ms); toks += tk
        frame_lat = sorted(max(v) for v in lat.values())
        p50 = frame_lat[len(frame_lat) // 2]
        p99 = frame_lat[max(0, int(0.99 * len(frame_lat)) - 1)]
        return wall, toks, p50, p99

    # warmup
    loop.run_until_complete(run_K(min(8, maxK)))
    print(f"{'K':>4} {'wall_ms':>8} {'frames/s':>9} {'tok/s':>8} {'P50_ms':>8} {'P99_ms':>8}", flush=True)
    rows = []
    for K in KS:
        best = None
        for _ in range(2):
            wall, toks, p50, p99 = loop.run_until_complete(run_K(K))
            if best is None or wall < best[0]:
                best = (wall, toks, p50, p99)
        wall, toks, p50, p99 = best
        fps = K / (wall / 1000); tps = toks / (wall / 1000)
        rows.append({"K": K, "wall_ms": round(wall), "fps": round(fps, 2), "tok_s": round(tps),
                     "p50": round(p50), "p99": round(p99)})
        print(f"{K:>4} {wall:>8.0f} {fps:>9.2f} {tps:>8.0f} {p50:>8.0f} {p99:>8.0f}", flush=True)
    print("MAXK@2s(P99):", max([r["K"] for r in rows if r["p99"] <= 2000], default=0))
    print("MAXK@3s(P99):", max([r["K"] for r in rows if r["p99"] <= 3000], default=0))
    json.dump(rows, open(os.environ.get("OUT", "/home/user/conc_sgl.json"), "w"))
    print("SGL_CONC_DONE", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
