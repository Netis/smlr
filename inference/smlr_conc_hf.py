"""Concurrency scaling — hand-rolled call_batch (runtime/vla_client.py, vllm env). Shipped realtime
config: SMLR_VLA_COMPILE=1 SMLR_VLA_POLICY_FIRST=1 (all-WAIT carry lanes). For each K, call_batch(K
WAIT frames) as ONE synchronized fused multi-head batch; measure wall (best-of-2). Per-frame latency =
batch wall (every frame waits for the slowest -> P50=P99=wall, the nature of the fused batch).
Base+adapter is the HF parity model (transformers 5.x reads rope_parameters natively)."""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "4"))
os.environ.setdefault("SMLR_VLA_COMPILE", "1")
os.environ.setdefault("SMLR_VLA_POLICY_FIRST", "1")
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)

BASE = os.environ.get("BASE", "/home/user/models/MiniCPM5-1B-SFT")
ADAPTER = os.environ.get("ADAPTER", "/home/user/models/smlr-1b-minicpm-vla-repro")
KS = [int(x) for x in os.environ.get("KS", "1,8,16,32,64,128").split(",")]


def wait_user_objs(n):
    pool = []
    for line in open(os.path.join(ROOT, "data/m5_eval_frames.jsonl")):
        r = json.loads(line.strip()) if line.strip() else None
        if not r or str(r.get("true_next_action")) != "WAIT":
            continue
        um = next((m for m in reversed(r["messages"]) if m.get("role") == "user"), None)
        pool.append(json.loads(um["content"]))
    return [pool[i % len(pool)] for i in range(n)], len(pool)


def main():
    import runtime.vla_client as vc
    vc.MAX_NEW["state_patch"] = 256          # match the SGLang workload cap (reasoning stays 48)
    from runtime.vla_client import VLAClient
    cli = VLAClient(BASE, ADAPTER)
    objs, ndistinct = wait_user_objs(max(KS))
    print(f"### HAND-ROLLED call_batch  distinct_WAIT_frames={ndistinct}  compile={vc.COMPILE} "
          f"policy_first={vc.POLICY_FIRST} ###", flush=True)

    def run_K(K):
        reqs = [{"user_obj": objs[i], "tier": ""} for i in range(K)]
        t0 = time.perf_counter()
        res = cli.call_batch(reqs)
        wall = (time.perf_counter() - t0) * 1000
        toks = sum(m.output_tokens for _, m in res)
        return wall, toks

    run_K(min(8, max(KS)))                   # warmup (pays torch.compile)
    run_K(min(8, max(KS)))
    print(f"{'K':>4} {'wall_ms':>8} {'frames/s':>9} {'tok/s':>8} {'P50=P99_ms':>11}", flush=True)
    rows = []
    for K in KS:
        best = None
        for _ in range(2):
            wall, toks = run_K(K)
            if best is None or wall < best[0]:
                best = (wall, toks)
        wall, toks = best
        fps = K / (wall / 1000); tps = toks / (wall / 1000)
        rows.append({"K": K, "wall_ms": round(wall), "fps": round(fps, 2), "tok_s": round(tps),
                     "p50": round(wall), "p99": round(wall)})
        print(f"{K:>4} {wall:>8.0f} {fps:>9.2f} {tps:>8.0f} {wall:>11.0f}", flush=True)
    print("MAXK@2s:", max([r["K"] for r in rows if r["wall_ms"] <= 2000], default=0))
    print("MAXK@3s:", max([r["K"] for r in rows if r["wall_ms"] <= 3000], default=0))
    json.dump(rows, open(os.environ.get("OUT", "/home/user/conc_hf.json"), "w"))
    print("HF_CONC_DONE", flush=True)


if __name__ == "__main__":
    main()
