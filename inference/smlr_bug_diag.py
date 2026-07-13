"""Bug-vs-drift diagnostic. Runs a FAILING held-out scenario through the SGLang client (closed loop),
recording per-frame (prompt, plen, hrows, chunk_crossed, policy_probs, mass, prompt-end hidden), and
scores recall. SGL_CHUNK controls chunked_prefill_size (Test 1: force huge). Dumps the trace to JSON so
the HF replay (smlr_bug_hf.py) can compare per-frame hidden/probs on the IDENTICAL prompts."""
import os, sys, json, asyncio
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "3"))
os.environ["SMLR_SGL_RECORD"] = "1"
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from types import SimpleNamespace

SCEN = os.environ.get("SCEN", "held_bufferbloat_01")
KIND = os.environ.get("KIND", "metrics")
OUT = os.environ.get("OUT", f"/home/user/bugdiag_{SCEN}.json")


def main():
    from runtime.player import Player
    from runtime.event_bus import EventBus
    from smlr import build_reasoner
    if KIND == "metrics":
        from scenarios.metrics import simulator as sim
        from evals import metrics_eval as ev
    else:
        from scenarios.logs import simulator as sim
        from evals import logs_eval as ev

    sc = sim.load_scenario(f"{ROOT}/scenarios/{KIND}/heldout/{SCEN}.json")
    ns = SimpleNamespace(tier="haiku", vla=None, vla_base=None, vla_remote=None)
    reasoner = build_reasoner(sc, ns)
    from smlr_sglang_client import SglangVLAClient
    client = SglangVLAClient(os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6"))
    reasoner.client = client

    bus = EventBus(); pl = Player(bus); pl.load(sc.protos)
    asyncio.run(pl.run(speed=float("inf"), hook=reasoner.hook()))
    score = ev.evaluate(pl.log, sc.ground_truth)

    tr = client.trace
    plens = [t["plen"] for t in tr]
    crossed = sum(1 for t in tr if t["chunk_crossed"])
    mismatch = sum(1 for t in tr if t["hrows"] != t["plen"])
    print(f"\n=== BUGDIAG {SCEN} chunk={tr[0]['chunk'] if tr else '?'} ===")
    print(f"recall={score.get('alert_recall')} fa={score.get('false_alerts')} frames={len(tr)}")
    print(f"prompt_len: min={min(plens)} max={max(plens)} (chunk_crossed frames={crossed})")
    print(f"hidden_rows != prompt_tokens (capture-length mismatch) frames: {mismatch}/{len(tr)}")
    print(f"max escalation mass: {max((t['mass'] for t in tr), default=0):.4f} (tau_warn=0.2 tau_alert=0.3)")
    for i, t in enumerate(tr):
        flag = ""
        if t["chunk_crossed"]: flag += " CHUNK_CROSSED"
        if t["hrows"] != t["plen"]: flag += f" HROWS={t['hrows']}!=PLEN"
        if t["mass"] >= 0.2 or t["next_action"] not in ("WAIT", "NOTE"): flag += " <FIRE>"
        print(f"  f{i:02d} plen={t['plen']:4d} hrows={t['hrows']} act={t['next_action']:8s} mass={t['mass']:.3f}{flag}")
    json.dump({"scenario": SCEN, "recall": score.get("alert_recall"),
               "chunk": tr[0]["chunk"] if tr else None, "trace": tr}, open(OUT, "w"))
    print("WROTE", OUT)
    client.shutdown()


if __name__ == "__main__":
    main()
