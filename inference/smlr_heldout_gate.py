"""FULL held-out detect/alert gate, head-to-head. Runs every held-out scenario (metrics n=15, logs
n=12) through the reasoner with a WARM client (loaded ONCE, reused across all scenarios), scores with
metrics_eval / logs_eval, and dumps per-scenario rows + per-tier aggregates + bar verdicts + failures.

Mirrors evals/generalization_eval.py (the harness behind the shipped MANIFEST). Backend via env:
  BACKEND=sglang CKPT=/home/user/models/smlr-1b-ml6   (sglang env)
  BACKEND=hf BASE=.../MiniCPM5-1B-SFT ADAPTER=.../smlr-1b-minicpm-vla-repro   (vllm env)
Reasoner 1b config passed via SMLR_POLICY_TAU_WARN/ALERT/SUSTAIN_K/M.
"""
import os, sys, json, asyncio, time, statistics
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "4"))
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from types import SimpleNamespace

from runtime.player import Player
from runtime.event_bus import EventBus
from smlr import build_reasoner
from scenarios.metrics import simulator as metrics_sim
from scenarios.logs import simulator as logs_sim
from evals import metrics_eval, logs_eval

BACKEND = os.environ.get("BACKEND", "sglang")
TIER = os.environ.get("TIER", "haiku")   # valid MODELS key: only names the repair-model string
_GF = os.environ.get("GROUP")            # optional: run only 'metrics' or 'logs'
GROUPS = [g for g in [("metrics", "scenarios/metrics/heldout", metrics_sim, metrics_eval),
                      ("logs", "scenarios/logs/heldout", logs_sim, logs_eval)]
          if not _GF or g[0] == _GF]


def collect():
    from pathlib import Path
    filt = os.environ.get("FILTER")
    jobs = []
    for group, d, sim, ev in GROUPS:
        for p in sorted(Path(os.path.join(ROOT, d)).glob("*.json")):
            if filt and filt not in p.name:
                continue
            jobs.append((group, sim, ev, str(p)))
    return jobs


def make_client():
    if BACKEND == "hf":
        return None    # built by build_reasoner from base+adapter on the first scenario
    from smlr_sglang_client import SglangVLAClient
    return SglangVLAClient(os.environ.get("CKPT", "/home/user/models/smlr-1b-ml6"))


def main():
    jobs = collect()
    print(f"### held-out gate backend={BACKEND} tier={TIER} n={len(jobs)} "
          f"(metrics={sum(1 for j in jobs if j[0]=='metrics')} logs={sum(1 for j in jobs if j[0]=='logs')}) ###",
          flush=True)
    shared = make_client()
    rows = []
    for group, sim, ev, path in jobs:
        sc = sim.load_scenario(path)
        ns = SimpleNamespace(tier=TIER,
                             vla=(os.environ.get("ADAPTER") if (BACKEND == "hf" and shared is None) else None),
                             vla_base=os.environ.get("BASE"), vla_remote=None)
        reasoner = build_reasoner(sc, ns)
        if shared is None:
            shared = reasoner.client
        else:
            reasoner.client = shared
        calls = []
        _o = reasoner.client.call
        def timed(t, s, uo, mt=1500, ex=None, _o=_o, calls=calls):
            t0 = time.perf_counter(); r = _o(t, s, uo, mt, ex)
            calls.append((time.perf_counter() - t0) * 1000); return r
        reasoner.client.call = timed
        bus = EventBus(); pl = Player(bus); pl.load(sc.protos)
        asyncio.run(pl.run(speed=float("inf"), hook=reasoner.hook()))
        reasoner.client.call = _o                     # unwrap
        score = ev.evaluate(pl.log, sc.ground_truth)
        row = {"group": group, "scenario": sc.name,
               "alert_recall": score.get("alert_recall"), "false_alerts": score.get("false_alerts"),
               "detected": score.get("detected"), "alerted": score.get("alerted"),
               "root_cause_correct": score.get("root_cause_correct"),
               "root_cause_accuracy": score.get("root_cause_accuracy"),
               "mttd": score.get("mttd_avg", score.get("mttd")),
               "frames": len(calls), "lat_p50": round(statistics.median(calls), 0) if calls else 0}
        # per-scenario bar
        if group == "metrics":
            row["bar_pass"] = (score.get("alert_recall", 0) >= 1.0 and score.get("false_alerts", 99) == 0)
        else:
            row["bar_pass"] = (bool(score.get("detected")) and bool(score.get("root_cause_correct"))
                               and score.get("false_alerts", 99) == 0)
        rows.append(row)
        print(f"  [{group}] {sc.name}: recall={row['alert_recall']} fa={row['false_alerts']} "
              f"det={row['detected']} rc={row['root_cause_correct']} mttd={row['mttd']} "
              f"BAR={'PASS' if row['bar_pass'] else 'FAIL'}", flush=True)
    out = f"/home/user/heldout_{BACKEND}.json"
    json.dump(rows, open(out, "w"))
    print("WROTE", out, flush=True)

    # per-tier aggregates
    for group, *_ in GROUPS:
        g = [r for r in rows if r["group"] == group]
        n = len(g)
        fa_total = sum((r["false_alerts"] or 0) for r in g)
        passes = sum(1 for r in g if r["bar_pass"])
        def mean(key):
            vals = [(1.0 if r[key] is True else 0.0 if r[key] is False else r[key]) for r in g if r[key] is not None]
            return round(sum(vals) / len(vals), 3) if vals else None
        mttds = [r["mttd"] for r in g if r["mttd"] is not None]
        print(f"\n=== {BACKEND} {group} n={n} ===")
        print(f"  bar_pass={passes}/{n}  false_alerts_total={fa_total}  detect_rate={mean('detected')}")
        print(f"  alert_recall(mean)={mean('alert_recall')}  root_cause_correct(mean)={mean('root_cause_correct')}"
              f"  root_cause_accuracy(mean)={mean('root_cause_accuracy')}")
        if mttds:
            print(f"  mttd mean={round(statistics.mean(mttds),1)} median={round(statistics.median(mttds),1)} "
                  f"min={min(mttds)} max={max(mttds)} vals={[round(m,1) for m in mttds]}")
        fails = [r["scenario"] for r in g if not r["bar_pass"]]
        if fails:
            print(f"  BAR FAILURES: {fails}")
    if shared is not None and hasattr(shared, "shutdown"):
        shared.shutdown()
    print("HELDOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
