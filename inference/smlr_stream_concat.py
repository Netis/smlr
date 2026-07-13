"""Sustained streaming test (proper concatenation). Chain scenarios into ONE continuous monotonic
timeline by OFFSETTING each scenario's proto sim_ts and incident onsets, then run ONE Player + ONE
reasoner over the whole stream via the LIVE HTTP server. State (reasoning/working_state) carries
across incident boundaries. Scores the combined log against a merged ground-truth. Reports latency
EARLY vs LATE across the long horizon, per-incident detect/alert/mttd, total false_alerts, health."""
import os, sys, time, asyncio, statistics, json, copy
import httpx
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
from types import SimpleNamespace

URL = os.environ.get("URL", "http://localhost:8137")
GAP = float(os.environ.get("GAP", "6"))
from scenarios.metrics import simulator as sim
from evals import metrics_eval as ev

CHAIN = [("scenarios", "congestion_l2_001"), ("scenarios", "multi_incident_001"),
         ("heldout", "held_bufferbloat_00"), ("heldout", "held_silent_drop_00"),
         ("heldout", "held_congestion_90"), ("heldout", "held_interface_error_91"),
         ("heldout", "held_bufferbloat_03"), ("heldout", "held_silent_drop_02")]


def pctl(xs, q):
    xs = sorted(xs); return xs[max(0, int(q * len(xs)) - 1)] if xs else 0


def main():
    from runtime.player import Player
    from runtime.event_bus import EventBus
    from smlr import build_reasoner

    protos = []; incidents = []; truth = {}; links = set(); offset = 0.0
    for sub, name in CHAIN:
        sc = sim.load_scenario(f"{ROOT}/scenarios/metrics/{sub}/{name}.json")
        for p in sc.protos:
            p.sim_ts = float(p.sim_ts) + offset
            protos.append(p)
        gt = sc.ground_truth
        for inc in gt.get("incidents", []):
            ic = copy.deepcopy(inc)
            ic["incident_id"] = f"{name}:{inc['incident_id']}"
            for k in ("onset_sec", "resolve_sec"):
                if ic.get(k) is not None:
                    ic[k] = ic[k] + offset
            incidents.append(ic)
        truth.update(gt.get("mock_tool_truth", {}))
        links.update(gt.get("links", []))
        dur = float(gt.get("duration_sec") or (sc.protos[-1].sim_ts - offset if sc.protos else 0))
        offset += dur + GAP
    protos.sort(key=lambda p: p.sim_ts)
    merged_gt = {"scenario_id": "CONCAT_STREAM", "duration_sec": offset, "links": sorted(links),
                 "incidents": incidents, "noise": {}, "mock_tool_truth": truth}
    print(f"[stream] concat {len(CHAIN)} scenarios -> {len(protos)} protos, {len(incidents)} incidents, "
          f"horizon={offset:.0f}s via {URL}", flush=True)

    ns = SimpleNamespace(tier="1b", vla=None, vla_base=None, vla_remote=URL)
    reasoner = build_reasoner(sim.load_scenario(f"{ROOT}/scenarios/metrics/scenarios/{CHAIN[0][1]}.json"), ns)
    reasoner.executor.truth = truth

    lat = []
    _o = reasoner.client.call
    def timed(t, s, uo, mt=1500, ex=None):
        t0 = time.perf_counter(); upd, meta = _o(t, s, uo, mt, ex)
        lat.append((time.perf_counter() - t0) * 1000)
        return upd, meta
    reasoner.client.call = timed

    bus = EventBus(); p = Player(bus); p.load(protos)
    t0 = time.perf_counter()
    asyncio.run(p.run(speed=float("inf"), hook=reasoner.hook()))
    wall = time.perf_counter() - t0
    score = ev.evaluate(p.log, merged_gt)

    N = len(lat)
    early = lat[:N // 3]; late = lat[2 * N // 3:]
    print(f"\n=== SUSTAINED CONCAT STREAM: {len(CHAIN)} scenarios, {N} model-frames, "
          f"{len(incidents)} incidents, horizon {offset:.0f}s ===")
    print(f"compute_wall={wall:.1f}s")
    print(f"latency EARLY(first {len(early)}): P50={round(statistics.median(early))}ms P99={round(pctl(early,0.99))}ms")
    print(f"latency LATE (last  {len(late)}): P50={round(statistics.median(late))}ms P99={round(pctl(late,0.99))}ms")
    print(f"latency WHOLE: P50={round(statistics.median(lat))}ms P99={round(pctl(lat,0.99))}ms max={round(max(lat))}ms")
    print(f"detect/alert (whole stream): alert_recall={score.get('alert_recall')} "
          f"false_alerts={score.get('false_alerts')} mttd_avg={score.get('mttd_avg')} "
          f"rc_acc={score.get('root_cause_accuracy')}")
    print("per-incident:")
    for inc in score.get("incidents", []):
        print(f"  {inc.get('incident_id'):40s} detected={inc.get('detected')} alerted={inc.get('alerted')} "
              f"mttd={inc.get('mttd')} rc={inc.get('root_cause_correct')}")
    try:
        h = httpx.get(URL + "/health", timeout=10).json()
        print(f"server /health: frames={h['frames']} empty_hidden={h['empty_hidden']} p50={h['p50_ms']} p99={h['p99_ms']}")
    except Exception as e:
        print("health fetch failed:", e)
    json.dump({"score": {k: str(v) for k, v in score.items() if k != "incidents"},
               "incidents": score.get("incidents"), "N": N, "horizon": offset,
               "early_p99": pctl(early, 0.99), "late_p99": pctl(late, 0.99)},
              open("/home/user/stream_concat_report.json", "w"))


if __name__ == "__main__":
    main()
