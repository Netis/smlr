"""Drive a live scenario end-to-end through the REAL HTTP serving path: reasoner -> RemoteVLAClient
-> HTTP -> SGLang 6-lane server -> update -> event log. Reports real per-frame HTTP-loop latency
(P50/P99), the live detect/alert timeline (per-frame action + escalation), and the metrics_eval
outcome (alert_recall / false_alerts / mttd). Run in sglang env (httpx only; no torch)."""
import os, sys, time, asyncio, statistics, json
ROOT = os.environ.get("SMLR_REPO", os.path.expanduser("~/streamingllm")); sys.path.insert(0, ROOT)
from types import SimpleNamespace

URL = os.environ.get("URL", "http://localhost:8137")
KIND = os.environ.get("KIND", "metrics")
SCEN = os.environ.get("SCEN", "congestion_l2_001")
HELDOUT = os.environ.get("HELDOUT", "0") == "1"


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
    sub = "heldout" if HELDOUT else "scenarios"
    sc = sim.load_scenario(f"{ROOT}/scenarios/{KIND}/{sub}/{SCEN}.json")

    ns = SimpleNamespace(tier="1b", vla=None, vla_base=None, vla_remote=URL)
    reasoner = build_reasoner(sc, ns)
    print(f"[live] driving {KIND}/{SCEN} via {URL}  model={reasoner.client.model_name} "
          f"lanes={reasoner.client.lanes}", flush=True)

    timeline = []
    _o = reasoner.client.call
    def timed(t, s, uo, mt=1500, ex=None):
        t0 = time.perf_counter(); upd, meta = _o(t, s, uo, mt, ex)
        dt = (time.perf_counter() - t0) * 1000
        timeline.append((upd.get("next_action"), dt,
                         round((meta.policy_probs or {}).get("WARN", 0) + (meta.policy_probs or {}).get("ALERT", 0), 3)))
        return upd, meta
    reasoner.client.call = timed

    bus = EventBus(); p = Player(bus); p.load(sc.protos)
    t0 = time.perf_counter()
    asyncio.run(p.run(speed=float("inf"), hook=reasoner.hook()))
    wall = time.perf_counter() - t0
    score = ev.evaluate(p.log, sc.ground_truth)

    lat = [d for _, d, _ in timeline]
    p50 = round(statistics.median(lat)); p99 = round(sorted(lat)[max(0, int(0.99 * len(lat)) - 1)])
    print(f"\n=== LIVE {KIND}/{SCEN} through HTTP serving path ===")
    print(f"frames={len(timeline)} compute_wall={wall:.1f}s  per-frame HTTP-loop P50={p50}ms P99={p99}ms "
          f"mean={round(statistics.mean(lat))}ms max={round(max(lat))}ms")
    if KIND == "metrics":
        print(f"detect/alert: alert_recall={score.get('alert_recall')} false_alerts={score.get('false_alerts')} "
              f"mttd={score.get('mttd_avg')} rc_acc={score.get('root_cause_accuracy')}")
        ok = score.get("alert_recall", 0) >= 1.0 and score.get("false_alerts", 99) == 0
    else:
        print(f"detect/alert: detected={score.get('detected')} rc={score.get('root_cause_correct')} "
              f"false_alerts={score.get('false_alerts')} mttd={score.get('mttd')}")
        ok = bool(score.get("detected")) and bool(score.get("root_cause_correct")) and score.get("false_alerts", 99) == 0
    print(f"BAR: {'PASS' if ok else 'FAIL'}")
    print("timeline (action, ms, esc_mass):")
    for i, (a, d, m) in enumerate(timeline):
        star = "  <== ESCALATION" if a not in ("WAIT", "NOTE") or m >= 0.2 else ""
        print(f"  f{i:02d} {a:8s} {d:7.0f}ms mass={m}{star}")


if __name__ == "__main__":
    main()
