"""Head-to-head compare of the full held-out gate: HF (shipped) vs SGLang. Reads the two per-scenario
JSON dumps and prints per-tier aggregates + bar verdicts + the regression breakdown (mttd blowups,
rc->0, NEW false alerts, recall misses where HF passes but SGLang fails)."""
import json, statistics

hf = {r["scenario"]: r for r in json.load(open("/home/user/heldout_hf.json"))}
sg = {r["scenario"]: r for r in json.load(open("/home/user/heldout_sglang.json"))}
scen = [r["scenario"] for r in json.load(open("/home/user/heldout_sglang.json"))]
grp = {r["scenario"]: r["group"] for r in json.load(open("/home/user/heldout_sglang.json"))}


def num(v):
    return 1.0 if v is True else 0.0 if v is False else v


def agg(rows, key):
    vals = [num(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


for group in ("metrics", "logs"):
    ss = [sg[s] for s in scen if grp[s] == group]
    hh = [hf[s] for s in scen if grp[s] == group and s in hf]
    n = len(ss)
    print(f"\n########## {group}  (n={n}) ##########")
    for tag, rows in (("HF   ", hh), ("SGL  ", ss)):
        fa = sum((r["false_alerts"] or 0) for r in rows)
        passes = sum(1 for r in rows if r["bar_pass"])
        mttds = [r["mttd"] for r in rows if r["mttd"] is not None]
        line = (f"  {tag} bar_pass={passes}/{len(rows)} fa_total={fa} "
                f"recall={agg(rows,'alert_recall')} detect={agg(rows,'detected')} "
                f"rc_correct={agg(rows,'root_cause_correct')} rc_acc={agg(rows,'root_cause_accuracy')}")
        if mttds:
            line += f" mttd(mean={round(statistics.mean(mttds),1)},med={round(statistics.median(mttds),1)})"
        print(line)
    # per-scenario head-to-head + regressions
    print("  --- per scenario (HF -> SGL) ---")
    reg_mttd = reg_rc = new_fa = miss = 0
    for s in [x for x in scen if grp[x] == group]:
        h, g = hf.get(s), sg[s]
        hm = h["mttd"] if h else None; gm = g["mttd"]
        mflag = ""
        if hm is not None and gm is not None and gm >= 3 * hm and gm - hm >= 10:
            mflag = " <MTTD_BLOWUP>"; reg_mttd += 1
        if h and num(h.get("root_cause_correct")) == 1.0 and num(g.get("root_cause_correct")) == 0.0:
            reg_rc += 1
        if h and num(h.get("root_cause_accuracy") or 0) >= 0.99 and num(g.get("root_cause_accuracy") or 0) == 0.0:
            reg_rc += 1
        if (g["false_alerts"] or 0) > (h["false_alerts"] or 0 if h else 0):
            new_fa += 1; mflag += " <NEW_FALSE_ALERT>"
        if h and h["bar_pass"] and not g["bar_pass"]:
            miss += 1; mflag += " <SGL_BAR_FAIL_HF_PASS>"
        print(f"    {s:28s} HF[bar={h['bar_pass'] if h else '?'} rec={h['alert_recall'] if h else '?'} "
              f"mttd={hm} rc={h.get('root_cause_correct') if h else '?'}] -> "
              f"SGL[bar={g['bar_pass']} rec={g['alert_recall']} mttd={gm} rc={g.get('root_cause_correct')}]{mflag}")
    print(f"  REGRESSIONS: mttd_blowups={reg_mttd}  rc->0={reg_rc}  new_false_alerts={new_fa}  "
          f"sgl_fail_where_hf_pass={miss}")
