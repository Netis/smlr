"""Netis SMLR Tech Preview — live streaming-monitor demo.

A thin client: it reads THIS machine's real telemetry (live CPU/memory/load and the
system's real-time logs) and streams frames to the SMLR SGLang inference server running
on the GPU host. No model runs locally.

Point it at the server with SMLR_SERVER_URL (default http://localhost:8100 — use an SSH
tunnel to the GPU host: `ssh -N -L 8100:localhost:8100 <gpu-host>`).

Run:  pip install -r requirements.txt  &&  SMLR_SERVER_URL=http://localhost:8100 python app.py
"""

from __future__ import annotations

import html
import time

import gradio as gr

from collectors import HostLogs, HostMetrics
from smlr_client import SmlrClient

BRAND = "Netis SMLR Tech Preview"
CADENCE = 3.0  # seconds between frames (inference runs on the GPU host)

print(f"[{BRAND}] connecting to SGLang server...")
metric_engine = SmlrClient()
log_engine = SmlrClient()          # same server, independent session state
host_metrics = HostMetrics()
host_logs = HostLogs()
_t0 = time.time()
print(f"[{BRAND}] connected to {metric_engine.url} (model={metric_engine.model_name}) | logs via {host_logs.source}")

_ACTION_COLOR = {
    "WAIT": "#6b7280", "NOTE": "#2563eb", "SUMMARY": "#2563eb", "QUESTION": "#2563eb",
    "VERIFY": "#7c3aed", "QUERY_TOOL": "#0891b2", "WARN": "#d97706",
    "ALERT": "#dc2626", "RESOLVE": "#059669", "REVISE": "#7c3aed",
}


def _badge(action: str) -> str:
    c = _ACTION_COLOR.get(action, "#6b7280")
    return (f"<span style='background:{c};color:#fff;padding:3px 12px;border-radius:6px;"
            f"font-weight:700;font-size:15px'>{html.escape(action)}</span>")


def _fmt_decision(res: dict) -> str:
    obs = res["observation"]
    obs = obs[0] if isinstance(obs, list) and obs else (obs if isinstance(obs, str) else "")
    tag = " <small style='color:#0891b2'>· from tool_result</small>" if res.get("event_type") == "tool_result" else ""
    parts = [f"{_badge(res['status'])}{tag}<br>"]
    if obs:
        parts.append(f"<b>observation:</b> {html.escape(str(obs))[:300]}<br>")
    if res["reasoning"]:
        parts.append(f"<b>reasoning:</b> {html.escape(res['reasoning'])[:400]}<br>")
    if res.get("tool_calls"):
        names = ", ".join(str(t.get("tool", t.get("call_id", "tool"))) for t in res["tool_calls"][:4])
        parts.append(f"<b>🔧 query_tool:</b> {html.escape(names)[:200]}<br>")
    if res["public_mode"] and res["public_mode"] != "SILENT" and res["public_text"]:
        parts.append(f"<b>📣 {res['public_mode']}:</b> {html.escape(res['public_text'])[:300]}<br>")
    return "".join(parts)


# ---- Metric tab ------------------------------------------------------------
_metric_log: list[str] = []


def tick_metrics():
    mev, raw = host_metrics.frame(metric_engine.seq + 1, time.time() - _t0)
    gauges = (f"**CPU** {raw['cpu_pct']:.0f}%  ·  **Mem** {raw['mem_pct']:.0f}% "
              f"({raw['mem_used_gb']}/{raw['mem_total_gb']} GB)  ·  "
              f"**Load/core** {raw['load_per_core']}  ·  **Net** {raw['net_mbps']} Mb/s  ·  "
              f"**Disk** {raw['disk_mbps']} MB/s")
    # if the model just queried a tool, feed the synthesized tool_result this tick instead
    pending = metric_engine.take_pending()
    res = metric_engine.step(pending if pending else mev)
    stamp = time.strftime("%H:%M:%S")
    _metric_log.insert(0, f"<div style='border-bottom:1px solid #eee;padding:6px 0'>"
                          f"<small style='color:#999'>{stamp} · frame {res['seq']}</small><br>"
                          f"{_fmt_decision(res)}</div>")
    del _metric_log[60:]
    return gauges, _badge(res["status"]), "".join(_metric_log)


# ---- Log tab ---------------------------------------------------------------
_log_log: list[str] = []


def tick_logs():
    pending = log_engine.take_pending()
    lines = host_logs.drain(limit=30)
    if pending:
        ev, recent = pending, "*(confirming via tool_result…)*"
    elif lines:
        ev = host_logs.frame(log_engine.seq + 1, time.time() - _t0, lines)
        recent = "\n".join(f"`{html.escape(l[:160])}`" for l in lines[-8:])
    else:
        return "*(no new log lines this tick)*", gr.update(), "".join(_log_log)
    res = log_engine.step(ev)
    stamp = time.strftime("%H:%M:%S")
    n = len(lines) if not pending else 0
    _log_log.insert(0, f"<div style='border-bottom:1px solid #eee;padding:6px 0'>"
                       f"<small style='color:#999'>{stamp} · {n} lines · frame {res['seq']}</small><br>"
                       f"{_fmt_decision(res)}</div>")
    del _log_log[60:]
    return recent, _badge(res["status"]), "".join(_log_log)


_BRANDBAR = ("background:linear-gradient(90deg,#0b1220,#1e3a5f);color:#fff;"
             "padding:14px 18px;border-radius:10px")

with gr.Blocks(title=BRAND) as demo:  # theme/css kept out of the constructor for gradio 4/5/6 compat
    gr.HTML(f"<div style='{_BRANDBAR}'><h2 style='margin:0'>🛰️ {BRAND}</h2>"
            "<div style='opacity:.85;font-size:14px'>Live streaming monitor — SMLR watching this "
            "machine's real CPU/memory and real-time logs. Research preview; keep a human in the loop.</div></div>")
    gr.Markdown(
        f"*{BRAND}: `netis-ai/smlr-metrics-1b` runs a shared prefill + policy head + 6 decode lanes per "
        "frame, carrying state forward. The metrics tier was trained on network-link telemetry, so real "
        "host resources are **projected** into that schema — it stays quiet on a healthy box and reacts to "
        "sustained load. Spike CPU/mem (e.g. `yes > /dev/null` or a build) to watch it escalate.*")

    with gr.Tab("📈 Metric Monitoring"):
        gr.Markdown(f"**{BRAND} · Metric Monitoring** — real CPU / memory / load projected into SMLR's "
                    "metrics schema, one frame every ~%.0fs." % CADENCE)
        m_gauges = gr.Markdown("*starting…*")
        with gr.Row():
            m_action = gr.HTML(_badge("WAIT"))
        m_feed = gr.HTML()
        with gr.Row():
            m_start = gr.Button("▶ Start", variant="primary")
            m_stop = gr.Button("⏸ Stop")
        m_timer = gr.Timer(CADENCE, active=False)
        m_timer.tick(tick_metrics, outputs=[m_gauges, m_action, m_feed])
        m_start.click(lambda: gr.Timer(active=True), outputs=m_timer)
        m_stop.click(lambda: gr.Timer(active=False), outputs=m_timer)

    with gr.Tab("📜 Log Monitoring"):
        gr.Markdown(f"**{BRAND} · Log Monitoring** — the system's real-time logs (`{host_logs.source}`) "
                    "streamed into SMLR.")
        l_recent = gr.Markdown("*starting…*")
        with gr.Row():
            l_action = gr.HTML(_badge("WAIT"))
        l_feed = gr.HTML()
        with gr.Row():
            l_start = gr.Button("▶ Start", variant="primary")
            l_stop = gr.Button("⏸ Stop")
        l_timer = gr.Timer(CADENCE, active=False)
        l_timer.tick(tick_logs, outputs=[l_recent, l_action, l_feed])
        l_start.click(lambda: gr.Timer(active=True), outputs=l_timer)
        l_stop.click(lambda: gr.Timer(active=False), outputs=l_timer)

    gr.HTML(f"<div style='text-align:center;color:#999;padding:10px;font-size:13px'>{BRAND} · "
            "<a href='https://github.com/Netis/smlr'>github.com/Netis/smlr</a> · "
            "<a href='https://huggingface.co/netis-ai/smlr-metrics-1b'>netis-ai/smlr-metrics-1b</a></div>")

if __name__ == "__main__":
    demo.queue().launch()
