"""Sentinel demo: replay a series point by point and watch alerts + evidence.

  streamlit run app/streamlit_app.py

This is a local replay of stored CSV files through the streaming detector (one update()
call per row), not a connection to a live data source. Ground-truth windows are drawn and
used for the running scores only ("benchmark view"); they are never passed to the detector.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from sentinel import StreamingSentinel  # noqa: E402
from sentinel.config import load_config  # noqa: E402
from sentinel.io import NAB_ROOT, Series, load_corpus, probation_length  # noqa: E402
from sentinel.metrics.trigger import trigger_counts, trigger_prf  # noqa: E402
from sentinel.metrics.windows import events_from_labels  # noqa: E402

GREY, RED, ORANGE, BLUE = "#6b7280", "#dc2626", "#f59e0b", "#2563eb"

st.set_page_config(page_title="Sentinel demo", page_icon="🛰️", layout="wide")


@st.cache_resource
def get_corpus():
    if not (NAB_ROOT / "labels" / "combined_windows.json").exists():
        # hosted deployments (e.g. Streamlit Community Cloud) start without data/raw/:
        # install the pinned NAB commit once, verified against configs/nab_manifest.json
        import subprocess
        with st.spinner("First start: downloading the NAB data (about 50 MB, once)..."):
            subprocess.run([sys.executable, str(ROOT / "scripts" / "download_nab.py")], check=False)
    if (NAB_ROOT / "labels" / "combined_windows.json").exists():
        return load_corpus()
    return {}


def get_config():
    final = ROOT / "configs" / "final.yaml"
    return load_config(final if final.exists() else None), ("final.yaml" if final.exists() else "default.yaml")


def uploaded_series(f):
    """Parse an uploaded CSV; returns (Series, None) or (None, error message)."""
    try:
        df = pd.read_csv(f)
    except Exception as e:  # noqa: BLE001
        return None, f"Could not read the file as CSV ({e})."
    if df.shape[1] < 2:
        return None, "Need two columns: a timestamp and a numeric value."
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    val_col = "value" if "value" in df.columns else df.columns[1]
    try:
        ts = pd.to_datetime(df[ts_col]).to_numpy(dtype="datetime64[ns]")
    except Exception:  # noqa: BLE001
        return None, f"Column '{ts_col}' does not contain parseable timestamps."
    vals = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
    if len(vals) < 20:
        return None, f"Only {len(vals)} rows; need at least 20."
    if np.isnan(vals).all():
        return None, f"Column '{val_col}' has no numeric values."
    return Series(f.name, ts, vals, [], [], probation_length(len(vals))), None


def make_figure(s, upto, surprise, labels, th_enter, th_exit, state=None):
    ts = pd.to_datetime(s.timestamps[:upto])          # Timestamps: plotly shapes need them
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.06, subplot_titles=("value", "surprise (−log10 p)"))
    # traces first: plotly skips add_vrect(row=...) on a subplot that has no traces yet
    fig.add_trace(go.Scatter(x=ts, y=s.values[:upto], mode="lines", line=dict(color=GREY, width=1),
                             name="value"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=surprise[:upto], mode="lines", line=dict(color=BLUE, width=1),
                             name="surprise"), row=2, col=1)
    p = min(s.probation, upto) - 1
    if p > 0:
        fig.add_vrect(x0=ts[0], x1=ts[p], fillcolor=GREY, opacity=0.12, line_width=0, layer="below",
                      row=1, col=1, annotation_text="probation", annotation_position="top left")
    for a, b in s.windows:
        if a < upto:
            fig.add_vrect(x0=ts[a], x1=ts[min(b, upto - 1)], fillcolor=ORANGE, opacity=0.25,
                          line_width=0, layer="below", row=1, col=1)
    if state is not None:
        for a, b in events_from_labels(np.asarray(state[:upto]) == "EVENT", s.probation):
            fig.add_vrect(x0=ts[a], x1=ts[b], fillcolor=RED, opacity=0.15, line_width=0,
                          layer="below", row=1, col=1)
    trig = [a for a, _ in events_from_labels(labels[:upto], s.probation)]
    if trig:
        fig.add_trace(go.Scatter(x=ts[trig], y=s.values[trig], mode="markers", name="alert",
                                 marker=dict(color=RED, size=9, symbol="triangle-down")), row=1, col=1)
        for a in trig:
            fig.add_vline(x=ts[a], line=dict(color=RED, width=1), row=1, col=1)
    for y, dash, name in ((th_enter, "solid", "θ enter"), (th_exit, "dot", "θ exit")):
        fig.add_hline(y=y, line=dict(color=RED, dash=dash, width=1), row=2, col=1,
                      annotation_text=name, annotation_position="right")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return fig


def running_metrics(s, labels, upto):
    """Benchmark view only: trigger-based event P/R against the windows seen so far."""
    if not s.windows:
        return None
    wins = [(a, min(b, upto - 1)) for a, b in s.windows if a < upto]
    c = trigger_counts(labels[:upto], wins, s.probation)
    return c, trigger_prf(c)


# ---------------------------------------------------------------- sidebar
cfg, cfg_name = get_config()
corpus = get_corpus()
st.sidebar.title("🛰️ Sentinel")
st.sidebar.caption(f"Config: `{cfg_name}` · θ_enter {cfg['events']['theta_enter']}")
source = st.sidebar.radio("Data", ["NAB file", "Upload CSV"] if corpus else ["Upload CSV"])
series = None
qp = st.query_params                      # deep link: ?file=realKnownCause/nyc_taxi.csv&run=1
if source == "NAB file":
    want = qp.get("file", "realKnownCause/nyc_taxi.csv")
    want = want if want in corpus else next(iter(corpus))
    cats = sorted({n.split("/")[0] for n in corpus})
    cat = st.sidebar.selectbox("Category", cats, index=cats.index(want.split("/")[0]))
    names = [n for n in corpus if n.startswith(cat + "/")]
    name = st.sidebar.selectbox("File", names, format_func=lambda n: n.split("/")[1],
                                index=names.index(want) if want in names else 0)
    series = corpus[name]
else:
    up = st.sidebar.file_uploader("CSV with timestamp,value", type="csv")
    if up is not None:
        series, err = uploaded_series(up)
        if err:
            st.sidebar.error(err)
step = st.sidebar.slider("Points per frame", 25, 2000, 250, step=25)
pause = st.sidebar.slider("Pause per frame (s)", 0.0, 0.5, 0.05, step=0.05)
go_stream = st.sidebar.button("▶ Stream", type="primary", width="stretch")
go_all = st.sidebar.button("⏩ Run whole file", width="stretch")
if qp.get("run") == "1" and "auto_ran" not in st.session_state:
    st.session_state["auto_ran"] = True
    go_all = True

st.title("Sentinel — calibrated, explainable streaming anomaly detection")
st.caption("Local replay of a stored series through the streaming detector, one row at a time.")
if series is None:
    st.info("Pick a NAB file or upload a CSV in the sidebar.")
    st.stop()

th_enter = cfg["events"]["theta_enter"]
th_exit = th_enter - cfg["events"]["exit_offset"]
n = len(series)
st.caption(f"`{series.name}` · {n:,} points · probation {series.probation} · "
           f"orange = labelled window (benchmark view, never shown to the detector) · "
           f"red marker = alert trigger · light red = open event · grey = probation")

chart = st.empty()
cols = st.columns(4)
stat_boxes = [c.empty() for c in cols]
key = (series.name, n, cfg_name)

if go_stream or go_all:
    det = StreamingSentinel(cfg, probation_len=series.probation)
    surprise = np.zeros(n)
    labels = np.zeros(n, dtype=np.int8)
    states = np.empty(n, dtype=object)
    states[:] = "PROBATION"
    t_ns = np.zeros(n, dtype=np.int64)
    secs, vals = series.seconds, series.values
    frame = n if go_all else step
    for start in range(0, n, frame):
        end = min(start + frame, n)
        for j in range(start, end):
            t0 = time.perf_counter_ns()
            r = det.update(secs[j], vals[j])
            t_ns[j] = time.perf_counter_ns() - t0
            surprise[j] = r.surprise
            labels[j] = r.label == "ANOMALOUS"
            states[j] = r.state
        chart.plotly_chart(make_figure(series, end, surprise, labels, th_enter, th_exit, states),
                           width="stretch", key=f"chart_{start}")
        live = t_ns[series.probation:end]
        n_events = len(det.events) + (det.current is not None)
        stat_boxes[0].metric("Points streamed", f"{end:,}")
        stat_boxes[1].metric("Alerts (events)", n_events)
        stat_boxes[2].metric("µs / point (mean)", f"{live.mean() / 1000:.0f}" if len(live) else "–")
        m = running_metrics(series, labels, end)
        stat_boxes[3].metric("Benchmark P / R so far", f"{m[1]['precision']:.2f} / {m[1]['recall']:.2f}"
                             if m and (m[0]['n_alerts'] or m[0]['n_windows']) else "–")
        if not go_all and pause:
            time.sleep(pause)
    st.session_state["result"] = {"key": key, "surprise": surprise, "labels": labels,
                                  "states": states, "events": det.finish(), "t_ns": t_ns,
                                  "season": det.season}

res = st.session_state.get("result")
if res and res["key"] == key:
    if not (go_stream or go_all):
        chart.plotly_chart(make_figure(series, n, res["surprise"], res["labels"], th_enter, th_exit,
                                       res.get("states")), width="stretch")
        live = res["t_ns"][series.probation:]
        stat_boxes[0].metric("Points streamed", f"{n:,}")
        stat_boxes[1].metric("Alerts (events)", len(res["events"]))
        stat_boxes[2].metric("µs / point (mean)", f"{live.mean() / 1000:.0f}")
        m = running_metrics(series, res["labels"], n)
        stat_boxes[3].metric("Benchmark P / R", f"{m[1]['precision']:.2f} / {m[1]['recall']:.2f}" if m else "–")
    st.subheader("Evidence for each alert")
    st.caption(f"Season mode detected after probation: **{res['season']['mode']}**")
    if not res["events"]:
        st.write("No alerts on this series.")
    else:
        evs = res["events"]
        pick = st.selectbox("Alert", range(len(evs)), format_func=lambda i: (
            f"#{evs[i]['event_id']} · {evs[i]['start_timestamp']} · {evs[i]['top_detector']} · "
            f"peak surprise {evs[i]['peak_surprise']:.1f}"))
        e = evs[pick]
        st.error(e["explanation"])
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Detector surprise at onset**")
            st.bar_chart(pd.Series(e["detector_surprise"]))
            st.markdown(f"Event length **{e['length_points']}** points (alert label on the first "
                        f"{cfg['events'].get('label_len') or 'all'} row(s)) · re-baselined: **{e['rebaselined']}**")
        with c2:
            st.markdown("**Evidence JSON**")
            st.json({k: v for k, v in e.items() if k != "trace"}, expanded=False)
elif not (go_stream or go_all):
    chart.plotly_chart(make_figure(series, n, np.zeros(n), np.zeros(n, dtype=np.int8), th_enter, th_exit),
                       width="stretch")
    st.info("Press ▶ Stream to replay this series point by point.")
