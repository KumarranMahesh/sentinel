"""Build slides/Sentinel_Challenge2.pptx (exactly 10 slides, PLAN.md 15.2).

  python slides/make_slides.py

Every number is read from results/ and configs/final.yaml at build time, so the deck
always matches the latest run. Figures come from figures/ (scripts/make_figures.py).
"""
import json
from pathlib import Path

import pandas as pd
import yaml
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
OUT = ROOT / "slides" / "Sentinel_Challenge2.pptx"

NAVY, WHITE, INK, INK2, MUTED = "0F1B2D", "FFFFFF", "0B0B0B", "52514E", "898781"
BLUE, RED, ORANGE, TINT, TINT2, GRID = "2A78D6", "E34948", "EB6834", "EEF4FC", "F4F4F2", "E1E0D9"
FONT = "Calibri"
TEAM = ("Team TBD  ·  Kommana Srinivasa Rao  ·  Panuganti Bhanu  ·  Vishnu Datta  ·  "
        "Kumarran Mahesh")


def rgb(h):
    return RGBColor.from_string(h)


# ------------------------------------------------------------------ data
def load():
    d = {
        "m": json.loads((RES / "metrics_summary.json").read_text()),
        "search": json.loads((RES / "search_summary.json").read_text()),
        "base": pd.read_csv(RES / "baselines_comparison.csv"),
        "abl": pd.read_csv(RES / "ablation.csv"),
        "rt": json.loads((RES / "runtime.json").read_text()),
        "cats": pd.read_csv(RES / "dataset_stats_by_category.csv"),
        "audit": pd.read_csv(RES / "metric_audit.csv") if (RES / "metric_audit.csv").exists() else None,
    }
    d["cfg"] = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())
    d["storm"] = storm_counts()
    d["evidence"] = pick_evidence()
    return d


STORM_FILE = "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv"


def storm_counts():
    """Alert counts on the slide-2 file, recomputed (same as figures/alert_storm.png)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from sentinel.baselines import baseline_labels
    from sentinel.config import load_config
    from sentinel.io import load_corpus
    from sentinel.metrics.windows import events_from_labels, scored_windows
    from sentinel.runner import run_sentinel
    s = load_corpus(files=[STORM_FILE])[STORM_FILE]
    gt = scored_windows(s.windows, s.probation)
    _, z = baseline_labels("rolling_z", s, "raw")
    sen = run_sentinel(s, load_config(ROOT / "configs" / "final.yaml"))["label"]
    out = {}
    for k, lab in (("z", z), ("sentinel", sen)):
        trig = [a for a, _ in events_from_labels(lab, s.probation)]
        out[k] = (len(trig), sum(any(a <= t <= b for a, b in gt) for t in trig))
    return out


def pick_evidence():
    """Two correct alerts (residual, different files) and one false alert, from events.jsonl."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from sentinel.io import load_corpus
    from sentinel.metrics.windows import scored_windows
    events = [json.loads(x) for x in (ROOT / "predictions" / "events.jsonl").read_text().splitlines()]
    corpus = load_corpus()
    tps, fps = [], []
    for e in events:
        s = corpus[e["file"]]
        t = e["start_index"]
        hit = any(a <= t <= b for a, b in scored_windows(s.windows, s.probation))
        (tps if hit else fps).append(e)
    pref = ["realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv", "realTweets/Twitter_volume_FB.csv",
            "realKnownCause/machine_temperature_system_failure.csv", "realAdExchange/exchange-2_cpc_results.csv"]
    good = []
    for f in pref:
        cand = [e for e in tps if e["file"] == f and e["top_detector"] == "residual"]
        if cand and len(good) < 2:
            good.append(max(cand, key=lambda e: e["peak_surprise"]))
    if len(good) < 2:
        good += sorted([e for e in tps if e not in good], key=lambda e: -e["peak_surprise"])[:2 - len(good)]
    bad = next((e for e in fps if e["file"] == "artificialNoAnomaly/art_noisy.csv"),
               max(fps, key=lambda e: e["peak_surprise"]) if fps else None)
    n_free = sum(e["file"].startswith("artificialNoAnomaly/") for e in fps)
    return {"tp": good, "fp": bad, "n_fp_anomaly_free": n_free}


# ------------------------------------------------------------------ helpers
def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, margin=0.0, name=None, spacing=None):
    """content: str, or list of paragraphs; a paragraph is str or list of (text, opts) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", FONT)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def box(slide, x, y, w, h, fill=TINT, radius=0.08, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def box_text(slide, x, y, w, h, content, fill=TINT, size=14, color=INK, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, line=None, margin=0.12, radius=0.08):
    s = box(slide, x, y, w, h, fill, radius=radius, line=line)
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", False)
            r.font.color.rgb = rgb(o.get("color", color))
    return s


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def title(slide, t, sub=None, dark=False):
    text(slide, 0.6, 0.38, 12.1, 0.75, t, size=32, bold=True, color=WHITE if dark else INK, name="Title")
    if sub:
        text(slide, 0.6, 1.08, 12.1, 0.45, sub, size=15, color="C9D1DE" if dark else INK2)


def footer(slide, n, dark=False):
    c = "8A94A6" if dark else MUTED
    text(slide, 0.6, 7.05, 6, 0.3, "Sentinel  ·  Utkrishta 2026  ·  Challenge 2", size=10, color=c)
    text(slide, 11.73, 7.05, 1.0, 0.3, str(n), size=10, color=c, align=PP_ALIGN.RIGHT)


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(color)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def bar_chart(slide, x, y, w, h, cats, vals, colors, number_format='0.00', font_size=11, max_v=None):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("F1", vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_title = False
    ch.has_legend = False
    ch.font.size = Pt(font_size)
    ch.font.name = FONT
    ch.font.color.rgb = rgb(INK2)
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = number_format
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(font_size)
    dl.font.color.rgb = rgb(INK)
    ser = plot.series[0]
    for i, c in enumerate(colors):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis
    va.visible = False
    va.has_major_gridlines = False
    va.minimum_scale = 0
    if max_v:
        va.maximum_scale = max_v
    ca = ch.category_axis
    ca.reverse_order = True
    ca.format.line.fill.background()
    ca.tick_labels.font.size = Pt(font_size)
    ca.tick_labels.font.color.rgb = rgb(INK2)
    return gf


def table(slide, x, y, w, rows, col_w, header_fill=NAVY, size=12, row_h=0.36, highlight_rows=()):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = str(rows[i][j])
            r.font.name = FONT
            r.font.size = Pt(size)
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = rgb(header_fill)
                r.font.color.rgb = rgb(WHITE)
                r.font.bold = True
            elif i in highlight_rows:
                cell.fill.fore_color.rgb = rgb(TINT)
                r.font.color.rgb = rgb(INK)
                r.font.bold = True
            else:
                cell.fill.fore_color.rgb = rgb(WHITE if i % 2 else "F7F7F5")
                r.font.color.rgb = rgb(INK2)
    return gf


def stat(slide, x, y, w, value, label, color=INK, fill=TINT):
    box(slide, x, y, w, 1.35, fill=fill)
    text(slide, x + 0.2, y + 0.12, w - 0.4, 0.75, value, size=34, bold=True, color=color)
    text(slide, x + 0.2, y + 0.82, w - 0.4, 0.45, label, size=12, color=INK2)


# ------------------------------------------------------------------ slides
def s1_title(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, NAVY)
    text(s, 0.6, 0.55, 8, 0.4, "UTKRISHTA 2026  ·  CHALLENGE 2  ·  STREAMING ANOMALY DETECTION",
         size=12, bold=True, color="8FB3E8")
    text(s, 0.6, 1.0, 9, 1.1, "Sentinel", size=60, bold=True, color=WHITE)
    text(s, 0.6, 2.1, 8.6, 1.0, "A calibrated, explainable streaming ensemble that raises one alert per incident",
         size=22, color="D6DBE4")
    text(s, 0.6, 3.25, 12.1, 0.4, TEAM, size=14, color="C9D1DE")
    picture(s, FIG / "hero.png", 0.6, 4.1, w=12.1)
    text(s, 0.6, 6.68, 12, 0.35, "machine_temperature_system_failure (NAB) · orange = labelled anomaly window · "
                                 "red line = Sentinel alert trigger · light red = open event",
         size=10, color="8A94A6")
    notes(s, "Sentinel reads a series one point at a time. For every point it gives a score, a label and, for "
             "every alert, the reason. Simple causal detectors, one calibrated scale, one state machine.")


def s2_problem(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Monitoring needs alerts people can trust",
          "Servers, machines, traffic: a stream arrives point by point and someone has to act on each alert")
    rows = [("1", "Decide now, from the past only", "The output at time t may use data up to t. No look-ahead, no second pass."),
            ("2", "One alert per incident", "Repeated and false alerts cause alert fatigue; real incidents then get ignored."),
            ("3", "Say why", "Each alert carries evidence a person can check: observed vs expected, shift, odd shape.")]
    for i, (n, head, body) in enumerate(rows):
        y = 1.85 + i * 1.55
        circ = box(s, 0.6, y, 0.62, 0.62, fill=BLUE, shape=MSO_SHAPE.OVAL)
        circ.text_frame.text = n
        p = circ.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.runs[0].font.size = Pt(20)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = rgb(WHITE)
        p.runs[0].font.name = FONT
        text(s, 1.45, y - 0.05, 4.3, 0.45, head, size=20, bold=True)
        text(s, 1.45, y + 0.42, 4.3, 0.9, body, size=14, color=INK2)
    picture(s, FIG / "alert_storm.png", 6.0, 1.75, w=6.75)
    text(s, 6.0, 4.65, 6.75, 0.7,
         "Same series, same labelled window. A plain threshold fires again and again; Sentinel alerts "
         "inside the window and stays quiet afterwards.", size=13, color=INK2)
    zc, sc = d["storm"]["z"], d["storm"]["sentinel"]
    stat(s, 6.0, 5.45, 3.25, f"{zc[0]} alerts", f"plain rolling z-score, {zc[0] - zc[1]} outside the window",
         color=INK2, fill=TINT2)
    stat(s, 9.5, 5.45, 3.25, f"{sc[0]} alert{'s' if sc[0] != 1 else ''}",
         f"Sentinel, {sc[1]} triggered inside the window", color=BLUE)
    footer(s, 2)
    notes(s, "Precision per alert is what an operator feels. The judging metric is event F1, so the design "
             "goal is: trigger once, inside the incident, early, and stay quiet on its aftermath.")


def s3_data(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    cats = d["cats"]
    title(s, f"NAB: {int(cats.files.sum())} series, {int(cats.windows.sum())} labelled anomaly windows",
          "7 categories of real and synthetic metrics; windows ≈ 10% of each file, centred on the labelled point")
    short = {"artificialNoAnomaly": "Artificial, no anomaly", "artificialWithAnomaly": "Artificial, with anomaly",
             "realAWSCloudwatch": "AWS CloudWatch", "realAdExchange": "Ad exchange",
             "realKnownCause": "Known cause", "realTraffic": "Traffic", "realTweets": "Tweets"}
    rows = [["Category", "Files", "Points", "Windows", "Seasonal"]]
    for _, r in cats.iterrows():
        rows.append([short.get(r.category, r.category), int(r.files), f"{int(r.points):,}", int(r.windows),
                     int(r.seasonal_files)])
    rows.append(["Total", int(cats.files.sum()), f"{int(cats.points.sum()):,}", int(cats.windows.sum()),
                 int(cats.seasonal_files.sum())])
    table(s, 0.6, 1.75, 6.0, rows, [2.3, 0.7, 1.05, 0.95, 1.0], size=12, row_h=0.34, highlight_rows=(len(rows) - 1,))
    steps = [("Rows kept as given", "49 duplicate timestamps; a stream cannot drop rows"),
             ("Probation = min(15% of N, 750)", "learn only, no alerts (NAB rule)"),
             ("Season: day / week / none", "ACF at the period lag after a linear detrend"),
             ("No global scaling", "every statistic is online and robust")]
    for i, (h, b) in enumerate(steps):
        x = 0.6 + (i % 2) * 3.05
        y = 5.05 + (i // 2) * 0.95
        box_text(s, x, y, 2.9, 0.82, [[(h, {"bold": True, "size": 13})], [(b, {"size": 11, "color": INK2})]],
                 fill=TINT2, anchor=MSO_ANCHOR.MIDDLE)
    picture(s, FIG / "eda" / "category_samples.png", 7.0, 1.7, h=5.25)
    footer(s, 3)
    notes(s, "Every category behaves differently: daily cycles, spikes, level shifts, bursts. That is why one "
             "detector is not enough and why the threshold must be calibrated per series, online.")


def s4_design(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "One pass per point: detect, calibrate, fuse, decide, explain",
          "StreamingSentinel.update(timestamp, value) → score, label, evidence")
    cols = [("Stream", "timestamp, value\none point at a time"),
            ("Probation", "season mode, scale floor;\nreplay the first P points once"),
            (None, None),
            ("Calibration", "per detector: tail p-value vs its recent NORMAL scores\nU = −log10 p"),
            ("Fusion", "U = max U_k − log10 K\nŪ = smoothed U"),
            ("Event state machine", "hysteresis · cooldown · max length · pause · re-baseline")]
    x0, wbox, gap, y, h = 0.6, 1.72, 0.36, 2.25, 2.3
    xs = [x0 + i * (wbox + gap) for i in range(len(cols))]
    for i, (head, body) in enumerate(cols):
        if head is None:
            dets = [("D1 Residual", "spikes · dips · seasonal breaks"), ("D2 Change", "level / variance shifts"),
                    ("D3 Shape", "odd patterns (off in final.yaml)")]
            for k, (hh, bb) in enumerate(dets):
                box_text(s, xs[i], y + k * 0.8, wbox, 0.7, [[(hh, {"bold": True, "size": 13, "color": WHITE})],
                                                           [(bb, {"size": 10, "color": "DCE8F8"})]],
                         fill=BLUE, margin=0.08)
            continue
        fill = NAVY if head == "Event state machine" else TINT
        c1 = WHITE if fill == NAVY else INK
        c2 = "C9D1DE" if fill == NAVY else INK2
        box_text(s, xs[i], y, wbox, h, [[(head, {"bold": True, "size": 14, "color": c1})],
                                        [(body, {"size": 11, "color": c2})]], fill=fill, margin=0.12)
    for i in range(len(cols) - 1):
        arrow(s, xs[i] + wbox + 0.04, y + h / 2, xs[i + 1] - 0.04, y + h / 2)
    box_text(s, xs[-1], y + h + 0.45, wbox, 1.2,
             [[("Outputs", {"bold": True, "size": 13})], [("score ∈ [0,1] · label · top detector · event id · "
                                                          "evidence JSON + sentence", {"size": 10, "color": INK2})]],
             fill=TINT2, margin=0.1)
    arrow(s, xs[-1] + wbox / 2, y + h + 0.03, xs[-1] + wbox / 2, y + h + 0.42)
    rt = d["rt"]["us_per_point"]
    facts = [("O(1)", "D1 rolling median (sorted list) and D2 CUSUM per point"),
             ("O(H·m)", "D3: one matrix-vector product over the last H windows"),
             (f"{rt['mean']:.0f} µs", f"mean time per point, one CPU core (p99 {rt['p99']:.0f} µs)"),
             ("0", "random numbers, labels or future points used at run time")]
    for i, (v, l) in enumerate(facts):
        x = 0.6 + i * 2.6
        text(s, x, 5.55, 2.4, 0.5, v, size=24, bold=True, color=BLUE)
        text(s, x, 6.05, 2.4, 0.8, l, size=12, color=INK2)
    footer(s, 4)
    notes(s, "Left to right is one update call. Learning pauses while an alert is open so the anomaly does not "
             "become the new normal; after a lasting shift the detectors re-baseline.")


def s5_algorithm(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    cfg = d["cfg"]
    ev = cfg["events"]
    title(s, "Cheap detectors, one calibrated scale, one state machine")
    cards = [("D1  Residual", "Spikes, dips, broken daily/weekly pattern",
              "r = x − (median of last W deseasonalized x + hour offset)\nS1 = |r| / σ_EW"),
             ("D2  Change", "Level shifts, slow drift, variance change",
              "z = (x − slow mean) / slow scale\nC± = max(0, C± ± z − k),  S2 = max(C+, C−, V+) / h"),
             ("D3  Shape", "Unusual patterns (optional; off in final.yaml)",
              f"distance of newest z-normalized {cfg['shape']['m']}-point window\nto its nearest of the last {cfg['shape']['history']} windows")]
    for i, (h, sub, f) in enumerate(cards):
        x = 0.6 + i * 4.1
        box(s, x, 1.3, 3.9, 1.95, fill=TINT)
        text(s, x + 0.2, 1.4, 3.5, 0.45, h, size=18, bold=True, color=BLUE)
        text(s, x + 0.2, 1.85, 3.5, 0.45, sub, size=13, color=INK2)
        text(s, x + 0.2, 2.35, 3.6, 0.85, f, size=12, color=INK, italic=True)
    box(s, 0.6, 3.5, 5.2, 2.85, fill=TINT2)
    text(s, 0.8, 3.62, 4.8, 0.45, "Calibration → surprise", size=18, bold=True)
    text(s, 0.8, 4.15, 4.9, 2.5, [
        [("p = (1 + #{buffer ≥ s}) / (1 + n)", {"italic": True})],
        [("far tail: p = 0.05 · exp(−(s − q95) / mean excess)", {"italic": True})],
        [("U = −log10 p   (U = 4 ≈ 1 in 10,000)", {"italic": True})],
        [("Buffer = recent scores in NORMAL state only, so an anomaly never teaches itself. "
          "The tail fit lets U pass the 1/(n+1) floor of an empirical p-value.", {"size": 12, "color": INK2})],
    ], size=13, spacing=6)
    # state machine
    sx, sy = 6.2, 4.3
    states = [("NORMAL", TINT, INK), ("EVENT", RED, WHITE), ("COOLDOWN", "D9DEE7", INK)]
    pos = [(sx, sy), (sx + 2.35, sy), (sx + 4.7, sy)]
    for (n, f, c), (x, y) in zip(states, pos):
        box_text(s, x, y, 1.75, 0.7, [[(n, {"bold": True, "size": 13, "color": c})]], fill=f,
                 align=PP_ALIGN.CENTER, radius=0.3)
    arrow(s, sx + 1.79, sy + 0.35, sx + 2.31, sy + 0.35, color=INK2)
    arrow(s, sx + 4.14, sy + 0.35, sx + 4.66, sy + 0.35, color=INK2)
    text(s, sx + 1.55, sy - 0.62, 1.4, 0.55, f"Ū ≥ θ_enter\n= {ev['theta_enter']}", size=11, color=INK2,
         align=PP_ALIGN.CENTER)
    text(s, sx + 3.85, sy - 0.62, 1.5, 0.55, f"Ū < {ev['theta_enter'] - ev['exit_offset']:.1f} or\nlength ≥ {ev['max_len']}",
         size=11, color=INK2, align=PP_ALIGN.CENTER)
    for (x1, y1, x2, y2, head) in ((sx + 5.575, sy + 0.72, sx + 5.575, sy + 1.0, False),
                                   (sx + 5.575, sy + 1.0, sx + 0.875, sy + 1.0, False),
                                   (sx + 0.875, sy + 1.0, sx + 0.875, sy + 0.74, True)):
        if head:
            arrow(s, x1, y1, x2, y2, color=INK2, width=1.25)
        else:
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            c.line.color.rgb = rgb(INK2)
            c.line.width = Pt(1.25)
    text(s, sx + 1.4, sy + 1.05, 4.0, 0.4, f"after {ev['cooldown']} quiet points", size=11, color=INK2,
         align=PP_ALIGN.CENTER)
    fus = d["cfg"]["fusion"]
    text(s, sx, sy + 1.55, 6.5, 1.0, [
        [("Fusion: ", {"bold": True}), (f"U = max_k U_k − log10 K (Bonferroni, K = {sum(d['cfg'][k]['enabled'] for k in ('residual', 'change', 'shape'))} detectors on); Ū = {fus.get('smooth_mode', 'mean').replace('_', ' ')} over {fus['smooth']} points", {})],
        [("Label: ", {"bold": True}), (f"ANOMALOUS on the first {ev.get('label_len') or 'every'} row(s) of an event (the trigger); "
                                        "the span is in state/event_id and events.jsonl.", {})],
        [("While EVENT: ", {"bold": True}), ("learning paused. At the end: re-baseline if the level moved > 3σ or D2 fired.", {})],
    ], size=12, color=INK2, spacing=4)
    footer(s, 5)
    notes(s, "All three raw scores live on different scales. Calibration turns each into a p-value against its "
             "own recent normal history, so one threshold means the same on every file.")


def held_out(srch):
    """(value, label) of the held-out estimate actually computed; never relabels a partial result."""
    if srch.get("nested_loco"):
        return srch["nested_loco"]["f1"], "ET F1, held-out categories (nested LOCO, 7 folds)"
    p = srch.get("nested_loco_partial")
    if p:
        return p["f1"], f"ET F1, held-out (nested LOCO, partial: {len(p['folds'])} of 7 categories)"
    return srch["pool_loco_cv_optimistic"]["f1"], "ET F1, pool LOCO (optimistic)"


def held_text(srch):
    v, lab = held_out(srch)
    return f"{lab}: {v:.3f}"


def s6_experiments(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    srch = d["search"]
    n = srch.get("pool_size", 0)
    title(s, "Staged search, logged run by run, judged on held-out categories",
          f"objective: trigger-based event F1; {n} configs in the final search, each run on all 58 files; "
          "every run in experiments/runs.csv")
    picture(s, FIG / "tuning.png", 0.6, 1.65, w=6.3)
    abl = d["abl"]
    short = {"1a": "D1 residual only", "1b": "D2 change only", "1c": "D3 shape only",
             "1d": "D1 + D2 + D3 (shape on)", "2": "D1 + D2, no calibration", "3": "+ calibration, plain threshold",
             "4": "+ hysteresis", "5": "+ cooldown, max length", "6": "+ pause, re-baseline (full)"}
    labels = [short.get(v.split(" ", 1)[0], v) for v in abl.variant]
    colors = [BLUE if v.startswith("6") else "B8B6AE" for v in abl.variant]
    bar_chart(s, 7.15, 1.55, 5.6, 4.0, labels, [round(v, 3) for v in abl.et_f1], colors, font_size=11,
              max_v=max(abl.et_f1) * 1.25)
    text(s, 7.15, 5.5, 5.6, 0.35, "Ablation: trigger-based event F1 (ET), each variant with its own best θ_enter",
         size=11, color=MUTED)
    what = {"stage1": "detector params", "stage2a": "θ, hysteresis, smoothing", "stage2b": "cooldown, max length",
            "stage3a": "Fisher, peak-or-mean, learn in cooldown", "stage3b": "season ACF, log1p",
            "stage3c": "capped CUSUM", "stage3d": "shape detector off", "stage3e": "deseasonalized CUSUM"}
    lines, prev = [], None
    for st in srch["stages"]:
        key = st["stage"].split("_")[0]
        res = (f"ET F1 {st['best_et_f1']:.3f}" if st["best_hash"] != prev else "no gain, kept")
        lines.append([(f"{key.replace('stage', 'S')} ", {"bold": True}),
                      (f"{what.get(key, '')} ({st['n_configs']}): {res}", {})])
        prev = st["best_hash"]
    fin = srch["final"]
    held = held_text(srch)
    lines.append([("Final ", {"bold": True}),
                  (f"ET F1 {fin['f1']:.3f} (stability rule); {held}", {})])
    text(s, 0.6, 4.85, 6.3, 2.1, lines, size=11, color=INK2, spacing=1)
    footer(s, 6)
    notes(s, "Each stage keeps the best of the previous one. The all-files F1 is optimistic; the nested LOCO number "
             "re-runs the whole search without the held-out category and is what a new category would see.")


def s7_evaluation(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Event F1 counted at the alert trigger, on held-out categories",
          "No official evaluator was given, so we defined the metric and audited it against degenerate strategies")
    x0, y0, w = 0.6, 2.2, 5.7
    box(s, x0, y0, w, 0.06, fill="C3C2B7", shape=MSO_SHAPE.RECTANGLE)
    box(s, x0 + 0.6, y0 - 0.55, 1.6, 0.5, fill="FBD9CB", shape=MSO_SHAPE.RECTANGLE)
    box(s, x0 + 3.6, y0 - 0.55, 1.4, 0.5, fill="FBD9CB", shape=MSO_SHAPE.RECTANGLE)
    for xx in (x0 + 1.1, x0 + 2.75, x0 + 0.35):
        box(s, xx, y0 + 0.12, 0.05, 0.4, fill=RED, shape=MSO_SHAPE.RECTANGLE)
    text(s, x0 + 0.95, y0 + 0.5, 1.3, 0.4, "true alert", size=11, color=INK2)
    text(s, x0 + 2.45, y0 + 0.5, 1.2, 0.4, "false alert", size=11, color=INK2)
    text(s, x0 - 0.05, y0 + 0.5, 1.0, 0.4, "false", size=11, color=INK2)
    text(s, x0 + 3.6, y0 + 0.5, 1.6, 0.4, "missed window", size=11, color=INK2)
    text(s, x0, y0 + 0.9, w, 1.4, [
        [("Precision", {"bold": True}), (" = alerts triggered inside a window / alerts", {})],
        [("Recall", {"bold": True}), (" = windows with ≥ 1 alert trigger / windows", {})],
        [("Delay", {"bold": True}), (" = first trigger in the window − window start", {})],
    ], size=13, color=INK, spacing=4)
    aud = d.get("audit")
    if aud is not None:
        def val(k, col):
            r = aud[aud.strategy == k]
            return float(r[col].iloc[0]) if len(r) else float("nan")
        text(s, x0, 4.2, w, 1.0, [
            [("Why not overlap F1? ", {"bold": True}),
             (f"flagging every row scores {val('all_anomalous', 'E1_f1'):.2f} under an overlap event F1, "
              f"{val('all_anomalous', 'ET_f1'):.2f} under ours; random 1% of rows: "
              f"{val('random_1pct', 'ET_f1'):.2f}. Details: reports/metric_audit.md.", {})]], size=12, color=INK2)
    text(s, x0, 5.15, w, 0.4, "Nested leave-one-category-out (LOCO)", size=16, bold=True)
    cats = ["Art. none", "Art. anom.", "AWS", "AdExch.", "Known", "Traffic", "Tweets"]
    for i, c in enumerate(cats):
        held = i == 4
        box_text(s, x0 + i * 0.81, 5.6, 0.75, 0.5, [[(c, {"size": 10, "bold": held,
                                                           "color": WHITE if held else INK2})]],
                 fill=BLUE if held else TINT2, align=PP_ALIGN.CENTER, margin=0.03)
    text(s, x0, 6.15, w, 0.8, "The whole staged search runs on 6 categories; its pick is scored on the 7th. "
                              "Run for 3 of 7 categories (cost); one global config, no per-file tuning.",
         size=12, color=INK2)
    pc = d["m"]["per_category"]
    cats2 = [c for c in pc if pc[c]["trigger"]["n_windows"]]
    cats2.sort(key=lambda c: -pc[c]["trigger"]["f1"])
    short = {"artificialWithAnomaly": "Artificial", "realAWSCloudwatch": "AWS", "realAdExchange": "Ad exchange",
             "realKnownCause": "Known cause", "realTraffic": "Traffic", "realTweets": "Tweets"}
    labels = [f"{short.get(c, c)} ({pc[c]['trigger']['tp']}/{pc[c]['trigger']['n_windows']})" for c in cats2]
    bar_chart(s, 6.9, 1.5, 5.85, 4.3, labels, [round(pc[c]["trigger"]["f1"], 3) for c in cats2],
              [BLUE] * len(cats2), font_size=12, max_v=1.15)
    fa = pc.get("artificialNoAnomaly", {}).get("trigger", {})
    nfa = fa.get("fp", 0)
    text(s, 6.9, 5.8, 5.85, 0.9, f"Sentinel trigger-based event F1 per category (windows found / windows), "
                                  f"final config on all files. On the 5 no-anomaly files: {nfa} false "
                                  f"alert{'' if nfa == 1 else 's'}.", size=12, color=INK2)
    footer(s, 7)
    notes(s, "No point adjustment. An alert is judged where it starts: one that starts before the window and "
             "runs into it is a false alert and the window counts as missed, exactly as NAB scores it.")


def pub_note():
    """Published NAB detectors scored with ET at their own NAB threshold (results/nab_published.csv)."""
    p = RES / "nab_published.csv"
    if not p.exists():
        return ""
    pub = pd.read_csv(p).set_index("detector")
    names = {"contextOSE": "CAD OSE", "numenta": "HTM", "ARTime": "ARTime"}
    parts = [f"{v} {pub.loc[k, 'et_f1_at_nab_threshold']:.2f}" for k, v in names.items() if k in pub.index]
    return ("Published NAB detectors, ET F1 at their NAB threshold: " + ", ".join(parts) +
            "; not causal (NAB gives each file's min/max in advance) and thresholds tuned on all labels.")


def s8_results(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    m, srch, base, rt = d["m"]["overall"], d["search"], d["base"], d["rt"]
    t = m["trigger"]
    hv, hl = held_out(srch)
    ci = d["m"]["bootstrap"]["f1_ci95"]
    title(s, "Results: Sentinel vs baselines on the same metric and protocol")
    stat(s, 0.6, 1.3, 2.85, f"{t['f1']:.2f}", f"ET F1, all files (95% CI {ci[0]:.2f}–{ci[1]:.2f})", color=BLUE)
    stat(s, 3.6, 1.3, 2.85, f"{hv:.2f}", hl, color=BLUE)
    stat(s, 6.6, 1.3, 2.85, f"{t['precision']:.2f} / {t['recall']:.2f}",
         f"precision / recall ({t['tp']}/{t['n_windows']} windows, {t['fp']} false)")
    stat(s, 9.6, 1.3, 3.1, f"{rt['us_per_point']['mean']:.0f} µs", "per point, one CPU core")
    keep = base[base.detector != "null"].copy()
    best = keep[keep.detector != "sentinel"].sort_values("et_f1", ascending=False).groupby("detector").head(1)
    rows = [["Method", "ET F1", "Held-out", "Precision", "Recall", "False alerts", "E1 F1", "NAB"]]
    sen = keep[keep.detector == "sentinel"].iloc[0]
    nice = {"rolling_z": "Rolling z-score", "iforest": "Isolation Forest", "lof": "LOF", "ocsvm": "One-Class SVM",
            "random": "Random"}
    vnice = {"raw": "plain threshold", "event": "+ event logic", "calibrated": "+ our calibration & event logic"}

    def fmt(v):
        return "–" if pd.isna(v) else f"{v:.3f}"
    rows.append(["Sentinel", fmt(sen.et_f1), fmt(sen.cv_et_f1), fmt(sen.et_precision), fmt(sen.et_recall),
                 int(sen.false_alerts), fmt(sen.e1_f1), f"{sen.nab_official:.1f}"])
    for _, r in best.sort_values("et_f1", ascending=False).iterrows():
        rows.append([f"{nice.get(r.detector, r.detector)} ({vnice.get(r.variant, r.variant)})", fmt(r.et_f1),
                     fmt(r.cv_et_f1), fmt(r.et_precision), fmt(r.et_recall), int(r.false_alerts),
                     fmt(r.e1_f1), f"{r.nab_official:.1f}"])
    table(s, 0.6, 2.95, 12.1, rows, [4.1, 1.0, 1.1, 1.15, 1.0, 1.35, 1.0, 1.4], size=12, row_h=0.4,
          highlight_rows=(1,))
    text(s, 0.6, 2.95 + 0.4 * len(rows) + 0.1, 12.1, 0.9,
         "Best variant of each baseline (full table: results/baselines_comparison.csv). Held-out column: θ chosen on "
         "6 categories, scored on the 7th, for every method (Sentinel's other settings were tuned on all files). NAB = NAB standard score, every ANOMALOUS row a detection "
         "(checked against NAB's own code). " + pub_note(),
         size=10.5, color=MUTED)
    footer(s, 8)
    notes(s, "The fairest comparison is the row where a baseline gets our calibration and event logic with its "
             "threshold chosen the same way: the detectors themselves still matter.")


def s9_demo(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Every alert comes with evidence a person can check",
          "streamlit run app/streamlit_app.py  ·  python scripts/run_stream.py --input file.csv --output out.csv --events ev.jsonl")
    shot = FIG / "demo_streamlit.png"
    picture(s, shot if shot.exists() else FIG / "examples_tp.png", 0.6, 1.65, h=5.2)
    ev = d["evidence"]
    cards = [(e, "Correct alert", BLUE, None) for e in ev["tp"]]
    fp_note = ("This file is pure noise with no anomaly (NAB artificialNoAnomaly): the variance CUSUM "
               f"crossed θ on a chance run of wide swings. False alerts on the 5 anomaly-free files: {ev['n_fp_anomaly_free']}."
               if ev["fp"]["file"] == "artificialNoAnomaly/art_noisy.csv" else
               "Triggered outside every labelled window (see figures/examples_fp.png for context).")
    cards.append((ev["fp"], "False alert: why", RED, fp_note))
    y = 1.65
    for e, tag, col, note in cards:
        hgt = 1.35 if note is None else 1.85
        box(s, 7.05, y, 5.7, hgt, fill=TINT2)
        text(s, 7.2, y + 0.08, 5.4, 0.35,
             [[(tag, {"bold": True, "size": 12, "color": col}),
               (f"  ·  {e['file'].split('/')[1]}  ·  {e['start_timestamp']}", {"size": 11, "color": INK2})]])
        text(s, 7.2, y + 0.45, 5.4, 1.4, e["explanation"] if note is None
             else [e["explanation"], [(note, {"size": 11, "color": INK2, "italic": True})]], size=12, color=INK)
        y += hgt + 0.15
    footer(s, 9)
    notes(s, "Each alert record also keeps every detector's surprise at onset, the score trace, and whether the "
             "detector re-baselined afterwards (events.jsonl).")


def s10_end(prs, d):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, NAVY)
    title(s, "What breaks it, and what comes next", dark=True)
    fails = [("Looks like normal", "anomalies that resemble the usual seasonal pattern"),
             ("Bursty series", "Tweets/AWS: many unlabelled spikes look like labelled ones"),
             ("Regime switching", "CUSUM fires on series that change level often"),
             ("Labels", "some windows show no visible change; some clear events are unlabelled")]
    for i, (h, b) in enumerate(fails):
        y = 1.45 + i * 1.12
        box_text(s, 0.6, y, 5.9, 0.95, [[(h, {"bold": True, "size": 15, "color": WHITE})],
                                       [(b, {"size": 12, "color": "C9D1DE"})]], fill="1C2B45", margin=0.15)
    text(s, 7.0, 1.45, 5.7, 0.45, "Next", size=18, bold=True, color="8FB3E8")
    text(s, 7.0, 1.95, 5.7, 1.9, [
        "Multivariate streams (shared context across metrics)",
        "Learn more seasonal periods online, not only day/week",
        "Operator feedback to adapt θ per service",
        "Compare with time-series foundation models",
    ], size=14, color="D6DBE4", spacing=6)
    m = d["m"]["overall"]["trigger"]
    hv, hl = held_out(d["search"])
    text(s, 7.0, 4.15, 5.7, 0.45, "Conclusion", size=18, bold=True, color="8FB3E8")
    text(s, 7.0, 4.65, 5.7, 2.2, [
        f"Two causal detectors (residual + CUSUM) + calibration + event logic: trigger-based event F1 {m['f1']:.2f} "
        f"on NAB, all files; {hv:.2f} {hl.replace('ET F1, ', '')}.",
        "One alert per incident, each with a reason; tens of microseconds per point.",
        "Metric audited; every number reproducible from one command per step.",
    ], size=14, color=WHITE, spacing=8)
    footer(s, 10, dark=True)
    notes(s, "Be honest about NAB: labels are imperfect, so some false positives are real events and some "
             "misses have no visible signal. The held-out number is the one to expect on new data.")


def main():
    d = load()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for fn in (s1_title, s2_problem, s3_data, s4_design, s5_algorithm, s6_experiments, s7_evaluation,
               s8_results, s9_demo, s10_end):
        fn(prs, d)
    prs.core_properties.title = "Sentinel — Utkrishta 2026 Challenge 2"
    prs.core_properties.author = "Kommana Srinivasa Rao"
    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    print("wrote", OUT.relative_to(ROOT), "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
