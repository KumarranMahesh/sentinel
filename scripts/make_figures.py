"""All figures for README and slides (PLAN.md T1.7, T8.1). 300 dpi PNG in figures/.

  python scripts/make_figures.py [--only eda,examples,baselines,ablation,tuning,delays,runtime,categories]

Colors: series grey, labelled window light orange, Sentinel alert red, Sentinel bars blue,
other methods grey.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.config import ROOT, load_config  # noqa: E402
from sentinel.io import categories, load_corpus  # noqa: E402
from sentinel.metrics.windows import events_from_labels, overlaps, scored_windows  # noqa: E402
from sentinel.preprocess import detect_season  # noqa: E402
from sentinel.runner import parallel_map, run_sentinel  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED, OTHER = "#2a78d6", "#eb6834", "#e34948", "#c3c2b7"
FIG = ROOT / "figures"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    path = FIG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def shade(ax, s, labels=None, upto=None):
    ax.axvspan(s.timestamps[0], s.timestamps[s.probation - 1], color=AXIS, alpha=0.25, lw=0)
    for a, b in s.windows:
        ax.axvspan(s.timestamps[a], s.timestamps[b], color=ORANGE, alpha=0.18, lw=0)
    if labels is not None:
        for a, b in events_from_labels(labels, s.probation):
            ax.axvspan(s.timestamps[a], s.timestamps[b], color=RED, alpha=0.45, lw=0)


# ------------------------------------------------------------------ EDA
def eda(corpus, acf_min=0.3):
    rows = []
    for name, s in corpus.items():
        p = s.probation
        info = detect_season(s.seconds[:p], s.values[:p], acf_min)
        v = s.values
        win_len = [b - a + 1 for a, b in s.windows]
        pos = [(a + b) / 2 / len(s) for a, b in s.windows]
        rows.append({
            "file": name, "category": s.category, "points": len(s), "probation": p,
            "start": str(s.timestamps[0])[:19], "end": str(s.timestamps[-1])[:19],
            "dt_seconds": info["dt_seconds"],
            "duplicate_timestamps": int(len(s) - len(np.unique(s.timestamps))),
            "missing_values": int(np.isnan(v).sum()), "min": float(np.nanmin(v)),
            "max": float(np.nanmax(v)),
            "skewness": float(pd.Series(v).skew()), "season_mode": info["mode"],
            "anomaly_labels": len(s.label_idx), "windows": len(s.windows),
            "mean_window_points": float(np.mean(win_len)) if win_len else 0.0,
            "frac_points_in_windows": float(sum(win_len) / len(s)),
            "window_positions": ";".join(f"{x:.2f}" for x in pos),
        })
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "dataset_stats.csv", index=False)
    cat = df.groupby("category").agg(files=("file", "count"), points=("points", "sum"),
                                     windows=("windows", "sum"),
                                     mean_window_points=("mean_window_points", "mean"),
                                     frac_in_windows=("frac_points_in_windows", "mean"),
                                     seasonal_files=("season_mode", lambda x: int((x != "none").sum())))
    cat.to_csv(ROOT / "results" / "dataset_stats_by_category.csv")
    print(cat.to_string())

    picks = {"artificialWithAnomaly": "artificialWithAnomaly/art_daily_jumpsdown.csv",
             "artificialNoAnomaly": "artificialNoAnomaly/art_daily_small_noise.csv",
             "realAWSCloudwatch": "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
             "realAdExchange": "realAdExchange/exchange-3_cpc_results.csv",
             "realKnownCause": "realKnownCause/machine_temperature_system_failure.csv",
             "realTraffic": "realTraffic/speed_7578.csv",
             "realTweets": "realTweets/Twitter_volume_GOOG.csv"}
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 11))
    for ax, (c, name) in zip(axes, sorted(picks.items())):
        s = corpus[name]
        shade(ax, s)
        ax.plot(s.timestamps, s.values, color=INK2, lw=0.6)
        ax.set_title(f"{c}  ·  {name.split('/')[1]}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(False)
    fig.suptitle("One series per NAB category  (grey = probation, orange = labelled anomaly window)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "eda/category_samples.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    lens = [b - a + 1 for s in corpus.values() for a, b in s.windows]
    ax.hist(lens, bins=25, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("window length (points)")
    ax.set_ylabel("windows")
    ax.set_title(f"NAB anomaly windows: {len(lens)} windows, median {int(np.median(lens))} points")
    save(fig, "eda/window_lengths.png")


# ------------------------------------------------------------------ examples
def _run(s, cfg):
    r = run_sentinel(s, cfg)
    return {k: r[k] for k in ("surprise", "label", "events", "detector_surprise", "detector_names")}


def examples(corpus, cfg):
    res = dict(zip(corpus, parallel_map(_run, [(s, cfg) for s in corpus.values()])))
    tps, fps, fns = [], [], []
    for name, s in corpus.items():
        gt = scored_windows(s.windows, s.probation)
        r = res[name]
        for e in r["events"]:
            span = (e["start_index"], e["end_index"])
            (tps if any(overlaps(span, w) for w in gt) else fps).append((e["peak_surprise"], name, e))
        evs = events_from_labels(r["label"], s.probation)
        for w in gt:
            if not any(overlaps(w, p) for p in evs):
                fns.append((float(r["surprise"][w[0]:w[1] + 1].max()), name, w))

    def pick(items, k):
        out, seen = [], set()
        for it in sorted(items, key=lambda t: -t[0]):
            cat = it[1].split("/")[0]
            if cat not in seen:
                out.append(it)
                seen.add(cat)
            if len(out) == k:
                break
        return out

    th = cfg["events"]["theta_enter"]
    th_exit = th - cfg["events"]["exit_offset"]

    def panel(items, title, fname, kind):
        fig, axes = plt.subplots(2 * len(items), 1, figsize=(11, 3.3 * len(items)),
                                 gridspec_kw={"height_ratios": [2, 1] * len(items)})
        for k, (_, name, obj) in enumerate(items):
            s, r = corpus[name], res[name]
            a0, a1 = axes[2 * k], axes[2 * k + 1]
            if kind == "fn":
                c = (obj[0] + obj[1]) // 2
            else:
                c = (obj["start_index"] + obj["end_index"]) // 2
            lo, hi = max(0, c - 700), min(len(s), c + 700)
            sl = slice(lo, hi)
            for a, b in s.windows:
                if b >= lo and a < hi:
                    a0.axvspan(s.timestamps[max(a, lo)], s.timestamps[min(b, hi - 1)], color=ORANGE, alpha=0.18, lw=0)
            for a, b in events_from_labels(r["label"], s.probation):
                if b >= lo and a < hi:
                    a0.axvspan(s.timestamps[max(a, lo)], s.timestamps[min(b, hi - 1)], color=RED, alpha=0.45, lw=0)
            a0.plot(s.timestamps[sl], s.values[sl], color=INK2, lw=0.8)
            a0.grid(False)
            text = obj["explanation"] if kind != "fn" else (
                f"Missed window: highest surprise inside was {_:.1f} (θ_enter {th})")
            a0.set_title(f"{name}\n{text}", fontsize=8.5, fontweight="normal")
            a1.plot(s.timestamps[sl], r["surprise"][sl], color=BLUE, lw=0.8)
            a1.axhline(th, color=RED, lw=0.8)
            a1.axhline(th_exit, color=RED, lw=0.8, ls=":")
            a1.set_ylabel("surprise", fontsize=8)
            a1.set_ylim(0, max(th * 1.6, float(np.nanmax(r["surprise"][sl])) * 1.05))
            for a in (a0, a1):
                a.tick_params(labelsize=7)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout()
        save(fig, fname)

    panel(pick(tps, 3), "Correct alerts (red) inside labelled windows (orange), with evidence text",
          "examples_tp.png", "tp")
    panel(pick(fps, 3), "False alerts: no labelled window nearby", "examples_fp.png", "fp")
    panel(pick(fns, 2), "Missed windows", "examples_fn.png", "fn")
    events = [e for r in res.values() for e in r["events"]]
    return res, events


# ------------------------------------------------------------------ results charts
def hbar(ax, labels, values, highlight, errors=None, fmt="{:.3f}"):
    y = np.arange(len(labels))[::-1]
    colors = [BLUE if h else OTHER for h in highlight]
    ax.barh(y, values, color=colors, height=0.62)
    if errors is not None:
        lo, hi = errors
        ax.errorbar(values, y, xerr=[np.array(values) - lo, hi - np.array(values)], fmt="none",
                    ecolor=INK2, elinewidth=0.8, capsize=2)
    right = values if errors is None else np.maximum(values, errors[1])
    for yi, v, r, h in zip(y, values, right, highlight):
        ax.text(r * (1.15 if ax.get_xscale() == "log" else 1) + (0 if ax.get_xscale() == "log" else 0.008),
                yi, fmt.format(v), va="center", fontsize=8,
                color=INK if h else INK2, fontweight="bold" if h else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.grid(axis="y", visible=False)


def baselines_chart():
    df = pd.read_csv(ROOT / "results" / "baselines_comparison.csv", keep_default_na=False,
                     na_values=[""])
    df = df[(df.detector != "null") & (df.event_f1 > 0)].copy()
    df["label"] = df.detector + " · " + df.variant
    df = df.sort_values("event_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.2))
    hbar(ax, df.label.tolist(), df.event_f1.to_numpy(), (df.detector == "sentinel").tolist(),
         (df.f1_ci95_low.to_numpy(), df.f1_ci95_high.to_numpy()))
    ax.set_xlabel("event F1, all 58 files (whiskers: 95% bootstrap CI over files)")
    ax.set_title("Sentinel vs baselines  (raw = plain threshold, event = + our event logic, "
                 "calibrated = + our calibration and event logic)", fontsize=9)
    ax.set_xlim(0, max(0.8, df.f1_ci95_high.max() + 0.08))
    save(fig, "baselines.png")


def ablation_chart():
    df = pd.read_csv(ROOT / "results" / "ablation.csv")
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(df) + 1.2))
    hbar(ax, df.variant.tolist(), df.event_f1.to_numpy(), [v.startswith("6") for v in df.variant])
    ax.set_xlabel("event F1, all files (each variant with its own best θ_enter)")
    ax.set_title("Ablation: what each part adds")
    ax.set_xlim(0, df.event_f1.max() + 0.12)
    save(fig, "ablation.png")


def tuning_chart():
    df = pd.read_csv(ROOT / "experiments" / "runs.csv")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(df))
    ax.scatter(x, df.all_event_f1, s=9, color=OTHER, linewidths=0, label="one config")
    ax.step(x, np.maximum.accumulate(df.all_event_f1.to_numpy()), where="post", color=BLUE,
            lw=2, label="best so far")
    starts = df.groupby("stage", sort=False).head(1).index
    for k, i in enumerate(starts):
        ax.axvline(i, color=AXIS, lw=0.8)
        if k == 0:
            continue
        ax.text(i + 1.5, 1.01 + 0.06 * (k % 2), df.stage[i].replace("stage", "S").split("_")[0],
                fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform())
    ax.set_xlabel("evaluated configs, in search order")
    ax.set_ylabel("event F1 (all files)")
    ax.set_title("Tuning trajectory (staged grid search)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Tuning trajectory (staged grid search)", pad=22)
    save(fig, "tuning.png")


def delays_chart(corpus, res):
    d = []
    for name, s in corpus.items():
        lab = res[name]["label"]
        for a, b in scored_windows(s.windows, s.probation):
            hit = np.flatnonzero(lab[a:b + 1])
            if len(hit):
                d.append(hit[0] / (b - a + 1))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(d, bins=20, range=(0, 1), color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0.5, color=INK2, lw=0.8)
    ax.text(0.51, ax.get_ylim()[1] * 0.92, "labelled anomaly point\n(window centre)", fontsize=7.5, color=INK2)
    ax.set_xlabel("first alert position inside window (0 = window start, 1 = end)")
    ax.set_ylabel("detected windows")
    ax.set_title(f"Detection delay, {len(d)} detected windows (median {np.median(d):.2f})")
    save(fig, "delays.png")


def runtime_chart():
    rt = json.loads((ROOT / "results" / "runtime.json").read_text())
    items = [("Sentinel · mean", rt["us_per_point"]["mean"], True),
             ("Sentinel · p99", rt["us_per_point"]["p99"], True)]
    items += [(f"{k} · single-point score", v, False) for k, v in rt["baseline_single_point_us"].items()]
    items.sort(key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(7, 3))
    hbar(ax, [t[0] for t in items], [t[1] for t in items], [t[2] for t in items], fmt="{:.0f} µs")
    ax.set_xscale("log")
    ax.set_xlim(1, max(t[1] for t in items) * 8)
    ax.set_xlabel("microseconds per streaming point (log scale)")
    ax.set_title(f"Runtime per point · {rt['machine'].get('cpu', '')[:40]}")
    save(fig, "runtime.png")


def categories_chart():
    m = json.loads((ROOT / "results" / "metrics_summary.json").read_text())
    pc = m["per_category"]
    cats = [c for c in pc if pc[c]["n_windows"]]
    cats.sort(key=lambda c: -pc[c]["f1"])
    fig, ax = plt.subplots(figsize=(7, 3))
    hbar(ax, [f"{c} ({pc[c]['tp_gt']}/{pc[c]['n_windows']} windows, {pc[c]['fp']} FP)" for c in cats],
         [pc[c]["f1"] for c in cats], [True] * len(cats))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("event F1")
    ax.set_title(f"Sentinel per category (overall F1 {m['overall']['f1']:.3f})")
    save(fig, "per_category.png")


def walkthrough(corpus, cfg, name="realKnownCause/machine_temperature_system_failure.csv"):
    """How each detector sees one series: value, surprise of D1/D2/D3, fused surprise."""
    s = corpus[name]
    r = run_sentinel(s, cfg)
    names = r["detector_names"]
    rows = 2 + len(names)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 1.55 * rows + 0.6), sharex=True)
    shade(axes[0], s, r["label"])
    axes[0].plot(s.timestamps, s.values, color=INK2, lw=0.6)
    axes[0].set_ylabel("value", fontsize=8)
    titles = {"residual": "D1 residual", "change": "D2 change (CUSUM)", "shape": "D3 shape (NN distance)"}
    for k, n in enumerate(names):
        ax = axes[1 + k]
        ax.plot(s.timestamps, r["detector_surprise"][:, k], color=BLUE, lw=0.6)
        ax.set_ylabel(titles[n].split(" (")[0], fontsize=8)
        ax.text(0.005, 0.85, f"{titles[n]} surprise", transform=ax.transAxes, fontsize=8, color=INK2)
    ax = axes[-1]
    ax.plot(s.timestamps, r["surprise"], color=BLUE, lw=0.6)
    th = cfg["events"]["theta_enter"]
    ax.axhline(th, color=RED, lw=0.8)
    ax.axhline(th - cfg["events"]["exit_offset"], color=RED, lw=0.8, ls=":")
    ax.set_ylabel("fused", fontsize=8)
    ax.text(0.005, 0.85, "fused surprise Ū with θ_enter (solid) and θ_exit (dotted)",
            transform=ax.transAxes, fontsize=8, color=INK2)
    for a in axes:
        a.tick_params(labelsize=7)
        a.grid(False)
    fig.suptitle(f"Detector walkthrough · {name}  (orange = labelled window, red = alert)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "walkthrough.png")


def hero(corpus, cfg, name="realKnownCause/machine_temperature_system_failure.csv"):
    """Title-slide image: one series on a dark background, windows orange, alerts red."""
    s = corpus[name]
    r = run_sentinel(s, cfg)
    dark = "#0f1b2d"
    fig, ax = plt.subplots(figsize=(12, 2.4), facecolor=dark)
    ax.set_facecolor(dark)
    for a, b in s.windows:
        ax.axvspan(a, b, color=ORANGE, alpha=0.55, lw=0)
    for a, b in events_from_labels(r["label"], s.probation):
        ax.axvspan(a, b, color=RED, alpha=0.75, lw=0)
    ax.plot(np.arange(len(s)), s.values, color="#d6dbe4", lw=0.7)
    ax.axis("off")
    ax.margins(x=0)
    path = FIG / "hero.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=dark, pad_inches=0.05)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def alert_storm(corpus, cfg, name="realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv"):
    """Same series: plain rolling-z threshold vs Sentinel."""
    from sentinel.baselines import baseline_labels
    s = corpus[name]
    _, z_lab = baseline_labels("rolling_z", s, "raw")
    r = run_sentinel(s, cfg)
    rows = [("Rolling z-score, |z| > 3 (plain threshold)", z_lab),
            ("Sentinel (calibrated surprise + event logic)", r["label"])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    for ax, (title, lab) in zip(axes, rows):
        n_ev = len(events_from_labels(lab, s.probation))
        shade(ax, s, lab)
        ax.plot(s.timestamps, s.values, color=INK2, lw=0.6)
        ax.set_title(f"{title}: {n_ev} alert{'s' if n_ev != 1 else ''}", fontsize=10)
        ax.grid(False)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{name.split('/')[1]}  ·  orange = labelled window, red = alert",
                 x=0.01, ha="left", fontsize=9, color=INK2, fontweight="normal")
    fig.tight_layout()
    save(fig, "alert_storm.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    def want(x):
        return only is None or x in only

    corpus = load_corpus()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    if want("eda"):
        eda(corpus, cfg["season"]["acf_min"])
    if want("walkthrough"):
        walkthrough(corpus, cfg)
    if want("hero"):
        hero(corpus, cfg)
    if want("storm"):
        alert_storm(corpus, cfg)
    res = None
    if want("examples") or want("delays"):
        res, _ = examples(corpus, cfg)
    if want("delays"):
        delays_chart(corpus, res)
    for key, fn in (("baselines", baselines_chart), ("ablation", ablation_chart),
                    ("tuning", tuning_chart), ("runtime", runtime_chart),
                    ("categories", categories_chart)):
        if want(key):
            try:
                fn()
            except FileNotFoundError as e:
                print(f"skip {key}: {e.filename} missing")


if __name__ == "__main__":
    main()
