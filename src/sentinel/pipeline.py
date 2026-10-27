"""StreamingSentinel: one instance per series; call update(timestamp, value) per point.

Causal by construction: the output at point t uses only points <= t.

Probation (first P points): points are stored, label is NORMAL, score 0. At the last
probation point the probation statistics (scale floor, robust level/scale, season mode)
are computed and the probation points are replayed through the detectors and
calibrators in order, so their state matches the chosen season mode. From point P on,
every point is scored, labelled and learned from (unless learning is paused).
"""
import math
from collections import deque
from typing import NamedTuple, Optional

import numpy as np

from .calibration import P99Calibrator, make_calibrator
from .config import load_config
from .detectors import ORDER, ChangeDetector, LofDetector, ResidualDetector, ShapeDetector
from .events import ANOMALOUS, COOLDOWN, NORMAL, PROBATION, EventMachine, PlainThreshold
from .evidence import explanation, fmt_ts, round_floats
from .fusion import FUSERS
from .preprocess import detect_season
from .robust import robust_stats, scale_floor


EVENT = "EVENT"                   # output state while an incident (event) is open


class PointResult(NamedTuple):
    anomaly_score: float          # in [0, 1]; see output.score in configs/default.yaml
    surprise: float               # smoothed fused surprise U-bar
    label: str                    # NORMAL / ANOMALOUS (alert rows: first label_len rows of an event)
    top_detector: str             # residual / change / shape / none
    event_id: Optional[int]       # set on every row of an open event
    state: str                    # PROBATION / NORMAL / EVENT / COOLDOWN
    detector_surprise: tuple      # U_k per enabled detector


def to_seconds(ts):
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return float(ts)
    return float(np.datetime64(ts, "s").astype(np.int64))


def _skew(x):
    sd = x.std()
    return float(np.mean((x - x.mean()) ** 3) / sd ** 3) if sd > 0 else 0.0


def _median(vals):
    v = sorted(vals)
    n = len(v)
    return 0.5 * (v[(n - 1) // 2] + v[n // 2]) if n else 0.0


class StreamingSentinel:
    def __init__(self, config=None, probation_len=None):
        c = self.cfg = config if config is not None else load_config()
        p = probation_len if probation_len is not None else c["probation"]["default_len"]
        self.P = max(int(p), 2)
        self.names = [n for n in ORDER if c.get(n, {}).get("enabled")]
        ev = c["events"]
        self.event_logic = ev["enabled"]
        # label_len: how many rows of each event are labelled ANOMALOUS, starting at the
        # trigger row (None = every row of the event, the v1 behaviour). The event itself
        # (pause, re-baseline, cooldown, evidence) is unchanged; see reports/metric_audit.md.
        ll = ev.get("label_len")
        self.label_len = None if ll is None else max(int(ll), 1)
        if self.event_logic:
            ro = ev.get("reenter_offset")       # optional: re-trigger during cooldown
            self.machine = EventMachine(ev["theta_enter"], ev["theta_enter"] - ev["exit_offset"],
                                        ev["cooldown"], ev["max_len"],
                                        reenter=None if ro is None else ev["theta_enter"] + ro)
            self.pause, self.rebase = ev["pause"], ev["rebaseline"]
        else:
            self.machine = PlainThreshold(ev["theta_enter"])
            self.pause = self.rebase = False
        self.rb_sigma = ev["rebaseline_sigma"]
        self.fuse = FUSERS[c["fusion"]["method"]]
        self.u_hist = deque(maxlen=c["fusion"]["smooth"])
        self.peak_penalty = (math.log10(c["fusion"]["smooth"])
                             if c["fusion"].get("smooth_mode", "mean") == "mean_or_peak" else None)
        self.learn_states = ((NORMAL, COOLDOWN) if c["calibration"].get("learn_in_cooldown")
                             else (NORMAL,))
        self.recent = deque(maxlen=max(ev["rebaseline_r"], 1))
        self.score_mode = c.get("output", {}).get("score", "surprise")
        self.i = 0
        self.state = PROBATION
        self._prob_sec, self._prob_x = [], []
        self.prev_sec = None
        self.nan_count = 0
        self.events = []
        self.current = None
        self.event_id = 0
        self.season = {"mode": "none", "dt_seconds": None}
        self.log1p = False
        self.dets, self.cals = [], []
        self.last = PointResult(0.0, 0.0, NORMAL, "none", None, PROBATION, (0.0,) * len(self.names))

    # ---------------------------------------------------------------- streaming
    def update(self, timestamp, value):
        sec = to_seconds(timestamp)
        x = float(value)
        i = self.i
        self.i += 1
        if self.state == PROBATION:
            if x == x:
                self._prob_sec.append(sec)
                self._prob_x.append(x)
            self.prev_sec = sec
            if i == self.P - 1:
                self._finish_probation()
            return self.last
        if x != x:                                   # NaN: carry state, no learning
            self.nan_count += 1
            self.last = self.last._replace(label=NORMAL, event_id=None)
            return self.last
        if self.log1p:
            return self._step(i, sec, math.log1p(max(x, 0.0)), raw_x=x)
        return self._step(i, sec, x)

    def finish(self):
        """Close an open event at the end of the stream; return all event records."""
        if self.current is not None:
            self._close_event(self.i, self.current["_end_sec"], at_end=True)
        return self.events

    # ---------------------------------------------------------------- internals
    def _finish_probation(self):
        c = self.cfg
        secs = np.asarray(self._prob_sec) if self._prob_sec else np.zeros(1)
        xs = np.asarray(self._prob_x) if self._prob_x else np.zeros(1)
        self.season = detect_season(secs, xs, c["season"]["acf_min"])
        self.dt = self.season["dt_seconds"]
        tr = c.get("transform", {})
        self.log1p = bool(tr.get("log1p")) and xs.min() >= 0 and _skew(xs) > tr.get("skew_min", 2.0)
        if self.log1p:
            xs = np.log1p(xs)
        floor = scale_floor(xs)
        med, rs = robust_stats(xs)
        rs = max(rs, floor)
        vmean = float(np.mean(np.log(((xs - med) / rs) ** 2 + 0.01)))
        self.d1 = self.d2 = None
        for name in self.names:
            if name == "residual":
                d = self.d1 = ResidualDetector(c["residual"], floor, self.season["mode"],
                                               recent=c["events"]["rebaseline_r"])
            elif name == "change":
                d = self.d2 = ChangeDetector(c["change"], med, rs, vmean, floor)
            elif name == "shape":
                d = ShapeDetector(c["shape"], floor)
            else:
                d = LofDetector(c["lof"], xs)
            self.dets.append(d)
            self.cals.append(make_calibrator(c["calibration"]))
        if (c["change"].get("deseasonalize") and self.d1 is not None and self.d2 is not None
                and self.season["mode"] != "none"):
            self.d2.deseasonalize_with(self.d1)
        for j, (s, x) in enumerate(zip(secs, xs)):
            for d, cal in zip(self.dets, self.cals):
                ready = d.ready
                raw = d.score(x, s, j)
                if ready:
                    cal.add(raw)
                d.update(x)
            self.recent.append(x)
        for cal in self.cals:
            if isinstance(cal, P99Calibrator):
                cal.freeze()
        self._prob_sec = self._prob_x = None
        self.state = NORMAL

    def _step(self, i, sec, x, raw_x=None):
        gap = self.prev_sec is not None and sec - self.prev_sec > 3 * self.dt
        self.prev_sec = sec
        raws, us = [], []
        for d, cal in zip(self.dets, self.cals):
            ready = d.ready
            r = d.score(x, sec, i)
            raws.append(r if ready else None)
            us.append(cal.surprise(r) if ready else 0.0)
        u, k = self.fuse(us)
        self.u_hist.append(u)
        ubar = sum(self.u_hist) / len(self.u_hist)
        if self.peak_penalty is not None:           # sustained OR one extreme point
            ubar = max(ubar, u - self.peak_penalty)
        self.recent.append(x)

        state, started, ended = self.machine.step(ubar)
        self.state = state
        if ended:
            self._close_event(i, self.current["_end_sec"])
        if started:
            self._open_event(i, sec, us, k, gap, x if raw_x is None else raw_x)
        if state == ANOMALOUS:
            cur = self.current
            cur["_end_sec"] = sec
            cur["trace"].append(round(ubar, 3))
            if ubar > cur["peak_surprise"]:
                cur["peak_surprise"] = ubar

        if not (self.pause and state == ANOMALOUS):
            for d in self.dets:
                d.update(x)
            if state in self.learn_states:
                for cal, r in zip(self.cals, raws):
                    if r is not None:
                        cal.add(r)

        in_event = state == ANOMALOUS
        alert = in_event and (self.label_len is None or not self.event_logic
                              or self.machine.length <= self.label_len)
        self.last = PointResult(
            self._score(ubar, alert), ubar,
            ANOMALOUS if alert else NORMAL,
            self.names[k] if u > 0 else "none",
            self.event_id if in_event else None,
            EVENT if in_event else state, tuple(us))
        return self.last

    def _score(self, ubar, alert):
        """'surprise': 1 - 10^-U. 'alert': same value squeezed into [0, 0.5) on normal rows and
        [0.5, 1] on alert rows, so a threshold sweep over the score at 0.5 reproduces the labels
        and lower thresholds add further candidates in surprise order."""
        s = 1.0 - 10.0 ** (-ubar)
        if self.score_mode == "alert":
            return 0.5 + 0.5 * s if alert else 0.5 * s
        return s

    def _open_event(self, i, sec, us, k, gap, raw_x):
        self.event_id += 1
        evid = {n: d.evidence() for n, d in zip(self.names, self.dets)}
        if self.log1p:                              # report values in original units
            if "residual" in evid:
                e = evid["residual"]
                e["observed"] = raw_x
                for key in ("expected", "level"):
                    e[key] = math.expm1(e[key])
                e["seasonal_offset"] = e["expected"] - e["level"]
                e["residual"] = raw_x - e["expected"]
            if "change" in evid:
                e = evid["change"]
                for key in ("baseline_mean", "recent_mean"):
                    if e[key] is not None:
                        e[key] = math.expm1(e[key])
        if "shape" in evid:
            cal = self.cals[self.names.index("shape")]
            evid["shape"]["typical_p95"] = cal.typical_p95()
            evid["shape"]["nn_timestamp"] = fmt_ts(evid["shape"].pop("nn_start_seconds"))
        dsur = dict(zip(self.names, us))
        if self.d1 is not None:
            level, sigma = self.d1.current_level(), self.d1.current_sigma()
        elif self.d2 is not None:
            level, sigma = self.d2.mu, max(self.d2.floor, self.d2.scale.std())
        else:
            level = sigma = None
        self.current = {
            "event_id": self.event_id,
            "start_index": i,
            "start_timestamp": fmt_ts(sec),
            "top_detector": self.names[k],
            "detector_surprise": dsur,
            "evidence": evid,
            "context": {"season_mode": self.season["mode"], "data_gap_before": bool(gap),
                        "transform": "log1p" if self.log1p else "none"},
            "explanation": explanation(dsur, evid),
            "peak_surprise": 0.0,
            "trace": [],
            "_end_sec": sec,
            "_level_before": level,
            "_sigma_before": sigma,
        }

    def _close_event(self, i, end_sec, at_end=False):
        ev = self.current
        rebased = False
        if self.event_logic and not at_end:
            if self.rebase:
                rec = list(self.recent)
                lvl, sig = ev["_level_before"], ev["_sigma_before"]
                now = self.d1.recent_level() if self.d1 is not None else _median(rec)
                shifted = lvl is not None and abs(now - lvl) > self.rb_sigma * sig
                if ev["top_detector"] == "change" or shifted:
                    for d in self.dets:
                        d.rebaseline(rec)
                    rebased = True
            if self.d2 is not None:
                self.d2.reset_cusum()
        out = {k: v for k, v in ev.items() if not k.startswith("_") and k != "trace"}
        out.update({
            "end_index": i - 1,
            "end_timestamp": fmt_ts(end_sec),
            "length_points": i - ev["start_index"],
            "peak_score": 1.0 - 10.0 ** (-ev["peak_surprise"]),
            "rebaselined": rebased,
            "trace": ev["trace"],
        })
        self.events.append(round_floats(out))
        self.current = None
