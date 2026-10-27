"""Compiled engine (Numba) for the submitted detector set: same algorithm as pipeline.py.

  from sentinel.fast import run_fast, supports
  res = run_fast(series, cfg)          # same dict as runner.run_sentinel

Supported: D1 residual + D2 change (+ optional D7 subspace), tail calibration, max-Bonferroni
fusion with mean smoothing, the full event machine (hysteresis, max length, cooldown,
optional re-entry), pause, re-baseline, label_len, output.score. Anything else (log1p/counts
transforms, other detectors, Fisher fusion, ...) -> `supports(cfg)` is False and callers fall
back to the reference Python engine. The numeric core runs in one compiled loop over the
series; the per-event evidence is rebuilt in Python from state snapshots taken at the trigger
row, with the same expressions as the reference, so events.jsonl matches it.
tests/test_fast.py checks labels, states, events and scores against pipeline.py.
"""
import math
import sys

import numpy as np
from numba import njit

from .config import deep_merge
from .detectors import ORDER
from .evidence import explanation, fmt_ts, round_floats
from .preprocess import detect_season
from .robust import robust_stats, scale_floor

MEANABS_TO_STD = 1.2533
P_MIN = 1e-12
EPS = 1e-9
ST_PROB, ST_NORMAL, ST_ANOM, ST_COOL = 0, 1, 2, 3
STATE_NAMES = np.array(["PROBATION", "NORMAL", "EVENT", "COOLDOWN"], dtype=object)


def supports(cfg):
    names = [n for n in ORDER if cfg.get(n, {}).get("enabled")]
    tr = cfg.get("transform", {})
    return (names[:2] == ["residual", "change"] and set(names[2:]) <= {"subspace"}
            and not tr.get("log1p") and not tr.get("counts")
            and cfg["fusion"]["method"] == "max_bonferroni"
            and cfg["fusion"].get("smooth_mode", "mean") == "mean"
            and cfg["calibration"].get("mode", "tail") == "tail"
            and cfg["change"].get("cap") is None and not cfg["change"].get("rebaseline_scale", False)
            and cfg["events"]["enabled"])


# ----------------------------------------------------------------------------- helpers
@njit(cache=True)
def _pw_leaf(a, lo, n):
    """numpy pairwise_sum for n <= 128 (8 accumulators)."""
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    m = n - (n % 8)
    while i < m:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(cache=True)
def _pairwise_sum(a, lo, n):
    """numpy's pairwise summation (float64, contiguous), without recursion: the split tree is
    walked in post-order with an explicit stack, so the additions happen in numpy's order."""
    if n <= 128:
        return _pw_leaf(a, lo, n)
    st_lo = np.empty(64, np.int64)
    st_n = np.empty(64, np.int64)
    st_state = np.empty(64, np.int64)       # 0: left pending, 1: right pending, 2: combine
    st_left = np.empty(64)
    vals = np.empty(64)
    top = 0
    st_lo[0], st_n[0], st_state[0] = lo, n, 0
    ret = 0.0
    while top >= 0:
        cl, cn, cs = st_lo[top], st_n[top], st_state[top]
        if cn <= 128:
            ret = _pw_leaf(a, cl, cn)
            top -= 1
            if top >= 0:
                if st_state[top] == 0:
                    st_left[top] = ret
                    st_state[top] = 1
                else:
                    vals[top] = st_left[top] + ret
                    st_state[top] = 2
            continue
        n2 = cn // 2
        n2 -= n2 % 8
        if cs == 0:
            top += 1
            st_lo[top], st_n[top], st_state[top] = cl, n2, 0
        elif cs == 1:
            top += 1
            st_lo[top], st_n[top], st_state[top] = cl + n2, cn - n2, 0
        else:
            ret = vals[top]
            top -= 1
            if top >= 0:
                if st_state[top] == 0:
                    st_left[top] = ret
                    st_state[top] = 1
                else:
                    vals[top] = st_left[top] + ret
                    st_state[top] = 2
    return ret


@njit(cache=True)
def _fsum_py(vals, n, compensated):
    """CPython sum() of floats: Neumaier-compensated since 3.12, plain left-to-right before."""
    f = 0.0
    c = 0.0
    if not compensated:
        for j in range(n):
            f += vals[j]
        return f
    for j in range(n):
        x = vals[j]
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c != 0.0 and math.isfinite(c):
        f += c
    return f


@njit(cache=True)
def _sorted_insert(arr, n, x):
    lo, hi = 0, n
    while lo < hi:                      # bisect_right
        mid = (lo + hi) // 2
        if arr[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    for j in range(n, lo, -1):
        arr[j] = arr[j - 1]
    arr[lo] = x


@njit(cache=True)
def _sorted_remove(arr, n, x):
    lo, hi = 0, n
    while lo < hi:                      # bisect_left
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    for j in range(lo, n - 1):
        arr[j] = arr[j + 1]


@njit(cache=True)
def _bisect_left(arr, n, x):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _quantile_sorted(s, n, q):
    """numpy.quantile(v, q), method 'linear', from the sorted values s[:n]."""
    vi = (n - 1) * q
    if vi >= n - 1:
        return s[n - 1]
    prev = math.floor(vi)
    a = s[int(prev)]
    b = s[int(prev) + 1]
    t = vi - prev
    d = b - a
    if t >= 0.5:
        return b - d * (1 - t)
    return a + d * t


# ----------------------------------------------------------------------------- state
# rolling median (D1 level): ring rm_buf[W] (deque order via rm_head/rm_n), sorted rm_sorted[W+1]
# calibrator k: ring cb[k, C], cpos[k], cn[k]; sorted cs[k, C+1]; cq[k], cme[k], csince[k], cqok[k]

@njit(cache=True)
def _rm_add(rm_buf, rm_sorted, st, W, x):
    head, n = st[0], st[1]
    _sorted_insert(rm_sorted, n, x)
    if n < W:
        rm_buf[(head + n) % W] = x
        st[1] = n + 1
    else:
        old = rm_buf[head]
        rm_buf[head] = x
        st[0] = (head + 1) % W
        _sorted_remove(rm_sorted, n + 1, old)


@njit(cache=True)
def _rm_median(rm_sorted, n):
    return 0.5 * (rm_sorted[(n - 1) // 2] + rm_sorted[n // 2])


@njit(cache=True)
def _cal_recalc(k, cb, cs, cn, cq, cme, csince, cqok, tail_q, tmp):
    n = cn[k]
    q = _quantile_sorted(cs[k], n, tail_q)
    m = 0
    for j in range(n):
        v = cb[k, j]
        if v > q:
            tmp[m] = v - q
            m += 1
    me = _pairwise_sum(tmp, 0, m) / m if m > 0 else 0.0
    cq[k] = q
    cme[k] = me if me > EPS else EPS
    csince[k] = 0
    cqok[k] = True


@njit(cache=True)
def _cal_surprise(k, s, cb, cs, cn, cq, cme, csince, cqok, tail_q, min_exceed, min_tail_n,
                  recalc_every, tmp):
    n = cn[k]
    if n == 0:
        return 0.0                       # p = 1
    if (not cqok[k]) or csince[k] >= recalc_every or n < 2 * recalc_every:
        _cal_recalc(k, cb, cs, cn, cq, cme, csince, cqok, tail_q, tmp)
    n_exceed = n - _bisect_left(cs[k], n, s)
    q = cq[k]
    if s <= q or n_exceed >= min_exceed or n < min_tail_n:
        p = (1 + n_exceed) / (1 + n)
    else:
        p = (1 - tail_q) * math.exp(-(s - q) / cme[k])
    return -math.log10(max(p, P_MIN))


@njit(cache=True)
def _cal_add(k, s, cb, cs, cn, cpos, csince, C):
    n = cn[k]
    pos = cpos[k]
    if n == C:
        _sorted_remove(cs[k], n, cb[k, pos])
        n -= 1
    cb[k, pos] = s
    cpos[k] = (pos + 1) % C
    _sorted_insert(cs[k], n, s)
    cn[k] = n + 1
    csince[k] += 1


@njit(cache=True)
def _bucket(sec, mode):
    if mode == 1:
        return int(sec // 3600) % 24
    if mode == 2:
        return ((int(sec // 86400.0) + 3) % 7) * 24 + int(sec // 3600) % 24
    return -1


# ----------------------------------------------------------------------------- kernel
@njit(cache=True)
def _kernel(secs, xs, P, fl, it, U, out_f, out_i, det_u, ev_i, ev_f, ev_prof, ev_cnt, ev_rec):
    """fl/it: float/int parameters (see run_fast). Returns the number of events opened."""
    W, min_count, mode, recent_r = it[0], it[1], it[2], it[3]
    deseason, C, min_exceed, min_tail_n, recalc_every = it[4], it[5], it[6], it[7], it[8]
    learn_cool, smooth, cooldown, max_len, label_len = it[9], it[10], it[11], it[12], it[13]
    pause, rebase, K, L, alert_score = it[14], it[15], it[16], it[17], it[18]
    has_reenter = it[19]
    ew_rate, beta, floor, slow_rate, kk, kv, h, mu0, sig0, vmean = (
        fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7], fl[8], fl[9])
    tail_q, th_enter, th_exit, rb_sigma, reenter, ss_s = (
        fl[10], fl[11], fl[12], fl[13], fl[14], fl[15])
    n_all = len(xs)
    nb = 168 if mode == 2 else 24
    log10K = math.log10(K)
    min_ready = min(16, W)
    # D1 state
    rm_buf = np.zeros(W)
    rm_sorted = np.zeros(W + 1)
    rm_st = np.zeros(2, np.int64)
    sc_n = 0
    sc_v = 0.0
    prof = np.zeros(nb)
    cnt = np.zeros(nb, np.int64)
    rd = np.zeros(recent_r)              # recent deseasonalized values (deque maxlen recent_r)
    rd_head = 0
    rd_n = 0
    d1_off = 0.0
    l_x = l_lvl = l_pred = l_r = l_sig = l_s = 0.0
    l_b = -1
    l_scored = False
    # D2 state
    mu = mu0
    s2_n = int(1 / slow_rate)
    s2_v = max(sig0, floor) / MEANABS_TO_STD
    cp = cn_ = vp = 0.0
    st_up = st_dn = st_var = -1
    r20 = np.zeros(20)
    r20_head = 0
    r20_n = 0
    d2_x = 0.0
    d2_i = 0
    # D7 state
    ss_win = np.zeros(max(L - 1, 1))
    ss_head = 0
    ss_n = 0
    ss_v = np.zeros(max(L, 1))
    ss_spe = 0.0
    ss_frac = 0.0
    use_ss = L > 0
    # calibrators
    cb = np.zeros((K, C))
    cs = np.zeros((K, C + 1))
    cnk = np.zeros(K, np.int64)
    cpos = np.zeros(K, np.int64)
    cq = np.zeros(K)
    cme = np.full(K, EPS)
    csince = np.zeros(K, np.int64)
    cqok = np.zeros(K, np.bool_)
    tmp = np.zeros(C)
    # fusion / events
    uh = np.zeros(smooth)
    uh_head = 0
    uh_n = 0
    rec = np.zeros(recent_r)             # pipeline.recent (raw values)
    rec_head = 0
    rec_n = 0
    m_state = ST_NORMAL
    m_len = 0
    m_cool = 0
    n_ev = 0
    cur = -1
    lvl_before = sig_before = 0.0
    us = np.zeros(K)
    raws = np.zeros(K)
    ready = np.zeros(K, np.bool_)
    last_score = last_ubar = 0.0
    last_top = -1
    last_state = ST_PROB
    prev_sec = np.nan
    dt = fl[16]

    # ---- probation: collect non-NaN values; replay at the last probation row
    for i in range(n_all):
        x = xs[i]
        sec = secs[i]
        if i < P:
            out_f[i, 0] = 0.0
            out_f[i, 1] = 0.0
            out_i[i, 0] = 0
            out_i[i, 1] = -1
            out_i[i, 2] = -1
            out_i[i, 3] = ST_PROB
            prev_sec = sec
            if i == P - 1:
                j = 0
                for q in range(P):
                    xq = xs[q]
                    if xq != xq:
                        continue
                    sq = secs[q]
                    # --- one replay step: score (ready checked first), calibrate, update
                    r1_ready = rm_st[1] >= min_ready
                    b = _bucket(sq, mode)
                    s_off = prof[b] if b >= 0 and cnt[b] >= min_count else 0.0
                    d1_off = s_off
                    rd[(rd_head + rd_n) % recent_r if rd_n < recent_r else rd_head] = xq - s_off
                    if rd_n < recent_r:
                        rd_n += 1
                    else:
                        rd_head = (rd_head + 1) % recent_r
                    if rm_st[1] == 0:
                        raw1 = 0.0
                        l_scored = False
                        l_lvl = xq
                        l_s = s_off
                        l_b = b
                    else:
                        l_lvl = _rm_median(rm_sorted, rm_st[1])
                        l_s = s_off
                        l_b = b
                        l_r = xq - (l_lvl + s_off)
                        sig = max(floor, MEANABS_TO_STD * sc_v)
                        raw1 = abs(l_r) / sig
                        l_scored = True
                    if r1_ready:
                        _cal_add(0, raw1, cb, cs, cnk, cpos, csince, C)
                    # D1 update
                    if l_scored:
                        sc_n += 1
                        a = max(ew_rate, 1.0 / sc_n)
                        sc_v += a * (abs(l_r) - sc_v)
                        if b >= 0:
                            dev = xq - l_lvl
                            if cnt[b] == 0:
                                prof[b] = dev
                            else:
                                prof[b] += beta * (dev - prof[b])
                            cnt[b] += 1
                    _rm_add(rm_buf, rm_sorted, rm_st, W, xq - s_off)
                    # D2 (always ready)
                    x2 = xq - d1_off if deseason else xq
                    sig2 = max(floor, MEANABS_TO_STD * s2_v)
                    z = (x2 - mu) / sig2
                    pcp, pcn, pvp = cp, cn_, vp
                    cp = max(0.0, cp + z - kk)
                    cn_ = max(0.0, cn_ - z - kk)
                    vp = max(0.0, vp + math.log(z * z + 0.01) - vmean - kv)
                    if cp == 0.0:
                        st_up = -1
                    elif pcp == 0.0:
                        st_up = j
                    if cn_ == 0.0:
                        st_dn = -1
                    elif pcn == 0.0:
                        st_dn = j
                    if vp == 0.0:
                        st_var = -1
                    elif pvp == 0.0:
                        st_var = j
                    if r20_n < 20:
                        r20[(r20_head + r20_n) % 20] = x2
                        r20_n += 1
                    else:
                        r20[r20_head] = x2
                        r20_head = (r20_head + 1) % 20
                    d2_i = j
                    raw2 = max(cp, max(cn_, vp)) / h
                    _cal_add(1, raw2, cb, cs, cnk, cpos, csince, C)
                    s2_n += 1
                    a = max(slow_rate, 1.0 / s2_n)
                    s2_v += a * (abs(x2 - mu) - s2_v)
                    mu += slow_rate * (x2 - mu)
                    # D7
                    if use_ss:
                        if ss_n >= L - 1:
                            for t in range(L - 1):
                                ss_v[t] = ss_win[(ss_head + t) % (L - 1)]
                            ss_v[L - 1] = xq
                            m = _pairwise_sum(ss_v, 0, L) / L
                            for t in range(L):
                                ss_v[t] = (ss_v[t] - m) / ss_s
                            tot = 0.0
                            for t in range(L):
                                tot += ss_v[t] * ss_v[t]
                            pr = 0.0
                            for c in range(U.shape[1]):
                                acc = 0.0
                                for t in range(L):
                                    acc += U[t, c] * ss_v[t]
                                pr += acc * acc
                            raw7 = max(tot - pr, 0.0) / L
                            _cal_add(2, raw7, cb, cs, cnk, cpos, csince, C)
                        if ss_n < L - 1:
                            ss_win[(ss_head + ss_n) % (L - 1)] = xq
                            ss_n += 1
                        else:
                            ss_win[ss_head] = xq
                            ss_head = (ss_head + 1) % (L - 1)
                    if rec_n < recent_r:
                        rec[(rec_head + rec_n) % recent_r] = xq
                        rec_n += 1
                    else:
                        rec[rec_head] = xq
                        rec_head = (rec_head + 1) % recent_r
                    j += 1
                last_state = ST_NORMAL
            continue

        if x != x:                                     # NaN: carry the last result
            out_f[i, 0] = last_score
            out_f[i, 1] = last_ubar
            out_i[i, 0] = 0
            out_i[i, 1] = last_top
            out_i[i, 2] = -1
            out_i[i, 3] = last_state
            for k in range(K):
                det_u[i, k] = det_u[i - 1, k] if i > 0 else 0.0
            continue

        gap = (prev_sec == prev_sec) and (sec - prev_sec > 3 * dt)
        prev_sec = sec
        # ---- score every detector (no learning)
        ready[0] = rm_st[1] >= min_ready
        b = _bucket(sec, mode)
        s_off = prof[b] if b >= 0 and cnt[b] >= min_count else 0.0
        d1_off = s_off
        if rd_n < recent_r:
            rd[(rd_head + rd_n) % recent_r] = x - s_off
            rd_n += 1
        else:
            rd[rd_head] = x - s_off
            rd_head = (rd_head + 1) % recent_r
        l_x = x
        l_b = b
        l_s = s_off
        if rm_st[1] == 0:
            raws[0] = 0.0
            l_scored = False
            l_lvl = x
            l_pred = x
            l_r = 0.0
            l_sig = floor
        else:
            l_lvl = _rm_median(rm_sorted, rm_st[1])
            l_pred = l_lvl + s_off
            l_r = x - l_pred
            l_sig = max(floor, MEANABS_TO_STD * sc_v)
            raws[0] = abs(l_r) / l_sig
            l_scored = True
        ready[1] = True
        x2 = x - d1_off if deseason else x
        d2_x = x2
        sig2 = max(floor, MEANABS_TO_STD * s2_v)
        z = (x2 - mu) / sig2
        pcp, pcn, pvp = cp, cn_, vp
        cp = max(0.0, cp + z - kk)
        cn_ = max(0.0, cn_ - z - kk)
        vp = max(0.0, vp + math.log(z * z + 0.01) - vmean - kv)
        if cp == 0.0:
            st_up = -1
        elif pcp == 0.0:
            st_up = i
        if cn_ == 0.0:
            st_dn = -1
        elif pcn == 0.0:
            st_dn = i
        if vp == 0.0:
            st_var = -1
        elif pvp == 0.0:
            st_var = i
        if r20_n < 20:
            r20[(r20_head + r20_n) % 20] = x2
            r20_n += 1
        else:
            r20[r20_head] = x2
            r20_head = (r20_head + 1) % 20
        d2_i = i
        raws[1] = max(cp, max(cn_, vp)) / h
        if use_ss:
            ready[2] = ss_n >= L - 1
            if ready[2]:
                for t in range(L - 1):
                    ss_v[t] = ss_win[(ss_head + t) % (L - 1)]
                ss_v[L - 1] = x
                m = _pairwise_sum(ss_v, 0, L) / L
                for t in range(L):
                    ss_v[t] = (ss_v[t] - m) / ss_s
                tot = 0.0
                for t in range(L):
                    tot += ss_v[t] * ss_v[t]
                pr = 0.0
                for c in range(U.shape[1]):
                    acc = 0.0
                    for t in range(L):
                        acc += U[t, c] * ss_v[t]
                    pr += acc * acc
                ss_spe = max(tot - pr, 0.0) / L
                ss_frac = 1.0 - ss_spe * L / tot if tot > 0 else 1.0
                raws[2] = ss_spe
            else:
                raws[2] = 0.0
        for k in range(K):
            us[k] = _cal_surprise(k, raws[k], cb, cs, cnk, cq, cme, csince, cqok, tail_q,
                                  min_exceed, min_tail_n, recalc_every, tmp) if ready[k] else 0.0
        top = 0
        for k in range(1, K):
            if us[k] > us[top]:
                top = k
        u = max(0.0, us[top] - log10K)
        if uh_n < smooth:
            uh[(uh_head + uh_n) % smooth] = u
            uh_n += 1
        else:
            uh[uh_head] = u
            uh_head = (uh_head + 1) % smooth
        for t in range(uh_n):
            tmp[t] = uh[(uh_head + t) % smooth]
        ubar = _fsum_py(tmp, uh_n, it[20] == 1) / uh_n
        if rec_n < recent_r:
            rec[(rec_head + rec_n) % recent_r] = x
            rec_n += 1
        else:
            rec[rec_head] = x
            rec_head = (rec_head + 1) % recent_r

        # ---- event machine
        started = False
        ended = False
        skip = False
        if m_state == ST_COOL:
            if has_reenter and ubar >= reenter:
                m_state = ST_ANOM
                m_len = 1
                m_cool = 0
                started = True
                skip = True
            elif m_cool > 0:
                m_cool -= 1
                skip = True
            else:
                m_state = ST_NORMAL
        if not skip:
            if m_state == ST_NORMAL:
                if ubar >= th_enter:
                    m_state = ST_ANOM
                    m_len = 1
                    started = True
            elif m_state == ST_ANOM:
                if ubar >= th_exit and m_len < max_len:
                    m_len += 1
                else:
                    ended = True
                    if cooldown > 0:
                        m_state = ST_COOL
                        m_cool = cooldown - 1
                    else:
                        m_state = ST_NORMAL
                        if ubar >= th_enter:
                            m_state = ST_ANOM
                            m_len = 1
                            started = True

        if ended:                                      # close the open event
            e = cur
            ev_i[e, 1] = i                             # index at close (end_index = i - 1)
            rebased = 0
            if rebase:
                for t in range(rd_n):
                    tmp[t] = rd[(rd_head + t) % recent_r]
                srt = np.sort(tmp[:rd_n])
                nr = rd_n
                now = 0.5 * (srt[(nr - 1) // 2] + srt[nr // 2]) if nr else _rm_median(rm_sorted, rm_st[1])
                shifted = abs(now - lvl_before) > rb_sigma * sig_before
                if ev_i[e, 2] == 1 or shifted:
                    # D1: level.refill(recent_d)
                    rm_st[0] = 0
                    rm_st[1] = 0
                    start = max(0, rd_n - W)
                    for t in range(start, rd_n):
                        _rm_add(rm_buf, rm_sorted, rm_st, W, rd[(rd_head + t) % recent_r])
                    # D2: mu = median(recent_d) (deseasonalized) or median(recent raw)
                    if deseason:
                        mu = 0.5 * (srt[(nr - 1) // 2] + srt[nr // 2]) if nr else mu
                    else:
                        for t in range(rec_n):
                            tmp[t] = rec[(rec_head + t) % recent_r]
                        sr = np.sort(tmp[:rec_n])
                        if rec_n:
                            mu = 0.5 * (sr[(rec_n - 1) // 2] + sr[rec_n // 2])
                    rebased = 1
            cp = cn_ = vp = 0.0
            st_up = st_dn = st_var = -1
            ev_i[e, 3] = rebased
            cur = -1
        if started:
            e = n_ev
            n_ev += 1
            cur = e
            ev_i[e, 0] = i
            ev_i[e, 2] = top
            ev_i[e, 4] = 1 if gap else 0
            ev_i[e, 5] = l_b
            ev_i[e, 6] = 1 if l_scored else 0
            ev_i[e, 7] = st_up
            ev_i[e, 8] = st_dn
            ev_i[e, 9] = st_var
            ev_i[e, 10] = d2_i
            ev_i[e, 11] = r20_n
            ev_f[e, 0] = l_x
            ev_f[e, 1] = l_lvl
            ev_f[e, 2] = l_pred
            ev_f[e, 3] = l_r
            ev_f[e, 4] = l_sig
            ev_f[e, 5] = l_s
            ev_f[e, 6] = cp
            ev_f[e, 7] = cn_
            ev_f[e, 8] = vp
            ev_f[e, 9] = mu
            ev_f[e, 10] = ss_spe
            ev_f[e, 11] = ss_frac
            for k in range(K):
                ev_f[e, 12 + k] = us[k]
            for t in range(nb):
                ev_prof[e, t] = prof[t]
                ev_cnt[e, t] = cnt[t]
            for t in range(r20_n):
                ev_rec[e, t] = r20[(r20_head + t) % 20]
            lvl_before = _rm_median(rm_sorted, rm_st[1])
            sig_before = max(floor, MEANABS_TO_STD * sc_v)

        # ---- learning
        if not (pause and m_state == ST_ANOM):
            if l_scored:
                sc_n += 1
                a = max(ew_rate, 1.0 / sc_n)
                sc_v += a * (abs(l_r) - sc_v)
                if b >= 0:
                    dev = x - l_lvl
                    if cnt[b] == 0:
                        prof[b] = dev
                    else:
                        prof[b] += beta * (dev - prof[b])
                    cnt[b] += 1
            _rm_add(rm_buf, rm_sorted, rm_st, W, x - s_off)
            s2_n += 1
            a = max(slow_rate, 1.0 / s2_n)
            s2_v += a * (abs(d2_x - mu) - s2_v)
            mu += slow_rate * (d2_x - mu)
            if use_ss:
                if ss_n < L - 1:
                    ss_win[(ss_head + ss_n) % (L - 1)] = x
                    ss_n += 1
                else:
                    ss_win[ss_head] = x
                    ss_head = (ss_head + 1) % (L - 1)
            if m_state == ST_NORMAL or (learn_cool and m_state == ST_COOL):
                for k in range(K):
                    if ready[k]:
                        _cal_add(k, raws[k], cb, cs, cnk, cpos, csince, C)

        in_event = m_state == ST_ANOM
        alert = in_event and (label_len < 0 or m_len <= label_len)
        sv = 1.0 - 10.0 ** (-ubar)
        if alert_score:
            sv = 0.5 + 0.5 * sv if alert else 0.5 * sv
        last_score = sv
        last_ubar = ubar
        last_top = top if u > 0 else -1
        last_state = ST_ANOM if in_event else m_state
        out_f[i, 0] = sv
        out_f[i, 1] = ubar
        out_i[i, 0] = 1 if alert else 0
        out_i[i, 1] = last_top
        out_i[i, 2] = n_ev if in_event else -1
        out_i[i, 3] = last_state
        for k in range(K):
            det_u[i, k] = us[k]
    if cur >= 0:
        ev_i[cur, 1] = n_all
        ev_i[cur, 3] = 0
    return n_ev


# ----------------------------------------------------------------------------- wrapper
def run_fast(series, cfg, det_probation=None):
    """Compiled equivalent of runner.run_sentinel for configs where supports(cfg)."""
    c = cfg
    nn = c.get("neural", {})
    if nn.get("enabled") and "{category}" in nn.get("weights", ""):
        c = deep_merge(c, {"neural": {"weights": nn["weights"].format(category=series.category)}})
    if not supports(c):
        raise ValueError("config not supported by the compiled engine")
    P = max(int(series.probation if det_probation is None else det_probation), 2)
    secs = np.asarray(series.seconds, float)
    xs = np.asarray(series.values, float)
    n = len(xs)
    pm = ~np.isnan(xs[:P])
    ps, px = (secs[:P][pm], xs[:P][pm]) if pm.any() else (np.zeros(1), np.zeros(1))
    season = detect_season(ps, px, c["season"]["acf_min"])
    floor = scale_floor(px)
    med, rs = robust_stats(px)
    rs = max(rs, floor)
    vmean = float(np.mean(np.log(((px - med) / rs) ** 2 + 0.01)))
    mode = {"none": 0, "day": 1, "week": 2}[season["mode"]]
    names = [k for k in ORDER if c.get(k, {}).get("enabled")]
    K = len(names)
    U = np.zeros((1, 1))
    L = 0
    if "subspace" in names:
        from .detectors.subspace import SubspaceDetector
        sd = SubspaceDetector(c["subspace"], px, rs)
        if sd.ok:
            U, L = np.ascontiguousarray(sd.U), sd.L
        else:
            return None                                   # rare: fall back to reference
    ev = c["events"]
    cal = c["calibration"]
    ro = ev.get("reenter_offset")
    ll = ev.get("label_len")
    deseason = bool(c["change"].get("deseasonalize")) and mode != 0
    fl = np.array([c["residual"]["ew_mad_rate"], c["residual"]["season_beta"], floor,
                   c["change"]["slow_rate"], c["change"]["k"], c["change"]["k_v"], c["change"]["h"],
                   med, rs, vmean, cal["tail_q"], ev["theta_enter"],
                   ev["theta_enter"] - ev["exit_offset"], ev["rebaseline_sigma"],
                   ev["theta_enter"] + ro if ro is not None else 0.0, rs,
                   season["dt_seconds"]], float)
    it = np.array([c["residual"]["window"], c["residual"]["season_min_count"], mode,
                   max(ev["rebaseline_r"], 1), int(deseason), int(cal["buffer"]), cal["min_exceed"],
                   cal["min_tail_n"], cal["recalc_every"], int(bool(cal.get("learn_in_cooldown"))),
                   c["fusion"]["smooth"], ev["cooldown"], ev["max_len"],
                   -1 if ll is None else max(int(ll), 1), int(ev["pause"]), int(ev["rebaseline"]),
                   K, L, int(c.get("output", {}).get("score", "surprise") == "alert"),
                   int(ro is not None), int(sys.version_info >= (3, 12))], np.int64)
    nb = 168 if mode == 2 else 24
    max_ev = n // 2 + 2
    out_f = np.zeros((n, 2))
    out_i = np.zeros((n, 4), np.int64)
    det_u = np.zeros((n, K))
    ev_i = np.zeros((max_ev, 12), np.int64)
    ev_f = np.zeros((max_ev, 12 + K))
    ev_prof = np.zeros((max_ev, nb))
    ev_cnt = np.zeros((max_ev, nb), np.int64)
    ev_rec = np.zeros((max_ev, 20))
    n_ev = _kernel(secs, xs, P, fl, it, U, out_f, out_i, det_u, ev_i, ev_f, ev_prof, ev_cnt, ev_rec)
    surprise = out_f[:, 1]
    events = [_event_record(e, ev_i, ev_f, ev_prof, ev_cnt, ev_rec, names, season, secs, xs,
                            surprise, deseason, L, U, c["residual"]["season_min_count"])
              for e in range(n_ev)]
    top = np.where(out_i[:, 1] >= 0, np.array(names + ["none"], dtype=object)[out_i[:, 1]], "none")
    return {"anomaly_score": out_f[:, 0], "surprise": surprise,
            "label": out_i[:, 0].astype(np.int8), "top_detector": top.astype(object),
            "event_id": out_i[:, 2], "state": STATE_NAMES[out_i[:, 3]], "detector_surprise": det_u,
            "detector_names": names, "events": [dict(e, file=series.name) for e in events],
            "t_ns": None, "season": season}


def _event_record(e, ev_i, ev_f, ev_prof, ev_cnt, ev_rec, names, season, secs, xs, surprise,
                  deseason, L, U, min_count):
    start, close_i, top = int(ev_i[e, 0]), int(ev_i[e, 1]), int(ev_i[e, 2])
    b, scored = int(ev_i[e, 5]), bool(ev_i[e, 6])
    x, lvl, pred, r, sig, s = (float(v) for v in ev_f[e, :6])
    mode = season["mode"]
    prof, cnt = ev_prof[e], ev_cnt[e]
    evid = {}
    for k, name in enumerate(names):
        if name == "residual":
            if not scored:
                x_, lvl_, pred_, r_, sig_ = x, x, x, 0.0, sig
            else:
                x_, lvl_, pred_, r_, sig_ = x, lvl, pred, r, sig
            learned = cnt >= min_count
            cc = float(prof[learned].mean()) if learned.any() else 0.0
            evid["residual"] = {"observed": float(x_), "expected": float(pred_), "level": float(lvl_ + cc),
                                "seasonal_offset": float(s - cc) if b >= 0 else 0.0,
                                "residual": float(r_), "z": float(r_ / sig_), "season_mode": mode,
                                "bucket": int(b)}
        elif name == "change":
            stats = {"up": float(ev_f[e, 6]), "down": float(ev_f[e, 7]), "variance": float(ev_f[e, 8])}
            direction = max(stats, key=stats.get)
            st = {"up": ev_i[e, 7], "down": ev_i[e, 8], "variance": ev_i[e, 9]}[direction]
            nr = int(ev_i[e, 11])
            rec = [float(v) for v in ev_rec[e, :nr]]
            evid["change"] = {"direction": direction, "cusum": stats[direction],
                              "baseline_mean": float(ev_f[e, 9]),
                              "recent_mean": float(sum(rec) / len(rec)) if rec else None,
                              "points_since_drift_start": int(ev_i[e, 10] - st + 1) if st >= 0 else 0,
                              "deseasonalized": deseason}
        elif name == "subspace":
            evid["subspace"] = {"spe": float(ev_f[e, 10]), "explained_fraction": float(ev_f[e, 11]),
                                "lag": int(L), "components": int(U.shape[1])}
    dsur = {n: float(ev_f[e, 12 + k]) for k, n in enumerate(names)}
    end_i = close_i - 1
    rows = np.arange(start, close_i)
    rows = rows[~np.isnan(xs[rows])]
    trace = [round(float(v), 3) for v in surprise[rows]]
    peak = 0.0
    for v in surprise[rows]:
        if v > peak:
            peak = float(v)
    end_sec = float(secs[rows[-1]]) if len(rows) else float(secs[start])
    out = {
        "event_id": e + 1,
        "start_index": start,
        "start_timestamp": fmt_ts(float(secs[start])),
        "top_detector": names[top],
        "detector_surprise": dsur,
        "evidence": evid,
        "context": {"season_mode": mode, "data_gap_before": bool(ev_i[e, 4]), "transform": "none"},
        "explanation": explanation(dsur, evid),
        "peak_surprise": peak,
        "end_index": end_i,
        "end_timestamp": fmt_ts(end_sec),
        "length_points": close_i - start,
        "peak_score": 1.0 - 10.0 ** (-peak),
        "rebaselined": bool(ev_i[e, 3]),
        "trace": trace,
    }
    return round_floats(out)

