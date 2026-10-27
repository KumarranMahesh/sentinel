"""Combine per-detector surprises into one surprise (PLAN.md 7.6)."""
import math

LN10 = math.log(10.0)


def max_bonferroni(us):
    """max_k U_k - log10(K), floored at 0. Returns (U, index of top detector)."""
    k = max(range(len(us)), key=us.__getitem__)
    return max(0.0, us[k] - math.log10(len(us))), k


def fisher(us):
    """Fisher's method: -2 sum ln p_k ~ chi2(2K) if detectors were independent (they are not)."""
    k = max(range(len(us)), key=us.__getitem__)
    x = 2.0 * LN10 * sum(us)            # -2 sum ln p
    # chi2 survival with even dof 2K: exp(-x/2) * sum_{i<K} (x/2)^i / i!
    half = x / 2.0
    term, total = 1.0, 1.0
    for i in range(1, len(us)):
        term *= half / i
        total += term
    log10_p = (-half + math.log(total)) / LN10
    return max(0.0, -log10_p), k


FUSERS = {"max_bonferroni": max_bonferroni, "fisher": fisher}
