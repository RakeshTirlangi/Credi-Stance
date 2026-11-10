"""Credal core: bounds, reachable intervals, intersection probability, entropies,
conformal credal calibration, EffCre single-model decalibration, and metrics.
NumPy only. All entropies in bits.
"""
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-12


# ------------------------------------------------------------------ basics
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(p, axis=-1):
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


# ------------------------------------------------------------------ credal sets from members
def credal_bounds(P):
    """P: (M, N, K) member probabilities -> lower, upper (N, K)  [Credal Wrapper, Wang et al. 2025]."""
    return P.min(axis=0), P.max(axis=0)


def reachable(l, u):
    """Tighten interval bounds to those actually attainable in the credal set (CreDE Eq. 12)."""
    K = l.shape[-1]
    sl, su = l.sum(-1, keepdims=True), u.sum(-1, keepdims=True)
    u2 = np.minimum(u, 1 - (sl - l))
    l2 = np.maximum(l, 1 - (su - u))
    return l2, u2


def intersection_probability(l, u):
    """Single representative distribution of a probability-interval credal set."""
    w = u - l
    ws = w.sum(-1, keepdims=True)
    alpha = np.where(ws > EPS, (1 - l.sum(-1, keepdims=True)) / np.maximum(ws, EPS), 0.0)
    p = l + alpha * w
    return p / p.sum(-1, keepdims=True)


def upper_entropy(l, u, iters=60):
    """max_{p in C(l,u)} H(p): water-filling p_k = clip(t, l_k, u_k), bisection on t."""
    lo = np.zeros(l.shape[0]); hi = np.ones(l.shape[0])
    for _ in range(iters):
        t = (lo + hi) / 2
        s = np.clip(t[:, None], l, u).sum(-1)
        hi = np.where(s > 1, t, hi); lo = np.where(s > 1, lo, t)
    return entropy(np.clip(((lo + hi) / 2)[:, None], l, u))


def _vertices_mask(K):
    """All assignments of K-1 coordinates to {lower, upper} with a free coordinate."""
    out = []
    for free in range(K):
        others = [j for j in range(K) if j != free]
        for choice in itertools.product([0, 1], repeat=K - 1):
            out.append((free, others, choice))
    return out


def lower_entropy(l, u):
    """min_{p in C} H(p): entropy is concave -> minimum at a vertex. Vectorised over N."""
    N, K = l.shape
    best = np.full(N, np.inf)
    for free, others, choice in _vertices_mask(K):
        p = np.empty((N, K))
        for j, c in zip(others, choice):
            p[:, j] = u[:, j] if c else l[:, j]
        p[:, free] = 1 - p[:, others].sum(-1)
        ok = (p[:, free] >= l[:, free] - 1e-9) & (p[:, free] <= u[:, free] + 1e-9)
        h = entropy(np.clip(p, 0, 1))
        best = np.where(ok, np.minimum(best, h), best)
    # fallback (numerical): if no vertex found, use intersection probability
    ip = intersection_probability(l, u)
    return np.where(np.isfinite(best), best, entropy(ip))


def decompose(l, u):
    l, u = reachable(l, u)
    H_up = upper_entropy(l, u)
    H_lo = lower_entropy(l, u)
    return {"TU": H_up, "AU": H_lo, "EU": np.maximum(H_up - H_lo, 0.0), "width": (u - l).sum(-1)}


def nondominated(l, u):
    """Interval dominance: class k kept unless some j has l_j > u_k. Returns bool (N,K)."""
    return u >= l.max(-1, keepdims=True)


def bayesian_decomposition(P):
    """Standard ensemble decomposition: TU = H(mean), AU = mean H, EU = MI."""
    pbar = P.mean(0)
    tu = entropy(pbar); au = entropy(P).mean(0)
    return {"TU": tu, "AU": au, "EU": tu - au}


# ------------------------------------------------------------------ EffCre (single model)
def effcre_bounds(train_logits, train_y, test_logits, alpha=0.5, iters=50, tmax=20.0):
    """Efficient Credal Prediction through Decalibration (Hofman et al., ICLR 2026), 1-D variant.
    For each class k, shift logits by t*e_k; feasible t satisfy  sum_n log p_{y_n}(t) - log p_{y_n}(0) >= log(alpha).
    Bounds = p_k(x; t_k^-), p_k(x; t_k^+), endpoints found by bisection.
    """
    K = train_logits.shape[1]
    base = np.log(softmax(train_logits)[np.arange(len(train_y)), train_y] + EPS).sum()
    budget = np.log(alpha)

    def dll(k, t):
        z = train_logits.copy(); z[:, k] += t
        return np.log(softmax(z)[np.arange(len(train_y)), train_y] + EPS).sum() - base

    lower = np.zeros((len(test_logits), K)); upper = np.zeros_like(lower)
    for k in range(K):
        # upper endpoint t+ in [0, tmax]
        lo, hi = 0.0, tmax
        for _ in range(iters):
            mid = (lo + hi) / 2
            if dll(k, mid) >= budget: lo = mid
            else: hi = mid
        tp = lo
        lo, hi = -tmax, 0.0
        for _ in range(iters):
            mid = (lo + hi) / 2
            if dll(k, mid) >= budget: hi = mid
            else: lo = mid
        tm = hi
        zp = test_logits.copy(); zp[:, k] += tp
        zm = test_logits.copy(); zm[:, k] += tm
        upper[:, k] = softmax(zp)[:, k]; lower[:, k] = softmax(zm)[:, k]
    return lower, upper


# ------------------------------------------------------------------ conformal credal calibration
def tv(a, b):
    return 0.5 * np.abs(a - b).sum(-1)


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(scores)[min(k, n) - 1]


def tv_ball_bounds(g, q):
    return np.maximum(0.0, g - q), np.minimum(1.0, g + q)


def conformal_credal(g_cal, lam_cal, g_test, alpha=0.1):
    """Conformalized credal sets (Javanmardi et al. 2024) with TV non-conformity on a first-order predictor."""
    q = conformal_quantile(tv(lam_cal, g_cal), alpha)
    l, u = tv_ball_bounds(g_test, q)
    return q, l, u


def credal_conformal(l_cal, u_cal, lam_cal, l_test, u_test, alpha=0.1):
    """Our variant: non-conformity = distance of lambda to the *credal set* (0 if inside), so the
    calibrated set is the credal set dilated by q. Uses the box distance (max class-wise violation)."""
    def dist_to_box(l, u, lam):
        return np.maximum(np.maximum(l - lam, lam - u), 0).max(-1)
    q = conformal_quantile(dist_to_box(l_cal, u_cal, lam_cal), alpha)
    return q, np.maximum(0, l_test - q), np.minimum(1, u_test + q)


# ------------------------------------------------------------------ metrics
def ece(p, y, bins=15):
    conf = p.max(-1); pred = p.argmax(-1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def nll(p, y):
    return -np.log(p[np.arange(len(y)), y] + EPS).mean()


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return ((p - oh) ** 2).sum(-1).mean()


def macro_f1(pred, y, classes=(0, 1, 2)):
    f = []
    for c in classes:
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def f_avg(pred, y):
    """Official SemEval-2016 metric: mean F1 of FAVOR and AGAINST."""
    return macro_f1(pred, y, classes=(0, 1))


def aurc(score, correct):
    """Area under risk-coverage curve; score = uncertainty (higher = reject first)."""
    order = np.argsort(score)  # most certain first
    err = 1 - correct[order].astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return risk.mean()


def auroc(score, positive):
    if positive.sum() == 0 or positive.sum() == len(positive): return float("nan")
    return roc_auc_score(positive, score)


def aupr(score, positive):
    return average_precision_score(positive, score)


def u65(nd, y):
    """Utility-discounted accuracy (Zaffalon et al. 2012) for set-valued predictions; nd bool (N,K)."""
    size = nd.sum(-1); hit = nd[np.arange(len(y)), y]
    u = np.where(hit, 1.6 / size - 0.6 / size ** 2, 0.0)
    return u.mean()


def u80(nd, y):
    size = nd.sum(-1); hit = nd[np.arange(len(y)), y]
    return np.where(hit, 2.2 / size - 1.2 / size ** 2, 0.0).mean()


def coverage_tv(g, lam, q):
    return (tv(lam, g) <= q).mean()


def temperature_scale(logits_cal, y_cal, grid=np.linspace(0.3, 5, 95)):
    best, bt = np.inf, 1.0
    for t in grid:
        v = nll(softmax(logits_cal / t), y_cal)
        if v < best: best, bt = v, t
    return bt
