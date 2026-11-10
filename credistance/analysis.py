"""Build all uncertainty methods from cached member outputs and compute every metric.
Writes results/results.json, results/tables.md and figures to results/fig/.
"""
import os, json, glob, sys, warnings
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import credal as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs"); RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "fig")
os.makedirs(FIG, exist_ok=True)
warnings.filterwarnings("ignore")
PROC = os.path.join(ROOT, "data", "processed")
TARGETS = {"ath": "Atheism", "cli": "Climate Change is a Real Concern", "fem": "Feminist Movement",
           "hil": "Hillary Clinton", "abo": "Legalization of Abortion"}
SEEN4 = {"Atheism", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"}
LLM_RUNS = [d for d in ("llm_qwen3b", "llm_qwen7b") if os.path.isdir(os.path.join(RUNS, d))]
METHOD_ORDER = ["Softmax", "TempScale", "MCD-MI", "MCD-Credal", "EDL", "DE-MI", "CreWrap-DE", "EffCre",
                "CreDE", "CreWrap-Soft", "CreDRO", "CreDRO-Soft"]


def load_meta(name):
    rows = [json.loads(l) for l in open(os.path.join(PROC, name + ".jsonl"), encoding="utf-8")]
    lam = np.array([r["label_dist"] if r["label_dist"] else [np.nan] * 3 for r in rows])
    H = np.array([r["human_entropy"] if r["human_entropy"] is not None else np.nan for r in rows])
    return rows, lam, H


MAX_M = int(os.environ.get("CREDAL_MAX_M", "5"))  # main tables use the first 5 members


def members(prefix, method, setname, M=None):
    files = sorted(glob.glob(os.path.join(RUNS, prefix, f"{method}_s*", setname + ".npz")))
    return [np.load(f) for f in files[:(M or MAX_M)]]


# ------------------------------------------------------------------ build every method on one eval set
def build(prefix, setname, temp_cal=None):
    """Returns dict name -> {p (N,K) point prediction, TU, AU, EU, l, u (optional)} and y, targets."""
    out = {}
    ce = members(prefix, "ce", setname)
    y = ce[0]["y"]; tg = ce[0]["target"]
    # --- single model + temperature
    z0 = ce[0]["logits"]; p0 = C.softmax(z0)
    out["Softmax"] = dict(p=p0, TU=C.entropy(p0), AU=C.entropy(p0), EU=1 - p0.max(-1))
    if temp_cal is not None:
        pt = C.softmax(z0 / temp_cal)
        out["TempScale"] = dict(p=pt, TU=C.entropy(pt), AU=C.entropy(pt), EU=1 - pt.max(-1))
    # --- MC dropout
    if "mc_logits" in ce[0].files:
        Pm = C.softmax(ce[0]["mc_logits"]); b = C.bayesian_decomposition(Pm)
        out["MCD-MI"] = dict(p=Pm.mean(0), **b)
        l, u = C.credal_bounds(Pm); d = C.decompose(l, u)
        out["MCD-Credal"] = dict(p=C.intersection_probability(*C.reachable(l, u)), l=l, u=u, **{k: d[k] for k in ("TU", "AU", "EU")})
    # --- EDL
    edl = members(prefix, "edl", setname)
    if edl:
        al = edl[0]["alpha"]; S = al.sum(-1); pe = al / S[:, None]
        out["EDL"] = dict(p=pe, TU=C.entropy(pe), AU=C.entropy(pe), EU=3.0 / S)
    # --- deep ensemble (Bayesian) and credal wrapper
    P = C.softmax(np.stack([m["logits"] for m in ce])); b = C.bayesian_decomposition(P)
    out["DE-MI"] = dict(p=P.mean(0), **b)
    for name, meth in [("CreWrap-DE", "ce"), ("CreWrap-Soft", "soft"), ("CreDRO", "dro"), ("CreDRO-Soft", "softdro")]:
        ms = members(prefix, meth, setname)
        if not ms: continue
        Pm = C.softmax(np.stack([m["logits"] for m in ms]))
        l, u = C.credal_bounds(Pm); d = C.decompose(l, u)
        out[name] = dict(p=C.intersection_probability(*C.reachable(l, u)), l=l, u=u, Pm=Pm, **{k: d[k] for k in ("TU", "AU", "EU")})
    # --- CreDE: average reachable intervals of CreNets
    cn = members(prefix, "crenet", setname)
    if cn:
        L = []; U = []
        for m in cn:
            l, u = C.reachable(m["qL"], m["qU"]); L.append(l); U.append(u)
        l, u = np.mean(L, 0), np.mean(U, 0); d = C.decompose(l, u)
        out["CreDE"] = dict(p=C.intersection_probability(*C.reachable(l, u)), l=l, u=u, **{k: d[k] for k in ("TU", "AU", "EU")})
    return out, y, tg


def llm_methods(setname, run="llm_qwen3b"):
    """Prompt-credal LLM: credal wrapper over prompt paraphrases + single-prompt baseline."""
    f = os.path.join(RUNS, run, setname + ".npz")
    if not os.path.exists(f): return {}
    d = np.load(f); P = d["P"]; out = {}
    p0 = P[0]; out[f"LLM-single ({run})"] = dict(p=p0, TU=C.entropy(p0), AU=C.entropy(p0), EU=1 - p0.max(-1))
    b = C.bayesian_decomposition(P); out[f"LLM-promptMI ({run})"] = dict(p=P.mean(0), **b)
    l, u = C.credal_bounds(P); dd = C.decompose(l, u)
    out[f"LLM-PromptCredal ({run})"] = dict(p=C.intersection_probability(*C.reachable(l, u)), l=l, u=u, **{k: dd[k] for k in ("TU", "AU", "EU")})
    return out


def effcre(prefix, setname, alpha, ref="semeval_cal", exclude=None):
    ce0 = np.load(os.path.join(RUNS, prefix, "ce_s0", setname + ".npz"))
    refd = np.load(os.path.join(RUNS, prefix, "ce_s0", ref + ".npz"))
    mask = np.ones(len(refd["y"]), bool) if exclude is None else refd["target"] != exclude
    l, u = C.effcre_bounds(refd["logits"][mask], refd["y"][mask], ce0["logits"], alpha=alpha)
    d = C.decompose(l, u)
    return dict(p=C.intersection_probability(*C.reachable(l, u)), l=l, u=u, **{k: d[k] for k in ("TU", "AU", "EU")})


# ------------------------------------------------------------------ metrics
def width_score(m):
    if "l" in m:
        l, u = C.reachable(m["l"], m["u"]); return (u - l).sum(-1)
    return m["EU"]


def point_metrics(m, y):
    p = m["p"]; pred = p.argmax(-1); correct = (pred == y)
    r = dict(acc=correct.mean(), macroF1=C.macro_f1(pred, y), Favg=C.f_avg(pred, y), ECE=C.ece(p, y),
             NLL=C.nll(p, y), Brier=C.brier(p, y),
             misclf_AUROC_TU=C.auroc(m["TU"], ~correct), misclf_AUROC_EU=C.auroc(m["EU"], ~correct),
             AURC_TU=C.aurc(m["TU"], correct), AURC_EU=C.aurc(m["EU"], correct))
    if "l" in m:
        l, u = C.reachable(m["l"], m["u"]); nd = C.nondominated(l, u)
        r.update(determinacy=(nd.sum(-1) == 1).mean(), set_size=nd.sum(-1).mean(),
                 set_acc=nd[np.arange(len(y)), y].mean(), u65=C.u65(nd, y), u80=C.u80(nd, y),
                 width=(u - l).sum(-1).mean())
    return {k: float(v) for k, v in r.items()}


def cost_curve(m, y, budgets=(0.0, 0.05, 0.1, 0.2, 0.3), c_defer=0.3, c_set=0.15):
    """Expected cost under credal routing (defer by EU, set-valued by dominance) vs confidence routing (defer by TU)."""
    N = len(y); res = {}
    for b in budgets:
        k = int(b * N)
        # confidence thresholding: defer top-k by TU, single prediction otherwise
        order = np.argsort(-m["TU"]); keep = np.ones(N, bool); keep[order[:k]] = False
        err = (m["p"].argmax(-1) != y)
        res[f"conf@{b}"] = float((err[keep].sum() + c_defer * k) / N)
        if "l" in m:
            orderE = np.argsort(-m["EU"]); keepE = np.ones(N, bool); keepE[orderE[:k]] = False
            l, u = C.reachable(m["l"], m["u"]); nd = C.nondominated(l, u)
            size = nd.sum(-1); hit = nd[np.arange(N), y]
            single = size == 1
            cost = np.where(single, (~hit).astype(float), np.where(hit, c_set * (size - 1), 1.0 + c_set * (size - 1)))
            res[f"credal@{b}"] = float((cost[keepE].sum() + c_defer * k) / N)
    return res


def human_alignment(m, lam, H):
    ok = ~np.isnan(H)
    r = {}
    for k in ("AU", "EU", "TU"):
        rho, pv = spearmanr(m[k][ok], H[ok]); r[f"rho_{k}_H"] = float(rho); r[f"p_{k}_H"] = float(pv)
    r["JSD_p_lambda"] = float(np.mean([jensenshannon(m["p"][i], lam[i], base=2) ** 2 for i in np.where(ok)[0]]))
    r["TV_p_lambda"] = float(C.tv(m["p"][ok], lam[ok]).mean())
    return r


def conformal_eval(m_cal, lam_cal, m_test, lam_test, alphas=(0.05, 0.1, 0.2), resplits=100, seed=0):
    """Pool cal+test voted items, re-split 100x (exchangeable), report coverage / efficiency."""
    okc = ~np.isnan(lam_cal[:, 0]); okt = ~np.isnan(lam_test[:, 0])
    g = np.concatenate([m_cal["p"][okc], m_test["p"][okt]]); lam = np.concatenate([lam_cal[okc], lam_test[okt]])
    has_box = "l" in m_cal
    if has_box:
        L = np.concatenate([m_cal["l"][okc], m_test["l"][okt]]); U = np.concatenate([m_cal["u"][okc], m_test["u"][okt]])
    rng = np.random.default_rng(seed); n = len(g); res = {}
    for a in alphas:
        cov, q, nd, cov2, q2, nd2 = [], [], [], [], [], []
        for _ in range(resplits):
            idx = rng.permutation(n); ic, it = idx[:n // 2], idx[n // 2:]
            qq, l, u = C.conformal_credal(g[ic], lam[ic], g[it], alpha=a)
            cov.append(C.coverage_tv(g[it], lam[it], qq)); q.append(qq); nd.append(C.nondominated(l, u).sum(-1).mean())
            if has_box:
                q2v, l2, u2 = C.credal_conformal(L[ic], U[ic], lam[ic], L[it], U[it], alpha=a)
                inside = ((lam[it] >= l2 - 1e-9) & (lam[it] <= u2 + 1e-9)).all(-1).mean()
                cov2.append(inside); q2.append(q2v); nd2.append(C.nondominated(l2, u2).sum(-1).mean())
        res[f"TVball@{a}"] = dict(coverage=float(np.mean(cov)), q=float(np.mean(q)), set_size=float(np.mean(nd)))
        if has_box:
            res[f"CredalDilate@{a}"] = dict(coverage=float(np.mean(cov2)), q=float(np.mean(q2)), set_size=float(np.mean(nd2)))
    return res


# ------------------------------------------------------------------ main
def main():
    results = {"intarget": {}, "cross_dataset": {}, "human": {}, "ambig": {}, "conformal": {}, "cost": {}, "loto": {}}
    rows_test, lam_test, H_test = load_meta("semeval_test")
    rows_cal, lam_cal, H_cal = load_meta("semeval_cal")
    rows_amb, lam_amb, H_amb = load_meta("semeval_ambig")

    # temperature from cal (single ce_s0)
    cal0 = np.load(os.path.join(RUNS, "intarget", "ce_s0", "semeval_cal.npz"))
    T = C.temperature_scale(cal0["logits"], cal0["y"])
    results["temperature"] = float(T)

    # ---------- E1 in-target
    M_test, y, tg = build("intarget", "semeval_test", temp_cal=T)
    M_cal, yc, _ = build("intarget", "semeval_cal", temp_cal=T)
    for a in (0.5, 1e-2, 1e-4, 1e-6):
        M_test[f"EffCre(a={a:g})"] = effcre("intarget", "semeval_test", a)
        M_cal[f"EffCre(a={a:g})"] = effcre("intarget", "semeval_cal", a)
    for run in LLM_RUNS:
        M_test.update(llm_methods("semeval_test", run)); M_cal.update(llm_methods("semeval_cal", run))
    for name, m in M_test.items():
        results["intarget"][name] = point_metrics(m, y)
        results["human"][name] = human_alignment(m, lam_test, H_test)
        results["cost"][name] = cost_curve(m, y)
        results["conformal"][name] = conformal_eval(M_cal[name], lam_cal, m, lam_test)

    # ---------- E4b cross-dataset unseen targets (seen = semeval test, unseen = pstance/covid test)
    for other in ("pstance_test", "covid_test"):
        M_o, yo, _ = build("intarget", other, temp_cal=T)
        for a in (0.5, 1e-2, 1e-4, 1e-6):
            M_o[f"EffCre(a={a:g})"] = effcre("intarget", other, a)
        for run in LLM_RUNS: M_o.update(llm_methods(other, run))
        for name, mo in M_o.items():
            m = M_test[name]
            lab = np.concatenate([np.zeros(len(y)), np.ones(len(yo))])
            r = {f"unseen_AUROC_{k}": float(C.auroc(np.concatenate([m[k], mo[k]]), lab)) for k in ("EU", "TU", "AU")}
            r["unseen_AUROC_width"] = float(C.auroc(np.concatenate([width_score(m), width_score(mo)]), lab))
            # accuracy on the other dataset (binary P-Stance: ignore NONE predictions by mapping to argmax of 2)
            po = mo["p"]
            if other == "pstance_test": po = po[:, :2]
            r["acc_other"] = float((po.argmax(-1) == yo).mean()); r["macroF1_other"] = float(C.macro_f1(po.argmax(-1), yo))
            r["mean_EU_seen"] = float(m["EU"].mean()); r["mean_EU_unseen"] = float(mo["EU"].mean())
            results["cross_dataset"].setdefault(other, {})[name] = r

    # ---------- E5b ambiguity stress test (4 seen targets only; Trump excluded)
    M_a, ya, tga = build("intarget", "semeval_ambig", temp_cal=T)
    for a in (0.5, 1e-2, 1e-4, 1e-6):
        M_a[f"EffCre(a={a:g})"] = effcre("intarget", "semeval_ambig", a)
    for run in LLM_RUNS: M_a.update(llm_methods("semeval_ambig", run))
    keep_a = np.isin(tga, list(SEEN4)); keep_t = np.isin(tg, list(SEEN4))
    for name, ma in M_a.items():
        m = M_test[name]
        lab = np.concatenate([np.zeros(keep_t.sum()), np.ones(keep_a.sum())])
        r = {f"ambig_AUROC_{k}": float(C.auroc(np.concatenate([m[k][keep_t], ma[k][keep_a]]), lab)) for k in ("AU", "EU", "TU")}
        r["ambig_AUROC_width"] = float(C.auroc(np.concatenate([width_score(m)[keep_t], width_score(ma)[keep_a]]), lab))
        r["mean_AU_test"] = float(m["AU"][keep_t].mean()); r["mean_AU_ambig"] = float(ma["AU"][keep_a].mean())
        r["mean_EU_test"] = float(m["EU"][keep_t].mean()); r["mean_EU_ambig"] = float(ma["EU"][keep_a].mean())
        rho, _ = spearmanr(ma["AU"][keep_a], H_amb[keep_a]); r["rho_AU_H_ambig"] = float(rho)
        r["acc_majority_ambig"] = float((ma["p"][keep_a].argmax(-1) == ya[keep_a]).mean())
        results["ambig"][name] = r

    # ---------- E2/E4 LOTO
    per_fold = {}
    for short, full in TARGETS.items():
        prefix = f"loto_{short}"
        if not os.path.exists(os.path.join(RUNS, prefix, "ce_s0", "config.json")): continue
        calf = np.load(os.path.join(RUNS, prefix, "ce_s0", "semeval_cal.npz"))
        mk = calf["target"] != full
        Tf = C.temperature_scale(calf["logits"][mk], calf["y"][mk])
        Mf, yf, tgf = build(prefix, "semeval_test", temp_cal=Tf)
        for a in (0.5, 1e-2, 1e-4, 1e-6):
            Mf[f"EffCre(a={a:g})"] = effcre(prefix, "semeval_test", a, exclude=full)
        unseen = (tgf == full)
        for name, m in Mf.items():
            r = {f"unseen_AUROC_{k}": float(C.auroc(m[k], unseen)) for k in ("EU", "TU", "AU")}
            r["unseen_AUROC_width"] = float(C.auroc(width_score(m), unseen))
            r["mean_AU_seen"] = float(m["AU"][~unseen].mean()); r["mean_AU_unseen"] = float(m["AU"][unseen].mean())
            pm_u = point_metrics({k: (v[unseen] if isinstance(v, np.ndarray) and len(v) == len(yf) else v) for k, v in m.items()}, yf[unseen])
            pm_s = point_metrics({k: (v[~unseen] if isinstance(v, np.ndarray) and len(v) == len(yf) else v) for k, v in m.items()}, yf[~unseen])
            r.update({f"unseen_{k}": v for k, v in pm_u.items()}); r.update({f"seen_{k}": v for k, v in pm_s.items()})
            r["mean_EU_seen"] = float(m["EU"][~unseen].mean()); r["mean_EU_unseen"] = float(m["EU"][unseen].mean())
            cc = cost_curve({k: (v[unseen] if isinstance(v, np.ndarray) and len(v) == len(yf) else v) for k, v in m.items()}, yf[unseen])
            r.update({f"cost_unseen_{k}": v for k, v in cc.items()})
            per_fold.setdefault(name, []).append(r)
        # figure: EU distributions for one fold
        if short == "hil":
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
            for i, name in enumerate(["DE-MI", "CreWrap-DE", "CreDRO-Soft"]):
                if name not in Mf: continue
                ax[i].hist(Mf[name]["EU"][~unseen], bins=30, alpha=.6, density=True, label="seen targets")
                ax[i].hist(Mf[name]["EU"][unseen], bins=30, alpha=.6, density=True, label=f"unseen ({short})")
                ax[i].set_title(name); ax[i].set_xlabel("epistemic uncertainty"); ax[i].legend(fontsize=7)
            plt.tight_layout(); plt.savefig(os.path.join(FIG, "eu_seen_vs_unseen.png"), dpi=150); plt.close()
    for name, lst in per_fold.items():
        keys = lst[0].keys()
        results["loto"][name] = {k: dict(mean=float(np.mean([d[k] for d in lst])), std=float(np.std([d[k] for d in lst]))) for k in keys}
    results["loto_folds"] = per_fold

    # ---------- figures
    # risk-coverage
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name in ["Softmax", "DE-MI", "CreWrap-DE", "CreDRO", "CreDRO-Soft", "CreDE"]:
        if name not in M_test: continue
        m = M_test[name]; correct = (m["p"].argmax(-1) == y); order = np.argsort(m["TU"])
        risk = np.cumsum(1 - correct[order]) / np.arange(1, len(y) + 1); cov = np.arange(1, len(y) + 1) / len(y)
        ax.plot(cov, risk, label=name)
    ax.set_xlabel("coverage"); ax.set_ylabel("selective risk"); ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "risk_coverage_intarget.png"), dpi=150); plt.close()
    # AU vs human entropy (binned)
    fig, ax = plt.subplots(figsize=(5, 3.6)); ok = ~np.isnan(H_test)
    bins = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.6])
    for name in ["CreWrap-DE", "CreWrap-Soft", "CreDRO-Soft"]:
        if name not in M_test: continue
        au = M_test[name]["AU"][ok]; h = H_test[ok]; idx = np.digitize(h, bins) - 1
        ax.plot([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)],
                [au[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)], marker="o", label=name)
    ax.set_xlabel("human label entropy (bits)"); ax.set_ylabel("predicted aleatoric uncertainty"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "au_vs_human_entropy.png"), dpi=150); plt.close()

    json.dump(results, open(os.path.join(RES, "results.json"), "w"), indent=1)
    write_tables(results)


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_tables(R):
    L = []
    names = [n for n in METHOD_ORDER if n in R["intarget"]] + [n for n in R["intarget"] if n.startswith("EffCre")] + [n for n in R["intarget"] if n.startswith("LLM")]
    L.append("## E1. In-target (SemEval-2016 test)\n")
    L.append("| Method | Acc | F_avg | macro-F1 | ECE | NLL | AUROC-err (TU) | AURC (TU) | Determ. | Set acc | u65 | width |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = R["intarget"][n]
        L.append(f"| {n} | {fmt(r['acc'])} | {fmt(r['Favg'])} | {fmt(r['macroF1'])} | {fmt(r['ECE'])} | {fmt(r['NLL'])} | {fmt(r['misclf_AUROC_TU'])} | {fmt(r['AURC_TU'])} | {fmt(r.get('determinacy'))} | {fmt(r.get('set_acc'))} | {fmt(r.get('u65'))} | {fmt(r.get('width'))} |")
    L.append("\n## E4a. Unseen-target detection, leave-one-target-out (mean ± std over 5 folds)\n")
    L.append("| Method | AUROC unseen (EU) | AUROC unseen (width) | AUROC unseen (TU) | Acc unseen | F_avg unseen | AURC unseen (TU) | mean AU seen→unseen | mean EU seen→unseen |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        if n not in R["loto"]: continue
        r = R["loto"][n]
        f = lambda k: f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}"
        L.append(f"| {n} | {f('unseen_AUROC_EU')} | {f('unseen_AUROC_width')} | {f('unseen_AUROC_TU')} | {f('unseen_acc')} | {f('unseen_Favg')} | {f('unseen_AURC_TU')} | {r['mean_AU_seen']['mean']:.2f}→{r['mean_AU_unseen']['mean']:.2f} | {r['mean_EU_seen']['mean']:.2f}→{r['mean_EU_unseen']['mean']:.2f} |")
    L.append("\n## E4b. Cross-dataset unseen targets (SemEval-trained; unseen = P-Stance / COVID-19 test)\n")
    L.append("| Method | AUROC P-Stance (EU) | (width) | (TU) | Acc P-Stance | AUROC COVID (EU) | (width) | (TU) | Acc COVID | macro-F1 COVID |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        a = R["cross_dataset"]["pstance_test"].get(n); b = R["cross_dataset"]["covid_test"].get(n)
        if not a: continue
        L.append(f"| {n} | {fmt(a['unseen_AUROC_EU'])} | {fmt(a['unseen_AUROC_width'])} | {fmt(a['unseen_AUROC_TU'])} | {fmt(a['acc_other'])} | {fmt(b['unseen_AUROC_EU'])} | {fmt(b['unseen_AUROC_width'])} | {fmt(b['unseen_AUROC_TU'])} | {fmt(b['acc_other'])} | {fmt(b['macroF1_other'])} |")
    L.append("\n## E5. Human-disagreement alignment (SemEval test items with raw votes, n≈1080)\n")
    L.append("| Method | ρ(AU, H) | ρ(EU, H) | ρ(TU, H) | JSD(p*, λ) | TV(p*, λ) |")
    L.append("|---|---|---|---|---|---|")
    for n in names:
        r = R["human"][n]
        L.append(f"| {n} | {fmt(r['rho_AU_H'])} | {fmt(r['rho_EU_H'])} | {fmt(r['rho_TU_H'])} | {fmt(r['JSD_p_lambda'])} | {fmt(r['TV_p_lambda'])} |")
    L.append("\n## E5b. Ambiguity stress test (1,101 low-agreement tweets on seen targets vs test)\n")
    L.append("| Method | AUROC ambig (AU) | AUROC ambig (EU) | AUROC ambig (width) | AUROC ambig (TU) | mean AU test | mean AU ambig | mean EU test | mean EU ambig | ρ(AU,H) on ambig |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = R["ambig"][n]
        L.append(f"| {n} | {fmt(r['ambig_AUROC_AU'])} | {fmt(r['ambig_AUROC_EU'])} | {fmt(r['ambig_AUROC_width'])} | {fmt(r['ambig_AUROC_TU'])} | {fmt(r['mean_AU_test'])} | {fmt(r['mean_AU_ambig'])} | {fmt(r['mean_EU_test'])} | {fmt(r['mean_EU_ambig'])} | {fmt(r['rho_AU_H_ambig'])} |")
    L.append("\n## E6. Conformal coverage of the human label distribution (100 random re-splits of cal ∪ test voted items)\n")
    L.append("| Method | cov@0.9 (TV ball) | q@0.9 | set size@0.9 | cov@0.8 | q@0.8 | set size@0.8 | cov@0.9 (credal dilate) | q | set size |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = R["conformal"][n]; a = r["TVball@0.1"]; b = r["TVball@0.2"]; c = r.get("CredalDilate@0.1")
        L.append(f"| {n} | {fmt(a['coverage'])} | {fmt(a['q'])} | {fmt(a['set_size'],2)} | {fmt(b['coverage'])} | {fmt(b['q'])} | {fmt(b['set_size'],2)} | {fmt(c['coverage']) if c else '-'} | {fmt(c['q']) if c else '-'} | {fmt(c['set_size'],2) if c else '-'} |")
    L.append("\n## E7. Expected decision cost (in-target test; c_err=1, c_defer=0.3, c_set=0.15 per extra class)\n")
    L.append("| Method | conf@0 | conf@5% | conf@10% | conf@20% | credal@0 | credal@5% | credal@10% | credal@20% |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = R["cost"][n]
        L.append(f"| {n} | {fmt(r['conf@0.0'])} | {fmt(r['conf@0.05'])} | {fmt(r['conf@0.1'])} | {fmt(r['conf@0.2'])} | {fmt(r.get('credal@0.0'))} | {fmt(r.get('credal@0.05'))} | {fmt(r.get('credal@0.1'))} | {fmt(r.get('credal@0.2'))} |")
    open(os.path.join(RES, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
