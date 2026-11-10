"""Ensemble-size ablation, bootstrap confidence intervals, paired fold tests, and the AU/EU map figure.
Writes results/ablation_stats.md and results/fig/au_eu_map.png, results/fig/ensemble_size.png
"""
import os, sys, json, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import credal as C
from analysis import RUNS, RES, FIG, TARGETS, SEEN4, load_meta, members, effcre

rng = np.random.default_rng(0)
L = []


def credal_from(ms, setname, M):
    P = C.softmax(np.stack([m["logits"] for m in ms[:M]]))
    l, u = C.credal_bounds(P); l, u = C.reachable(l, u); d = C.decompose(l, u)
    return dict(p=C.intersection_probability(l, u), l=l, u=u, width=(u - l).sum(-1), **{k: d[k] for k in ("TU", "AU", "EU")}), P


# ------------------------------------------------------------------ 1. ensemble size ablation
L.append("## Ensemble-size ablation (members 1..M; in-target ECE, COVID unseen AUROC (TU), LOTO unseen AUROC (TU), mean width)\n")
L.append("| Method | M | ECE | AURC (TU) | misclf AUROC (TU) | COVID unseen AUROC (TU) | COVID unseen AUROC (width) | LOTO unseen AUROC (TU) | LOTO (width) | width |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
sizes = {}
for meth, name in [("ce", "CreWrap-DE"), ("dro", "CreDRO"), ("softdro", "CreDRO-Soft"), ("soft", "CreWrap-Soft")]:
    ms_t = members("intarget", meth, "semeval_test", M=8); ms_c = members("intarget", meth, "covid_test", M=8)
    y = ms_t[0]["y"]; Mmax = len(ms_t)
    for M in [1, 2, 3, 4, 5, 8]:
        if M > Mmax: continue
        mt, _ = credal_from(ms_t, "semeval_test", M); mc, _ = credal_from(ms_c, "covid_test", M)
        correct = mt["p"].argmax(-1) == y
        lab = np.concatenate([np.zeros(len(y)), np.ones(len(mc["p"]))])
        au_tu = C.auroc(np.concatenate([mt["TU"], mc["TU"]]), lab); au_w = C.auroc(np.concatenate([mt["width"], mc["width"]]), lab)
        lo_tu, lo_w = [], []
        for short, full in TARGETS.items():
            msf = members(f"loto_{short}", meth, "semeval_test")
            if len(msf) < M: continue
            mf, _ = credal_from(msf, "semeval_test", M); unseen = msf[0]["target"] == full
            lo_tu.append(C.auroc(mf["TU"], unseen)); lo_w.append(C.auroc(mf["width"], unseen))
        row = dict(M=M, ECE=C.ece(mt["p"], y), AURC=C.aurc(mt["TU"], correct), mis=C.auroc(mt["TU"], ~correct),
                   covid_tu=au_tu, covid_w=au_w, loto_tu=np.mean(lo_tu) if lo_tu else np.nan, loto_w=np.mean(lo_w) if lo_w else np.nan, width=mt["width"].mean())
        sizes.setdefault(name, []).append(row)
        L.append(f"| {name} | {M} | {row['ECE']:.3f} | {row['AURC']:.3f} | {row['mis']:.3f} | {row['covid_tu']:.3f} | {row['covid_w']:.3f} | {row['loto_tu']:.3f} | {row['loto_w']:.3f} | {row['width']:.3f} |")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for name, rows in sizes.items():
    Ms = [r["M"] for r in rows]
    ax[0].plot(Ms, [r["ECE"] for r in rows], marker="o", label=name)
    ax[1].plot(Ms, [r["loto_tu"] for r in rows], marker="o", label=name)
    ax[2].plot(Ms, [r["covid_w"] for r in rows], marker="o", label=name)
ax[0].set_title("in-target ECE"); ax[1].set_title("LOTO unseen AUROC (TU)"); ax[2].set_title("COVID unseen AUROC (width)")
for a in ax: a.set_xlabel("ensemble size M")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIG, "ensemble_size.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ 2. bootstrap CIs on in-target test
L.append("\n## In-target bootstrap 95% CIs (1,000 resamples of the test set)\n")
L.append("| Method | Acc | ECE | AURC (TU) | misclf AUROC (TU) |")
L.append("|---|---|---|---|---|")
ce = members("intarget", "ce", "semeval_test"); y = ce[0]["y"]
systems = {}
p0 = C.softmax(ce[0]["logits"]); systems["Softmax"] = dict(p=p0, TU=C.entropy(p0))
P = C.softmax(np.stack([m["logits"] for m in ce])); systems["DE-MI"] = dict(p=P.mean(0), TU=C.entropy(P.mean(0)))
for meth, name in [("ce", "CreWrap-DE"), ("soft", "CreWrap-Soft"), ("dro", "CreDRO"), ("softdro", "CreDRO-Soft")]:
    systems[name], _ = credal_from(members("intarget", meth, "semeval_test"), "semeval_test", 5)
systems["EffCre(a=1e-06)"] = effcre("intarget", "semeval_test", 1e-6)
boot = {}
for name, m in systems.items():
    correct = m["p"].argmax(-1) == y; vals = {"Acc": [], "ECE": [], "AURC": [], "mis": []}
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        vals["Acc"].append(correct[idx].mean()); vals["ECE"].append(C.ece(m["p"][idx], y[idx]))
        vals["AURC"].append(C.aurc(m["TU"][idx], correct[idx])); vals["mis"].append(C.auroc(m["TU"][idx], ~correct[idx]))
    boot[name] = {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in vals.items()}
    f = lambda k: f"{boot[name][k][0]:.3f} [{boot[name][k][1]:.3f}, {boot[name][k][2]:.3f}]"
    L.append(f"| {name} | {f('Acc')} | {f('ECE')} | {f('AURC')} | {f('mis')} |")

# ------------------------------------------------------------------ 3. paired fold tests on LOTO unseen-target AUROC (TU)
R = json.load(open(os.path.join(RES, "results.json")))
folds = R["loto_folds"]
L.append("\n## Paired tests over the 5 LOTO folds: unseen-target AUROC (TU)\n")
L.append("| Comparison | mean diff | folds won | paired t-test p | Wilcoxon p |")
L.append("|---|---|---|---|---|")
for a, b in [("CreDRO-Soft", "Softmax"), ("CreDRO-Soft", "DE-MI"), ("CreDRO", "DE-MI"), ("CreDRO-Soft", "CreWrap-DE"), ("CreDE", "DE-MI"), ("CreDRO-Soft", "CreDRO")]:
    if a not in folds or b not in folds: continue
    va = np.array([d["unseen_AUROC_TU"] for d in folds[a]]); vb = np.array([d["unseen_AUROC_TU"] for d in folds[b]])
    t = stats.ttest_rel(va, vb); w = stats.wilcoxon(va, vb) if len(va) >= 5 else None
    L.append(f"| {a} vs {b} | {np.mean(va-vb):+.3f} | {(va>vb).sum()}/{len(va)} | {t.pvalue:.3f} | {w.pvalue:.3f} |")

L.append("\n## Per-fold unseen-target AUROC (TU) and unseen accuracy\n")
L.append("| Method | " + " | ".join(f"{s} AUROC/Acc" for s in TARGETS) + " |")
L.append("|---|" + "---|" * len(TARGETS))
for name in ["Softmax", "DE-MI", "CreWrap-DE", "CreDE", "CreWrap-Soft", "CreDRO", "CreDRO-Soft", "EffCre(a=1e-06)"]:
    if name not in folds: continue
    L.append(f"| {name} | " + " | ".join(f"{d['unseen_AUROC_TU']:.3f}/{d['unseen_acc']:.3f}" for d in folds[name]) + " |")

# ------------------------------------------------------------------ 4. AU/EU map: test vs unseen (LOTO) vs ambiguous
rows_amb, lam_amb, H_amb = load_meta("semeval_ambig")
fig, ax = plt.subplots(figsize=(5.5, 4))
markers = {"CreWrap-DE": "o", "CreDRO": "s", "CreDRO-Soft": "^", "CreWrap-Soft": "D"}
for meth, name in [("ce", "CreWrap-DE"), ("dro", "CreDRO"), ("softdro", "CreDRO-Soft"), ("soft", "CreWrap-Soft")]:
    mt, _ = credal_from(members("intarget", meth, "semeval_test"), "semeval_test", 5)
    ma, _ = credal_from(members("intarget", meth, "semeval_ambig"), "semeval_ambig", 5)
    keep = np.isin(members("intarget", meth, "semeval_ambig")[0]["target"], list(SEEN4))
    pts = {"seen test": (mt["AU"].mean(), mt["width"].mean()), "ambiguous": (ma["AU"][keep].mean(), ma["width"][keep].mean())}
    us = []
    for short, full in TARGETS.items():
        msf = members(f"loto_{short}", meth, "semeval_test"); mf, _ = credal_from(msf, "semeval_test", 5); unseen = msf[0]["target"] == full
        us.append((mf["AU"][unseen].mean(), mf["width"][unseen].mean()))
    pts["unseen target"] = tuple(np.mean(us, 0))
    cols = {"seen test": "C0", "ambiguous": "C1", "unseen target": "C3"}
    for k, (a, w) in pts.items():
        ax.scatter(a, w, marker=markers[name], color=cols[k], s=60, label=f"{name} – {k}")
ax.set_xlabel("mean aleatoric uncertainty (lower entropy, bits)"); ax.set_ylabel("mean credal width Σ(u−l)")
ax.legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig(os.path.join(FIG, "au_eu_map.png"), dpi=150); plt.close()

open(os.path.join(RES, "ablation_stats.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
