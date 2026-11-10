"""Build unified JSONL splits for all stance datasets + human label distributions.

Outputs (data/processed/):
  semeval_train.jsonl, semeval_cal.jsonl, semeval_test.jsonl   (official split; 20% of train -> cal)
  semeval_ambig.jsonl                                           (1,385 low-agreement tweets, raw votes only)
  pstance_test.jsonl, covid_test.jsonl                          (cross-dataset unseen targets)
  pstance_train.jsonl, covid_train.jsonl                        (for optional in-domain runs)
Labels: FAVOR=0, AGAINST=1, NONE=2.  `label_dist` = human label distribution when raw votes exist.
"""
import csv, json, math, os, re, random, collections
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
LABELS = {"FAVOR": 0, "AGAINST": 1, "NONE": 2, "NEUTRAL": 2}
random.seed(0)


def clean(t):
    t = str(t).replace("#SemST", "").strip()
    t = re.sub(r"http\S+", "HTTPURL", t)
    t = re.sub(r"@\w+", "@USER", t)
    return re.sub(r"\s+", " ", t).strip()


def dump(rows, name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name:22s} {len(rows):6d}")


# ---------------- SemEval raw annotations -> label distributions ----------------
raw = list(csv.DictReader(open(os.path.join(ROOT, "data", "semeval_raw_annotations",
                                            "SemEval2016-Task6-raw-annotations-stance.csv"),
                               encoding="utf-8", errors="ignore")))
votes_by_id = collections.defaultdict(list)
votes_noid = collections.defaultdict(list)
noid_meta = {}
for r in raw:
    lab = LABELS[r["Stance"].strip().upper()]
    iid = r["Instance ID"].strip()
    if iid:
        votes_by_id[iid].append(lab)
    else:
        key = (r["Target"].strip(), r["Tweet"].strip())
        votes_noid[key].append(lab)


def dist(votes):
    c = [votes.count(k) for k in range(3)]
    n = sum(c)
    d = [x / n for x in c]
    h = -sum(p * math.log2(p) for p in d if p > 0)
    return d, n, h


# ---------------- SemEval official ----------------
tr = pd.read_csv(os.path.join(RAW, "semeval_train.txt"), sep="\t", encoding="latin-1")
te = pd.read_csv(os.path.join(RAW, "semeval_test", "SemEval2016-Task6-subtaskA-testdata-gold.txt"),
                 sep="\t", encoding="latin-1")


def semeval_rows(df, split):
    rows = []
    for _, r in df.iterrows():
        iid = str(r["ID"])
        rec = {"id": f"semeval-{iid}", "source": "semeval", "text": clean(r["Tweet"]),
               "target": r["Target"], "label": LABELS[r["Stance"]], "split": split,
               "label_dist": None, "n_votes": 0, "human_entropy": None}
        if iid in votes_by_id:
            d, n, h = dist(votes_by_id[iid])
            rec.update(label_dist=d, n_votes=n, human_entropy=h)
        rows.append(rec)
    return rows


train_all = semeval_rows(tr, "train")
# 20% calibration split, stratified by target, held out from all training
by_t = collections.defaultdict(list)
for r in train_all:
    by_t[r["target"]].append(r)
train, cal = [], []
for t, rs in by_t.items():
    random.shuffle(rs)
    k = int(round(0.2 * len(rs)))
    for r in rs[:k]:
        r["split"] = "cal"
    cal += rs[:k]
    train += rs[k:]
test = semeval_rows(te, "test")
dump(train, "semeval_train.jsonl")
dump(cal, "semeval_cal.jsonl")
dump(test, "semeval_test.jsonl")

ambig = []
for i, ((tgt, tw), v) in enumerate(votes_noid.items()):
    d, n, h = dist(v)
    ambig.append({"id": f"semeval-ambig-{i}", "source": "semeval_ambig", "text": clean(tw), "target": tgt,
                  "label": int(max(range(3), key=lambda k: d[k])), "split": "ambig",
                  "label_dist": d, "n_votes": n, "human_entropy": h})
dump(ambig, "semeval_ambig.jsonl")

cov = sum(1 for r in train + cal + test if r["label_dist"])
print(f"semeval items with raw votes: {cov}/{len(train)+len(cal)+len(test)}")

# ---------------- P-Stance (binary favor/against) ----------------
for split in ["train", "test"]:
    rows = []
    for tg in ["trump", "biden", "bernie"]:
        df = pd.read_csv(os.path.join(RAW, "pstance", "PStance", f"raw_{split}_{tg}.csv"))
        for i, r in df.iterrows():
            rows.append({"id": f"pstance-{tg}-{split}-{i}", "source": "pstance", "text": clean(r["Tweet"]),
                         "target": r["Target"], "label": LABELS[r["Stance"].strip().upper()], "split": split,
                         "label_dist": None, "n_votes": 0, "human_entropy": None})
    dump(rows, f"pstance_{split}.jsonl")

# ---------------- COVID-19-Stance ----------------
for split in ["train", "test"]:
    df = pd.read_csv(os.path.join(RAW, "covid", "Covid19_dataset", f"raw_{split}_all_onecol.csv"))
    rows = []
    for i, r in df.iterrows():
        rows.append({"id": f"covid-{split}-{i}", "source": "covid", "text": clean(r["Tweet"]),
                     "target": r["Target 1"], "label": LABELS[r["Stance 1"].strip().upper()], "split": split,
                     "label_dist": None, "n_votes": 0, "human_entropy": None,
                     "opinion_towards": str(r["Opinion Towards"])[:2].strip()})
    dump(rows, f"covid_{split}.jsonl")
