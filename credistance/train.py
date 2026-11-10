"""Train one ensemble member and cache its outputs on every evaluation set.

Methods
  ce      : cross-entropy on hard labels                                   (Deep-Ensemble member)
  soft    : cross-entropy on human label distributions where available     (label-distribution learning)
  dro     : CreDRO member - CE on the top-floor(delta*B) hardest samples per batch
  crenet  : CreDE member  - interval head (midpoint/half-length -> interval softmax),
            loss = CE(qU) + CE(qL) on hardest delta fraction            (Wang et al., NeurIPS 2024)
  edl     : evidential deep learning (Dirichlet head, Sensoy et al. 2018)
Outputs  : <out>/<set>.npz with logits (N,K) [or qL,qU for crenet; alpha for edl]; mc_logits (T,N,K) if --mc_samples
"""
import argparse, json, os, random, time, math, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
EVAL_SETS = ["semeval_train", "semeval_cal", "semeval_test", "semeval_ambig", "pstance_test", "covid_test"]
K = 3


def load(name):
    return [json.loads(l) for l in open(os.path.join(PROC, name + ".jsonl"), encoding="utf-8")]


class StanceModel(nn.Module):
    def __init__(self, backbone, method, dropout=0.1):
        super().__init__()
        # transformers>=5 loads the checkpoint's stored dtype (fp16 for deberta-v3); force fp32 master weights
        self.enc = AutoModel.from_pretrained(backbone, dtype=torch.float32)
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.method = method
        self.head = nn.Linear(H, 2 * K if method == "crenet" else K)

    def forward(self, **b):
        h = self.enc(**b).last_hidden_state[:, 0]
        return self.head(self.drop(h))


def interval_softmax(out):
    """CreNet: out (B,2K) -> qL,qU via Eq. (4) of Wang et al. 2024."""
    m, h = out[:, :K], F.softplus(out[:, K:])
    aL, aU = m - h, m + h
    mid = m  # (aU + aL) / 2
    # q_i = exp(a_i) / (exp(a_i) + sum_{k!=i} exp(mid_k))  computed stably as sigmoid(a_i - logsumexp_{k!=i} mid_k)
    B = out.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=out.device)
    mid_others = mid[:, None, :].expand(B, K, K).masked_fill(~mask, float("-inf"))
    logS = torch.logsumexp(mid_others, dim=2)  # (B,K): logsumexp over k != i
    qL = torch.sigmoid(aL - logS)
    qU = torch.sigmoid(aU - logS)
    return qL, qU


def edl_loss(out, y_onehot, epoch, anneal_epochs=5):
    evidence = F.softplus(out)
    alpha = evidence + 1
    S = alpha.sum(1, keepdim=True)
    p = alpha / S
    err = ((y_onehot - p) ** 2).sum(1)
    var = (p * (1 - p) / (S + 1)).sum(1)
    alpha_t = y_onehot + (1 - y_onehot) * alpha
    St = alpha_t.sum(1, keepdim=True)
    kl = (torch.lgamma(St.squeeze(1)) - math.lgamma(K) - torch.lgamma(alpha_t).sum(1)
          + ((alpha_t - 1) * (torch.digamma(alpha_t) - torch.digamma(St))).sum(1))
    lam = min(1.0, epoch / anneal_epochs)
    return (err + var + lam * kl).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ce", choices=["ce", "soft", "dro", "softdro", "crenet", "edl"])
    ap.add_argument("--backbone", default="microsoft/deberta-v3-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=1.0, help="hard-sample fraction for dro/crenet")
    ap.add_argument("--bootstrap", type=float, default=0.0, help="fraction of train sampled per member (0=all)")
    ap.add_argument("--exclude_target", default="", help="LOTO: target held out from training")
    ap.add_argument("--train_sets", default="semeval_train")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--mc_samples", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.backbone)
    model = StanceModel(a.backbone, a.method).to(dev)

    # ---------------- training data
    train = []
    for s in a.train_sets.split(","):
        train += load(s)
    if a.exclude_target:
        train = [r for r in train if r["target"] != a.exclude_target]
    if a.bootstrap > 0:
        rng = random.Random(a.seed)
        train = rng.sample(train, int(len(train) * a.bootstrap))
    print(f"[{a.method} seed={a.seed} delta={a.delta}] train items: {len(train)}", flush=True)

    def encode(rows):
        return tok([r["target"] for r in rows], [r["text"] for r in rows], truncation=True,
                   max_length=a.max_len, padding=True, return_tensors="pt")

    def targets(rows):
        y = torch.tensor([r["label"] for r in rows])
        soft = torch.tensor([r["label_dist"] if (a.method in ("soft", "softdro") and r["label_dist"]) else
                             [1.0 if r["label"] == k else 0.0 for k in range(K)] for r in rows])
        return y, soft

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); random.shuffle(train); tot = 0.0
        for i in range(0, len(train), a.bs):
            rows = train[i:i + a.bs]
            b = {k: v.to(dev) for k, v in encode(rows).items()}
            y, soft = targets(rows); y, soft = y.to(dev), soft.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**b).float()
            if a.method in ("ce", "soft"):
                loss = -(soft * F.log_softmax(out, -1)).sum(1).mean()
            elif a.method in ("dro", "softdro"):
                # CreDRO: CVaR-style DRO, closed form = top-floor(delta*B) hardest samples
                per = -(soft * F.log_softmax(out, -1)).sum(1)
                k = max(1, int(a.delta * len(rows)))
                loss = torch.topk(per, k).values.mean()
            elif a.method == "crenet":
                qL, qU = interval_softmax(out)
                ceU = -torch.log(qU.gather(1, y[:, None]).squeeze(1) + 1e-8)
                ceL = -torch.log(qL.gather(1, y[:, None]).squeeze(1) + 1e-8)
                k = max(1, int(a.delta * len(rows)))
                loss = ceU.mean() + torch.topk(ceL, k).values.mean()
            elif a.method == "edl":
                loss = edl_loss(out, soft, ep)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item() * len(rows)
        print(f"  epoch {ep+1}/{a.epochs} loss {tot/len(train):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- inference on every evaluation set
    @torch.no_grad()
    def infer(rows, mc=False):
        model.eval()
        if mc:  # enable dropout only
            for m in model.modules():
                if isinstance(m, nn.Dropout): m.train()
        outs = []
        for i in range(0, len(rows), 128):
            b = {k: v.to(dev) for k, v in encode(rows[i:i + 128]).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(model(**b).float().cpu())
        return torch.cat(outs)

    for s in EVAL_SETS:
        rows = load(s)
        out = infer(rows)
        y = np.array([r["label"] for r in rows])
        save = {"y": y, "target": np.array([r["target"] for r in rows]), "ids": np.array([r["id"] for r in rows])}
        if a.method == "crenet":
            qL, qU = interval_softmax(out); save["qL"] = qL.numpy(); save["qU"] = qU.numpy()
            save["logits"] = out[:, :K].numpy()  # midpoint logits as point estimate
        elif a.method == "edl":
            save["alpha"] = (F.softplus(out) + 1).numpy(); save["logits"] = torch.log(F.softplus(out) + 1).numpy()
        else:
            save["logits"] = out.numpy()
        if a.mc_samples:
            save["mc_logits"] = np.stack([infer(rows, mc=True).numpy() for _ in range(a.mc_samples)])
        np.savez_compressed(os.path.join(a.out, s + ".npz"), **save)
        if s == "semeval_test":
            acc = (save["logits"].argmax(1) == y).mean()
            print(f"  {s}: acc={acc:.4f}", flush=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # avoid occasional hang in CUDA/tokenizer teardown
