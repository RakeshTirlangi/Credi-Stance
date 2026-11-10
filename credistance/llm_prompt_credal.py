"""Zero-shot prompt-credal stance detection with an instruction-tuned LLM.
For each of P prompt paraphrases we read the next-token logits of the three verbalizer tokens
(favor / against / none), restrict the softmax to them, and cache a (P, N, 3) tensor per eval set.
The credal set is then built across prompts (credal wrapper over prompt variants).
"""
import argparse, json, os, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
SETS = ["semeval_test", "semeval_ambig", "pstance_test", "covid_test", "semeval_cal"]

SYSTEMS = [
    "You are an expert annotator for stance detection on social media.",
    "You are a careful linguist who labels the stance of tweets.",
    "You are a political science researcher coding opinions in text.",
    "You are a content analyst classifying attitudes towards topics.",
]
QUESTIONS = [
    "What is the author's stance towards the target? Answer with exactly one word: favor, against, or none.",
    "Is the author in favor of, against, or neutral/unrelated to the target? Reply with one word: favor, against, or none.",
    "Decide whether the tweet supports (favor), opposes (against), or neither (none) the target. One word only.",
    "Classify the author's attitude to the target as favor, against, or none. Respond with a single word.",
    "Does the tweet express support (favor), opposition (against), or no clear stance (none) toward the target? Answer: favor, against, or none.",
    "Label the stance: 'favor' if the author supports the target, 'against' if the author opposes it, 'none' otherwise. Reply with one word.",
    "From the tweet, infer the writer's position on the target. Output exactly one of: favor, against, none.",
    "Read the tweet and give the stance toward the target using one word from {favor, against, none}.",
]


def build_prompts(P):
    out = []
    for i in range(P):
        out.append((SYSTEMS[i % len(SYSTEMS)], QUESTIONS[i % len(QUESTIONS)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="runs/llm_qwen3b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    verb = {}
    for w, k in [("favor", 0), ("against", 1), ("none", 2)]:
        ids = tok.encode(w, add_special_tokens=False); ids2 = tok.encode(" " + w, add_special_tokens=False)
        verb[k] = sorted(set([ids[0], ids2[0]]))
    print("verbalizer token ids:", verb, flush=True)
    prompts = build_prompts(a.prompts)
    t0 = time.time()
    for s in SETS:
        rows = [json.loads(l) for l in open(os.path.join(PROC, s + ".jsonl"), encoding="utf-8")]
        allP = np.zeros((len(prompts), len(rows), 3), dtype=np.float32)
        for pi, (sysm, q) in enumerate(prompts):
            texts = []
            for r in rows:
                msgs = [{"role": "system", "content": sysm},
                        {"role": "user", "content": f"Target: {r['target']}\nTweet: {r['text']}\n{q}"}]
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            for i in range(0, len(texts), a.bs):
                b = tok(texts[i:i + a.bs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad():
                    lg = model(**b).logits[:, -1, :].float()
                cls = torch.stack([torch.logsumexp(lg[:, verb[k]], dim=1) for k in range(3)], dim=1)
                allP[pi, i:i + a.bs] = torch.softmax(cls, -1).cpu().numpy()
            acc = (allP[pi].argmax(-1) == np.array([r["label"] for r in rows])).mean()
            print(f"{s} prompt {pi}: acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(os.path.join(a.out, s + ".npz"), P=allP, y=np.array([r["label"] for r in rows]),
                            target=np.array([r["target"] for r in rows]))
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
