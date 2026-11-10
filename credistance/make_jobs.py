"""Emit the list of training commands (one per line) for the full experiment grid."""
import os, sys

M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DELTA_G = 0.5
TARGETS = ["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton",
           "Legalization of Abortion"]
SHORT = {"Atheism": "ath", "Climate Change is a Real Concern": "cli", "Feminist Movement": "fem",
         "Hillary Clinton": "hil", "Legalization of Abortion": "abo"}
PY = "python3 credistance/train.py"
jobs = []


def deltas(M):
    return [DELTA_G + (1 - DELTA_G) * i / (M - 1) for i in range(M)]


def grid(prefix, excl=""):
    ex = f' --exclude_target "{excl}"' if excl else ""
    for i in range(M):
        bs = " --bootstrap 0.8"
        mc = " --mc_samples 10" if (i == 0 and not excl) else ""
        jobs.append(f'{PY} --method ce --seed {i}{bs}{mc}{ex} --out runs/{prefix}/ce_s{i}')
        jobs.append(f'{PY} --method soft --seed {i}{bs}{ex} --out runs/{prefix}/soft_s{i}')
        jobs.append(f'{PY} --method dro --seed {i} --delta {deltas(M)[i]:.3f}{bs}{ex} --out runs/{prefix}/dro_s{i}')
        jobs.append(f'{PY} --method softdro --seed {i} --delta {deltas(M)[i]:.3f}{bs}{ex} --out runs/{prefix}/softdro_s{i}')
        jobs.append(f'{PY} --method crenet --seed {i} --delta 0.5{bs}{ex} --out runs/{prefix}/crenet_s{i}')
    if not excl:
        jobs.append(f'{PY} --method edl --seed 0{ex} --out runs/{prefix}/edl_s0')


grid("intarget")
for t in TARGETS:
    grid(f"loto_{SHORT[t]}", excl=t)

with open("jobs.txt", "w") as f:
    for j in jobs:
        f.write(j + "\n")
print(len(jobs), "jobs written to jobs.txt")
