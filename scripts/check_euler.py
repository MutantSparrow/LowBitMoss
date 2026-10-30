"""Check the sampler convention of the captured 4-step trajectories (euler / simple, CFG 1) against the stored teacher outputs, no model needed:
x_{k+1} = x_k + (t_{k+1} - t_k) * s * y_k with y_k = the raw network output (data/teacher_out_turbo768.pt) and t_4 = 0. Prints the relative error for s = +1 and -1."""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA

targets = torch.load(os.path.join(DATA, "teacher_out_turbo768.pt"))
files = sorted(glob.glob(os.path.join(DATA, "calib_turbo_768", "p*.pt")))
n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 12
idx = 0
errs = {1: [[] for _ in range(4)], -1: [[] for _ in range(4)]}
ts = []
for f in files[:n_files]:
    s = torch.load(f)["samples"]
    assert len(s) == 4, len(s)
    t = [float(v["t"]) for v in s] + [0.0]
    ts.append(t)
    for k in range(4):
        y = targets[idx + k].float()
        x = s[k]["x"].float()
        nxt = s[k + 1]["x"].float() if k < 3 else None
        for sign in (1, -1):
            xn = x + (t[k + 1] - t[k]) * sign * y
            if nxt is not None:
                errs[sign][k].append(((xn - nxt).norm() / (nxt - x).norm()).item())
            else:
                errs[sign][k].append(xn.abs().mean().item())  # final clean latent: just its scale
    idx += 4
print("t per step (first file):", ts[0])
for sign in (1, -1):
    print("s=%+d  rel error of x_{k+1} vs captured (k=0,1,2): %s   mean |x_final| %.4f" % (sign, ["%.4f" % (sum(errs[sign][k]) / len(errs[sign][k])) for k in range(3)], sum(errs[sign][3]) / len(errs[sign][3])))
