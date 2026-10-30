"""Teacher (bf16) final outputs for every captured calibration sample, in sorted-file / in-file order -> data/teacher_out_<res>.pt"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from _paths import DATA
from common import *  # noqa: F401,F403

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(DATA, "calib_512"))
ap.add_argument("--out", default=os.path.join(DATA, "teacher_out_512_plus.pt"), help="Output file; default matches the released expanded pool")
args = ap.parse_args()

m = build_model(load_teacher_sd(), device="cuda")
outs = []
files = sorted(glob.glob(os.path.join(args.data, "p*.pt")))
with torch.no_grad():
    for fi, f in enumerate(files):
        for s in torch.load(f)["samples"]:
            x, t, ctx = s["x"].unsqueeze(0).cuda(), s["t"].reshape(1).cuda(), s["ctx"].unsqueeze(0).cuda()
            tok, mod, pe, head, hw = prefix(m, x, t, ctx)
            for blk in m.layers:
                tok = blk(tok, mod, pe)
            outs.append(suffix(m, tok, head, hw)[0].cpu())
        if fi % 20 == 0:
            print(fi, len(outs), flush=True)
torch.save(outs, args.out)
print("saved", len(outs), "targets ->", args.out)
