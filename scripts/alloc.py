import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import LOGS

sens = json.load(open(os.path.join(LOGS, "sens_tern.json")))
LIN = ["qkv", "o", "ffn_up_gate", "ffn_down"]
PARAMS = {"qkv": 3072*9216, "o": 3072*3072, "ffn_up_gate": 3072*16384, "ffn_down": 8192*3072}
# relative weight-MSE of optimal group-128 scalar quantizers (measured: bin .61^2, tern .44^2, int2 .35^2, int4 .105^2) and bits/weight incl. scale
MSE = {"int3": 0.0377, "bin": 0.372, "tern5": 0.194, "tern": 0.194, "int2": 0.125, "int4": 0.011}
BPW = {"int3": 3.125, "bin": 1.125, "tern5": 1.725, "tern": 2.125, "int2": 2.125, "int4": 4.125}
def size(kind): return lambda n: PARAMS[n] * BPW[kind] / 8
def pred(b, n, kind): return sens[str(b)][n] * MSE[kind] / MSE["tern"]
def allocate(target, kinds=("bin", "int2", "int4"), per_linear=True, tern_kind=None):
    # start everything at int4, then greedily downgrade the cheapest-damage-per-byte-saved step until the budget is used
    ks = list(kinds)
    ks.sort(key=lambda k: -BPW[k])
    cur = {(b, n): ks[0] for b in range(42) for n in LIN}
    err = sum(pred(b, n, cur[(b, n)]) for (b, n) in cur)
    while True:
        best = None
        for (b, n), k in cur.items():
            i = ks.index(k)
            if i + 1 >= len(ks): continue
            k2 = ks[i + 1]
            d_err = pred(b, n, k2) - pred(b, n, k); d_sz = size(k)(n) - size(k2)(n)
            ratio = d_err / d_sz
            if best is None or ratio < best[0]: best = (ratio, (b, n), k2, d_err)
        if best is None or err + best[3] > target: break
        cur[best[1]] = best[2]; err += best[3]
    tot = sum(size(k)(n) for (b, n), k in cur.items())
    return cur, err, tot
if __name__ == "__main__":
    for kinds in (("bin", "int2", "int4"), ("bin", "tern5", "int4"), ("bin", "int4")):
        print("kinds", kinds)
        for target in (0.02, 0.04, 0.08, 0.15):
            cur, err, tot = allocate(target, kinds)
            per_block = {}
            for (b, n), k in cur.items(): per_block.setdefault(b, set()).add(k)
            cnt = {k: sum(1 for v in cur.values() if v == k) for k in kinds}
            print("  predicted PTQ-level err %.3f  block-linear bytes %.0f MB (+0.64GB bf16 rest)  counts %s" % (err, tot / 1e6, cnt))
