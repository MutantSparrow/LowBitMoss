"""Write the per-linear bit allocations (alloc_<tag>.json, used as train_blocks.py --kind @alloc_<tag>.json).

mixA..mixE: greedy allocation from the sensitivity map (alloc.py) at a predicted-error target; needs logs/sens_tern.json.
mixG / mixH: the attention kinds of mixA with the FFN set by block range (FFN is where the text/count quality lives, see docs/METHOD.md)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alloc

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOL = {"bin": "1", "tern5": "t", "int2": "2", "int4": "4"}


def write(tag, cfg):
    json.dump(cfg, open(os.path.join(HERE, "alloc_%s.json" % tag), "w"))


for tag, target, kinds in (("mixA", 0.15, ("bin", "int2", "int4")), ("mixB", 0.10, ("bin", "int2", "int4")), ("mixC", 0.30, ("bin", "int2", "int4")),
                           ("mixD", 0.50, ("bin", "int2", "int4")), ("mixE", 0.30, ("bin", "tern5", "int4"))):
    cur, err, tot = alloc.allocate(target, kinds)
    cfg = {"default": "int4", "blocks": {}}
    for (b, n), k in cur.items():
        cfg["blocks"].setdefault(str(b), {})[n] = "tern" if k == "tern5" else k
    write(tag, cfg)
    counts = {k: sum(1 for v in cur.values() if v == k) for k in kinds}
    print("%s: predicted PTQ err %.3f, block-linear bytes %.0f MB, counts %s" % (tag, err, tot / 1e6, counts))
    for n in alloc.LIN:
        print("   %-12s %s" % (n, "".join(SYMBOL[cur[(b, n)]] for b in range(42))))

# FFN kind per block range (first, last, kind); blocks not listed keep the default int4
FFN = {"mixG": ((2, 39, "int3"),),
       "mixH": ((2, 5, "int3"), (6, 35, "int2"), (36, 39, "int3"))}
for tag, ranges in FFN.items():
    cfg = json.load(open(os.path.join(HERE, "alloc_mixA.json")))
    for first, last, kind in ranges:
        for b in range(first, last + 1):
            cfg["blocks"][str(b)]["ffn_up_gate"] = cfg["blocks"][str(b)]["ffn_down"] = kind
    for b in (0, 1, 40, 41):
        cfg["blocks"][str(b)]["ffn_up_gate"] = cfg["blocks"][str(b)]["ffn_down"] = "int4"
    write(tag, cfg)
    print("%s: attention of mixA, FFN %s" % (tag, ", ".join("blocks %d-%d %s" % r for r in ranges)))
