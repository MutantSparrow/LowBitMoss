import json, struct, sys, re, collections
p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
h.pop("__metadata__", None)
tot = collections.Counter(); cnt = collections.Counter()
rows = []
for k, v in h.items():
    numel = 1
    for s in v["shape"]: numel *= s
    cls = re.sub(r"layers\.\d+\.", "layers.N.", k)
    tot[cls] += numel; cnt[cls] += 1
    rows.append((k, v["shape"], v["dtype"]))
print("total params: %.3fB" % (sum(tot.values())/1e9), "tensors:", len(h))
for c, t in sorted(tot.items(), key=lambda x: -x[1]):
    ex = next(r for r in rows if re.sub(r"layers\.\d+\.", "layers.N.", r[0]) == c)
    print("%-48s x%-3d %8.1fM  %s %s" % (c, cnt[c], t/1e6, ex[1], ex[2]))
