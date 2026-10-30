"""Distance of encoder conditionings from a reference (bf16 encoder), per prompt over all tokens: relative MSE (mean, median, p95 and max over the prompts) and the token cosine (mean, and the worst prompt).
    python -I cond_error.py ../data/enc_cond/bf16.pt ../data/enc_cond/<name>.pt [...]"""
import sys

import torch

ref = torch.load(sys.argv[1])["conds"]
for path in sys.argv[2:]:
    other = torch.load(path)["conds"]
    rel, cos, per_prompt = [], [], []
    for a, b in zip(ref, other):
        a, b = a.float().reshape(-1, a.shape[-1]), b.float().reshape(-1, b.shape[-1])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        r = ((a - b) ** 2).sum() / (a ** 2).sum()
        c = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
        rel.append(r.item())
        cos.append(c.item())
    t = torch.tensor(rel)
    print("%-34s rel MSE mean %.5f  median %.5f  p95 %.5f  max %.5f | token cosine mean %.5f  min-prompt %.5f  (%d prompts)" % (
        path.split("\\")[-1].split("/")[-1], t.mean(), t.median(), t.quantile(0.95), t.max(), sum(cos) / len(cos), min(cos), len(rel)))
