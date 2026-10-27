"""Stage 2 (v2): end-to-end distillation of a moss_lowbit_v1 model against the bf16 teacher's outputs on captured sampling trajectories.
Packed codes stay frozen. Trainable: group scales (log), the bf16 low-rank branches lora_a/lora_b (if the file has them), optionally the small bf16 tensors.
Loss per (cond, uncond) pair of one sampler step: rel-MSE(cond) + rel-MSE(uncond) + cfg_w * rel-MSE(CFG-combined velocity), each weighted by a
noise-level weight w(t) = 1 + tw * (1 - t) (low-noise steps carry the fine detail). Writes a new moss_lowbit_v1 file."""
import argparse
import glob
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

import lowbit_pack as L
from _paths import DATA
from common import *  # noqa: F401,F403

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", default=None, help="omit with --epochs 0 for an evaluation only")
ap.add_argument("--data", default=os.path.join(DATA, "calib_512"))
ap.add_argument("--targets", default=os.path.join(DATA, "teacher_out_512.pt"))
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--mb", type=int, default=2, help="(cond, uncond) pairs per optimizer step")
ap.add_argument("--lr", type=float, default=3e-4, help="lr of the log group scales")
ap.add_argument("--lora_lr", type=float, default=5e-5)
ap.add_argument("--train_small", action="store_true", help="also train the bf16 non-block tensors")
ap.add_argument("--tw", type=float, default=2.0, help="noise-level weight w(t) = 1 + tw*(1-t)")
ap.add_argument("--cfg", type=float, default=2.5)
ap.add_argument("--cfg_w", type=float, default=1.0, help="weight of the CFG-combined velocity loss (0 = off)")
ap.add_argument("--holdout_text", action="store_true", help="holdout = the text-rich prompts (p-t*), train on the rest")
ap.add_argument("--max_pairs", type=int, default=0)
ap.add_argument("--text_boost", type=int, default=1, help="oversample calibration prompts that contain quoted text by this factor")
ap.add_argument("--eval_every", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
dev = torch.device("cuda")


class PackedLinear(nn.Module):
    def __init__(self, codes, scale, kind, bias=None, lora_a=None, lora_b=None):
        super().__init__()
        self.kind = kind
        self.register_buffer("codes", codes)
        self.log_s = nn.Parameter(scale.float().clamp_min(1e-8).log())
        self.bias = bias
        self.lora_a = None if lora_a is None else nn.Parameter(lora_a.float())
        self.lora_b = None if lora_b is None else nn.Parameter(lora_b.float())

    @property
    def weight(self):
        return None

    def forward(self, x):
        _, bits, off = L.KINDS[self.kind]
        c = L.unpack_codes(self.codes, bits, self.log_s.shape[1] * L.GROUP)
        out, n = c.shape
        w = ((c.float() - off).reshape(out, n // L.GROUP, L.GROUP) * self.log_s.exp().unsqueeze(-1)).reshape(out, n)
        y = F.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        if self.lora_a is not None:
            y = y + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return y


def get_parent(root, parts):
    for p in parts:
        root = root[int(p)] if p.isdigit() else root[p] if isinstance(root, nn.ModuleDict) else getattr(root, p)
    return root


# ---- model
sd = load_file(args.inp)
quant = {k[:-len(".weight_kind")] for k in sd if k.endswith(".weight_kind")}
QUANT_SUFFIX = ("weight", "weight_scale", "weight_kind", "lora_a", "lora_b")  # a quantized linear's bias stays a plain tensor
plain = {k: v for k, v in sd.items() if not (k.rsplit(".", 1)[0] in quant and k.rsplit(".", 1)[1] in QUANT_SUFFIX)}
m = MOSSImage(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, latent_channels=128, fused_qkv=True,
              dtype=torch.bfloat16, device="cpu", operations=comfy.ops.disable_weight_init)
missing, unexpected = m.load_state_dict({k: v for k, v in plain.items() if k != "rope.bands"}, strict=False)
assert not unexpected, unexpected
assert all(k[:-len(".weight")] in quant for k in missing), [k for k in missing if k[:-len(".weight")] not in quant]
for name in quant:
    parts = name.split(".")
    parent = get_parent(m, parts[:-1])
    old = parent[parts[-1]] if isinstance(parent, nn.ModuleDict) else getattr(parent, parts[-1])
    pl = PackedLinear(sd[name + ".weight"], sd[name + ".weight_scale"], int(sd[name + ".weight_kind"]), old.bias,
                      sd.get(name + ".lora_a"), sd.get(name + ".lora_b"))
    if isinstance(parent, nn.ModuleDict):
        parent[parts[-1]] = pl
    else:
        setattr(parent, parts[-1], pl)
m = m.to(dev).requires_grad_(False).eval()
pls = [mod for mod in m.modules() if isinstance(mod, PackedLinear)]
scale_params = [mod.log_s for mod in pls]
lora_params = [p for mod in pls for p in (mod.lora_a, mod.lora_b) if p is not None]
for p in scale_params + lora_params:
    p.requires_grad_(True)
groups = [{"params": scale_params, "lr": args.lr}]
if lora_params:
    groups.append({"params": lora_params, "lr": args.lora_lr})
if args.train_small:
    small = [p for n, p in m.named_parameters() if not p.requires_grad and p.dtype == torch.bfloat16 and "layers." not in n]
    for p in small:
        p.data = p.data.float()
        p.requires_grad_(True)
    groups.append({"params": small, "lr": args.lr * 0.05})
print("trainable: %d scale tensors (%.1fM), %d lora tensors (%.1fM)%s" % (
    len(scale_params), sum(p.numel() for p in scale_params) / 1e6, len(lora_params), sum(p.numel() for p in lora_params) / 1e6,
    ", small tensors" if args.train_small else ""), flush=True)

# ---- data: samples, teacher targets, (cond, uncond) pairs of the same sampler step
files = sorted(glob.glob(os.path.join(args.data, "p*.pt")))
targets = torch.load(args.targets)
samples, has_text, is_text_file = [], [], []
for f in files:
    d = torch.load(f)
    samples += d["samples"]
    is_text_file += [os.path.basename(f).startswith("p-t")] * len(d["samples"])
    has_text += [bool(re.search(r'"[^"]{2,}"', d["prompt"]))] * len(d["samples"])  # prompts that ask for rendered text
assert len(samples) == len(targets), (len(samples), len(targets))
n_hold_files = max(2, len(files) // 20)
n_hold = sum(len(torch.load(f)["samples"]) for f in files[-n_hold_files:])
n_train_end = len(samples) - n_hold


def make_pairs(lo, hi):
    pairs, i = [], lo
    while i < hi:
        if i + 1 < hi and float(samples[i]["t"]) == float(samples[i + 1]["t"]) and torch.equal(samples[i]["x"], samples[i + 1]["x"]):
            pairs.append((i, i + 1))  # captured order: positive, then negative
            i += 2
        else:
            pairs.append((i, None))
            i += 1
    return pairs


if args.holdout_text:  # hold out the text-rich prompts (files p-t*) instead of the last ordinary files
    all_pairs = make_pairs(0, len(samples))
    pairs_hold = [p for p in all_pairs if is_text_file[p[0]]]
    pairs_train = [p for p in all_pairs if not is_text_file[p[0]]]
    pairs_hold = pairs_hold[::max(1, len(pairs_hold) // 48)]
else:
    pairs_train, pairs_hold = make_pairs(0, n_train_end), make_pairs(n_train_end, len(samples))
    pairs_hold = pairs_hold[::max(1, len(pairs_hold) // 36)]
if args.max_pairs:
    random.shuffle(pairs_train)
    pairs_train = pairs_train[:args.max_pairs]
if args.text_boost > 1:
    pairs_train += [p for p in pairs_train if has_text[p[0]]] * (args.text_boost - 1)
    print("text prompts oversampled x%d" % args.text_boost, flush=True)
print("pairs: train %d, holdout %d (%d fully paired)" % (len(pairs_train), len(pairs_hold), sum(1 for p in pairs_hold if p[1] is not None)), flush=True)


def forward_sample(i, train):
    s = samples[i]
    x, t, ctx = s["x"].unsqueeze(0).to(dev), s["t"].reshape(1).to(dev), s["ctx"].unsqueeze(0).to(dev)
    tok, mod, pe, head, hw = prefix(m, x, t, ctx)
    for blk in m.layers:
        tok = checkpoint(blk, tok, mod, pe, use_reentrant=False) if train else blk(tok, mod, pe)
    return suffix(m, tok, head, hw)[0].float()


def rel(o, y):
    return ((o - y) ** 2).sum() / (y ** 2).sum()


def evaluate(report=False):
    num = den = cnum = cden = 0.0
    buckets = {}
    with torch.no_grad():
        for ic, iu in pairs_hold:
            oc, yc = forward_sample(ic, False), targets[ic].to(dev).float()
            ec, dc = ((oc - yc) ** 2).sum().item(), (yc ** 2).sum().item()
            num += ec
            den += dc
            b = buckets.setdefault(min(2, int(float(samples[ic]["t"]) * 3)), [0.0, 0.0])
            b[0] += ec
            b[1] += dc
            if iu is not None:
                ou, yu = forward_sample(iu, False), targets[iu].to(dev).float()
                num += ((ou - yu) ** 2).sum().item()
                den += (yu ** 2).sum().item()
                comb_o, comb_y = ou + args.cfg * (oc - ou), yu + args.cfg * (yc - yu)
                cnum += ((comb_o - comb_y) ** 2).sum().item()
                cden += (comb_y ** 2).sum().item()
    if report:
        print("  by noise level (t<1/3, 1/3-2/3, >2/3): " + "  ".join("%.5f" % (buckets[k][0] / buckets[k][1]) for k in sorted(buckets)), flush=True)
    return num / den, cnum / max(cden, 1e-9)


base = evaluate(report=True)
print("holdout rel mse before: %.5f   cfg-combined: %.5f" % base, flush=True)
if args.epochs == 0:
    sys.exit(0)
assert args.out, "--out is required for training"
steps = int(args.epochs * len(pairs_train) / args.mb)
opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.99))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * (0.5 * (1 + math.cos(math.pi * s / max(1, steps))) * 0.98 + 0.02))


def export(path):
    out = dict(sd)
    for name in quant:
        pl = get_parent(m, name.split("."))
        out[name + ".weight_scale"] = pl.log_s.detach().exp().half().cpu()
        if pl.lora_a is not None:
            out[name + ".lora_a"] = pl.lora_a.detach().to(torch.bfloat16).cpu()
            out[name + ".lora_b"] = pl.lora_b.detach().to(torch.bfloat16).cpu()
    if args.train_small:
        for n, p in m.named_parameters():
            if n in plain and p.requires_grad:
                out[n] = p.detach().to(torch.bfloat16).cpu()
    save_file({k: v.contiguous() for k, v in out.items()}, path, metadata={"moss_lowbit_v1": "1"})


order, t0, run, best = [], time.time(), 0.0, (base[0], None)
print("steps: %d (%d pairs per step)" % (steps, args.mb), flush=True)
for step in range(steps):
    if len(order) < args.mb:
        order = pairs_train[:]
        random.shuffle(order)
    batch, order = order[:args.mb], order[args.mb:]
    loss_acc = 0.0
    for ic, iu in batch:
        w = 1.0 + args.tw * (1.0 - float(samples[ic]["t"]))
        oc, yc = forward_sample(ic, True), targets[ic].to(dev).float()
        loss = rel(oc, yc)
        if iu is not None:
            ou, yu = forward_sample(iu, True), targets[iu].to(dev).float()
            loss = loss + rel(ou, yu)
            if args.cfg_w:
                loss = loss + args.cfg_w * rel(ou + args.cfg * (oc - ou), yu + args.cfg * (yc - yu))
        loss = loss * w / (1.0 + args.tw * 0.5) / len(batch)
        loss.backward()
        loss_acc += loss.item()
    torch.nn.utils.clip_grad_norm_(scale_params + lora_params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    run += loss_acc
    if (step + 1) % 50 == 0:
        print("step %d/%d  train loss %.5f  %.1f min" % (step + 1, steps, run / 50, (time.time() - t0) / 60), flush=True)
        run = 0.0
    if (step + 1) % args.eval_every == 0 and step + 1 < steps:
        ev = evaluate()
        print("  holdout rel mse %.5f   cfg-combined: %.5f" % ev, flush=True)
final = evaluate(report=True)
print("holdout rel mse after: %.5f   cfg-combined: %.5f   (before %.5f / %.5f)" % (final + base), flush=True)
export(args.out)
print("wrote", args.out, "%.3f GB" % (os.path.getsize(args.out) / 1e9))
