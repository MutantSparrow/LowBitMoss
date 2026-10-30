"""Block-wise distillation QAT of MOSS-Image SFT.

For every transformer block i (sequentially): the student block gets fake-quantized linears (latent fp32 weights + learnable
group scales, straight-through rounding), its input is the activation produced by the already-quantized blocks <i, and its
target is the bf16 teacher block applied to the teacher's own activation. Loss = relative MSE over all tokens.
Output: a moss_lowbit_v1 safetensors file (and optionally a dequantized bf16 'fake' checkpoint for the stock loader)."""
import argparse
import copy
import glob
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors.torch import save_file

import lowbit_pack as L
from _paths import DATA
from common import *  # noqa: F401,F403
from comfy.ldm.moss_image.model import Block
import qlayer
from qlayer import QLinear

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(DATA, "calib_512"))
ap.add_argument("--kind", default="tern", help="bin|tern|int2|int4, or a json {linear_name: kind} / {'default':..., 'blocks':{i:{name:kind}}}")
ap.add_argument("--n_samples", type=int, default=512)
ap.add_argument("--holdout", type=int, default=32)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--mb", type=int, default=8, help="samples per optimizer step")
ap.add_argument("--lr_w", type=float, default=2e-5)
ap.add_argument("--lr_s", type=float, default=2e-3, help="lr of log-scale")
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--tern5", action="store_true", help="export ternary linears with the compact 5-trits-per-byte packing (kind 5)")
ap.add_argument("--lora_rank", type=int, default=0, help="rank of the bf16 low-rank correction added to every quantized linear (0 = none)")
ap.add_argument("--lora_rank_low", type=int, default=0, help="rank for the bin/ternary/int2 linears (0 = same as --lora_rank)")
ap.add_argument("--lora_lr", type=float, default=1e-4)
ap.add_argument("--loss", default="token", choices=["global", "token", "std"], help="token: mean over tokens of ||o-y||^2/||y||^2; global: sum over all tokens; std: per-channel-variance normalized")
ap.add_argument("--resume", default=None, help="partial pack file written by a previous run (blocks already done are reused)")
ap.add_argument("--blocks", default="0-41")
ap.add_argument("--out", default=None, help="moss_lowbit_v1 safetensors path")
ap.add_argument("--fake_out", default=None, help="dequantized bf16 safetensors path")
ap.add_argument("--log", default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dev = torch.device("cuda")
qlayer.EXPORT_TERN5 = args.tern5
rng = random.Random(args.seed)
torch.manual_seed(args.seed)
lo, hi = [int(v) for v in args.blocks.split("-")]
cfg = json.load(open(args.kind[1:])) if args.kind.startswith("@") else json.loads(args.kind) if args.kind.strip().startswith("{") else {"default": args.kind}


def kind_of(block_i, name):
    d = cfg.get("blocks", {}).get(str(block_i), {})
    k = d.get(name, cfg.get(name, cfg.get("default", "tern")))
    return L.KIND_BY_NAME[k] if isinstance(k, str) else k


# ---- calibration samples -> block-0 inputs (teacher prefix, bf16)
sd = load_teacher_sd()
model = build_model(sd)
files = sorted(glob.glob(os.path.join(args.data, "p*.pt")))
n_hold_files = max(2, len(files) // 20)  # holdout prompts are never trained on
hold, train = [], []
for f in files[-n_hold_files:]:
    hold += torch.load(f)["samples"]
for f in files[:-n_hold_files]:
    train += torch.load(f)["samples"]
rng.shuffle(hold)
rng.shuffle(train)
hold, train = hold[:args.holdout], train[:args.n_samples]
samples = None
print("samples: train %d holdout %d (holdout prompts: last %d of %d files)" % (len(train), len(hold), n_hold_files, len(files)), flush=True)

small = copy.deepcopy(model)
small.layers = torch.nn.ModuleList()
small.to(dev)


def to_inputs(ss):
    out = []
    with torch.no_grad():
        for s in ss:
            x = s["x"].unsqueeze(0).to(dev)
            t = s["t"].reshape(1).to(dev)
            ctx = s["ctx"].unsqueeze(0).to(dev)
            tok, mod, pe, head, hw = prefix(small, x, t, ctx)
            out.append({"tok": tok[0].cpu(), "mod": torch.stack([m[0, 0] for m in mod]).cpu(), "hw": hw, "L": ctx.shape[1]})
    return out


tr, ho = to_inputs(train), to_inputs(hold)
del small, samples
torch.cuda.empty_cache()

pe_cache = {}


def get_pe(hw, Lt):
    key = (hw, Lt)
    if key not in pe_cache:
        if len(pe_cache) > 64:
            pe_cache.clear()
        pe_cache[key] = pe_model.rope(hw[0], hw[1], Lt, dev)
    return pe_cache[key]


pe_model = model  # rope() only uses the parameter-free embedder


def run(block, item, tok):
    mod = tuple(m.to(dev).view(1, 1, -1) for m in item["mod"].unbind(0))
    return block(tok.to(dev).unsqueeze(0), mod, get_pe(item["hw"], item["L"]))[0]


def sample_loss(o, y):
    if args.loss == "global":
        return ((o - y) ** 2).sum() / (y ** 2).sum()
    if args.loss == "std":  # error relative to each channel's spread over tokens (not to the large constant offsets)
        sigma = (y - y.mean(0, keepdim=True)).pow(2).mean(0).sqrt()
        return (((o - y) / sigma.clamp_min(0.1 * sigma.median())) ** 2).mean()
    den = (y ** 2).sum(-1)
    return (((o - y) ** 2).sum(-1) / den.clamp_min(1e-3 * den.mean())).mean()


def rel_err(block, xs, ys, items):
    tot = 0.0
    with torch.no_grad():
        for item, x, y in zip(items, xs, ys):
            tot += sample_loss(run(block, item, x).float(), y.to(dev).float()).item()
    return tot / len(items)


packed = torch.load(args.resume) if args.resume else {}
fake = {}
Xs_tr = [it.pop("tok") for it in tr]  # student stream; the streams are the only copy of the tokens (they are ~22 GB per stream at 768 px / 1536 samples)
Xt_tr = list(Xs_tr)  # teacher stream
Xs_ho = [it.pop("tok") for it in ho]
Xt_ho = list(Xs_ho)
log = []
t_start = time.time()
for bi in range(42):
    tb = Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, device=dev, operations=comfy.ops.disable_weight_init)
    tb.load_state_dict({k[len("layers.%d." % bi):]: v for k, v in sd.items() if k.startswith("layers.%d." % bi)}, strict=True)
    tb.requires_grad_(False).eval()
    with torch.no_grad():
        Yt_tr = [run(tb, it, x).cpu() for it, x in zip(tr, Xt_tr)]
        Yt_ho = [run(tb, it, x).cpu() for it, x in zip(ho, Xt_ho)]
    if not (lo <= bi <= hi):
        Xs_tr, Xs_ho = Yt_tr, Yt_ho
        Xt_tr, Xt_ho = Yt_tr, Yt_ho
        continue
    if all("layers.%d.proj.%s.weight" % (bi, n) in packed for n in LINEARS):  # resumed block: only propagate the stream
        sb = copy.deepcopy(tb)
        for name in LINEARS:
            key = "layers.%d.proj.%s" % (bi, name)
            wd = L.dequantize(packed[key + ".weight"].to(dev), packed[key + ".weight_scale"].to(dev), int(packed[key + ".weight_kind"]), torch.float32)
            if key + ".lora_a" in packed:
                wd = wd + packed[key + ".lora_b"].to(dev).float() @ packed[key + ".lora_a"].to(dev).float()
            sb.proj[name].weight.data = wd.to(torch.bfloat16)
        with torch.no_grad():
            for i, (it, x) in enumerate(zip(tr, Xs_tr)):
                Xs_tr[i] = run(sb, it, x).cpu()  # in place: a second copy of the stream does not fit in RAM
            Xs_ho = [run(sb, it, x).cpu() for it, x in zip(ho, Xs_ho)]
        Xt_tr, Xt_ho = Yt_tr, Yt_ho
        print("block %2d  resumed" % bi, flush=True)
        continue
    sb = copy.deepcopy(tb)
    qls = {}
    for name in LINEARS:
        qls[name] = QLinear(tb.proj[name].weight, kind_of(bi, name)).to(dev)
        sb.proj[name] = qls[name]
    params_w = [q.w for q in qls.values()]
    params_s = [q.log_s for q in qls.values()]
    groups = [{"params": params_w, "lr": args.lr_w}, {"params": params_s, "lr": args.lr_s}]
    if args.lora_rank:
        covs = {n: torch.zeros(3072 if n != "ffn_down" else 8192, 3072 if n != "ffn_down" else 8192, device=dev) for n in LINEARS}
        cnt = [0]
        def collect(mod, inp, n):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            covs[n].addmm_(x.T, x)

        hooks = [tb.proj[n].register_forward_pre_hook(lambda mod, inp, n=n: collect(mod, inp, n)) for n in LINEARS]
        with torch.no_grad():
            for j in range(min(96, len(tr))):
                run(tb, tr[j], Xs_tr[j])
                cnt[0] += Xs_tr[j].shape[0]
        for h in hooks:
            h.remove()
        for n, q in qls.items():
            if q.kind != L.KIND_BY_NAME["int4"]:
                low_bit = q.kind in (L.KIND_BY_NAME["bin"], L.KIND_BY_NAME["tern"], L.KIND_BY_NAME["int2"], L.KIND_BY_NAME["tern5"])
                q.init_lora(args.lora_rank_low if (args.lora_rank_low and low_bit) else args.lora_rank, covs[n] / cnt[0])
        del covs
        groups.append({"params": [p for q in qls.values() if q.lora_a is not None for p in (q.lora_a, q.lora_b)], "lr": args.lora_lr})
    ptq = rel_err(sb, Xs_ho, Yt_ho, ho)
    opt = torch.optim.AdamW(groups, weight_decay=args.wd, betas=(0.9, 0.99))
    steps = int(args.epochs * len(tr) / args.mb)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(1, steps))) * 0.99 + 0.01)
    order = []
    t0 = time.time()
    run_loss = 0.0
    for step in range(steps):
        if len(order) < args.mb:
            order = list(range(len(tr)))
            rng.shuffle(order)
        idx, order = order[:args.mb], order[args.mb:]
        # all four quantized weights are rebuilt once per step and shared by the micro-batch
        wq = {n: q.quantized_weight().to(torch.bfloat16) for n, q in qls.items()}
        for n, q in qls.items():
            q.cached_w = wq[n]
        loss = 0.0
        for j in idx:
            loss = loss + sample_loss(run(sb, tr[j], Xs_tr[j]).float(), Yt_tr[j].to(dev).float())
        loss = loss / len(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.item()
        for q in qls.values():
            q.cached_w = None
        del wq
    post = rel_err(sb, Xs_ho, Yt_ho, ho)
    # advance both streams
    with torch.no_grad():
        for i, (it, x) in enumerate(zip(tr, Xs_tr)):
            Xs_tr[i] = run(sb, it, x).cpu()
        Xs_ho = [run(sb, it, x).cpu() for it, x in zip(ho, Xs_ho)]
    Xt_tr, Xt_ho = Yt_tr, Yt_ho
    drift = rel_err(sb, Xs_ho, Yt_ho, ho) if False else None
    for name, q in qls.items():
        p, sc, k = q.export()
        key = "layers.%d.proj.%s" % (bi, name)
        packed[key + ".weight"], packed[key + ".weight_scale"], packed[key + ".weight_kind"] = p, sc, k
        if q.lora_a is not None:
            packed[key + ".lora_a"], packed[key + ".lora_b"] = q.lora_a.detach().to(torch.bfloat16).cpu(), q.lora_b.detach().to(torch.bfloat16).cpu()
        if args.fake_out:
            wd = L.dequantize(p.to(dev), sc.to(dev), int(k), torch.float32)
            if q.lora_a is not None:
                wd = wd + packed[key + ".lora_b"].to(dev).float() @ packed[key + ".lora_a"].to(dev).float()
            fake[key + ".weight"] = wd.to(torch.bfloat16).cpu()
    rec = {"block": bi, "ptq_rel_mse": ptq, "qat_rel_mse": post, "steps": steps, "sec": time.time() - t0,
           "kinds": {n: L.KINDS[q.kind][0] for n, q in qls.items()}}
    # accumulated error of the student stream vs teacher stream
    num = sum(((a.float() - b.float()) ** 2).sum().item() for a, b in zip(Xs_ho, Xt_ho))
    den = sum((b.float() ** 2).sum().item() for b in Xt_ho)
    rec["stream_rel_mse"] = num / den
    log.append(rec)
    print("block %2d  ptq %.4f -> qat %.4f  (stream %.4f)  %d steps %.0fs  elapsed %.0fm" % (bi, ptq, post, rec["stream_rel_mse"], steps, rec["sec"], (time.time() - t_start) / 60), flush=True)
    del tb, sb, qls, opt
    torch.cuda.empty_cache()
    if args.log:
        json.dump({"args": vars(args), "blocks": log}, open(args.log, "w"), indent=1)
    if args.out and (bi + 1) % 6 == 0:
        torch.save(packed, args.out + ".partial.pt")

if args.out or args.fake_out:
    base = {k: v for k, v in sd.items()}
    if args.out:
        out = {k: v for k, v in base.items() if k not in {kk.rsplit(".weight_", 1)[0] + ".weight" for kk in packed}}
        out.update(packed)
        save_file({k: v.contiguous() for k, v in out.items()}, args.out, metadata={"moss_lowbit_v1": "1"})
        print("wrote", args.out, "%.2f GB" % (os.path.getsize(args.out) / 1e9))
    if args.fake_out:
        out = dict(base)
        out.update(fake)
        save_file({k: v.contiguous() for k, v in out.items()}, args.fake_out)
        print("wrote", args.fake_out)
