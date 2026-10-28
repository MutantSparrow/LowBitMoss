"""Rewrite the bf16 lora_a / lora_b of a moss_lowbit_v1 file in the compact form: float8_e4m3fn + a scalar fp16 lora_a_scale / lora_b_scale
(per tensor, scale = amax / 448; see lowbit_pack.lora_to_fp8). Everything else is copied unchanged.

    python -I -B compact_lora.py --inp X.safetensors --out Y.safetensors [--teacher moss_image_sft_bf16.safetensors] [--check 5] [--seed 0]

CPU only. Prints the byte saving, the relative reconstruction error of every factor (||fp8 - orig|| / ||orig||: mean, max) and, for --check random
linears, the error of the folded correction B A and of the whole weight W = Q + B A (fp8-roundtripped vs original), plus, with --teacher, the error of
the weight against the teacher weight before and after. Factors that are already compact are copied; the output is written to a temp file and
renamed, then read back and compared with the input.
"""
import argparse
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # never touch the GPU (an empty value is dropped by Windows, -1 hides all devices)
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import lowbit_pack as L  # noqa: E402

assert not torch.cuda.is_available()
ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--teacher", help="bf16 teacher checkpoint: also report the weight error against it")
ap.add_argument("--check", type=int, default=5, help="number of random linears whose folded weight is compared")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def rel(a, b):
    """||a - b|| / ||b||"""
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def raw(t):
    return t.contiguous().reshape(-1).view(torch.uint8)  # bit comparison, works for NaN, fp8 and 0-dim tensors


with safe_open(args.inp, "pt") as f:
    metadata = f.metadata()
sd = load_file(args.inp)
names = sorted(k[:-len(".lora_a")] for k in sd if k.endswith(".lora_a"))
assert names == sorted(k[:-len(".lora_b")] for k in sd if k.endswith(".lora_b")), "lora_a and lora_b keys do not pair up"
out = dict(sd)
before = after = compact = 0
errs = {"lora_a": [], "lora_b": []}  # (relative error, name)
for name in names:
    for part in ("lora_a", "lora_b"):
        key = f"{name}.{part}"
        t = sd[key]
        if t.dtype == torch.float8_e4m3fn:
            assert key + "_scale" in sd, key + " is float8_e4m3fn without a scale"
            before += t.nbytes + sd[key + "_scale"].nbytes
            after += t.nbytes + sd[key + "_scale"].nbytes
            compact += 1
            continue
        assert key + "_scale" not in sd, key + "_scale on a non-fp8 factor"
        q, scale = L.lora_to_fp8(t)
        out[key], out[key + "_scale"] = q, scale
        before += t.nbytes
        after += q.nbytes + scale.nbytes
        errs[part].append((rel(L.lora_from_stored(q, scale), t.float()), key))
print("%d linears with a low-rank correction (%d factors, %d of them already compact)" % (len(names), 2 * len(names), compact))
print("lora bytes: %.1f MB -> %.1f MB (saved %.1f MB = %.1f%% of the lora tensors, %.1f%% of the file)" % (
    before / 1e6, after / 1e6, (before - after) / 1e6, 100 * (before - after) / before, 100 * (before - after) / os.path.getsize(args.inp)))
for part, e in errs.items():
    if e:
        worst = max(e)
        print("%s relative error ||fp8 - orig|| / ||orig||: mean %.5f  max %.5f (%s)  min %.5f  [%d tensors]" % (part, sum(v for v, _ in e) / len(e), worst[0], worst[1], min(e)[0], len(e)))

# folded weight of a few random linears
teacher = safe_open(args.teacher, "pt") if args.teacher else None
for name in random.Random(args.seed).sample(names, min(args.check, len(names))):
    A, B = (L.lora_from_stored(sd[f"{name}.lora_{p}"], sd.get(f"{name}.lora_{p}_scale")) for p in "ab")
    A2 = L.lora_from_stored(out[f"{name}.lora_a"], out.get(f"{name}.lora_a_scale"))
    B2 = L.lora_from_stored(out[f"{name}.lora_b"], out.get(f"{name}.lora_b_scale"))
    delta, delta2 = B @ A, B2 @ A2
    line = "  %-34s r=%-3d rel err of B A %.5f" % (name, A.shape[0], rel(delta2, delta))
    if f"{name}.weight_kind" in sd:
        Q = L.dequantize(sd[f"{name}.weight"], sd[f"{name}.weight_scale"], int(sd[f"{name}.weight_kind"]), torch.float32)
        line += "  of W = Q + B A %.5f" % rel(Q + delta2, Q + delta)
        if teacher is not None:
            wt = teacher.get_tensor(f"{name}.weight").float()
            line += "  | error vs teacher: %.5f -> %.5f" % (rel(Q + delta, wt), rel(Q + delta2, wt))
    print(line)

tmp = args.out + ".tmp"
save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata=metadata)
os.replace(tmp, args.out)

# read back: same keys plus the scales, untouched tensors bit-identical, compact factors as computed
with safe_open(args.out, "pt") as f:
    assert set(f.keys()) == set(out), "key set differs"
    for k, v in out.items():
        t = f.get_tensor(k)
        assert t.dtype == v.dtype and t.shape == v.shape and torch.equal(raw(t), raw(v)), k
    assert f.metadata() == metadata
print("%s -> %s: %.3f GB -> %.3f GB, read back identical" % (os.path.basename(args.inp), os.path.basename(args.out), os.path.getsize(args.inp) / 1e9, os.path.getsize(args.out) / 1e9))
