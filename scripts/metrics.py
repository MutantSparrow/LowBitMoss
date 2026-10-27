"""Compare a student image dir against the teacher dir (matched by file name) and score text-image alignment.

Per image and mean: PSNR, SSIM (gaussian 11x11, sigma 1.5, per RGB channel), DINOv2-small CLS cosine, CLIP ViT-B/16 text-image
score (100*cosine, prompt truncated to CLIP's 77 tokens) for teacher and student, and, when both dirs have latents/, latent
relative-MSE and cosine. With no --student only the teacher CLIP score is computed.
Models come from the HF hub (facebook/dinov2-small, openai/clip-vit-base-patch16) and are cached by huggingface_hub.

  python -I metrics.py --teacher eval/teacher_bf16 --student eval/mixG_x --prompts eval [--device cuda] [--out x.json]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import PROMPTS


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for c in range(3):
        x, y = a[..., c], b[..., c]
        f = lambda t: gaussian_filter(t, 1.5, truncate=3.5)
        mx, my = f(x), f(y)
        vx, vy, cxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
        m = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        vals.append(m[5:-5, 5:-5].mean())
    return float(np.mean(vals))


@torch.no_grad()
def dino_embed(model, proc, paths, device):
    out = []
    for i in range(0, len(paths), 8):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
        x = proc(images=imgs, return_tensors="pt").to(device)
        out.append(torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0].float(), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def clip_scores(model, proc, paths, texts, device):
    out = []
    for p, t in zip(paths, texts):
        x = proc(text=[t], images=Image.open(p).convert("RGB"), return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        o = model(**x)
        ie, te = (torch.nn.functional.normalize(e.float(), dim=-1) for e in (o.image_embeds, o.text_embeds))
        out.append(float(100 * (ie * te).sum()))
    return out


def prompt_table(which):
    f = os.path.join(PROMPTS, "eval_prompts.json" if which == "eval" else "calib_prompts.json")
    return {it["id"]: it["prompt"] for it in json.load(open(f, encoding="utf-8"))}


def latent_stats(tdir, sdir, names):
    from safetensors.torch import load_file
    res = {}
    for n in names:
        tp, sp = tdir / "latents" / f"{n}.latent", sdir / "latents" / f"{n}.latent"
        if tp.exists() and sp.exists():
            t, s = (load_file(p)["latent_tensor"].float().flatten() for p in (tp, sp))
            res[n] = {"latent_rel_mse": float(((t - s) ** 2).mean() / (t ** 2).mean()), "latent_cos": float(torch.nn.functional.cosine_similarity(t, s, dim=0))}
    return res


def mean(rows, k):
    v = [r[k] for r in rows.values() if k in r and np.isfinite(r[k])]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default=None)
    ap.add_argument("--prompts", default="eval", choices=["eval", "calib"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="json path; default <student>/metrics.json")
    a = ap.parse_args()

    tdir = Path(a.teacher)
    sdir = Path(a.student) if a.student else None
    names = sorted(p.stem for p in tdir.glob("*.png"))
    if sdir:
        names = [n for n in names if (sdir / f"{n}.png").exists()]
    assert names, "no matching images"
    prompts = prompt_table(a.prompts)
    texts = [prompts[int(re.match(r"(\d+)_", n).group(1))] for n in names]
    tp = [tdir / f"{n}.png" for n in names]
    sp = [sdir / f"{n}.png" for n in names] if sdir else []
    dev = a.device

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    rows = {n: {} for n in names}
    for n, s in zip(names, clip_scores(clip, cproc, tp, texts, dev)):
        rows[n]["clip_teacher"] = s
    if sdir:
        for n, s in zip(names, clip_scores(clip, cproc, sp, texts, dev)):
            rows[n]["clip_student"] = s
        del clip
        dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        te, se = dino_embed(dino, dproc, tp, dev), dino_embed(dino, dproc, sp, dev)
        for n, c in zip(names, (te * se).sum(-1).tolist()):
            rows[n]["dino_cos"] = c
        for n, t, s in zip(names, tp, sp):
            A, B = load_rgb(t), load_rgb(s)
            rows[n]["psnr"], rows[n]["ssim"] = psnr(A, B), ssim(A, B)
        for n, d in latent_stats(tdir, sdir, names).items():
            rows[n].update(d)

    keys = ["psnr", "ssim", "dino_cos", "clip_teacher", "clip_student", "latent_rel_mse", "latent_cos"]
    means = {k: mean(rows, k) for k in keys if mean(rows, k) is not None}
    res = {"teacher": str(tdir), "student": str(sdir) if sdir else None, "n": len(names), "mean": means, "per_image": rows}
    out = Path(a.out) if a.out else (sdir or tdir) / "metrics.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"n={len(names)} " + " ".join(f"{k}={v:.4f}" for k, v in means.items()) + f" -> {out}")


if __name__ == "__main__":
    main()
