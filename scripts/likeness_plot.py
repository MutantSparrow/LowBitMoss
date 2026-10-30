"""Likeness numbers behind the likeness-vs-size charts -> RELEASE_CANDIDATES/charts/*.csv (likeness_charts.py draws them).
Likeness = how close the output is to the full-precision reference, reference = 100:
  DiTs: mean DINOv2-small CLS cosine of the 24 evaluation images to the 25-step bf16 teacher's image of the same prompt and seed (x100); local HPSv3-based means and paired differences with intervals
  encoders: mean token cosine of the conditioning to the bf16 encoder's (296 prompts, x100; logs/cond_error.log); DINOv2 of the images made with tmixG_x to the bf16-encoder images (x100)
  VAEs: DINOv2 and SSIM of the small decoder's images to the full VAE's decodes of the same latents (x100), with the level of two decodes of the SAME configuration for reference
    python -I -B likeness_plot.py"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import dino_embed, load_rgb, ssim  # noqa: E402

QW = str(Path(os.environ.get("LOWBITMOSS_HOME",Path(__file__).resolve().parents[1])))
EV = QW + r"\eval"
OUT = str(Path(__file__).resolve().parents[1] / "docs" / "charts")
os.makedirs(OUT, exist_ok=True)
proc, model = AutoImageProcessor.from_pretrained("facebook/dinov2-small"), AutoModel.from_pretrained("facebook/dinov2-small").eval()


def pngs(tag):
    return sorted(glob.glob(os.path.join(EV, tag, "*.png")))


def dino(a, b):
    return float((dino_embed(model, proc, a, "cpu") * dino_embed(model, proc, b, "cpu")).sum(-1).mean()) * 100


from review_reporting import paired

hps = {}
hps_by_prompt = {}
for f in glob.glob(os.path.join(EV, "hps3_*.json")):
    for k, v in json.load(open(f)).items():
        if isinstance(v, dict) and "mean_mu" in v:
            hps.setdefault(k, v["mean_mu"])
            if "mu_by_prompt" in v:
                hps_by_prompt.setdefault(k, v["mu_by_prompt"])

# ---- DiTs: (tag, size GB, family, label)
OK, DIST, FOLD, BAD, REF, DREF = "non-distilled, quantized", "distilled, quantized", "distilled, turbo factors folded (no retraining)", "failed / not acceptable", "bf16 teacher (25 steps)", "distilled bf16 teacher (4 steps)"
dits = [
    ("teacher_bf16", 10.16, REF, "bf16 teacher"), ("turbo4_1024", 10.16, DREF, "distilled bf16"),
    ("mixG_tuned", 2.14, OK, "mixG_tuned"), ("mixG_x", 2.14, OK, "mixG_x"), ("mixH_x", 1.95, OK, "mixH_x"), ("mixH_tuned", 1.95, OK, "mixH_tuned"),
    ("mixH2_x", 1.95, OK, "mixH2_x"), ("mixH2_tuned", 1.95, OK, "mixH2_tuned"), ("mixH2_x_f8", 1.85, OK, "mixH2_x_f8"), ("mixH2_xm", 1.83, OK, "mixH2_xm"),
    ("mixH2_xm_f8", 1.72, OK, "mixH2_xm_f8"), ("mixA_x_tuned", 1.67, OK, "mixA_x_tuned"), ("mixF_ptq", 2.42, OK, "mixF_ptq (plain PTQ)"),
    ("mixI_x", 1.74, BAD, "mixI_x (combined 1-bit)"), ("mixI_tuned", 1.74, BAD, "mixI_tuned"), ("tern_v1", 1.92, BAD, "tern_v1"), ("bin_v1", 1.32, BAD, "bin_v1"),
    ("tmixG_x_1024", 2.14, DIST, "tmixG_x"), ("tmixG_tuned_1024", 2.14, DIST, "tmixG_tuned"), ("tmixG_traj_1024", 2.14, BAD, "tmixG_traj (blur)"),
    ("tmixH_x_f8_1024", 1.85, DIST, "tmixH_x_f8"), ("tmixH_xm_f8_1024", 1.72, DIST, "tmixH_xm_f8"),
    ("turboF_mixG_tuned", 2.93, FOLD, "mixG_tuned_t"), ("turboF_mixH2_x_f8", 2.74, FOLD, "mixH2_x_f8_t"),
]
with open(os.path.join(OUT, "dit_likeness.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "label", "family", "size_gb", "dino_likeness_x100", "hpsv3_local_mean", "hpsv3_delta_to_teacher", "hpsv3_paired_se", "hpsv3_ci95_low", "hpsv3_ci95_high"])
    for tag, size, fam, label in dits:
        q = hps.get(tag)
        v = hps_by_prompt.get(tag)
        fields = [""] * 5
        if v is not None:
            z = paired(v, hps_by_prompt["teacher_bf16"])
            fields = ["%.6f" % value for value in [np.mean(list(v.values())), z["mean"], z["se"], *z["ci95"]]]
        w.writerow([tag, label, fam, size, "%.2f" % dino(pngs("teacher_bf16"), pngs(tag))] + fields)

# ---- text encoders (images made with tmixG_x; reference = its bf16-encoder images)
cond = {}
for line in open(os.path.join(QW, "logs", "cond_error.log")):
    parts = line.split()
    cond[parts[0].replace(".pt", "")] = float(line.split("token cosine mean")[1].split()[0])
encs = [("tmixG_x_1024", 19.31, "bf16 encoder (reference)", None), ("enc_int8_convrot", 9.75, "Comfy int8 convrot", "int8"), ("enc_w4a8_gptq", 7.34, "Comfy W4A8 GPTQ", "gptq"),
        ("enc_gptq_pruned", 4.93, "Comfy GPTQ pruned (recommended)", "gptq"), ("enc_i4e8", 4.60, "own int4 + int8 embedding", "i4e8"), ("enc_i4", 5.61, "own int4", "i4"),
        ("enc_m3", 5.01, "own mlp-int3", "m3"), ("enc_a3", 4.54, "own attn-int3", "a3"), ("enc_l3", 4.40, "own lin-attn-int3", "l3"), ("enc_i3", 4.75, "own all-int3", "i3")]
with open(os.path.join(OUT, "encoder_likeness.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "label", "size_gb", "conditioning_token_cosine_x100", "image_dino_x100_vs_bf16_encoder_images", "hpsv3_with_tmixG_x"])
    for tag, size, label, ck in encs:
        q = hps.get(tag)
        w.writerow([tag, label, size, "%.2f" % (100.0 if ck is None else cond[ck] * 100), "%.2f" % dino(pngs("tmixG_x_1024"), pngs(tag)), "" if q is None else "%.2f" % q])

# ---- VAEs (24 latents of the recommended package; full-VAE decodes = the PNGs of the generation run)
full, small, same = pngs("enc_gptq_pruned"), pngs("enc_gptq_pruned_flux2-small-decoder-vae"), pngs("enc_w4a8_gptq")
mean_ssim = lambda a, b: float(np.mean([ssim(load_rgb(x), load_rgb(y)) for x, y in zip(a, b)])) * 100
v_dino, v_ssim, n_dino, n_ssim = dino(full, small), mean_ssim(full, small), dino(full, same), mean_ssim(full, same)
with open(os.path.join(OUT, "vae_likeness.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["vae", "size_gb", "dino_x100_vs_full_vae_decode", "ssim_x100_vs_full_vae_decode"])
    w.writerow(["FLUX.2 VAE (full)", 0.336, "100.00", "100.00"])
    w.writerow(["small decoder + original encoder (official)", 0.250, "%.2f" % v_dino, "%.2f" % v_ssim])
    w.writerow(["small decoder only (built)", 0.112, "%.2f" % v_dino, "%.2f" % v_ssim])
    w.writerow(["reference: two runs of the SAME full-VAE pipeline", 0.336, "%.2f" % n_dino, "%.2f" % n_ssim])
print("wrote chart CSVs")
