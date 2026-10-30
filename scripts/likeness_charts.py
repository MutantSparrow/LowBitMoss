"""Draw the likeness-vs-size charts from the CSVs written by likeness_plot.py (RELEASE_CANDIDATES/charts/*.csv) -> *.png next to them.
    python -I -B likeness_charts.py"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script directory from sys.path
from review_reporting import charts  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

OUT = str(Path(__file__).resolve().parents[1] / "docs" / "charts")
REF, DREF, OK, DIST, FOLD, BAD = ("bf16 teacher (25 steps)", "distilled bf16 teacher (4 steps)", "non-distilled, quantized", "distilled, quantized",
                                   "distilled, turbo factors folded (no retraining)", "failed / not acceptable")
COL = {REF: "#111111", DREF: "#e08000", OK: "#1f77b4", DIST: "#d62728", FOLD: "#9467bd", BAD: "#8a8a8a"}
MK = {REF: "*", DREF: "s", OK: "o", DIST: "o", FOLD: "D", BAD: "X"}


def rd(name):
    return list(csv.DictReader(open(os.path.join(OUT, name), newline="")))


def num_labels(ax, pts, fs=8):
    """put each point's number next to it, choosing the offset that overlaps least with the numbers already placed"""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cands = [(5, 4), (5, -11), (-12, 4), (-12, -11), (0, 8), (0, -15), (11, 0), (-17, 0), (10, 10), (10, -16), (-18, 8), (-18, -16)]
    boxes = []
    for x, y, text, color in pts:
        best = None
        for dx, dy in cands:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fs, color=color, fontweight="bold")
            bb = t.get_window_extent(rend)
            ov = sum(max(0, min(bb.x1, o.x1) - max(bb.x0, o.x0)) * max(0, min(bb.y1, o.y1) - max(bb.y0, o.y0)) for o in boxes)
            if best is None or ov < best[0]:
                if best is not None:
                    best[1].remove()
                best = (ov, t, bb)
            else:
                t.remove()
            if ov == 0:
                break
        boxes.append(best[2])


def plain_log(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


# ---------------- DiTs: raw scores and comparison-specific intervals, no quality ratios.
rows = rd("dit_likeness.csv")
charts(rows, Path(OUT), None)

# ---------------- text encoders
er = rd("encoder_likeness.csv")
for r in er:
    r["size"], r["c"], r["i"] = float(r["size_gb"]), float(r["conditioning_token_cosine_x100"]), float(r["image_dino_x100_vs_bf16_encoder_images"])
short = {"bf16 encoder (reference)": "bf16 encoder", "Comfy int8 convrot": "Comfy int8", "Comfy W4A8 GPTQ": "Comfy GPTQ (full file)", "Comfy GPTQ pruned (recommended)": "Comfy GPTQ pruned\n(recommended)",
         "own int4 + int8 embedding": "own int4+int8 emb.", "own int4": "own int4", "own mlp-int3": "own mlp-int3", "own attn-int3": "own attn-int3", "own lin-attn-int3": "own lin-attn-int3", "own all-int3": "own all-int3"}
off = {"bf16 encoder": (-8, -14, "right"), "Comfy int8": (0, 9, "center"), "Comfy GPTQ (full file)": (0, -15, "center"), "Comfy GPTQ pruned\n(recommended)": (0, 12, "center"), "own int4+int8 emb.": (-8, 4, "right"),
       "own int4": (7, 3, "left"), "own mlp-int3": (7, -3, "left"), "own attn-int3": (-8, -9, "right"), "own lin-attn-int3": (7, 3, "left"), "own all-int3": (7, 3, "left")}
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
for ax, key, ylab, title in ((axes[0], "c", "conditioning likeness\n(token cosine to the bf16 encoder's output, x100)", "What the encoder produces"),
                             (axes[1], "i", "image likeness\n(DINOv2 of tmixG_x images vs the bf16-encoder images, x100)", "What the DiT makes of it")):
    for r in er:
        s = short[r["label"]]
        c = "#111111" if s == "bf16 encoder" else "#2ca02c" if "Comfy" in s else "#d62728"
        ax.scatter(r["size"], r[key], s=190 if s == "bf16 encoder" or "recommended" in s else 60, marker="*" if s == "bf16 encoder" else "o", color=c, zorder=3, edgecolor="white", linewidth=0.6)
        dx, dy, ha = off[s]
        ax.annotate(s, (r["size"], r[key]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=c, ha=ha)
    ax.set_xlim(2.2, 21)
    ax.set_ylim(top=107)
    ax.set_xticks([5, 6, 7.34, 9.75, 19.3])
    ax.set_xlabel("text encoder size on disk (GB)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
fig.suptitle("Qwen3.5-9B text encoder versions (green = Comfy-Org's files, red = my own post-training quantization; bf16 = 100)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "encoder_likeness_vs_size.png"), dpi=140)
plt.close(fig)

# ---------------- VAEs
vr = rd("vae_likeness.csv")
noise = [r for r in vr if r["vae"].startswith("reference")][0]
small = [r for r in vr if r["vae"].startswith("small decoder + ")][0]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
for ax, col, ylab, title in ((axes[0], "dino_x100_vs_full_vae_decode", "semantic likeness (DINOv2 x100)", "DINOv2"), (axes[1], "ssim_x100_vs_full_vae_decode", "structural likeness (SSIM x100)", "SSIM")):
    b, n = float(small[col]), float(noise[col])
    ax.axhline(n, color="#888888", linestyle="--", linewidth=1)
    ax.text(0.045, n + (0.06 if col.startswith("dino") else 0.25), "two decodes of the same pipeline: %.1f" % n, fontsize=8, color="#666666", ha="left", va="bottom")
    ax.scatter([0.336], [100], s=190, marker="*", color="#111111", zorder=3)
    ax.annotate("FLUX.2 VAE (full) 336 MB\n= reference 100", (0.336, 100), xytext=(0, 9), textcoords="offset points", fontsize=8, ha="center")
    ax.scatter([0.250], [b], s=75, color="#2ca02c", zorder=3)
    ax.annotate("small decoder + encoder\n(official) 250 MB: %.1f" % b, (0.250, b), xytext=(0, -30), textcoords="offset points", fontsize=8, ha="center", color="#2ca02c")
    ax.scatter([0.112], [b], s=75, color="#d62728", zorder=3)
    ax.annotate("small decoder only\n(built) 112 MB: %.1f" % b, (0.112, b), xytext=(0, -30), textcoords="offset points", fontsize=8, ha="center", color="#d62728")
    lo = min(b, n)
    ax.set_ylim(lo - (1.0 if col.startswith("dino") else 4.5), 101.2 if col.startswith("dino") else 103)
    ax.set_xlim(0.04, 0.4)
    ax.set_xlabel("VAE size on disk (GB)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
fig.suptitle("FLUX.2 VAE vs the small decoder (24 latents of the recommended package; the two small-decoder files decode identically)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "vae_likeness_vs_size.png"), dpi=140)
plt.close(fig)
print("charts written")
