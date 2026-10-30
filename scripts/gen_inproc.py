"""In-process twin of gen_eval.py (no server): same ComfyUI nodes, seeds, sampler settings, file names.

    python -I gen_inproc.py --unet out/mixG_x.safetensors --tag mixG_x [--res 768] [--prompts eval|calib] [--bf16]
    python -I gen_inproc.py --unet out/tmixG_x.safetensors --tag tmixG_x_1024 --res 1024 --steps 4 --cfg 1.0 --sampler euler --scheduler simple   # distilled model
    python -I gen_inproc.py --bf16 --unet <ComfyUI>/models/diffusion_models/moss_image_sft_bf16.safetensors --lora $LOWBITMOSS_AUX/turbo4/lora.safetensors \
        --steps 4 --cfg 1.0 --sampler euler --scheduler simple --res 1024 --tag turbo4_1024                                           # distilled reference
    python -I gen_inproc.py --unet out/tmixG_x.safetensors --clip <abs path>/enc_i4e8.safetensors --clip-lowbit --steps 4 --cfg 1.0 --sampler euler --scheduler simple --res 1024 --tag enc_i4e8   # own text encoder
--clip is a name in ComfyUI's models/text_encoders or an absolute path (with --clip-lowbit); --extra-seeds N adds N more seeds per selected prompt (seed + 100 * j).
Writes $LOWBITMOSS_HOME/eval/<tag>/NN_seedSSSS.png and .../latents/NN_seedSSSS.latent (same format as gen_eval.py --save-latent).

On Windows, when VRAM runs short (the desktop, another job, or a process that has run for a while) the driver pages VRAM to system RAM and sampling
gets 15-40x slower (a normal image takes about 30 s) and does not recover. So the script runs as a supervisor around a worker process: the worker leaves 2.5 GB
of VRAM free (reserve_vram), exits with code 3 when one image takes longer than --watchdog seconds, and also exits on purpose after --recycle images;
the supervisor then restarts it and it skips the images that already exist."""
import argparse
import importlib
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import EVAL, NODES, PROMPTS, comfy_dir

COMFY = comfy_dir()  # fail before the supervisor starts workers
if "--child" not in sys.argv:  # supervisor: relaunch the worker (skipping finished images) when its watchdog fires
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--child", "--skip-existing"] + sys.argv[1:]
    for attempt in range(1, 13):
        rc = subprocess.call(cmd)
        if rc == 0:
            sys.exit(0)
        print("gen_inproc: worker exited with %d, restarting (%d/12)" % (rc, attempt), flush=True)
    sys.exit(rc)

sys.path.insert(0, COMFY)
sys.path.insert(0, NODES)
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

import comfy.cli_args

comfy.cli_args.args.reserve_vram = 2.5  # headroom for the desktop: without it Windows pages VRAM to RAM and sampling slows ~40x
import comfy.sd
import comfy.utils
import folder_paths
import nodes

ap = argparse.ArgumentParser()
ap.add_argument("--unet", required=True, help="path to a moss_lowbit_v1 file, or a bf16 file with --bf16")
ap.add_argument("--tag", required=True)
ap.add_argument("--res", type=int, default=768)
ap.add_argument("--prompts", default="eval")
ap.add_argument("--n", type=int, default=0, help="only the first n eval prompts")
ap.add_argument("--bf16", action="store_true", help="stock loader instead of the low-bit loader")
ap.add_argument("--lora", default=None, help="LoRA applied with ComfyUI's patcher (bf16 round-to-nearest merge), e.g. the turbo/DMD LoRA; bf16 models only, the low-bit loader refuses LoRA patches")
ap.add_argument("--lora-strength", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--cfg", type=float, default=2.5)
ap.add_argument("--sampler", default="er_sde")
ap.add_argument("--scheduler", default="simple")
ap.add_argument("--clip", default="qwen3.5_9b_bf16.safetensors", help="text encoder file in models/text_encoders")
ap.add_argument("--clip-lowbit", action="store_true", help="the --clip file is a moss_lowbit_v1 text encoder (quantize_encoder.py): load it with the low-bit ops; --clip may then be an absolute path")
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--ids", default="", help="comma-separated prompt ids to generate (default: all)")
ap.add_argument("--extra-seeds", type=int, default=0, help="also generate each selected prompt with this many more seeds (seed + 100 * j)")
ap.add_argument("--child", action="store_true", help="internal: run the worker (the supervisor adds this)")
ap.add_argument("--recycle", type=int, default=5, help="planned restart of the worker after this many images (memory state degrades over a run); 0 = never")
ap.add_argument("--watchdog", type=int, default=150, help="seconds after which a single image is considered stalled and the process exits with code 3")
a = ap.parse_args()

NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
items = json.load(open(os.path.join(PROMPTS, "eval_prompts.json" if a.prompts == "eval" else "calib_prompts.json"), encoding="utf-8"))
if a.prompts != "eval":
    for it in items:
        it["seed"] = 7000 + it["id"]
if a.n:
    items = items[:a.n]
if a.ids:
    items = [it for it in items if it["id"] in {int(v) for v in a.ids.split(",")}]
items = items + [dict(it, seed=it["seed"] + 100 * j) for j in range(1, a.extra_seeds + 1) for it in items]
out = os.path.join(EVAL, a.tag)
os.makedirs(os.path.join(out, "latents"), exist_ok=True)

if not a.bf16 or a.clip_lowbit:
    lowbit = importlib.import_module("ComfyUI-MOSS-LowBit.lowbit")
if a.bf16:
    model = comfy.sd.load_diffusion_model(a.unet, model_options={"dtype": torch.bfloat16})
else:
    model = comfy.sd.load_diffusion_model(a.unet, model_options={"dtype": torch.bfloat16, "custom_operations": lowbit.lowbit_ops(torch.bfloat16)})
if a.lora:
    model, _ = comfy.sd.load_lora_for_models(model, None, comfy.utils.load_torch_file(a.lora), a.lora_strength, 0)
if a.clip_lowbit:
    clip = comfy.sd.load_clip(ckpt_paths=[a.clip if os.path.isabs(a.clip) else folder_paths.get_full_path_or_raise("text_encoders", a.clip)], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.MOSS_IMAGE, model_options={"custom_operations": lowbit.lowbit_ops(torch.bfloat16)})
else:
    clip = nodes.CLIPLoader().load_clip(a.clip, "moss_image")[0]
vae = nodes.VAELoader().load_vae("flux2-vae.safetensors")[0]
neg = nodes.CLIPTextEncode().encode(clip, NEG)[0]

def stalled():  # a normal image takes ~30 s; a VRAM-paging stall is 15-40x slower and never recovers, so restart the process
    print("watchdog: image took > %d s, exiting for a restart" % a.watchdog, flush=True)
    os._exit(3)


t_all = time.time()
done_here = 0
for it in items:
    name = "%02d_seed%d" % (it["id"], it["seed"])
    png = os.path.join(out, name + ".png")
    if a.skip_existing and os.path.exists(png):
        continue
    t0 = time.time()
    watchdog = threading.Timer(a.watchdog, stalled)
    watchdog.start()
    pos = nodes.CLIPTextEncode().encode(clip, it["prompt"])[0]
    lat = {"samples": torch.zeros(1, 128, a.res // 16, a.res // 16)}
    lat = nodes.KSampler().sample(model, it["seed"], a.steps, a.cfg, a.sampler, a.scheduler, pos, neg, lat)[0]
    img = nodes.VAEDecode().decode(vae, lat)[0]
    Image.fromarray((img[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()).save(png)
    save_file({"latent_tensor": lat["samples"].float().cpu().contiguous()}, os.path.join(out, "latents", name + ".latent"))
    watchdog.cancel()
    print("%s  %.1fs" % (name, time.time() - t0), flush=True)
    done_here += 1
    if a.recycle and a.steps >= 20 and done_here >= a.recycle and it is not items[-1]:  # few-step runs finish before the memory state degrades; a restart costs ~100 s
        print("recycling the worker after %d images" % done_here, flush=True)
        os._exit(3)
print("done %d images in %.1f min -> %s" % (len(items), (time.time() - t_all) / 60, out))
