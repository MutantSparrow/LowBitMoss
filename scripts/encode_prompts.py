"""Run a text encoder over a fixed prompt set the way the sampler sees it (CLIPTextEncode conditioning) and save the tensors for cond_error.py.
The prompts are the 24 evaluation, 192 calibration and 80 text-rich calibration prompts (296 in all). GPU.

    python -I encode_prompts.py --clip <file in models/text_encoders | absolute path> [--clip-lowbit] --out ../data/enc_cond/<name>.pt

--clip-lowbit: the file is a moss_lowbit_v1 text encoder (quantize_encoder.py), loaded with the low-bit ops."""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import NODES, PROMPTS, comfy_dir

sys.path.insert(0, comfy_dir())
sys.path.insert(0, NODES)
import torch  # noqa: E402

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.reserve_vram = 2.5
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--clip-lowbit", action="store_true")
ap.add_argument("--out", required=True)
args = ap.parse_args()

prompts = []
for f in ("eval_prompts.json", "calib_prompts.json", "calib_text_prompts.json"):
    prompts += [p["prompt"] if isinstance(p, dict) else p for p in json.load(open(os.path.join(PROMPTS, f), encoding="utf-8"))]
if args.clip_lowbit:
    lowbit = importlib.import_module("ComfyUI-MOSS-LowBit.lowbit")
    clip = comfy.sd.load_clip(ckpt_paths=[args.clip if os.path.isabs(args.clip) else folder_paths.get_full_path_or_raise("text_encoders", args.clip)],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
                              model_options={"custom_operations": lowbit.lowbit_ops(torch.bfloat16)})
else:
    clip = nodes.CLIPLoader().load_clip(args.clip, "moss_image")[0]
conds = []
for i, text in enumerate(prompts):
    conds.append(nodes.CLIPTextEncode().encode(clip, text)[0][0][0].detach().to(torch.bfloat16).cpu())
    if i % 50 == 0:
        print(i, len(prompts), tuple(conds[-1].shape), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save({"prompts": prompts, "conds": conds}, args.out)
print("saved", len(conds), "conditionings ->", args.out)
