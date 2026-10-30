"""Drop the tensors MOSS conditioning never reads (lm_head, mtp head, vision tower) from a ComfyUI-format Qwen3.5-9B text encoder file; everything else is copied byte for byte.
The result loads in the stock CLIPLoader (type moss_image): MOSS reads the last hidden state with the MTP head off (comfy builds the MTP head only when mtp.fc.weight exists).
CPU only. --inp defaults to $COMFYUI_DIR/models/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors (Comfy-Org's W4A8 GPTQ encoder, 7.34 GB -> 4.93 GB).

    python -I -B prune_encoder.py --out ../out/qwen3.5_9b_w4a8_gptq_pruned.safetensors"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import comfy_dir

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

DROP = ("lm_head.", "mtp.", "model.visual.")
ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=os.path.join(comfy_dir(), "models", "text_encoders", "qwen3.5_9b_w4a8_gptq.safetensors"))
ap.add_argument("--out", required=True)
args = ap.parse_args()

out, dropped = {}, 0
with safe_open(args.inp, "pt") as f:
    metadata = f.metadata()
    for key in f.keys():
        if key.startswith(DROP):
            dropped += 1
            continue
        out[key] = f.get_tensor(key)
tmp = args.out + ".tmp"
save_file(out, tmp, metadata=metadata)
os.replace(tmp, args.out)
print("kept %d tensors, dropped %d -> %s (%.2f GB, input %.2f GB)" % (len(out), dropped, args.out, os.path.getsize(args.out) / 1e9, os.path.getsize(args.inp) / 1e9))
