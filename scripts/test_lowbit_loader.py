"""Numerical test of the moss_lowbit loader: packed forward vs a plain bf16 model built from the dequantized weights.

    python -I test_lowbit_loader.py <file.safetensors> [blocks]      (ComfyUI .venv python, COMFYUI_DIR set, needs a GPU)
blocks: keep only the first N blocks, so the bf16 reference fits next to other GPU users
"""
import gc
import importlib.util
import os
import sys
import time

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import NODES, comfy_dir

COMFY = comfy_dir()
sys.path[:0] = [COMFY, HERE]
os.chdir(COMFY)
path = sys.argv[1]
blocks = int(sys.argv[2]) if len(sys.argv) > 2 else None
sys.argv = sys.argv[:1] + ["--disable-auto-launch"]

import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
from lowbit_pack import dequantize  # noqa: E402

pack_dir = os.path.join(NODES, "ComfyUI-MOSS-LowBit")
spec = importlib.util.spec_from_file_location("moss_lowbit_pack", os.path.join(pack_dir, "__init__.py"), submodule_search_locations=[pack_dir])
pack = importlib.util.module_from_spec(spec)
sys.modules["moss_lowbit_pack"] = pack
spec.loader.exec_module(pack)


dev = mm.get_torch_device()
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 128, 32, 32, generator=g).to(dev, torch.bfloat16)
ctx = torch.randn(1, 200, 4096, generator=g).to(dev, torch.bfloat16)
t = torch.tensor([0.5], device=dev)


def run(patcher, label, full=True):
    mm.load_models_gpu([patcher], force_full_load=full)
    net = patcher.model.diffusion_model
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    out = net(x, t, ctx)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        out = net(x, t, ctx)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 5 * 1000
    print(f"{label}: model_size {patcher.model_size() / 2**30:.2f} GiB, resident {resident:.2f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, {ms:.0f} ms/forward")
    res = out.float().cpu()
    mm.unload_all_models()
    del net, patcher, out
    gc.collect()
    torch.cuda.empty_cache()
    return res


sd = load_file(path)
if blocks is not None:
    sd = {k: v for k, v in sd.items() if not k.startswith("layers.") or int(k.split(".")[1]) < blocks}
packed = run(comfy.sd.load_diffusion_model_state_dict(dict(sd), model_options={"dtype": torch.bfloat16, "custom_operations": pack.lowbit_ops(torch.bfloat16)}), "packed")

kinds = {k[:-len("weight_kind")]: int(sd.pop(k)) for k in list(sd) if k.endswith(".weight_kind")}
for prefix, kind in kinds.items():
    sd[prefix + "weight"] = dequantize(sd[prefix + "weight"].to(dev), sd.pop(prefix + "weight_scale").to(dev), kind).cpu()
print("kinds used:", {k: sum(1 for v in kinds.values() if v == k) for k in sorted(set(kinds.values()))})
ref = run(comfy.sd.load_diffusion_model_state_dict(sd, model_options={"dtype": torch.bfloat16}), "bf16  ", full=False)

d = (packed - ref).abs()
print(f"output std {ref.std():.4f}; max|diff| {d.max():.3e}; mean|diff| {d.mean():.3e}; rel L2 {((packed - ref).norm() / ref.norm()):.3e}; bit-identical {torch.equal(packed, ref)}")
