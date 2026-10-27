"""Generate eval / calibration images with a MOSS-Image diffusion model through a running ComfyUI server (HTTP API).

Workflow per prompt: UNETLoader -> KSampler(25 steps, cfg 2.5, er_sde, simple) ; CLIPLoader(moss_image) -> CLIPTextEncode +/- ;
EmptyFlux2LatentImage ; VAEDecode with flux2-vae. Images are downloaded to <out>/NN_seedSSSS.png, final latents (optional) to
<out>/latents/NN_seedSSSS.latent (safetensors, key "latent_tensor"). NN = prompt id, so names sort identically across tags.
Only the UNETLoader differs between tags, so ComfyUI reuses the cached text-encoder output.

  python gen_eval.py --unet moss_image_sft_bf16.safetensors --tag teacher_bf16 --res 768 --save-latent
  python gen_eval.py --unet student_w4.safetensors --tag w4 --prompts calib --n 16 --out /tmp/w4_images
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import EVAL, PROMPTS

NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
CLIENT_ID = str(uuid.uuid4())


class Server:
    def __init__(self, host, port):
        self.base = f"http://{host}:{port}"

    def get(self, path, timeout=60):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return r.read()

    def get_json(self, path):
        return json.loads(self.get(path))

    def post_json(self, path, obj):
        req = urllib.request.Request(self.base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"HTTP {e.code} from {path}: {e.read().decode(errors='replace')[:3000]}")

    def view(self, f):
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
        return self.get("/view?" + q, timeout=300)


def build_workflow(unet, prompt, seed, res, steps, cfg, latent_prefix, lowbit=False):
    wf = {
        "1": ({"class_type": "MossLowBitUNETLoader", "inputs": {"unet_name": unet}} if lowbit else {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}),
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_bf16.safetensors", "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": res, "height": res, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "er_sde",
                                                   "scheduler": "simple", "positive": ["4", 0], "negative": ["5", 0],
                                                   "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
    }
    if latent_prefix:
        wf["10"] = {"class_type": "SaveLatent", "inputs": {"samples": ["7", 0], "filename_prefix": latent_prefix}}
    return wf


def wait_for(server, pid, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = server.get_json(f"/history/{pid}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            h = {}
        if pid in h:
            st = h[pid]["status"]
            if st["status_str"] == "error":
                errs = [m[1] for m in st["messages"] if m[0] == "execution_error"]
                sys.exit("execution error: " + json.dumps(errs, indent=1)[:4000])
            if st.get("completed"):
                return h[pid]
        time.sleep(1.0)
    sys.exit(f"timeout waiting for prompt {pid}")


def exec_seconds(entry):
    ts = {m[0]: m[1]["timestamp"] for m in entry["status"]["messages"] if "timestamp" in m[1]}
    if "execution_start" in ts and "execution_success" in ts:
        return (ts["execution_success"] - ts["execution_start"]) / 1000.0
    return None


def select_prompts(which, n):
    if which == "eval":
        items = json.load(open(os.path.join(PROMPTS, "eval_prompts.json"), encoding="utf-8"))
    else:
        items = json.load(open(os.path.join(PROMPTS, "calib_prompts.json"), encoding="utf-8"))
        for it in items:
            it["seed"] = 7000 + it["id"]
        if n and n < len(items):
            items = [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)] if n > 1 else items[:1]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True, help="file name in models/diffusion_models (subfolders allowed)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--res", type=int, default=768, choices=[512, 768, 1024])
    ap.add_argument("--prompts", default="eval", help="eval | calib | calib:N (N evenly spaced calibration prompts)")
    ap.add_argument("--n", type=int, default=0, help="calibration subset size (same as calib:N)")
    ap.add_argument("--out", default=None, help="default $LOWBITMOSS_HOME/eval/<tag>")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--lowbit", action="store_true", help="load with the MOSS LowBit UNET Loader node (moss_lowbit_v1 files)")
    ap.add_argument("--save-latent", action="store_true", help="also save the final latent to <out>/latents/")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds to wait for one image (server may be busy)")
    a = ap.parse_args()

    which, _, k = a.prompts.partition(":")
    n = int(k) if k else a.n
    items = select_prompts(which, n)
    out = Path(a.out) if a.out else Path(EVAL) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if a.save_latent:
        (out / "latents").mkdir(exist_ok=True)

    server = Server(a.host, a.port)
    loader = "MossLowBitUNETLoader" if a.lowbit else "UNETLoader"
    unets = server.get_json("/object_info/" + loader)[loader]["input"]["required"]["unet_name"][0]
    if a.unet not in unets:
        sys.exit(f"unet {a.unet!r} not known to the server; available: {unets}")

    run_id = uuid.uuid4().hex[:8]
    timings = []
    t_all = time.time()
    for it in items:
        name = f"{it['id']:02d}_seed{it['seed']}"
        png = out / f"{name}.png"
        if a.skip_existing and png.exists():
            continue
        prefix = f"quant_eval/{a.tag}_{run_id}/{name}" if a.save_latent else None
        wf = build_workflow(a.unet, it["prompt"], it["seed"], a.res, a.steps, a.cfg, prefix, a.lowbit)
        t0 = time.time()
        pid = server.post_json("/prompt", {"prompt": wf, "client_id": CLIENT_ID})["prompt_id"]
        entry = wait_for(server, pid, a.timeout)
        wall = time.time() - t0
        outputs = entry["outputs"]
        png.write_bytes(server.view(outputs["9"]["images"][0]))
        if a.save_latent:
            (out / "latents" / f"{name}.latent").write_bytes(server.view(outputs["10"]["latents"][0]))
        ex = exec_seconds(entry)
        timings.append({"name": name, "wall_s": round(wall, 2), "exec_s": ex})
        print(f"[{len(timings)}/{len(items)}] {name} wall {wall:.1f}s exec {ex}s", flush=True)

    if timings:
        walls = [t["wall_s"] for t in timings]
        steady = walls[1:] if len(walls) > 1 else walls
        meta = {"unet": a.unet, "tag": a.tag, "res": a.res, "steps": a.steps, "cfg": a.cfg, "prompts": a.prompts, "images": len(timings),
                "total_s": round(time.time() - t_all, 1), "mean_wall_s": round(sum(walls) / len(walls), 2),
                "mean_wall_s_excl_first": round(sum(steady) / len(steady), 2), "timings": timings}
        (out / "gen_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{a.tag}: {len(timings)} images, mean {meta['mean_wall_s']} s/img, excl. first {meta['mean_wall_s_excl_first']} s/img -> {out}")


if __name__ == "__main__":
    main()
