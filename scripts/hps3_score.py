"""HPSv3 (MizzenAI/HPSv3, MIT) human-preference score for the eval images, own minimal inference code.

The checkpoint is Qwen2-VL-7B-Instruct (not 2.5) + new token <|Reward|> + rm_head MLP (3584-1024-16-2). Output = [mu, log_sigma]
of the head at the <|Reward|> position of the last hidden state (after the final norm). Like the reference code, the LLM gets plain
sequential position ids (no 3D mrope index). The prompt text is parsed (ast.literal_eval, never imported) from the reference source.
Layers that do not fit on the GPU stay in pinned CPU RAM and are streamed in by accelerate (device_map from infer_auto_device_map):
--gpu_gib 11 (default) keeps about 11 GiB on the GPU, the remaining decoder layers live in CPU RAM.

Files the user downloads (never fetched by this script), under $LOWBITMOSS_AUX (default $LOWBITMOSS_HOME/models_aux):
  HPSv3/HPSv3.safetensors   https://huggingface.co/MizzenAI/HPSv3 (16.6 GB, MIT); only this file is read
  qwen2vl_proc/             config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, preprocessor_config.json,
                            chat_template.json, generation_config.json of https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct (no weights)
  hpsv3_src/                git clone of https://github.com/MizzenAI/HPSv3 (only hpsv3/dataset/data_collator_qwen.py and assets/example*.png are read)
The teacher size in the summary table is read from $COMFYUI_DIR/models/diffusion_models; the other sizes from $LOWBITMOSS_HOME/out.

  python -I scripts/hps3_score.py --tags teacher_bf16,mixH2_x --out eval/hps3_summary.json [--examples] [--wait_gpu]

Scores are cached per image in <eval_dir>/<tag>/hps3.json; a tag is scored once all prompts of prompts/eval_prompts.json have an image.
"""
import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import AUX, EVAL, OUT, PROMPTS, teacher_path

CKPT = Path(AUX) / "HPSv3" / "HPSv3.safetensors"
QWEN_DIR = Path(AUX) / "qwen2vl_proc"  # config / tokenizer / preprocessor of Qwen/Qwen2-VL-7B-Instruct
REF_SRC = Path(AUX) / "hpsv3_src"
REF_COLLATOR = REF_SRC / "hpsv3" / "dataset" / "data_collator_qwen.py"
EXAMPLES = [(REF_SRC / "assets" / f"example{i}.png",
             "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker") for i in (1, 2)]
PIXELS = 256 * 28 * 28
REWARD_TOKEN = "<|Reward|>"
CATEGORIES = ["text", "portrait", "detail", "counting", "varied"]
SIZE_FILE = {  # tag -> checkpoint in out/ whose size is reported (default: <tag>.safetensors)
    "mixH2_x_f8": "mixH2_x_fp8lora.safetensors",
    "tern_v1": "moss_sft_tern_v1.safetensors",
}


def reference_strings():
    consts = {}
    for node in ast.parse(REF_COLLATOR.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant):
            consts[node.targets[0].id] = node.value.value
    return consts["INSTRUCTION"], consts["prompt_with_special_token"]


def smart_resize(h, w, factor=28, min_pixels=PIXELS, max_pixels=PIXELS):
    h_bar, w_bar = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar, w_bar = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def size_gb(tag):
    f = Path(teacher_path()) if tag == "teacher_bf16" else Path(OUT) / SIZE_FILE.get(tag, f"{tag}.safetensors")
    return round(f.stat().st_size / 1e9, 2) if f.exists() else None


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def wait_for_gpu(limit_mib=2000, poll=60, max_wait=2400):
    t0 = time.time()
    while (used := gpu_used_mib()) >= limit_mib:
        if time.time() - t0 > max_wait:
            sys.exit(f"GPU still busy after {max_wait}s ({used} MiB used)")
        print(f"GPU busy ({used} MiB used), waiting", flush=True)
        time.sleep(poll)


class HPSv3:
    def __init__(self, gpu_gib=11, cpu_gib=90):
        self.processor = AutoProcessor.from_pretrained(QWEN_DIR)
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [REWARD_TOKEN]})
        self.reward_id = self.processor.tokenizer.convert_tokens_to_ids(REWARD_TOKEN)
        self.instruction, self.suffix = reference_strings()

        cfg = Qwen2VLConfig.from_pretrained(QWEN_DIR)
        cfg.text_config.vocab_size = len(self.processor.tokenizer)
        with init_empty_weights():
            model = Qwen2VLModel._from_config(cfg, attn_implementation="sdpa", dtype=torch.bfloat16)
        model.eval()
        device_map = infer_auto_device_map(
            model, max_memory={0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"},
            no_split_module_classes=["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"], dtype=torch.bfloat16)
        n_cpu = sum(v == "cpu" for k, v in device_map.items() if ".layers." in k)
        print(f"device_map: {len(device_map)} entries, decoder layers on cpu: {n_cpu}", flush=True)

        def where(name):
            return device_map[max((k for k in device_map if k == "" or name == k or name.startswith(k + ".")), key=len)]

        head = {}
        loaded = set()
        with safe_open(str(CKPT), "pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("rm_head."):
                    head[key[len("rm_head."):]] = f.get_tensor(key).float()
                    continue
                if key == "lm_head.weight":
                    continue
                name = "language_model." + key[len("model."):] if key.startswith("model.") else key
                dev = where(name)
                t = f.get_tensor(key)
                set_module_tensor_to_device(model, name, dev, value=t.pin_memory() if dev == "cpu" else t, dtype=torch.bfloat16)
                loaded.add(name)
        missing = set(model.state_dict()) - loaded
        assert not missing, f"not in checkpoint: {sorted(missing)[:5]}"
        self.model = dispatch_model(model, device_map=device_map, main_device=0)

        hidden = cfg.text_config.hidden_size
        self.head = torch.nn.Sequential(  # the Dropout holds the checkpoint's index 2, it has no weights
            torch.nn.Linear(hidden, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.05),
            torch.nn.Linear(1024, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        self.head.load_state_dict(head, strict=True)
        self.head.eval().to("cuda:0")

    def inputs(self, prompt, image_path):
        img = Image.open(image_path).convert("RGB")
        h, w = smart_resize(img.height, img.width)
        img = img.resize((w, h), Image.BICUBIC)
        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                + self.instruction.format(text_prompt=prompt) + self.suffix + "<|im_end|>\n<|im_start|>assistant\n")
        batch = self.processor(text=[text], images=[img], return_tensors="pt")
        assert (batch["input_ids"] == self.reward_id).sum() == 1
        return batch

    @torch.no_grad()
    def score(self, prompt, image_path):
        """returns (mu, sigma); sigma = exp of the second head output"""
        mu, log_sigma = self.raw(prompt, image_path)
        return mu, math.exp(log_sigma)

    @torch.no_grad()
    def raw(self, prompt, image_path):
        batch = self.inputs(prompt, image_path)
        ids = batch["input_ids"]
        pos = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        hs = self.model(input_ids=ids.to("cuda:0"), attention_mask=batch["attention_mask"].to("cuda:0"), position_ids=pos.to("cuda:0"),
                        pixel_values=batch["pixel_values"].to("cuda:0"), image_grid_thw=batch["image_grid_thw"].to("cuda:0")).last_hidden_state
        h = hs[0, (ids[0] == self.reward_id).nonzero().item()].float().to("cuda:0")
        mu, log_sigma = self.head(h).tolist()
        return mu, log_sigma


def score_tag(scorer, tag, eval_dir, prompts):
    folder = eval_dir / tag
    images = sorted(folder.glob("[0-9][0-9]_seed*.png"))
    if len(images) < len(prompts):
        print(f"{tag}: {len(images)}/{len(prompts)} images, skipped", flush=True)
        return
    path = folder / "hps3.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for p in images:
        if p.name in res:
            continue
        pid = int(p.name[:2])
        t0 = time.time()
        mu, log_sigma = scorer.raw(prompts[pid]["prompt"], p)
        res[p.name] = {"id": pid, "mu": mu, "log_sigma": log_sigma}
        path.write_text(json.dumps(res, indent=1))
        print(f"{tag} {p.name} mu={mu:.4f} log_sigma={log_sigma:.3f} ({time.time() - t0:.1f}s)", flush=True)


def boot_ci(x, n=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = np.asarray(x)[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(tags, eval_dir, prompts, sizes, out_json):
    cats = {pid: p["category"] for pid, p in prompts.items()}
    per = {}
    for tag in tags:
        path = eval_dir / tag / "hps3.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if len(r) < len(prompts):
            continue
        per[tag] = {v["id"]: v["mu"] for v in r.values()}
    ref = per.get("teacher_bf16")
    ids = sorted(prompts)
    summary = {}
    for tag, mu in per.items():
        x = np.array([mu[i] for i in ids])
        s = {"n": len(x), "mean_mu": float(x.mean()), "sd_mu": float(x.std(ddof=1)), "ci95": boot_ci(x),
             "category_mean_mu": {c: float(np.mean([mu[i] for i in ids if cats[i] == c])) for c in CATEGORIES},
             "mu_by_prompt": {str(i): mu[i] for i in ids}}
        if ref is not None and tag != "teacher_bf16":
            d = x - np.array([ref[i] for i in ids])
            s["diff_to_teacher"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "sd": float(d.std(ddof=1)),
                                    "wins": int((d > 0).sum())}
        s["size_gb"] = sizes[tag]
        summary[tag] = s
    out_json.write_text(json.dumps(summary, indent=1))

    lines = ["| tag | size GB | mean mu [bootstrap 95%] | " + " | ".join(CATEGORIES) + " | diff to teacher +- SE (wins/24) |",
             "|---|---|---|" + "---|" * len(CATEGORIES) + "---|"]
    for tag, s in summary.items():
        d = s.get("diff_to_teacher")
        lines.append(f"| {tag} | {s['size_gb'] if s['size_gb'] is not None else '-'} | {s['mean_mu']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}] | "
                     + " | ".join(f"{s['category_mean_mu'][c]:.2f}" for c in CATEGORIES) + " | "
                     + (f"{d['mean']:+.2f} +- {d['se']:.2f} ({d['wins']}/{s['n']})" if d else "-") + " |")
    mus = {t: np.array([per[t][i] for i in ids]) for t in per}
    paired_se = [(a, b, (mus[a] - mus[b]).std(ddof=1) / math.sqrt(len(ids))) for a in mus for b in mus if a < b]
    cand = [se for a, b, se in paired_se if "teacher_bf16" not in (a, b) and "tern_v1" not in (a, b)]
    sd = np.array([s["sd_mu"] for s in summary.values()])
    ci_half = np.array([(s["ci95"][1] - s["ci95"][0]) / 2 for s in summary.values()])
    notes = [
        "", "mu = HPSv3 raw score (higher = preferred), mean over 24 prompts, one seed per prompt, images resized to the HPSv3 pixel budget (256 x 28 x 28; 420x420 for a 768px square image).",
        "diff to teacher = mean of (student mu - teacher mu) per prompt, SE = sd of the 24 paired differences / sqrt(24); wins = prompts where student mu > teacher mu.",
        "", "Noise: only 24 prompts and ONE seed per prompt, so seed-to-seed variation of a model is not measured at all and the intervals only reflect which 24 prompts were drawn.",
        f"- sd of mu across prompts is {sd.min():.2f}-{sd.max():.2f}; bootstrap 95% half-width of a mean is {ci_half.min():.2f}-{ci_half.max():.2f}.",
        f"- paired SE between two candidate models (excluding teacher and tern_v1) is {min(cand):.2f}-{max(cand):.2f} (median {float(np.median(cand)):.2f});"
        f" a difference below about 2 x that ({2 * float(np.median(cand)):.2f} mu) cannot be told apart from prompt-sampling noise.",
    ]
    return summary, "\n".join(lines + notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", default=str(Path(EVAL) / "hps3_summary.json"))
    ap.add_argument("--eval_dir", default=EVAL)
    ap.add_argument("--examples", action="store_true", help="score the two example images of the HPSv3 repo and print")
    ap.add_argument("--wait_gpu", action="store_true", help="poll nvidia-smi until < 2000 MiB used (max 40 min)")
    ap.add_argument("--gpu_gib", type=float, default=11, help="GiB of the model kept on the GPU, the rest of the decoder layers stays in pinned CPU RAM")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    tags = [t for t in args.tags.split(",") if t]
    prompts = {p["id"]: p for p in json.loads((Path(PROMPTS) / "eval_prompts.json").read_text(encoding="utf-8"))}
    sizes = {t: size_gb(t) for t in tags}

    pending = any(len(list((eval_dir / t).glob("[0-9][0-9]_seed*.png"))) >= len(prompts)
                  and len(json.loads((eval_dir / t / "hps3.json").read_text()) if (eval_dir / t / "hps3.json").exists() else {}) < len(prompts)
                  for t in tags)
    if args.examples or pending:
        if args.wait_gpu:
            wait_for_gpu()
        t0 = time.time()
        scorer = HPSv3(gpu_gib=args.gpu_gib)
        print(f"loaded in {time.time() - t0:.0f}s, GPU used {gpu_used_mib()} MiB", flush=True)
        if args.examples:
            for path, prompt in EXAMPLES:
                print(path.name, scorer.raw(prompt, path), flush=True)
        for tag in tags:
            score_tag(scorer, tag, eval_dir, prompts)
        print(f"peak GPU alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        del scorer
        torch.cuda.empty_cache()
    summary, table = summarize(tags, eval_dir, prompts, sizes, Path(args.out))
    Path(args.out).with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
