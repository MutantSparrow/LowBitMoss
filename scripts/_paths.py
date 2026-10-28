"""Locations shared by all scripts.

LOWBITMOSS_HOME  work directory holding data/ out/ eval/ logs/ (default: the repository root)
LOWBITMOSS_AUX   folder with the extra model files of hps3_score.py (default: $LOWBITMOSS_HOME/models_aux)
COMFYUI_DIR     ComfyUI checkout (the folder with main.py); only needed by scripts that import comfy, read it through comfy_dir()
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.abspath(os.environ.get("LOWBITMOSS_HOME") or REPO)
DATA, OUT, EVAL, LOGS = (os.path.join(HOME, d) for d in ("data", "out", "eval", "logs"))
AUX = os.path.abspath(os.environ.get("LOWBITMOSS_AUX") or os.path.join(HOME, "models_aux"))
PROMPTS = os.path.join(REPO, "prompts")
NODES = os.path.join(REPO, "custom_nodes")  # parent folder of ComfyUI-MOSS-LowBit


def comfy_dir():
    d = os.environ.get("COMFYUI_DIR")
    if not d or not os.path.isfile(os.path.join(d, "main.py")):
        raise SystemExit("set COMFYUI_DIR to your ComfyUI checkout (the folder that contains main.py)")
    return os.path.abspath(d)


def teacher_path():
    return os.path.join(comfy_dir(), "models", "diffusion_models", "moss_image_sft_bf16.safetensors")
