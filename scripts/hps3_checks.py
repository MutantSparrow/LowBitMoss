"""Plausibility checks for hps3_score.py (there are no reference values to compare with): determinism, prompt sensitivity, degraded
images, mrope vs sequential positions. Needs the GPU, the HPSv3 files described in hps3_score.py and the two example images of the HPSv3 repo.
Test images go to $LOWBITMOSS_HOME/logs/hps3_check_imgs.

  python -I scripts/hps3_checks.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hps3_score as H
from _paths import LOGS

s = H.HPSv3()
(p1, prompt), (p2, _) = H.EXAMPLES
tmp = Path(LOGS) / "hps3_check_imgs"
tmp.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (768, 768), (128, 128, 128)).save(tmp / "gray.png")
Image.fromarray(np.random.default_rng(0).integers(0, 256, (768, 768, 3), dtype=np.uint8)).save(tmp / "noise.png")
Image.open(p1).convert("RGB").resize((96, 96)).resize((768, 768)).save(tmp / "ex1_blur.png")

for name, path, pr in [("ex1", p1, prompt), ("ex1 again", p1, prompt), ("ex2", p2, prompt),
                       ("ex1 wrong prompt", p1, "a red sports car driving on a highway at night, photograph"),
                       ("ex2 wrong prompt", p2, "a red sports car driving on a highway at night, photograph"),
                       ("ex1 blurred 96px", tmp / "ex1_blur.png", prompt),
                       ("gray", tmp / "gray.png", prompt), ("noise", tmp / "noise.png", prompt)]:
    print(f"{name:20s} mu={s.raw(pr, path)[0]:.3f}", flush=True)

# 3D mrope positions computed by transformers instead of the sequential ids of the reference code
for name, path in [("ex1", p1), ("ex2", p2)]:
    b = s.inputs(prompt, path)
    ids = b["input_ids"]
    with torch.no_grad():
        hs = s.model(input_ids=ids.cuda(), attention_mask=b["attention_mask"].cuda(), pixel_values=b["pixel_values"].cuda(),
                     image_grid_thw=b["image_grid_thw"].cuda(), mm_token_type_ids=(ids == s.model.config.image_token_id).int().cuda()).last_hidden_state
    h = hs[0, (ids[0] == s.reward_id).nonzero().item()].float().cuda()
    print(f"{name} mrope3d mu={s.head(h).tolist()[0]:.3f}", flush=True)
