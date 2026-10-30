# Method guide

The current whitepaper in `paper/` is the source for the experimental method, results, limitations, and exact tested configurations. This guide maps its stages to the code.

1. **Capture calibration states.** `capture.py` records non-distilled teacher inputs at selected sampling steps. `capture_turbo.py` records all four distilled conditional steps. Prompt definitions are in `prompts/`; captured states and matching teacher outputs are distributed on Hugging Face under `data/`.
2. **Measure sensitivity and allocate bits.** `sensitivity.py`, `sensitivity_extra.py`, `make_alloc.py`, and the `alloc_mix*.json` recipes define per-linear choices. Output-MSE sensitivity does not reliably predict text rendering and counting.
3. **Train block by block.** `train_blocks.py` fits quantized weights, group scales, and optional low-rank corrections, with student-stream inputs. `qlayer.py`, `lowbit_pack.py`, and `common.py` implement the training and packed-weight behavior. `e2e_tune.py` optionally adds a tuning round.
4. **Compact and assemble.** `compact_lora.py`, `quantize_extras.py`, and `add_turbo_factors.py` implement storage reductions and folded corrections. `prune_encoder.py` removes unused tensors from the existing GPTQ encoder; `make_small_vae.py` builds the decoder component.
5. **Generate and evaluate.** `gen_eval.py` and `gen_inproc.py` generate saved comparisons. `metrics.py` computes likeness metrics; `hps3_score.py` implements the local HPS-based scoring path. `review_reporting.py` and the chart scripts produce corrected summaries.

The non-distilled evaluation uses 25 steps, CFG 2.5, `er_sde` / `simple`, and 768 px. The primary distilled evaluation uses four steps, CFG 1, `euler` / `simple`, and 1024 px, with secondary 768 px runs. Six-step sampling is an informal author tip, not the source of the published scores.

All package sizes are file storage rather than peak memory. Exact encoder/decoder combinations matter: the 7.97 GB quality package is unmeasured, and the 7.18 GB package has no complete-package judge pass. Larger-data experiments also change optimizer steps. The combined 1-bit attention-and-FFN allocation fails. The distilled teacher is harder to quantize in this setting. Local HPS numerical parity with official HPSv3 remains unvalidated; separate judge instances and one seed per prompt limit comparisons.

For the binary layout, see `SPEC.md`. For environment and test commands, see `TESTING.md`. For measured configurations and paired intervals, see whitepaper Tables I7-I9.
