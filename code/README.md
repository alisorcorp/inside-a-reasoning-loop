# Code

These are the scripts behind the numbers on the page, as they were run. They expect:

- `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, downloaded to a folder (`base_hf` below)
- the authors' logs repo, at least `exp8_math/` (github.com/RikiyaT/lm_understand-logs)
- the GSM8K test set as JSONL (github.com/openai/grade-school-math)
- Neuronpedia's lens `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt` from the Hugging Face
  repo `neuronpedia/jacobian-lens`
- Python with `torch`, `transformers` (5.x, for `qwen3_5`), `safetensors`, `numpy`, `httpx`, `math-verify`, and
  llama.cpp for the generation evals

Rebuild the edit with the authors' own `exp8_math/code/apply_edit.py` (it checks every tensor hash), then convert
with llama.cpp's `convert_hf_to_gguf.py --outtype bf16`.

| Script | What it does |
|---|---|
| `astra_direction.py` | Recovers the edited direction u from the diff and shows the edit is rank one with the component at -5x |
| `make_alpha_ckpt.py` | Writes a checkpoint with u applied at any strength; `--check` at alpha 6 reproduces Astra's file |
| `lens_read.py` | Reads a direction through the Jacobian lens (both signs) and through the plain logit lens |
| `calibrate_loop.py` | Calibrates the compression-based loop detector on the authors' released generations |
| `capture.py` | Window-averaged activations in looping vs finishing generations (and `apply_edit` for dose tests) |
| `analyze.py` | Mean-difference directions vs Astra's, and which way the 64 edited neurons write |
| `capture_decision.py` | Activations and log-probs at the exact decision tokens; `--edit --alpha` for the dose test |
| `analyze_decision.py`, `lens_layers.py` | The decision-point comparison and per-layer lens readouts |
| `jlens_decisions.py` | Averaged lens readouts at loop vs stop decisions (Figure 2), base or edited |
| `fig_data.py` | Single-example readouts for Figure 1 |
| `eval_gsm8k.py`, `campaign.sh` | GSM8K generation eval against llama-server with the authors' prompt and sampling |
| `summarize_eval.py` | The eval table and paired counts |

`campaign.sh` and the eval assume a llama-server per model on port 8301 and a `~/wsp` working folder; adjust the
paths at the top. Scripts that import `capture` need this folder on the Python path.
