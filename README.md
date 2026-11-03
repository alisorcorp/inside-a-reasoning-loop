# What Qwen is thinking when it gets stuck saying "Wait"

A one-page write-up, published at https://alisorcorp.github.io/inside-a-reasoning-loop/

Qwen3.5-0.8B gets stuck re-checking its own answer, writing "Wait" again and again until it runs out of tokens. This reads its residual stream
through Anthropic's Jacobian lens at the token where it decides to loop, before and after the weight edit GPT-6 Astra
made to fix it ([Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)).

Analysis, code and evals by Claude Opus 5.5 (Anthropic).

- `index.html`: the page (static, no build step)
- `assets/`: each figure as a PNG
- `code/`: the scripts behind every number, with a short guide
- `data/`: the single-example lens readouts behind Figure 1, the dose and averaged readout logs, the eval summary

The edit, generations and logs come from the authors' [released logs](https://github.com/RikiyaT/lm_understand-logs).
The lens is Neuronpedia's pre-fitted one for this model, applied as in
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).
