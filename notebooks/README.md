# Notebooks

Walk-throughs for each sub-package of **neuro-ai-bridge**. They are
intended to be read *in order* — each one builds on concepts
introduced in the previous notebook.

| # | Notebook | Package it explores |
|---|----------|---------------------|
| 01 | [`01_hebbian_learning.py`](01_hebbian_learning.py) | `src/plasticity` — Hebb, Oja, BCM |
| 02 | [`02_spiking_neural_networks.py`](02_spiking_neural_networks.py) | `src/snn` — LIF neurons, surrogate gradients |
| 03 | [`03_predictive_processing.py`](03_predictive_processing.py) | `src/predictive_coding` — PC network, free-energy VAE |
| 04 | [`04_hippocampal_memory.py`](04_hippocampal_memory.py) | `src/memory` — episodic buffer, replay, CLS |
| 05 | [`05_attention_as_gating.py`](05_attention_as_gating.py) | Attention ↔ biased competition |

## Why `.py` and not `.ipynb`?

Each notebook is stored as a plain Python script using the
[`# %%`](https://jupyter.readthedocs.io/en/latest/interactive.html#cell-mode)
cell markers that Jupyter, VS Code and PyCharm all understand
natively.

There are three reasons for this choice:

1. **Clean diffs.** Git diffs over `.ipynb` JSON are almost
   unreadable; diffs over these `.py` files are trivial to review.
2. **Fast editing.** You can open them in any editor and run them
   cell-by-cell just like a notebook — no Jupyter required.
3. **Reproducible.** They don't carry stale execution output,
   so cloning the repo always gives you a clean starting state.

## Running a notebook

### Option A — directly as a script

```bash
python notebooks/01_hebbian_learning.py
```

Each notebook's `__main__` section is safe to run top-to-bottom and
will produce matplotlib figures.

### Option B — as a real Jupyter notebook

Use the included helper to convert any (or all) of the cell
scripts into `.ipynb` files:

```bash
pip install nbformat jupyter
python scripts/py_to_ipynb.py                        # convert all
python scripts/py_to_ipynb.py notebooks/02_spiking_neural_networks.py
jupyter notebook notebooks/01_hebbian_learning.ipynb
```

### Option C — in VS Code / PyCharm

Both IDEs detect the `# %%` markers automatically and offer a
"Run Cell" gutter icon on each cell. No conversion step required.

## Authoring conventions

If you add a new notebook, please follow the same conventions so
the converter and readers can rely on them:

- Start every markdown cell with `# %% [markdown]`.
- Start every code cell with `# %%`.
- Keep `sys.path` bootstrapping at the top of each file so the
  notebook works whether the repo root is on `PYTHONPATH` or not.
- End with a short **Take-aways** and **Suggested next steps**
  markdown block.

---

Part of a broader research programme on neuroscience-inspired AI
and the humanisation of artificial intelligence.
