
# jadeGPT

A small GPT learning lab: prepare text, train a decoder-only Transformer, inspect its loss, and generate continuations in a Gradio app or a guided notebook. Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).

jadeGPT is for understanding how language models learn. Its tiny models predict the next token; they are not instruction-tuned chat assistants. The included corpus makes a first experiment possible on a CPU without downloading a dataset or pretrained weights.

## What's new

- Python **3.13** by default, with **3.14** support and a CI matrix for both versions.
- **Gradio 6.28+**, PyTorch 2.14+, NumPy 2.5+, and Transformers 5.17+ with installable package metadata.
- A guided **Start here → Data → Train → Generate** workflow, CPU-sized presets, dataset statistics, training progress, and loss charts.
- Automatic device and precision selection, portable paths, validated training settings, and JSON tokenizer metadata.
- Four rewritten notebooks with explanations, runnable examples, experiments, and shared backend code.

Dependency baselines were checked against the stable [Gradio](https://pypi.org/project/gradio/), [PyTorch](https://pypi.org/project/torch/), [NumPy](https://pypi.org/project/numpy/), and [Transformers](https://pypi.org/project/transformers/) releases on **2026-09-27**. `pyproject.toml` defines dependency constraints; `uv.lock` records exact versions and artifact hashes for a reproducible PyPI installation.

## Install

Install a standard 64-bit **Python 3.13 or 3.14** interpreter from [python.org](https://www.python.org/downloads/) and Git. Python 3.13 is the default in `.python-version`; the commands below use it. Create a new environment when upgrading from the old app.

```console
git clone https://github.com/phatdatnguyen/jadeGPT.git
cd jadeGPT
```

**Windows / PowerShell** — these commands do not require activating scripts:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[notebooks]"
.\.venv\Scripts\python jadegpt_ui.py
```

**macOS / Linux**:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
python jadegpt_ui.py
```

Use `3.14` instead of `3.13` to try Python 3.14. Install with `-e .` if you only need the web app. `python -m pip install -r requirements.txt` is equivalent to that base installation. After activating the environment, the `jadegpt` command also launches the app.

Open **http://127.0.0.1:7860**. To choose a different port:

```console
python jadegpt_ui.py --port 7861
```

### CPU and GPU installation

The small character model works on a CPU. PyTorch is the largest dependency; the correct GPU build depends on your operating system, hardware, and driver. For NVIDIA CUDA or AMD ROCm, install the matching current `torch` build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) **before** installing jadeGPT. You do not need `torchvision` or `torchaudio` for this project. The selected build must satisfy the version constraint in `pyproject.toml`.

For a smaller, CPU-only install on Windows or Linux, run this in the new environment before installing jadeGPT:

```console
python -m pip install "torch>=2.14.0,<3" --index-url https://download.pytorch.org/whl/cpu
```

On Windows without activation, replace `python` with `.\.venv\Scripts\python`. On Apple Silicon, use the normal PyTorch installation and select `auto` or `mps` in the app. Available kernels and memory limits vary by device; CPU `float32` is the simplest fallback.

### Optional: install the locked environment with uv

If you already use [uv](https://docs.astral.sh/uv/), run these commands from the repository root:

```console
uv sync --locked --extra notebooks
uv run --locked --extra notebooks jadegpt
```

The lock covers Python 3.13 and 3.14 and uses PyPI's platform-specific PyTorch builds. Those builds can include large GPU libraries on Linux. For a particular CPU, CUDA, or ROCm wheel index, use the pip installation above; `uv sync` would replace a manually selected build with the locked one. Maintainers can refresh package versions deliberately with `uv lock --upgrade` and run the checks below.

## Your first experiment

1. Open **Start here** for a short explanation of tokens, next-token prediction, and the workflow.
2. In **Data**, use the bundled sample or upload your own UTF-8 `.txt` file. Choose character tokenization for a small model, inspect the preview and counts, then prepare the dataset. A 90/10 train/validation split is a useful starting point.
3. In **Train**, keep the **Tiny · CPU practice** preset, click **Initialize new model**, then **Train / fine-tune**. Watch training and validation loss. Increase the number of steps after the first successful run.
4. In **Generate**, enter a short prompt such as `the ` and generate text. The model you just trained remains loaded; to reuse a saved checkpoint later, first load it in **Train → Load checkpoint or pretrained GPT-2**. Try temperature `0.7` and `1.0` with the same seed, then compare.
5. Change one setting at a time: more data, more steps, a longer context, or a larger model. Record both validation loss and the quality of the samples.

The bundled [tiny corpus](examples/tiny_corpus.txt) is an original, deliberately small teaching dataset. A brief run will produce rough text and may memorize its patterns. Useful general language ability needs substantially more diverse data, parameters, and training.

### Tokenization and model compatibility

| Choice | Useful for | What to remember |
| --- | --- | --- |
| Characters | Fast, transparent experiments from scratch | Each distinct character is a token. Prompts must use characters in the saved vocabulary. |
| GPT-2 BPE | GPT-2 weights and subword experiments | Requires the GPT-2 vocabulary and a larger embedding table. Tokenizer assets may download on first use. |

Keep a checkpoint together with the exact tokenizer used for its training. Two character vocabularies with the same size can still assign different IDs to the same characters. GPT-2 checkpoints must use GPT-2 tokenization. Changing a vocabulary is not a compatible fine-tuning operation.

### Reading the loss chart

Loss measures next-token prediction error; lower is better on the same held-out dataset and tokenizer. Early noisy samples are normal. Falling training loss with rising validation loss suggests overfitting. A lower character-level loss cannot be compared directly with a GPT-2 BPE loss. The contiguous validation split is deliberately simple; for serious evaluation, separate documents and remove duplicates across splits.

The context window limits how many preceding tokens the model sees. A longer context, larger batch, more layers, and wider embeddings all increase memory use. Gradient accumulation increases the effective batch size without requiring the entire batch in memory at once.

## Notebooks

Start JupyterLab **from the repository root**, using the same environment as the app:

```console
python -m jupyter lab
```

On Windows without activation, use `.\.venv\Scripts\python -m jupyter lab`. Select the Python kernel belonging to that environment. If necessary, register it explicitly:

```console
python -m ipykernel install --user --name jadegpt --display-name "Python (jadeGPT)"
```

| Notebook | What you learn | Prerequisite |
| --- | --- | --- |
| [train-gpt.ipynb](train-gpt.ipynb) | Inspect a corpus, encode tokens, train a tiny Transformer, and interpret losses | Installed notebook extras; no network needed for the character example |
| [sample-gpt.ipynb](sample-gpt.ipynb) | Load a trained checkpoint and compare sampling settings | Run the training notebook first |
| [finetune-gpt.ipynb](finetune-gpt.ipynb) | Continue training with a compatible tokenizer and compare results | Run the training notebook first |
| [sample-gpt2.ipynb](sample-gpt2.ipynb) | Explore pretrained GPT-2 text continuation | Explicitly enable the pretrained download cell; internet and enough memory |

Run cells from top to bottom. Notebook artifacts live under `artifacts/notebooks/`; manifests connect the training, sampling, and fine-tuning examples. The GPT-2 notebook defaults to `ENABLE_PRETRAINED = False` so opening or executing it does not unexpectedly download model weights. Set it to `True` when you want that experiment. GPT-2 is a base continuation model, so prompt it with text to continue.

## Settings, files, and upgrading existing work

`config.json` starts with relative `data/` and `models/` directories and automatic device/precision selection. Relative paths are resolved from the directory where you launch the app. Use **Settings** to choose locations and hardware. Generated datasets, weights, environments, and notebook outputs are excluded from Git.

- Prepared data consists of training and validation token files plus readable JSON tokenizer metadata. Keep these files together.
- New checkpoints use names such as `model-100.ckpt`, where the number counts completed optimizer updates. They record the model configuration and, when attached, the tokenizer metadata. Save your datasets separately to continue training later.
- Fine-tuning restores the model weights and starts a new optimizer; it is not an exact continuation of the old optimizer or random-number state.
- Keep a copy of old checkpoints and metadata before migrating. Old Windows-specific paths in a personal `config.json` may need updating.
- Legacy `meta.pkl` files use Python pickle. Only explicitly migrate a file you created or trust; unpickling a malicious file can execute code. Prefer re-preparing the original text into JSON metadata when possible.
- Checkpoint loading uses PyTorch's restricted weights loading. Unsupported legacy objects are not silently loaded with unrestricted pickle.

For **trusted files only**, the Python API provides an explicit legacy migration path. This example keeps the originals and writes modern replacements:

```python
import json
from pathlib import Path
import jadegpt

metadata = jadegpt.load_metadata("data/meta.pkl", trusted_legacy=True)
Path("data/meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
model = jadegpt.resume_gpt("models/old.ckpt", device="cpu", trusted_legacy=True)
model.tokenizer_metadata = metadata
jadegpt.save_checkpoint(model, None, 0, None, "models", "converted.ckpt")
```

This is a local, single-user teaching app. It binds to `127.0.0.1` and does not create a public sharing link. Browser sessions have separate working state, and prepared datasets and training runs get separate subfolders. The process still shares hardware and filesystem access. Running it on a shared server requires deliberate authentication, filesystem isolation, and resource limits; changing `--host` alone does not provide those controls.

## Troubleshooting

| Symptom | Try this |
| --- | --- |
| `No matching distribution found` | Verify `python --version`, use a standard 64-bit Python 3.13/3.14 environment, and upgrade pip. Check that your OS has wheels for the selected PyTorch build. |
| CUDA is unavailable | Check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`; install the right GPU build or select CPU. |
| Out of memory | Reduce batch size, context length, or model width; use the smallest preset first. GPT-2 is much larger than the character demo. |
| Validation data is too short | Add more text, reduce the context window, or reserve more tokens for validation. Both splits need more tokens than the context length. |
| Unknown character in a prompt | Use characters present in the training corpus; preserve the original tokenizer when fine-tuning. |
| Repetitive or nonsensical output | Train longer on better data, compare held-out loss, and vary sampling settings. A tiny model cannot behave like a modern chat model. |
| A notebook cannot import jadegpt | Install the editable project in the selected kernel's environment and restart the kernel. |

## Development

```console
python -m pip install -e ".[notebooks,dev]"
python -m pip check
python -m ruff check .
python -m pytest -q
python -m build
```

CI installs CPU PyTorch and runs dependency, code, test, and packaging checks on Python 3.13 and 3.14. Tests use tiny models and local fixtures; pretrained downloads and GPU training are separate, opt-in experiments.

The main implementation is intentionally easy to explore:

- `model.py`: Transformer blocks, attention, model initialization, and sampling.
- `jadegpt.py`: dataset preparation, tokenizer metadata, training, checkpoint loading, and generation.
- `jadegpt_ui.py`: Gradio workflow and command-line entry point.
- `examples/` and the four notebooks: guided experiments using the same backend.

The project retains nanoGPT's educational spirit and credits Andrej Karpathy for the original architecture and training approach. See [LICENSE.txt](LICENSE.txt) for the project license and [nanoGPT](https://github.com/karpathy/nanoGPT) for the upstream implementation.
