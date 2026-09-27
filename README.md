# jadeGPT

A hands-on language model lab inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). Prepare text, train a small Transformer, watch its loss, and generate continuations through Gradio or guided Jupyter notebooks.

The app includes model presets, a practice dataset, live loss charts, checkpoint downloads, and automatic CPU/CUDA/Apple MPS selection. Tiny models are for learning next-token prediction; short training runs produce rough text rather than a chat assistant.

## Quick start

Use **64-bit Python 3.13 or 3.14**. Python 3.13 is the project default. Dependencies, including Gradio 6 and PyTorch, are defined in [pyproject.toml](pyproject.toml); [uv.lock](uv.lock) records exact versions.

```console
git clone https://github.com/phatdatnguyen/jadeGPT.git
cd jadeGPT
```

**Windows / PowerShell** — no environment activation required:

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

Open **http://127.0.0.1:7860**. Use `--port 7861` to choose another port. For Python 3.14, replace `3.13` in the environment creation command. Install with `-e .` if you only need the web app.

### Hardware options

The Tiny preset runs on a CPU. For a particular NVIDIA CUDA or AMD ROCm build, install PyTorch using its [official installation selector](https://pytorch.org/get-started/locally/) **before** installing jadeGPT. The build must satisfy the project's PyTorch version requirement.

For CPU-only Windows/Linux environments, install this first:

```console
python -m pip install "torch>=2.14.0,<3" --index-url https://download.pytorch.org/whl/cpu
```

On Windows without activation, use `.\.venv\Scripts\python` in place of `python`. On Apple Silicon, use the normal installation and select `auto` or `mps` in Settings.

### Using uv

If you use [uv](https://docs.astral.sh/uv/), install and launch the locked environment from the repository root:

```console
uv sync --locked --extra notebooks
uv run --locked --extra notebooks jadegpt
```

The lock uses PyPI's platform-specific PyTorch builds. Use the pip instructions above for a custom CPU/CUDA/ROCm index; `uv sync` replaces a manually selected build with the locked one.

## Your first experiment

1. Open **Start here** for a short introduction to tokens and next-token prediction.
2. In **Data**, click **Use practice dataset**, keep **Characters**, then click **Prepare dataset**. You can also upload UTF-8 text or paste your own writing.
3. In **Train**, keep **Tiny · CPU practice**, click **Initialize new model**, then **Train / fine-tune**. Watch the loss chart; **Stop and save** preserves completed updates.
4. In **Generate**, try the prompt `the `. Compare temperatures such as `0.7` and `1.0`, or use `0` for greedy decoding.
5. Change one setting at a time: training steps, context length, model size, or dataset. Compare validation loss and generated text.

The bundled [practice corpus](examples/tiny_corpus.txt) works offline. It is deliberately small; more varied data and longer training are needed for useful text.

### Load and fine-tune a model

In **Train → Load checkpoint or pretrained GPT-2**, load a saved checkpoint or choose a GPT-2 variant. Pretrained weights download on first use; start with the smallest `gpt2` variant.

- **Character models:** load the model before preparing new text and keep **Reuse the loaded model’s character vocabulary** enabled. Prompts and training text must use its known characters.
- **Pretrained GPT-2:** prepare data with **GPT-2 BPE**. Tokenizer assets may also download on first use.
- **Fine-tuning:** restores learned weights with a fresh optimizer. It does not resume the previous optimizer or random-number state.

Checkpoints created by the app include their tokenizer. Older checkpoints may need their original metadata file and the explicit trusted-legacy option; enable that only for files you trust.

## Notebooks

Launch JupyterLab from the repository root in the same environment:

```console
python -m jupyter lab
```

On Windows without activation, use `.\.venv\Scripts\python -m jupyter lab`. Select that environment's Python kernel and run cells from top to bottom.

| Notebook | Lesson | Prerequisite |
| --- | --- | --- |
| [train-gpt.ipynb](train-gpt.ipynb) | Tokenize text, train a tiny GPT, and interpret loss | Notebook extras; no downloads for the default experiment |
| [sample-gpt.ipynb](sample-gpt.ipynb) | Reload a checkpoint and compare sampling settings | Run the training notebook first |
| [finetune-gpt.ipynb](finetune-gpt.ipynb) | Fine-tune with the original vocabulary and compare results | Run the training notebook first |
| [sample-gpt2.ipynb](sample-gpt2.ipynb) | Generate text with pretrained GPT-2 | Set `ENABLE_PRETRAINED = True` to download weights |

Notebook outputs are saved under `artifacts/notebooks/`. The notebooks share the app's training and generation code.

## Settings and saved files

Use **Settings** to choose the device, precision, and output folders. Defaults are saved in `config.json` in the directory where you launch the app; relative paths resolve from there.

- `data/`: tokenized datasets and JSON tokenizer metadata, separated by session and preparation.
- `models/`: checkpoints, tokenizer metadata, and training metrics, separated by run.
- `artifacts/notebooks/`: notebook experiments and manifests connecting the lessons.

Models remain loaded within their browser session. Save checkpoints to reuse them later. The app binds to localhost by default and is intended for local use; exposing it on a shared server requires authentication and resource controls.

## Experiment tips

- **Out of memory:** reduce batch size, context length, or model width. GPT-2 needs much more memory than the Tiny preset.
- **Dataset too short:** both training and validation splits need more tokens than the context length. Add text or shorten the context.
- **Unknown character:** use the checkpoint's original character vocabulary and prompts containing known characters.
- **Training loss falls but validation loss rises:** try more diverse data, fewer steps, or more dropout. Compare losses only with the same tokenizer and held-out data.

## Development

```console
python -m pip install -e ".[notebooks,dev]"
python -m pip check
python -m ruff check .
python -m pytest -q
python -m build
```

CI runs CPU tests and packaging checks on Python 3.13 and 3.14. Tests use tiny models and local fixtures; pretrained downloads are optional experiments.

For the underlying ideas, watch Karpathy's [Let's build GPT, from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY). See [LICENSE.txt](LICENSE.txt) for the project license.
