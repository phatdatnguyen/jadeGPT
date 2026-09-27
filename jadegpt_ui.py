"""A local, guided nanoGPT learning lab, built with Gradio 6."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import re
import sysconfig
import threading
import uuid

import gradio as gr
import pandas as pd
import torch

import jadegpt

ROOT = Path(__file__).resolve().parent
DEFAULTS = {"data_dir": "data", "model_dir": "models", "device": "auto", "dtype": "auto"}
PRESETS = {
    "Tiny · CPU practice": (2, 2, 64, 64, 4, 100, 0.001),
    "Small · longer experiments": (4, 4, 128, 128, 8, 500, 0.0006),
    "nanoGPT · GPU recommended": (6, 6, 384, 256, 8, 1000, 0.0003),
}
CSS = """
.gradio-container { max-width: 1240px !important; margin: auto; }
#hero { padding: 30px; border: 1px solid var(--border-color-primary);
  border-radius: 18px; background: linear-gradient(125deg, #0b302c, #174d46); }
#hero h1, #hero p { color: #f0fdf8 !important; }
#hero h1 { font-size: 2.5rem; letter-spacing: -0.04em; }
#hero p { max-width: 760px; }
.guide { border-left: 3px solid #34b399; padding: 8px 16px; }
footer { opacity: .75; }
"""


@dataclass
class LabSession:
    """Models live server-side, keyed by a browser session."""

    model: object = None
    model_meta: dict | None = None
    dataset: dict | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    busy: bool = False


_sessions: dict[str, LabSession] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str) -> LabSession:
    with _sessions_lock:
        return _sessions.setdefault(session_id, LabSession())


def drop_session(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is not None:
        session.stop.set()


def load_settings(path: Path | None = None) -> dict:
    path = path or Path.cwd() / "config.json"
    settings = dict(DEFAULTS)
    if path.exists():
        try:
            settings.update({k: v for k, v in json.loads(path.read_text("utf-8")).items() if k in settings})
        except (ValueError, OSError, AttributeError):
            pass
    settings["device"] = {"GPU": "cuda", "CPU": "cpu"}.get(settings["device"], settings["device"])
    return settings


def save_settings(data_dir, model_dir, device, dtype):
    if not str(data_dir).strip() or not str(model_dir).strip():
        raise gr.Error("Choose a data folder and a checkpoint folder.")
    selected_device = jadegpt.resolve_device(device)
    selected_dtype = jadegpt.resolve_dtype(dtype, selected_device)
    settings = dict(data_dir=str(data_dir), model_dir=str(model_dir), device=device, dtype=dtype)
    (Path.cwd() / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return f"Settings saved. Runtime: **{selected_device} / {selected_dtype}**. Changes apply to your next operation."


def runtime_description() -> str:
    device = jadegpt.resolve_device("auto")
    name = torch.cuda.get_device_name() if device == "cuda" else ("Apple Silicon" if device == "mps" else "CPU")
    return f"**Available runtime:** {name} · PyTorch {torch.__version__} · Gradio {gr.__version__}"


def sample_text():
    path = ROOT / "examples" / "tiny_corpus.txt"
    if not path.exists():
        path = Path(sysconfig.get_path("data")) / "share" / "jadegpt" / "examples" / "tiny_corpus.txt"
    return path.read_text(encoding="utf-8")


def _model_description(session):
    if session.model is None:
        return "**No model loaded.** Prepare data and initialize a model, or load a checkpoint."
    config = session.model.config
    return (f"**Model ready** · {session.model.get_num_params():,} parameters · "
            f"{config.n_layer} layers / {config.n_head} heads / {config.n_embd} embedding width · "
            f"context {config.block_size} · vocabulary {config.vocab_size:,}")


def _compatible(left, right):
    if not left or not right or left["encoding"] != right["encoding"]:
        return False
    return left["encoding"] == "gpt2" or left["stoi"] == right["stoi"]


def prepare_data(session_id, upload, text, split, tokenizer, data_dir, reuse_vocabulary=True):
    session = get_session(session_id)
    data = jadegpt.open_dataset_file(upload) if upload else text
    if not data or not data.strip():
        raise gr.Error("Upload a UTF-8 text file, paste text, or click Use practice dataset.")
    if not str(data_dir).strip():
        raise gr.Error("Choose a data folder in Settings.")
    training, validation = jadegpt.split_dataset(data, float(split))
    # A new directory prevents stale memmaps and cross-tab overwrites.
    folder = Path(data_dir).expanduser() / session_id / uuid.uuid4().hex[:8]
    original = session.model_meta if reuse_vocabulary and session.model_meta and tokenizer == "Characters" and session.model_meta["encoding"] == "custom" else None
    stats = jadegpt.export_data_to_files(data, training, validation, tokenizer == "GPT-2 BPE",
                                       folder, "train.bin", "val.bin", "meta.json", tokenizer_metadata=original)
    metadata = jadegpt.load_metadata(folder / "meta.json")
    session.dataset = {**stats, "directory": str(folder), "metadata": metadata}
    compatible = session.model is None or _compatible(session.model_meta, metadata)
    note = "" if compatible else "\n\n**The loaded model uses a different tokenizer. Initialize a new model before training.**"
    summary = (f"### Dataset ready\n{len(data):,} characters → **{stats['train_tokens']:,} training tokens** "
               f"+ **{stats['val_tokens']:,} validation tokens**.\n\n"
               f"Vocabulary: **{stats['vocab_size']:,}** · tokenizer: **{tokenizer}**. "
               "Both splits must contain more tokens than the training context." + note)
    return summary, data[:2500], str(folder / "meta.json")


def initialize_model(session_id, seed, layers, heads, width, context, dropout, bias):
    session = get_session(session_id)
    if session.dataset is None:
        raise gr.Error("Prepare a dataset in the Data tab first so the vocabulary is known.")
    metadata = session.dataset["metadata"]
    model = jadegpt.init_gpt(int(seed), int(layers), int(heads), int(width), float(dropout),
                            bool(bias), int(context), metadata["vocab_size"])
    model.tokenizer_metadata = metadata
    session.model, session.model_meta = model, metadata
    return _model_description(session)


def load_model(session_id, source, checkpoint, metadata_file, pretrained, seed, device, trusted):
    session = get_session(session_id)
    if source == "Pretrained GPT-2":
        model = jadegpt.init_gpt2(pretrained, int(seed))
        metadata = {"encoding": "gpt2", "vocab_size": 50257, "actual_vocab_size": 50257}
    else:
        if not checkpoint:
            raise gr.Error("Choose a .pt or .ckpt checkpoint first.")
        model = jadegpt.resume_gpt(checkpoint, int(seed), device, trusted_legacy=trusted)
        metadata = getattr(model, "tokenizer_metadata", None)
        if metadata_file:
            supplied = jadegpt.load_metadata(metadata_file, trusted_legacy=trusted)
            if metadata and not _compatible(metadata, supplied):
                raise gr.Error("The supplied tokenizer differs from the tokenizer stored in this checkpoint.")
            metadata = supplied
        if metadata is None:
            raise gr.Error("This older checkpoint needs its original meta.json or trusted meta.pkl file.")
        if metadata["encoding"] == "gpt2":
            valid_vocabulary = model.config.vocab_size in {50257, 50304}
        else:
            valid_vocabulary = int(metadata["vocab_size"]) == model.config.vocab_size
        if not valid_vocabulary:
            raise gr.Error("The tokenizer vocabulary does not match this checkpoint.")
    model.tokenizer_metadata = metadata
    session.model, session.model_meta = model, metadata
    return _model_description(session)


def loss_frame(history):
    rows = []
    for entry in history:
        for key, label in (("train_loss", "Train"), ("val_loss", "Validation")):
            if entry.get(key) is not None:
                rows.append({"step": entry["step"], "loss": float(entry[key]), "split": label})
    return pd.DataFrame(rows, columns=["step", "loss", "split"])


def train_model(session_id, device, dtype, context, batch_size, steps, learning_rate,
                accumulation, eval_interval, eval_iters, decay_lr, model_dir, model_name):
    """Stream metrics while the backend checks cancellation between optimizer steps."""
    session = get_session(session_id)
    if session.model is None or session.dataset is None:
        raise gr.Error("Prepare data and initialize or load a model before training.")
    if not _compatible(session.model_meta, session.dataset["metadata"]):
        raise gr.Error("The dataset and model tokenizers differ. Use the model's original vocabulary or initialize a new model.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", model_name or ""):
        raise gr.Error("Run name must be 1–64 letters, numbers, hyphens or underscores, starting with a letter or number.")
    if int(context) > session.model.config.block_size:
        raise gr.Error(f"Training context must be at most {session.model.config.block_size} for this model.")
    if not str(model_dir).strip():
        raise gr.Error("Choose a checkpoint folder in Settings.")
    session.stop.clear()
    session.busy = True
    messages = queue.Queue()
    history = []
    run_dir = Path(model_dir).expanduser() / f"{model_name}-{uuid.uuid4().hex[:8]}"

    def worker():
        try:
            folder = session.dataset["directory"]
            result = jadegpt.train_gpt(
                session.model, dtype, device,
                jadegpt.load_data_file_to_memmap(folder, "train.bin"),
                jadegpt.load_data_file_to_memmap(folder, "val.bin"),
                int(context), int(batch_size), int(steps), 0.1, float(learning_rate), 0.9, 0.95,
                min(10, int(steps) // 10), int(steps), float(learning_rate) / 10,
                bool(decay_lr), int(eval_interval), int(eval_iters), int(accumulation), 1.0,
                max(1, int(steps) // 100), True, max(1, int(steps)), run_dir, model_name,
                callback=lambda metrics: messages.put(("metrics", metrics)), should_stop=session.stop.is_set,
            )
            (run_dir / "meta.json").write_text(json.dumps(session.model_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            messages.put(("done", result))
        except Exception as exc:
            messages.put(("error", exc))
        finally:
            session.busy = False

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield "**Training started.** The first evaluation may take a moment.", loss_frame([]), None
        while True:
            kind, payload = messages.get()
            if kind == "error":
                raise gr.Error(str(payload)) from payload
            if kind == "done":
                state = "Stopped and saved" if payload["cancelled"] else "Training complete"
                validation = payload["best_val_loss"]
                validation_label = f"{validation:.4f}" if validation is not None else "not evaluated yet"
                yield (f"**{state}** after {payload['iterations']:,} optimizer steps. "
                       f"Best validation loss: **{validation_label}**. "
                       "Try the model in Generate; checkpoint and tokenizer are saved together.",
                       loss_frame(payload["history"]), payload["checkpoint_path"])
                break
            history.append(payload)
            yield (f"**Step {payload['step']:,} / {int(steps):,}** · "
                   f"training loss {payload['train_loss']:.4f} · "
                   f"{payload.get('tokens_per_second', 0):,.0f} tokens/s",
                   loss_frame(history), None)
    finally:
        session.stop.set()
        thread.join()


def stop_training(session_id):
    session = get_session(session_id)
    session.stop.set()
    return "Stop requested. Saving completed updates; an active evaluation may finish first." if session.busy else "No training run is active."


def generate(session_id, prompt, samples, tokens, temperature, top_k, seed, device, dtype):
    session = get_session(session_id)
    if session.model is None or session.model_meta is None:
        raise gr.Error("Initialize or load a model in Train first.")
    torch.manual_seed(int(seed))
    directory = Path.cwd() / "artifacts" / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(session.model_meta, ensure_ascii=False), encoding="utf-8")
    return jadegpt.generate_text(session.model, prompt, session.model_meta["encoding"] == "gpt2",
                                 directory, "meta.json", int(samples), int(tokens), float(temperature),
                                 int(top_k), device, dtype)


def build_app():
    settings = load_settings()
    with gr.Blocks(title="jadeGPT · A small language model lab", analytics_enabled=False) as app:
        session_id = gr.State(value=lambda: uuid.uuid4().hex, delete_callback=drop_session)
        gr.Markdown("# jadeGPT\n### Understand a language model by building one.\n"
                    "Turn a text file into tokens, train a small transformer, and watch it learn to continue your words. "
                    "An interactive companion to Andrej Karpathy’s nanoGPT.", elem_id="hero")
        gr.Markdown(runtime_description())
        with gr.Tabs():
            with gr.Tab("Start here"):
                gr.Markdown("## Your first experiment\n"
                            "**1. Data** — Load the practice text and prepare a character vocabulary.\n\n"
                            "**2. Train** — Keep the Tiny preset, initialize a model, then train for 100 steps.\n\n"
                            "**3. Generate** — Enter a short prompt and compare samples at different temperatures.\n\n"
                            "A short run teaches the mechanics. Expect fragments and invented words; useful text requires "
                            "more varied data, more training, and a larger model.", elem_classes="guide")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### What the model learns\n"
                                    "A GPT predicts the next token from the tokens before it. During training, the target "
                                    "is the same sequence shifted by one position. Causal attention keeps future tokens hidden.\n\n"
                                    "`input: jade` → `target: ade!`\n\n"
                                    "Characters make this process easy to inspect. GPT-2 uses byte-pair encoding (BPE), "
                                    "which combines common byte sequences into tokens.")
                    with gr.Column():
                        gr.Markdown("### Read the learning curve\n"
                                    "**Training loss** measures next-token errors on the examples used to update weights. "
                                    "**Validation loss** uses a held-out final portion of the text. Lower is better.\n\n"
                                    "If training loss falls while validation loss rises, the model may be memorizing. "
                                    "Try more data, fewer steps, or more dropout. Repeated practice text makes validation "
                                    "optimistic; use a separate, diverse corpus for meaningful experiments.")
                with gr.Accordion("Experiments to try", open=False):
                    gr.Markdown("- Compare 100 and 500 steps with the same seed and prompt.\n"
                                "- Compare context lengths of 32 and 64. What patterns fit in each window?\n"
                                "- Set temperature to 0 for greedy decoding, then try 0.7 and 1.2.\n"
                                "- Change one setting at a time and record validation loss as well as samples.\n\n"
                                "For a code walkthrough, open `train-gpt.ipynb`, then `sample-gpt.ipynb`. "
                                "Continue with `finetune-gpt.ipynb` and `sample-gpt2.ipynb`.")
                gr.Markdown("Built on [nanoGPT](https://github.com/karpathy/nanoGPT). "
                            "Watch [Let’s build GPT, from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY). "
                            "These are text-completion models; the small models here have no instruction or chat training.")
            with gr.Tab("Data"):
                gr.Markdown("## Give your model something to learn\n"
                            "Use UTF-8 plain text. An uploaded file takes priority over the text box. "
                            "The split preserves order; the final portion is held out for validation.")
                with gr.Row():
                    with gr.Column():
                        upload = gr.File(label="Upload a text file", file_types=[".txt"], type="filepath")
                        text = gr.Textbox(label="Or paste text", lines=9, placeholder="Paste a story, poems, or your own writing…")
                        sample = gr.Button("Use practice dataset")
                        tokenizer = gr.Radio(["Characters", "GPT-2 BPE"], value="Characters", label="Tokenizer",
                                             info="Characters for a tiny model. GPT-2 BPE is required for pretrained GPT-2.")
                        reuse_vocabulary = gr.Checkbox(True, label="Reuse the loaded model’s character vocabulary",
                                                       info="Keep enabled for fine-tuning. Disable to build a vocabulary for a new model.")
                        split = gr.Slider(0.5, 0.99, value=0.9, step=0.01, label="Fraction used for training")
                        prepare = gr.Button("Prepare dataset", variant="primary")
                    with gr.Column():
                        data_status = gr.Markdown("Your token counts will appear here.")
                        preview = gr.Textbox(label="Dataset preview (first 2,500 characters)", lines=10, interactive=False)
                        metadata_download = gr.File(label="Tokenizer metadata — keep this with older checkpoints", interactive=False)
            with gr.Tab("Train"):
                gr.Markdown("## Build a model, or continue from learned weights\n"
                            "Fine-tuning uses the loaded weights with a fresh optimizer. Preparing new text does not "
                            "change a model’s vocabulary; character models must retain the same character mapping.")
                model_status = gr.Markdown(_model_description(LabSession()))
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("New model", open=True):
                            preset = gr.Dropdown(list(PRESETS), value=next(iter(PRESETS)), label="Model preset")
                            with gr.Row():
                                layers = gr.Number(2, precision=0, minimum=1, maximum=48, label="Transformer layers")
                                heads = gr.Number(2, precision=0, minimum=1, maximum=32, label="Attention heads")
                                width = gr.Number(64, precision=0, minimum=8, maximum=2048, label="Embedding width")
                            gr.Markdown("Embedding width must be divisible by the number of heads. Larger models need more memory.")
                            dropout = gr.Slider(0, 0.8, value=0.1, step=0.05, label="Dropout", info="Regularization used during training only.")
                            bias = gr.Checkbox(False, label="Include linear and normalization biases")
                            seed = gr.Number(1337, precision=0, minimum=0, label="Initialization seed")
                            initialize = gr.Button("Initialize new model")
                        with gr.Accordion("Load checkpoint or pretrained GPT-2", open=False):
                            source = gr.Radio(["Checkpoint", "Pretrained GPT-2"], value="Checkpoint", label="Model source")
                            checkpoint = gr.File(label="Checkpoint", file_types=[".pt", ".ckpt", ".pth"], type="filepath")
                            metadata_file = gr.File(label="Original tokenizer (only needed for older checkpoints)",
                                                    file_types=[".json", ".pkl"], type="filepath")
                            pretrained = gr.Dropdown(["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], value="gpt2", label="Pretrained variant")
                            gr.Markdown("GPT-2 downloads weights on first use. Even the smallest variant has 124M parameters; "
                                        "start with it and a small batch. The larger variants need substantially more RAM/VRAM.")
                            trusted = gr.Checkbox(False, label="I trust this legacy checkpoint / pickle metadata",
                                                   info="Only needed for old files you created or trust. Legacy pickle loading can execute code.")
                            load = gr.Button("Load model")
                    with gr.Column():
                        gr.Markdown("### Training controls")
                        context = gr.Slider(8, 1024, value=64, step=8, label="Context length (tokens)",
                                             info="Used to initialize a new model and to sample training windows.")
                        with gr.Row():
                            batch = gr.Number(4, precision=0, minimum=1, maximum=256, label="Batch size")
                            steps = gr.Number(100, precision=0, minimum=1, maximum=100000, label="Optimizer steps")
                        learning_rate = gr.Number(0.001, minimum=0.000001, maximum=0.1, label="Learning rate",
                                                  info="Try 0.001 from scratch; start near 0.00003 when fine-tuning GPT-2.")
                        with gr.Accordion("Evaluation and optimization", open=False):
                            accumulation = gr.Number(1, precision=0, minimum=1, maximum=128, label="Gradient accumulation",
                                                      info="Effective batch = batch size × accumulation. Loss is averaged over these micro-batches.")
                            eval_interval = gr.Number(20, precision=0, minimum=1, label="Evaluate every N steps")
                            eval_iters = gr.Number(5, precision=0, minimum=1, label="Batches per evaluation split")
                            decay_lr = gr.Checkbox(True, label="Warm up, then decay the learning rate")
                        model_name = gr.Textbox("experiment", label="Run name")
                        with gr.Row():
                            train = gr.Button("Train / fine-tune", variant="primary")
                            stop = gr.Button("Stop and save", variant="stop")
                        stop_status = gr.Markdown()
                training_status = gr.Markdown("The final checkpoint is saved automatically in a new run folder.")
                chart = gr.LinePlot(x="step", y="loss", color="split", title="Next-token prediction loss",
                                    x_title="Optimizer step", y_title="Cross entropy", height=300)
                saved_checkpoint = gr.File(label="Download checkpoint (includes tokenizer)", interactive=False)
            with gr.Tab("Generate"):
                gr.Markdown("## What comes next?\n"
                            "Generation uses the model loaded in this browser session. The prompt must use known "
                            "characters for a character model; GPT-2 can encode arbitrary text.")
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox("the ", lines=5, label="Prompt")
                        tokens = gr.Slider(1, 1024, value=120, step=1, label="New tokens per sample",
                                            info="A token is one character for character models, or a BPE piece for GPT-2.")
                        temperature = gr.Slider(0, 2, value=0.8, step=0.05, label="Temperature",
                                                 info="0 = greedy. Lower values are more predictable; higher values add variety.")
                        top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k", info="Sample from the k most likely tokens. 0 keeps all tokens.")
                        with gr.Row():
                            samples = gr.Number(1, precision=0, minimum=1, maximum=5, label="Samples")
                            sampling_seed = gr.Number(1337, precision=0, minimum=0, label="Sampling seed")
                        generate_button = gr.Button("Generate text", variant="primary")
                    with gr.Column():
                        output = gr.Textbox(label="Generated continuation", lines=20, interactive=False, buttons=["copy"])
            with gr.Tab("Settings"):
                gr.Markdown("## Local workspace\nFiles stay on the machine running this app. "
                            "Prepared data and checkpoints have separate run folders. Auto chooses CUDA, Apple MPS, or CPU.")
                with gr.Row():
                    data_dir = gr.Textbox(settings["data_dir"], label="Data folder")
                    model_dir = gr.Textbox(settings["model_dir"], label="Checkpoint folder")
                with gr.Row():
                    device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value=settings["device"], label="Device")
                    dtype = gr.Dropdown(["auto", "float32", "bfloat16", "float16"], value=settings["dtype"], label="Precision",
                                         info="Auto uses full precision on CPU/MPS and supported mixed precision on CUDA.")
                save = gr.Button("Save defaults")
                settings_status = gr.Markdown()

        compute = dict(concurrency_id="compute", concurrency_limit=1, api_visibility="private")
        sample.click(lambda: (None, sample_text()), outputs=[upload, text], api_visibility="private")
        prepare.click(prepare_data, [session_id, upload, text, split, tokenizer, data_dir, reuse_vocabulary],
                      [data_status, preview, metadata_download], **compute)
        preset.change(lambda name: PRESETS[name], preset, [layers, heads, width, context, batch, steps, learning_rate], api_visibility="private")
        initialize.click(initialize_model, [session_id, seed, layers, heads, width, context, dropout, bias], model_status, **compute)
        load.click(load_model, [session_id, source, checkpoint, metadata_file, pretrained, seed, device, trusted], model_status, **compute)
        train.click(train_model, [session_id, device, dtype, context, batch, steps, learning_rate, accumulation,
                                 eval_interval, eval_iters, decay_lr, model_dir, model_name],
                    [training_status, chart, saved_checkpoint], **compute)
        stop.click(stop_training, session_id, stop_status, queue=False, api_visibility="private")
        generate_button.click(generate, [session_id, prompt, samples, tokens, temperature, top_k, sampling_seed, device, dtype], output, **compute)
        save.click(save_settings, [data_dir, model_dir, device, dtype], settings_status, api_visibility="private")
    return app.queue(default_concurrency_limit=1, max_size=16)


def main():
    parser = argparse.ArgumentParser(description="Run the local jadeGPT learning lab.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost)")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=False,
                       theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="teal"), css=CSS,
                       show_error=True)


if __name__ == "__main__":
    main()
