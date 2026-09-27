"""Small, readable GPT training utilities shared by the app and notebooks.

Binary token datasets use uint16, tokenizer metadata uses JSON, and checkpoints
store a plain configuration dictionary so modern PyTorch can load them safely.
"""

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig


def resolve_device(device="auto"):
    """Choose CUDA, then Apple MPS, then CPU; reject unavailable explicit devices."""
    if str(device).lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    result = torch.device(device)
    if result.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, cuda:N, or mps.")
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Choose auto or cpu.")
        if result.index is not None and result.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {result.index} is unavailable.")
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable. Choose auto or cpu.")
    return str(result)


def resolve_dtype(dtype="auto", device="auto"):
    """Use CUDA mixed precision where supported and float32 on CPU/MPS."""
    device = resolve_device(device)
    device_type = torch.device(device).type
    if dtype == "auto":
        if device_type == "cuda":
            with torch.cuda.device(device):
                return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be auto, float32, bfloat16, or float16.")
    if device_type != "cuda" and dtype != "float32":
        raise ValueError("Use float32 on CPU/MPS, or auto to choose a supported dtype.")
    if device_type == "cuda" and dtype == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support bfloat16; use auto or float16.")
    return dtype


def _precision_context(device, dtype):
    if dtype == "float32":
        return nullcontext()
    return torch.amp.autocast(torch.device(device).type, dtype=getattr(torch, dtype))


def _positive_int(value, name, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _gpt2_encoding():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def open_dataset_file(input_file_path):
    return Path(input_file_path).read_text(encoding="utf-8-sig")


def split_dataset(data, split):
    if not data:
        raise ValueError("The dataset is empty.")
    if not 0 < split < 1:
        raise ValueError("The training fraction must be strictly between 0 and 1.")
    boundary = int(len(data) * split)
    if boundary == 0 or boundary == len(data):
        raise ValueError("The dataset must contain both training and validation text.")
    return data[:boundary], data[boundary:]


def get_vocab_size(data, use_gpt2_encoding):
    if not data:
        raise ValueError("The dataset is empty.")
    return 50304 if use_gpt2_encoding else len(set(data))


def export_data_to_files(data, train_data, val_data, use_gpt2_encoding, data_dir,
                         train_file_name, val_file_name, meta_file_name, *, tokenizer_metadata=None):
    """Write tokens and JSON metadata; return sizes and paths for the UI.

    GPT-2 has 50,257 real tokens; its model embedding table is padded to 50,304.
    A legacy .pkl filename is accepted, but all new metadata is written as JSON.
    Pass tokenizer_metadata when fine-tuning to retain the model's exact token IDs.
    """
    if not data or not train_data or not val_data:
        raise ValueError("Full text, training text, and validation text must be nonempty.")
    if use_gpt2_encoding:
        encoder = _gpt2_encoding()
        encode = lambda value: encoder.encode(value, allowed_special={"<|endoftext|>"})
        metadata = {"encoding": "gpt2", "vocab_size": 50304,
                    "actual_vocab_size": encoder.n_vocab, "stoi": {}, "itos": {}}
    else:
        if tokenizer_metadata is not None:
            preserved = _validate_metadata(dict(tokenizer_metadata))
            if preserved["encoding"] != "custom":
                raise ValueError("Character datasets require character tokenizer metadata.")
            stoi = dict(preserved["stoi"])
            characters = sorted(stoi, key=stoi.get)
        else:
            characters = sorted(set(data))
            stoi = {character: index for index, character in enumerate(characters)}
        if len(characters) > np.iinfo(np.uint16).max + 1:
            raise ValueError("The character vocabulary exceeds uint16 capacity (65,536 tokens).")
        def encode(value):
            unknown = set(value) - stoi.keys()
            if unknown:
                raise ValueError("Text contains characters absent from the selected vocabulary: "
                                 f"{sorted(unknown)[:12]!r}. Use the original vocabulary or initialize a new model.")
            return [stoi[character] for character in value]
        metadata = {"encoding": "custom", "vocab_size": len(characters),
                    "actual_vocab_size": len(characters), "stoi": stoi,
                    "itos": {str(index): character for index, character in enumerate(characters)}}
    train_ids, val_ids = encode(train_data), encode(val_data)
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    train_path, val_path, meta_path = (directory / name for name in
                                      (train_file_name, val_file_name, meta_file_name))
    if len({path.resolve() for path in (train_path, val_path, meta_path)}) != 3:
        raise ValueError("Training, validation, and metadata filenames must be different.")
    metadata.update(format_version=1, dtype="uint16", train_tokens=len(train_ids),
                    val_tokens=len(val_ids))
    np.asarray(train_ids, dtype=np.uint16).tofile(train_path)
    np.asarray(val_ids, dtype=np.uint16).tofile(val_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**metadata, "train_path": str(train_path), "val_path": str(val_path),
            "meta_path": str(meta_path)}


def load_metadata(path, *, trusted_legacy=False):
    """Read tokenizer JSON. Pickled legacy metadata requires explicit trust."""
    path = Path(path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        if not trusted_legacy:
            raise ValueError("This is not JSON metadata. Convert trusted legacy pickle files, "
                             "or explicitly set trusted_legacy=True for a file you trust.") from error
        with path.open("rb") as handle:
            metadata = pickle.load(handle)
    return _validate_metadata(metadata)


def _validate_metadata(metadata):
    """Validate metadata and normalize JSON's stringified integer keys."""
    if not isinstance(metadata, dict) or metadata.get("encoding") not in {"custom", "gpt2"}:
        raise ValueError("Invalid tokenizer metadata.")
    if metadata["encoding"] == "custom":
        stoi = metadata.get("stoi")
        if not isinstance(stoi, dict) or not stoi:
            raise ValueError("Character metadata must contain a nonempty stoi mapping.")
        if any(not isinstance(key, str) or len(key) != 1 or type(value) is not int
               for key, value in stoi.items()) or set(stoi.values()) != set(range(len(stoi))):
            raise ValueError("Character token IDs must be unique contiguous integers starting at zero.")
        metadata["itos"] = {index: character for character, index in stoi.items()}
        if metadata.get("vocab_size") != len(stoi):
            raise ValueError("Metadata vocabulary size does not match its character mapping.")
        metadata["actual_vocab_size"] = len(stoi)
    else:
        metadata["actual_vocab_size"] = 50257
    return metadata


def load_data_file_to_memmap(data_dir, data_file_name):
    path = Path(data_dir) / data_file_name
    if path.stat().st_size == 0 or path.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError(f"{path.name} must contain nonempty uint16 token data.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def init_gpt(random_seed=1337, n_layer=6, n_head=6, n_embd=384, dropout=0.0,
             bias=False, block_size=32, vocab_size=50304):
    torch.manual_seed(random_seed)
    return GPT(GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                         block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout))


def init_gpt2(gpt2_model="gpt2", random_seed=1337):
    """Download GPT-2 weights from Hugging Face on first use (optional dependency)."""
    torch.manual_seed(random_seed)
    model = GPT.from_pretrained(gpt2_model, {"dropout": 0.0}).eval()
    model.tokenizer_metadata = {"encoding": "gpt2", "vocab_size": 50257,
                                "actual_vocab_size": 50257, "stoi": {}, "itos": {}}
    return model


def resume_gpt(model_file_path, random_seed=1337, device="auto", *, trusted_legacy=False):
    """Load model weights for inference/fine-tuning, not optimizer continuation.

    New checkpoints load with weights_only=True. Old checkpoints containing a
    pickled GPTConfig require trusted_legacy=True; only use this for trusted files.
    """
    torch.manual_seed(random_seed)
    device = resolve_device(device)
    try:
        checkpoint = torch.load(model_file_path, map_location="cpu", weights_only=not trusted_legacy)
    except pickle.UnpicklingError as error:
        raise ValueError("Checkpoint could not be loaded safely. For a trusted old checkpoint "
                         "containing GPTConfig, pass trusted_legacy=True, then save it again.") from error
    config = checkpoint["model_args"]
    if isinstance(config, dict):
        config = GPTConfig(**config)
    elif not trusted_legacy or not isinstance(config, GPTConfig):
        raise ValueError("Checkpoint model_args must be a plain configuration dictionary.")
    model = GPT(config)
    state_dict = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()}
    # Old PyTorch checkpoints may contain a pre-SDPA causal attention buffer.
    state_dict = {key: value for key, value in state_dict.items()
                  if not key.endswith((".attn.bias", ".attn.masked_bias"))}
    model.load_state_dict(state_dict)
    if checkpoint.get("tokenizer_metadata") is not None:
        model.tokenizer_metadata = checkpoint["tokenizer_metadata"]
    return model.to(device).eval()


def get_batch(split, device, block_size, batch_size):
    _positive_int(block_size, "block_size")
    _positive_int(batch_size, "batch_size")
    if len(split) <= block_size:
        raise ValueError(f"Each split needs at least block_size + 1 ({block_size + 1}) tokens; "
                         f"received {len(split)}. Add text or lower the context length.")
    device = resolve_device(device)
    indices = torch.randint(len(split) - block_size, (batch_size,)).tolist()
    x = torch.from_numpy(np.stack([np.asarray(split[i:i + block_size], dtype=np.int64) for i in indices]))
    y = torch.from_numpy(np.stack([np.asarray(split[i + 1:i + 1 + block_size], dtype=np.int64) for i in indices]))
    if torch.device(device).type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters, ctx, train_data, val_data, device, block_size, batch_size):
    """Estimate mean token cross-entropy and restore the caller's train/eval mode."""
    _positive_int(eval_iters, "eval_iters")
    was_training = model.training
    model.eval()
    result = {}
    try:
        for name, data in (("train", train_data), ("val", val_data)):
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(data, device, block_size, batch_size)
                with ctx:
                    _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return result


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """Cosine decay with warmup, including the zero/equal-length edge cases."""
    if warmup_iters > 0 and it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it >= lr_decay_iters:
        return min_lr
    if lr_decay_iters <= warmup_iters:
        return learning_rate
    decay_ratio = max(0.0, (it - warmup_iters) / (lr_decay_iters - warmup_iters))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (learning_rate - min_lr)


def train_gpt(model, dtype, device, train_data, val_data, block_size, batch_size,
              max_iters, weight_decay, learning_rate, beta1, beta2, warmup_iters,
              lr_decay_iters, min_lr, decay_lr, eval_interval, eval_iters,
              gradient_accumulation_steps, grad_clip, log_interval, only_save_on_finish,
              save_interval, model_dir, model_name, *, callback=None, should_stop=None):
    """Run exactly max_iters optimizer updates and save a portable checkpoint.

    Effective tokens/update = batch_size * block_size * gradient_accumulation_steps.
    Loss is averaged across the explicitly requested microbatches. callback receives
    a metrics dictionary at log/evaluation steps. should_stop is checked before each
    update and microbatch; an interrupted partial update is discarded and completed
    updates are saved. Loading model weights starts a fresh optimizer and schedule.
    """
    for name, value in (("block_size", block_size), ("batch_size", batch_size),
                        ("max_iters", max_iters), ("eval_interval", eval_interval),
                        ("eval_iters", eval_iters), ("gradient_accumulation_steps", gradient_accumulation_steps),
                        ("log_interval", log_interval), ("save_interval", save_interval)):
        _positive_int(value, name)
    _positive_int(warmup_iters, "warmup_iters", allow_zero=True)
    _positive_int(lr_decay_iters, "lr_decay_iters", allow_zero=True)
    if block_size > model.config.block_size:
        raise ValueError("Training context exceeds the model's context length.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if not math.isfinite(min_lr) or not 0 <= min_lr <= learning_rate:
        raise ValueError("min_lr must be between zero and learning_rate.")
    if not all(math.isfinite(value) and value >= 0 for value in (weight_decay, grad_clip)):
        raise ValueError("weight_decay and grad_clip must be finite and nonnegative.")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1).")
    if decay_lr and lr_decay_iters < warmup_iters:
        raise ValueError("lr_decay_iters must be >= warmup_iters.")
    for name, data in (("Training", train_data), ("Validation", val_data)):
        if len(data) <= block_size:
            raise ValueError(f"{name} split needs at least {block_size + 1} tokens; received {len(data)}.")
        if not np.issubdtype(np.asarray(data).dtype, np.integer):
            raise ValueError(f"{name} data must contain integer token IDs.")
        if int(np.min(data)) < 0 or int(np.max(data)) >= model.config.vocab_size:
            raise ValueError(f"{name} token IDs exceed the model vocabulary; check the tokenizer.")
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    model.to(device).train()
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), torch.device(device).type)
    scaler = torch.amp.GradScaler("cuda", enabled=(torch.device(device).type == "cuda" and dtype == "float16"))
    history = []
    best_val_loss = float("inf")
    started = time.perf_counter()
    iterations = 0
    cancelled = False
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, max_iters + 1):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        step_started = time.perf_counter()
        lr = get_lr(step, warmup_iters, learning_rate, lr_decay_iters, min_lr) if decay_lr else learning_rate
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            if should_stop is not None and should_stop():
                cancelled = True
                break
            x, y = get_batch(train_data, device, block_size, batch_size)
            with _precision_context(device, dtype):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss became nonfinite. Lower the learning rate or use float32.")
            total_loss += loss.item()
            scaler.scale(loss / gradient_accumulation_steps).backward()
        if cancelled:
            optimizer.zero_grad(set_to_none=True)
            break
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iterations = step
        step_seconds = time.perf_counter() - step_started
        should_evaluate = step % eval_interval == 0 or step == max_iters
        if step % log_interval == 0 or should_evaluate or step == 1:
            metrics = {"step": step, "train_loss": total_loss / gradient_accumulation_steps,
                       "val_loss": None, "learning_rate": lr,
                       "tokens_per_second": batch_size * block_size * gradient_accumulation_steps / max(step_seconds, 1e-9),
                       "elapsed_seconds": time.perf_counter() - started}
            if should_evaluate:
                losses = estimate_loss(model, eval_iters, _precision_context(device, dtype),
                                       train_data, val_data, device, block_size, batch_size)
                metrics.update(train_loss=losses["train"], val_loss=losses["val"])
                best_val_loss = min(best_val_loss, losses["val"])
            history.append(metrics)
            if callback is not None:
                callback(dict(metrics))
        if not only_save_on_finish and step % save_interval == 0 and step != max_iters:
            save_checkpoint(model, optimizer, step, best_val_loss, model_dir, f"{model_name}-{step}.ckpt")
    checkpoint_path = save_checkpoint(model, optimizer, iterations, best_val_loss,
                                      model_dir, f"{model_name}-{iterations}.ckpt")
    return {"history": history, "iterations": iterations,
            "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
            "checkpoint_path": checkpoint_path, "cancelled": cancelled,
            "device": device, "dtype": dtype}


def save_checkpoint(model, optimizer, iter_num, best_val_loss, model_dir, model_name="ckpt.pt"):
    """Save tensors plus primitive metadata, compatible with weights_only=True."""
    checkpoint = {"format_version": 2, "model": model.state_dict(),
                  "optimizer": optimizer.state_dict() if optimizer is not None else None,
                  "model_args": asdict(model.config), "iter_num": int(iter_num),
                  "best_val_loss": float(best_val_loss) if best_val_loss is not None else None}
    if getattr(model, "tokenizer_metadata", None) is not None:
        # JSON round trip ensures portable primitive types, even if the caller
        # attached metadata obtained from a legacy pickle file.
        checkpoint["tokenizer_metadata"] = json.loads(json.dumps(model.tokenizer_metadata))
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / model_name
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
    return str(path)


def generate_text(model, start, use_gpt2_encoding, meta_dir, meta_file_name,
                  num_samples, max_new_tokens, temperature, top_k, device, dtype,
                  *, trusted_legacy=False):
    """Sample in evaluation mode, then restore the caller's previous model mode."""
    _positive_int(num_samples, "num_samples")
    _positive_int(max_new_tokens, "max_new_tokens", allow_zero=True)
    if not start:
        raise ValueError("Enter a nonempty prompt to seed generation.")
    if use_gpt2_encoding:
        encoder = _gpt2_encoding()
        encode = lambda value: encoder.encode(value, allowed_special={"<|endoftext|>"})
        decode = encoder.decode
        actual_vocab_size = encoder.n_vocab
        if model.config.vocab_size not in {50257, 50304}:
            raise ValueError("GPT-2 tokenizer requires a model vocabulary of 50,257 or 50,304 tokens.")
    else:
        metadata = load_metadata(Path(meta_dir) / meta_file_name, trusted_legacy=trusted_legacy)
        if metadata["encoding"] != "custom":
            raise ValueError("Choose GPT-2 tokenization for GPT-2 metadata.")
        stoi, itos = metadata["stoi"], metadata["itos"]
        embedded_metadata = getattr(model, "tokenizer_metadata", None)
        if embedded_metadata is not None and embedded_metadata.get("stoi") != stoi:
            raise ValueError("Tokenizer character mapping differs from the mapping saved with this model.")
        unknown = sorted(set(start) - stoi.keys())
        if unknown:
            raise ValueError(f"Prompt contains characters absent from the training vocabulary: {unknown[:12]!r}")
        encode = lambda value: [stoi[character] for character in value]
        decode = lambda ids: "".join(itos[index] for index in ids)
        actual_vocab_size = metadata["actual_vocab_size"]
        if model.config.vocab_size != actual_vocab_size:
            raise ValueError("Tokenizer vocabulary does not match the loaded character model.")
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    x = torch.tensor(encode(start), dtype=torch.long, device=device).unsqueeze(0)
    was_training = model.training
    model.to(device).eval()
    outputs = []
    try:
        with torch.inference_mode(), _precision_context(device, dtype):
            for _ in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                   valid_vocab_size=actual_vocab_size)
                outputs.append(decode(y[0].tolist()))
    finally:
        model.train(was_training)
    return "\n\n---------------\n\n".join(outputs)
