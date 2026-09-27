"""Fast CPU regression tests for the shared training and sampling path."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

import jadegpt
from model import GPT, GPTConfig


@pytest.fixture(autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    return jadegpt.init_gpt(n_layer=1, n_head=2, n_embd=16, block_size=4, vocab_size=4)


@pytest.fixture
def train_args(tmp_path):
    return dict(dtype="auto", device="cpu", train_data=np.arange(64, dtype=np.uint16) % 4,
                val_data=np.arange(32, dtype=np.uint16) % 4, block_size=4, batch_size=2,
                max_iters=3, weight_decay=0.01, learning_rate=1e-3, beta1=0.9, beta2=0.95,
                warmup_iters=0, lr_decay_iters=3, min_lr=1e-4, decay_lr=False,
                eval_interval=2, eval_iters=1, gradient_accumulation_steps=2, grad_clip=1.0,
                log_interval=1, only_save_on_finish=True, save_interval=2,
                model_dir=tmp_path, model_name="tiny")


def test_character_dataset_json_and_portable_paths(tmp_path):
    data = "a b\né🙂" * 30
    train, val = jadegpt.split_dataset(data, 0.8)
    summary = jadegpt.export_data_to_files(data, train, val, False, tmp_path,
                                         "train.bin", "val.bin", "meta.json")
    assert Path(summary["train_path"]) == tmp_path / "train.bin"
    assert summary["train_tokens"] == len(train)
    meta = jadegpt.load_metadata(summary["meta_path"])
    ids = jadegpt.load_data_file_to_memmap(tmp_path, "train.bin")
    assert "".join(meta["itos"][int(index)] for index in ids) == train
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["encoding"] == "custom"


def test_finetune_preserves_vocabulary_when_corpus_omits_characters(tmp_path):
    summary = jadegpt.export_data_to_files("abcd", "aba" * 10, "baa" * 4, False,
                                         tmp_path, "train.bin", "val.bin", "meta.json")
    assert summary["stoi"] == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_finetune_preserves_explicit_non_alphabetical_token_ids(tmp_path):
    original = {"encoding": "custom", "vocab_size": 4,
                "stoi": {"d": 0, "b": 1, "c": 2, "a": 3}}
    summary = jadegpt.export_data_to_files("aba" * 10, "aba" * 8, "baa" * 2, False,
                                         tmp_path, "train.bin", "val.bin", "meta.json",
                                         tokenizer_metadata=original)
    assert summary["stoi"] == original["stoi"]
    assert list(jadegpt.load_data_file_to_memmap(tmp_path, "train.bin")[:3]) == [3, 1, 3]
    assert jadegpt.load_metadata(summary["meta_path"])["itos"][0] == "d"


def test_legacy_metadata_requires_explicit_trust(tmp_path):
    path = tmp_path / "meta.pkl"
    path.write_bytes(pickle.dumps({"encoding": "custom", "vocab_size": 2,
                                  "stoi": {"a": 0, "b": 1}, "itos": {0: "a", 1: "b"}}))
    with pytest.raises(ValueError, match="trusted_legacy"):
        jadegpt.load_metadata(path)
    assert jadegpt.load_metadata(path, trusted_legacy=True)["stoi"]["b"] == 1


def test_training_exact_updates_and_safe_checkpoint(model, train_args):
    before = model.transformer.wte.weight.detach().clone()
    rows = []
    result = jadegpt.train_gpt(model, **train_args, callback=rows.append)
    assert result["iterations"] == 3
    assert not result["cancelled"]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert result["best_val_loss"] is not None
    assert not torch.equal(before, model.transformer.wte.weight)
    checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
    assert checkpoint["iter_num"] == 3
    assert isinstance(checkpoint["model_args"], dict)
    optimizer_steps = [value["step"].item() for value in checkpoint["optimizer"]["state"].values()]
    assert set(optimizer_steps) == {3}
    restored = jadegpt.resume_gpt(result["checkpoint_path"], device="cpu")
    assert not restored.training
    assert torch.equal(model.transformer.wte.weight, restored.transformer.wte.weight)


def test_accumulation_averages_loss_without_hidden_multiplier(model, train_args, monkeypatch):
    # With identical microbatches and SGD, one update must be identical for 1 vs 3
    # accumulation steps. Adam can hide a gradient scaling bug on its first step.
    other = copy.deepcopy(model)
    x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    y = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 1]])
    calls = 0

    def fixed_batch(*args):
        nonlocal calls
        if torch.is_grad_enabled():
            calls += 1
        return x, y

    monkeypatch.setattr(jadegpt, "get_batch", fixed_batch)
    for current in (model, other):
        monkeypatch.setattr(current, "configure_optimizers",
                            lambda *args, current=current: torch.optim.SGD(current.parameters(), lr=0.1))
    args = {**train_args, "max_iters": 1, "learning_rate": 0.1, "grad_clip": 0.0}
    jadegpt.train_gpt(model, **{**args, "gradient_accumulation_steps": 1})
    assert calls == 1
    calls = 0
    jadegpt.train_gpt(other, **{**args, "gradient_accumulation_steps": 3})
    assert calls == 3
    for actual, expected in zip(other.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_cancellation_saves_only_completed_updates(model, train_args):
    rows = []
    result = jadegpt.train_gpt(model, **train_args, callback=rows.append,
                               should_stop=lambda: bool(rows))
    assert result["cancelled"]
    assert result["iterations"] == 1
    assert Path(result["checkpoint_path"]).name == "tiny-1.ckpt"


def test_cancel_during_accumulation_discards_partial_update(model, train_args):
    before = {key: value.clone() for key, value in model.state_dict().items()}
    checks = iter([False, False, True])
    result = jadegpt.train_gpt(model, **train_args, should_stop=lambda: next(checks))
    assert result["cancelled"] and result["iterations"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_dropout_disabled_in_eval_and_crop_with_sdpa():
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=4, dropout=0.7)).eval()
    x = torch.tensor([[0, 1, 2, 3]])
    first, _ = model(x)
    second, _ = model(x)
    assert torch.equal(first, second)
    model.crop_block_size(4)
    assert model.transformer.wpe.weight.shape[0] == 4
    assert model(x)[0].shape == (1, 1, 4)


def test_attention_is_causal(model):
    model.eval()
    x = torch.tensor([[0, 1, 2, 3]])
    changed = torch.tensor([[0, 1, 0, 0]])
    first, _ = model(x, x)
    second, _ = model(changed, changed)
    torch.testing.assert_close(first[:, :2], second[:, :2])


def test_estimate_loss_restores_eval_mode(model, train_args):
    model.eval()
    result = jadegpt.estimate_loss(model, 1, nullcontext(), train_args["train_data"],
                                  train_args["val_data"], "cpu", 4, 2)
    assert not model.training
    assert result["train"] > 0 and result["val"] > 0


def test_sampling_excludes_padded_tokens(model, monkeypatch):
    def predict_padding(idx):
        logits = torch.zeros((idx.shape[0], 1, 4))
        logits[:, :, 3] = 1e6
        return logits, None

    monkeypatch.setattr(model, "forward", predict_padding)
    result = model.generate(torch.tensor([[0]]), 20, top_k=1, valid_vocab_size=3)
    assert result.max() < 3
    model.tokenizer_metadata = {"actual_vocab_size": 3}
    assert model.generate(torch.tensor([[0]]), 20, top_k=1).max() < 3


def test_zero_temperature_is_greedy_and_zero_top_k_keeps_all(model, monkeypatch):
    monkeypatch.setattr(model, "forward", lambda idx: (torch.tensor([[[1.0, 2.0, 9.0, 4.0]]]), None))
    result = model.generate(torch.tensor([[0]]), 4, temperature=0, top_k=0)
    assert result.tolist() == [[0, 2, 2, 2, 2]]
    # The stochastic path also accepts zero as the no-filter setting.
    assert model.generate(torch.tensor([[0]]), 2, temperature=1, top_k=0).shape == (1, 3)


def test_generation_preserves_mode_and_tokenizer_checkpoint(model, tmp_path):
    summary = jadegpt.export_data_to_files("abcd" * 10, "abcd" * 8, "abcd" * 2,
                                         False, tmp_path, "train.bin", "val.bin", "meta.json")
    model.tokenizer_metadata = jadegpt.load_metadata(summary["meta_path"])
    path = jadegpt.save_checkpoint(model, None, 0, None, tmp_path)
    restored = jadegpt.resume_gpt(path, device="cpu")
    assert restored.tokenizer_metadata["stoi"] == model.tokenizer_metadata["stoi"]
    model.train()
    text = jadegpt.generate_text(model, "ab", False, tmp_path, "meta.json", 1, 3, 1.0, 2, "cpu", "auto")
    assert text.startswith("ab") and len(text) == 5
    assert model.training
    with pytest.raises(ValueError, match="absent"):
        jadegpt.generate_text(model, "z", False, tmp_path, "meta.json", 1, 1, 1.0, 2, "cpu", "auto")


def test_generation_rejects_same_sized_different_tokenizer(model, tmp_path):
    model.tokenizer_metadata = {"encoding": "custom", "stoi": {"a": 0, "b": 1, "c": 2, "d": 3}}
    jadegpt.export_data_to_files("bcde", "bcde", "bcde", False, tmp_path,
                                "train.bin", "val.bin", "meta.json")
    with pytest.raises(ValueError, match="mapping differs"):
        jadegpt.generate_text(model, "b", False, tmp_path, "meta.json", 1, 1, 1.0, 2, "cpu", "auto")


def test_legacy_checkpoint_requires_trust(model, tmp_path):
    path = tmp_path / "legacy.ckpt"
    torch.save({"model_args": model.config, "model": model.state_dict()}, path)
    with pytest.raises(ValueError, match="trusted_legacy"):
        jadegpt.resume_gpt(path, device="cpu")
    assert jadegpt.resume_gpt(path, device="cpu", trusted_legacy=True).config == model.config


@pytest.mark.parametrize("key,value", [("max_iters", 0), ("gradient_accumulation_steps", 0),
                                       ("learning_rate", -1), ("beta1", 1), ("block_size", 5)])
def test_training_validates_inputs(model, train_args, key, value):
    with pytest.raises(ValueError):
        jadegpt.train_gpt(model, **{**train_args, key: value})


def test_short_dataset_and_invalid_generation(model):
    with pytest.raises(ValueError, match="block_size"):
        jadegpt.get_batch(np.arange(4), "cpu", 4, 1)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.tensor([[0]]), 1, temperature=-1)
    with pytest.raises(ValueError, match="top_k"):
        model.generate(torch.tensor([[0]]), 1, top_k=-1)
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(n_embd=15, n_head=2)


def test_device_dtype_and_scheduler_edges():
    assert jadegpt.resolve_device("cpu") == "cpu"
    assert jadegpt.resolve_dtype("auto", "cpu") == "float32"
    with pytest.raises(ValueError, match="float32"):
        jadegpt.resolve_dtype("float16", "cpu")
    assert jadegpt.get_lr(0, 0, 1e-3, 0, 1e-4) == 1e-4
    assert jadegpt.get_lr(10, 10, 1e-3, 10, 1e-4) == 1e-4


def test_pretrained_conversion_matches_tiny_transformers_gpt2(monkeypatch):
    """Exercise real Transformers weights offline, including Conv1D transposes."""
    transformers = pytest.importorskip("transformers")
    import model as model_module

    config = GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=8, vocab_size=31, bias=True)
    hf_config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=8,
                                        vocab_size=31, resid_pdrop=0.0, embd_pdrop=0.0,
                                        attn_pdrop=0.0)
    reference = transformers.GPT2LMHeadModel(hf_config).eval()
    # Avoid fetching large public weights; retain the real HF model/state_dict.
    monkeypatch.setattr(transformers.GPT2LMHeadModel, "from_pretrained", lambda *args, **kwargs: reference)
    monkeypatch.setattr(model_module, "GPTConfig", lambda **kwargs: config)
    converted = GPT.from_pretrained("gpt2").eval()
    tokens = torch.tensor([[0, 1, 4, 8], [3, 2, 7, 9]])
    with torch.no_grad():
        actual, _ = converted(tokens, tokens)
        expected = reference(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
