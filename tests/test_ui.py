"""Exercise the learning workflow and Gradio's actual callback serialization."""

import asyncio
import json
from pathlib import Path

import gradio as gr
from gradio.state_holder import SessionState
import pytest
import torch

import jadegpt_ui as ui


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield tmp_path
    for session_id in list(ui._sessions):
        ui.drop_session(session_id)
    torch.set_num_threads(previous)


def ready_model(workspace, session_id="test"):
    ui.prepare_data(session_id, None, ui.sample_text(), 0.9, "Characters", str(workspace / "data"))
    ui.initialize_model(session_id, 1337, 1, 2, 16, 8, 0.1, False)
    return session_id


def training_inputs(workspace, session_id):
    return [session_id, "cpu", "auto", 8, 2, 3, 0.001, 1, 2, 1, False,
            str(workspace / "models"), "test-run"]


def test_complete_workflow_checkpoint_and_session_isolation(workspace):
    session_id = ready_model(workspace)
    results = list(ui.train_model(*training_inputs(workspace, session_id)))
    status, curve, checkpoint = results[-1]
    assert "Training complete" in status
    assert set(curve["split"]) == {"Train", "Validation"}
    assert Path(checkpoint).is_file()
    result = ui.generate(session_id, "the ", 1, 8, 0, 0, 10, "cpu", "auto")
    assert result.startswith("the ") and len(result) == 12
    ui.load_model("second", "Checkpoint", checkpoint, None, "gpt2", 10, "cpu", False)
    assert ui.get_session("second").model is not ui.get_session(session_id).model
    assert ui.generate("second", "the ", 1, 8, 0, 0, 10, "cpu", "auto") == result
    assert ui.get_session("unrelated").model is None


def test_stop_before_evaluation_saves_loadable_checkpoint(workspace):
    session_id = ready_model(workspace)
    stream = ui.train_model(*training_inputs(workspace, session_id))
    next(stream)
    ui.stop_training(session_id)
    status, _, checkpoint = list(stream)[-1]
    assert "Stopped and saved" in status
    assert Path(checkpoint).is_file()
    ui.load_model("stopped", "Checkpoint", checkpoint, None, "gpt2", 1, "cpu", False)
    assert not ui.get_session(session_id).busy


def test_finetuning_keeps_original_character_ids(workspace):
    session_id = ready_model(workspace)
    before = dict(ui.get_session(session_id).model_meta["stoi"])
    ui.prepare_data(session_id, None, "the " * 100, 0.9, "Characters", str(workspace / "data"))
    assert ui.get_session(session_id).dataset["metadata"]["stoi"] == before
    list(ui.train_model(*training_inputs(workspace, session_id)))
    ui.prepare_data(session_id, None, "the " * 100, 0.9, "Characters", str(workspace / "data"), False)
    with pytest.raises(gr.Error, match="tokenizers differ"):
        list(ui.train_model(*training_inputs(workspace, session_id)))


def test_state_cleanup_and_errors(workspace):
    with pytest.raises(gr.Error, match="Prepare a dataset"):
        ui.initialize_model("empty", 1, 1, 2, 16, 8, 0, False)
    ready_model(workspace)
    session = ui.get_session("test")
    ui.drop_session("test")
    assert session.stop.is_set()
    assert "test" not in ui._sessions


def test_gradio_training_callback_serializes_chart_and_download(workspace):
    session_id = ready_model(workspace)
    app = ui.build_app()
    state = SessionState(app)
    state_component = next(block for block in app.blocks.values() if isinstance(block, gr.State))
    state[state_component._id] = session_id
    callback = next(fn for fn in app.fns.values() if fn.fn is ui.train_model)
    args = training_inputs(workspace, session_id)
    args[0] = None  # Gradio obtains this value from server-side session state.

    async def exercise():
        response = await app.process_api(callback, args, state=state, session_hash=session_id)
        seen = 0
        while True:
            assert len(response["data"]) == 3
            json.dumps(response["data"])
            seen += 1
            if not response["is_generating"]:
                break
            response = await app.process_api(callback, args, state=state,
                                             iterator=response["iterator"], session_hash=session_id)
        return seen

    assert asyncio.run(exercise()) >= 3


def test_settings_written_in_working_directory(workspace):
    ui.save_settings("data", "models", "cpu", "auto")
    assert (workspace / "config.json").exists()
    assert ui.load_settings()["device"] == "cpu"
