"""Execute the bilingual example and CLI workflow against a real local encoder."""

from pathlib import Path
import re
import runpy
import socket

import torch

import neembed.model as model_module
from test_role_prompts import local_prompt_encoder


ROOT = Path(__file__).parents[1]


def reject_network(*args, **kwargs):
    raise AssertionError("documentation examples must run with local fixtures")


def test_bilingual_role_example_is_identical_and_executes_offline(local_prompt_encoder, monkeypatch, tmp_path):
    pattern = r"<!-- v11-role-example -->\n```python\n(.*?)\n```"
    english = re.search(pattern, (ROOT / "README.md").read_text(), re.S).group(1)
    japanese = re.search(pattern, (ROOT / "docs/README_ja.md").read_text(), re.S).group(1)
    assert english == japanese
    encoder = model_module.SentenceTransformer

    def load_local(name, **kwargs):
        if name == "intfloat/e5-small-v2":
            assert kwargs["revision"] == "ffb93f3bd4047442299a41ebb6fa998a38507c52"
            assert kwargs["local_files_only"] is True
            name = str(local_prompt_encoder)
        return encoder(name, **kwargs)

    monkeypatch.setattr(model_module, "SentenceTransformer", load_local)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(english, "README role example", "exec"), namespace)
    assert torch.isfinite(namespace["loss"])
    assert namespace["embeddings"].shape == (2, 16)
    assert not namespace["model"].training
    assert [len(rows) for rows in namespace["results"]] == [2, 2]
    for rows in namespace["results"]:
        assert [row["distance"] for row in rows] == sorted(row["distance"] for row in rows)
    assert (tmp_path / "saved_role_model/neembed_config.json").exists()


def test_runnable_role_example_with_local_encoder(local_prompt_encoder, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    namespace = runpy.run_path(str(ROOT / "examples/v11_role_workflow.py"))
    result = namespace["run_example"](str(local_prompt_encoder), revision=None, local_files_only=True)
    assert 0 <= result["loss"] < float("inf")
    assert result["embedding_shape"] == [2, 16]
    assert [len(rows) for rows in result["results"]] == [2, 2]
