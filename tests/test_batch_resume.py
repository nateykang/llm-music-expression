"""Batch resume: recorded cells are skipped, missing ones generated."""

import json

from llm_music import cli, store
from llm_music.generate import PieceResult


class _FakeClient:
    def __init__(self, name):
        self.name = name


def _fake_env(monkeypatch, tmp_path, calls):
    def fake_gen(client, prompt, mode, wd, **kw):
        calls.append((client.name, prompt))
        return PieceResult(ok=True, model=client.name, prompt=prompt, mode=mode,
                           title=f"{client.name}-{prompt}")
    monkeypatch.setattr(cli, "generate_piece", fake_gen)
    monkeypatch.setattr(cli, "get_client", _FakeClient)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)


def test_resume_runs_only_missing_cells(monkeypatch, tmp_path):
    calls = []
    _fake_env(monkeypatch, tmp_path, calls)
    batch, results = cli._run_matrix(["m1", "m2"], ["p1"], "abc", 5,
                                     samples=2, workers=2, bake_audio=False)
    assert len(results) == 4 and len(calls) == 4

    # Simulate an interrupted run: drop two pieces from the manifest.
    manifest = json.loads((batch / "data.json").read_text())
    kept = manifest["pieces"][:2]
    manifest["pieces"] = kept
    (batch / "data.json").write_text(json.dumps(manifest))

    calls.clear()
    batch2, results2 = cli._run_matrix(["m1", "m2"], ["p1"], "abc", 5,
                                       samples=2, workers=2, bake_audio=False,
                                       resume=batch)
    assert batch2 == batch
    assert len(results2) == 2 and len(calls) == 2
    done = {(e["model"], e["prompt"], e["sample"]) for e in kept}
    assert all((m, "p1") == (m, "p1") and (m, "p1", None) not in done for m, _ in calls)
    final = json.loads((batch / "data.json").read_text())
    cells = {(e["model"], e["prompt"], e.get("sample", 0)) for e in final["pieces"]}
    assert len(final["pieces"]) == 4
    assert cells == {(m, "p1", s) for m in ("m1", "m2") for s in (0, 1)}


def test_resume_extends_sample_count(monkeypatch, tmp_path):
    calls = []
    _fake_env(monkeypatch, tmp_path, calls)
    batch, _ = cli._run_matrix(["m1"], ["p1"], "abc", 5,
                               samples=1, workers=1, bake_audio=False)
    calls.clear()
    _, results = cli._run_matrix(["m1"], ["p1"], "abc", 5,
                                 samples=3, workers=1, bake_audio=False, resume=batch)
    assert len(results) == 2  # only the two new sample indices
    final = json.loads((batch / "data.json").read_text())
    assert {e["sample"] for e in final["pieces"]} == {0, 1, 2}
