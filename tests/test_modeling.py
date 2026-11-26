from __future__ import annotations

import json
from pathlib import Path

import pytest

from routelab.bundle import BundleError, load_bundle, save_bundle
from routelab.modeling import ModelConfig, train_model

from .conftest import synthetic_examples


def test_prediction_ranks_alternatives_by_score(tiny_bundle_dir: Path) -> None:
    bundle = load_bundle(tiny_bundle_dir)
    prediction = bundle.predict("how can i change my pin number", top_k=3)
    assert prediction.label == "change_pin"
    scores = [prediction.confidence] + [score for _, score in prediction.alternatives]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert prediction.label not in [label for label, _ in prediction.alternatives]


def test_training_requires_two_labels() -> None:
    with pytest.raises(ValueError, match="two different labels"):
        train_model(["a", "b"], ["x", "x"], ModelConfig())


def test_bundle_round_trip_preserves_predictions(tmp_path: Path) -> None:
    texts, labels = synthetic_examples()
    pipeline = train_model(texts, labels, ModelConfig())
    saved = save_bundle(tmp_path, pipeline, {"training": {"examples": len(texts)}})
    loaded = load_bundle(tmp_path)
    assert loaded.model_version == saved.model_version
    assert loaded.model_version.startswith("lr-")
    message = "please refund my order"
    assert loaded.predict(message) == saved.predict(message)


def test_missing_bundle_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="routelab train"):
        load_bundle(tmp_path / "does-not-exist")


def test_tampered_model_file_is_rejected(tmp_path: Path, tiny_bundle_dir: Path) -> None:
    copy = tmp_path / "bundle"
    copy.mkdir()
    for name in ("model.joblib", "bundle.json"):
        (copy / name).write_bytes((tiny_bundle_dir / name).read_bytes())
    with (copy / "model.joblib").open("ab") as handle:
        handle.write(b"extra bytes")
    with pytest.raises(BundleError, match="does not match the hash"):
        load_bundle(copy)


def test_scikit_learn_version_mismatch_is_rejected(tmp_path: Path, tiny_bundle_dir: Path) -> None:
    copy = tmp_path / "bundle"
    copy.mkdir()
    (copy / "model.joblib").write_bytes((tiny_bundle_dir / "model.joblib").read_bytes())
    meta = json.loads((tiny_bundle_dir / "bundle.json").read_text())
    meta["environment"]["scikit-learn"] = "0.0.1"
    (copy / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(BundleError, match="scikit-learn 0.0.1"):
        load_bundle(copy)
