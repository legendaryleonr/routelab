"""Save and load a trained model together with its metadata.

A bundle is a directory with two files:

* ``model.joblib`` - the fitted scikit-learn pipeline
* ``bundle.json``  - metadata, including the SHA-256 of ``model.joblib``

Loading verifies the hash before unpickling. joblib/pickle files can run code
when loaded, so only load bundles you built yourself.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

BUNDLE_FORMAT = 1
MODEL_FILE = "model.joblib"
META_FILE = "bundle.json"


class BundleError(RuntimeError):
    """The model bundle is missing, corrupted, or incompatible."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    alternatives: list[tuple[str, float]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "platform": f"{sys.platform}-{platform.machine()}",
    }


class ModelBundle:
    """A loaded, verified model that can score messages."""

    def __init__(self, pipeline: Pipeline, metadata: dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.labels: list[str] = [str(label) for label in pipeline.classes_]

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        scores = self.predict_proba([text])[0]
        order = np.argsort(-scores, kind="stable")[: max(1, top_k)]
        ranked = [(self.labels[i], float(scores[i])) for i in order]
        best_label, best_score = ranked[0]
        return Prediction(label=best_label, confidence=best_score, alternatives=ranked[1:])


def save_bundle(directory: Path, pipeline: Pipeline, metadata: dict[str, Any]) -> ModelBundle:
    """Write the pipeline and metadata. ``bundle.json`` is written last."""
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".model-")
    os.close(fd)
    tmp_model = Path(tmp_name)
    try:
        joblib.dump(pipeline, tmp_model)
        os.chmod(tmp_model, 0o644)  # mkstemp creates 0600; the server may run as another user
        model_sha = _sha256(tmp_model)
        tmp_model.replace(directory / MODEL_FILE)
    finally:
        tmp_model.unlink(missing_ok=True)

    full = {
        "format_version": BUNDLE_FORMAT,
        "model_version": f"lr-{model_sha[:12]}",
        "model_sha256": model_sha,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "labels": [str(label) for label in pipeline.classes_],
        "environment": environment_info(),
        **metadata,
    }
    meta_tmp = directory / f".{META_FILE}.tmp"
    meta_tmp.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_tmp.replace(directory / META_FILE)
    return ModelBundle(pipeline, full)


def load_bundle(directory: Path) -> ModelBundle:
    model_path = directory / MODEL_FILE
    meta_path = directory / META_FILE
    if not meta_path.is_file() or not model_path.is_file():
        raise BundleError(
            f"No model bundle found in {directory}. Train one with `routelab train`."
        )
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{meta_path} is not valid JSON: {exc}") from exc

    if metadata.get("format_version") != BUNDLE_FORMAT:
        raise BundleError(f"Unsupported bundle format: {metadata.get('format_version')!r}")
    actual_sha = _sha256(model_path)
    if actual_sha != metadata.get("model_sha256"):
        raise BundleError(f"{model_path} does not match the hash recorded in {meta_path}.")
    trained_with = metadata.get("environment", {}).get("scikit-learn")
    if trained_with != sklearn.__version__:
        raise BundleError(
            f"Model was trained with scikit-learn {trained_with}, but {sklearn.__version__} "
            "is installed. Retrain the model or install the pinned requirements."
        )

    pipeline = joblib.load(model_path)
    classes = [str(label) for label in pipeline.classes_]
    if classes != metadata.get("labels"):
        raise BundleError("Model classes do not match the labels recorded in bundle.json.")
    return ModelBundle(pipeline, metadata)
