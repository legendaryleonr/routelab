"""Text classification model: TF-IDF features plus logistic regression."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the TF-IDF + logistic regression pipeline."""

    C: float = 10.0
    max_iter: int = 3000

    def to_dict(self) -> dict:
        return asdict(self)


def build_pipeline(config: ModelConfig) -> Pipeline:
    """Create an unfitted pipeline.

    The vectorizer lives inside the pipeline, so its vocabulary and IDF weights
    are learned only from the texts passed to ``fit`` (the training data).
    """
    features = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    classifier = LogisticRegression(C=config.C, max_iter=config.max_iter)
    return Pipeline([("features", features), ("classifier", classifier)])


def train_model(texts: Sequence[str], labels: Sequence[str], config: ModelConfig) -> Pipeline:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if len(set(labels)) < 2:
        raise ValueError("training data needs at least two different labels")
    pipeline = build_pipeline(config)
    pipeline.fit(list(texts), list(labels))
    return pipeline
