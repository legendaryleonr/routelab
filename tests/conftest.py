"""Shared test fixtures.

Unit tests never download BANKING77. They train a tiny model on synthetic
messages so the whole suite runs offline in a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from routelab.bundle import save_bundle
from routelab.modeling import ModelConfig, train_model

_TEMPLATES = {
    "card_arrival": [
        "where is my new card", "my card has not arrived yet", "card still not delivered",
        "when will my card arrive", "i am still waiting for my card", "has my card been sent",
    ],
    "change_pin": [
        "how do i change my pin", "i want a new pin number", "can i reset my pin",
        "change the pin on my card", "update my pin code please", "where do i change my pin",
    ],
    "request_refund": [
        "i want a refund", "please refund my purchase", "how do i get my money back",
        "refund for an item i returned", "can i request a refund", "i need a refund for this order",
    ],
    "exchange_rate": [
        "what is the exchange rate", "which currency rate do you use", "how much is a euro in dollars",
        "tell me the exchange rate for pounds", "what rate do you use to convert money",
        "exchange rate for japanese yen",
    ],
}
_SUFFIXES = ["", " please", " thanks", " today", " now", " asap"]


def synthetic_examples() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for label, phrases in _TEMPLATES.items():
        for phrase in phrases:
            for suffix in _SUFFIXES:
                texts.append(phrase + suffix)
                labels.append(label)
    return texts, labels


@pytest.fixture(scope="session")
def tiny_bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    texts, labels = synthetic_examples()
    pipeline = train_model(texts, labels, ModelConfig(C=10.0))
    directory = tmp_path_factory.mktemp("bundle")
    save_bundle(directory, pipeline, {"config": ModelConfig().to_dict(), "training": {"examples": len(texts)}})
    return directory
