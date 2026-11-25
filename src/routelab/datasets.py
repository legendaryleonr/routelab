"""Pinned dataset sources, checksum-verified downloads, and loaders.

Every file is fetched from a fixed Git commit and checked against a SHA-256
hash recorded here, so the data used for training cannot silently change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import ssl
import tempfile
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

RAW_BASE_URL = "https://raw.githubusercontent.com"

BANKING77_REPO = "PolyAI-LDN/task-specific-datasets"
BANKING77_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77_LICENSE = "CC-BY-4.0"
BANKING77_EXPECTED_ROWS = {"train": 10003, "test": 3080}


@dataclass(frozen=True)
class DataFile:
    """One downloadable file, pinned to a commit and a content hash."""

    local_name: str
    repo: str
    commit: str
    repo_path: str
    sha256: str
    license: str

    def url(self, base_url: str = RAW_BASE_URL) -> str:
        return f"{base_url.rstrip('/')}/{self.repo}/{self.commit}/{self.repo_path}"


BANKING77_FILES: tuple[DataFile, ...] = (
    DataFile(
        "banking77/train.csv",
        BANKING77_REPO,
        BANKING77_COMMIT,
        "banking_data/train.csv",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        BANKING77_LICENSE,
    ),
    DataFile(
        "banking77/test.csv",
        BANKING77_REPO,
        BANKING77_COMMIT,
        "banking_data/test.csv",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        BANKING77_LICENSE,
    ),
    DataFile(
        "banking77/categories.json",
        BANKING77_REPO,
        BANKING77_COMMIT,
        "banking_data/categories.json",
        "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
        BANKING77_LICENSE,
    ),
)

ALL_FILES: tuple[DataFile, ...] = BANKING77_FILES


class DataError(RuntimeError):
    """Raised when a dataset file is missing, corrupted, or malformed."""


@dataclass(frozen=True)
class Example:
    """A labelled message. ``id`` is stable: source split plus row number."""

    id: str
    text: str
    label: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise DataError(f"{path} is missing. Run `routelab data download` first.")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise DataError(
            f"{path} has SHA-256 {actual}, expected {expected_sha256}. "
            "Delete the file and download it again."
        )


def _ssl_context() -> ssl.SSLContext:
    # Use certifi's CA bundle so downloads work with Python builds that do not
    # ship system certificates (for example some macOS installers).
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi is a pinned dependency
        return ssl.create_default_context()


def download_files(
    raw_dir: Path,
    files: Iterable[DataFile] = ALL_FILES,
    *,
    base_url: str = RAW_BASE_URL,
    timeout: float = 60.0,
) -> list[Path]:
    """Download each file unless a verified copy already exists.

    Files are written to a temporary name and only moved into place after the
    hash matches, so an interrupted download never leaves a half-written file.
    """
    written: list[Path] = []
    context = _ssl_context()
    for item in files:
        target = raw_dir / item.local_name
        if target.is_file() and sha256_file(target) == item.sha256:
            written.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = item.url(base_url)
        request = urllib.request.Request(url, headers={"User-Agent": "routelab-data/1"})
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".download-")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out:
                open_kwargs = {"timeout": timeout}
                if url.startswith("https://"):
                    open_kwargs["context"] = context
                with urllib.request.urlopen(request, **open_kwargs) as response:  # noqa: S310
                    for chunk in iter(lambda: response.read(1 << 16), b""):
                        out.write(chunk)
            actual = sha256_file(tmp_path)
            if actual != item.sha256:
                raise DataError(
                    f"Downloaded {url} but its SHA-256 is {actual}, expected {item.sha256}."
                )
            os.chmod(tmp_path, 0o644)  # mkstemp creates files readable only by the owner
            tmp_path.replace(target)
        except OSError as exc:
            raise DataError(f"Could not download {url}: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        written.append(target)
    return written


def _file(name: str) -> DataFile:
    for item in ALL_FILES:
        if item.local_name == name:
            return item
    raise KeyError(name)


def load_banking77_labels(raw_dir: Path) -> list[str]:
    item = _file("banking77/categories.json")
    path = raw_dir / item.local_name
    verify_file(path, item.sha256)
    labels = json.loads(path.read_text(encoding="utf-8"))
    if len(labels) != 77 or len(set(labels)) != 77:
        raise DataError(f"Expected 77 unique BANKING77 labels, found {len(set(labels))}")
    return list(labels)


def read_examples_csv(path: Path, split: str, labels: set[str]) -> list[Example]:
    """Parse a BANKING77-style CSV with ``text`` and ``category`` columns.

    The csv module handles quoted fields, including the few messages that
    contain line breaks.
    """
    examples: list[Example] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise DataError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row_number, row in enumerate(reader):
            if row["category"] not in labels:
                raise DataError(f"Unknown label {row['category']!r} in {path}")
            examples.append(Example(f"{split}-{row_number:05d}", row["text"], row["category"]))
    return examples


def load_banking77_split(raw_dir: Path, split: str) -> list[Example]:
    """Load the official ``train`` or ``test`` split after verifying its hash."""
    if split not in BANKING77_EXPECTED_ROWS:
        raise ValueError(f"split must be one of {sorted(BANKING77_EXPECTED_ROWS)}")
    item = _file(f"banking77/{split}.csv")
    path = raw_dir / item.local_name
    verify_file(path, item.sha256)
    examples = read_examples_csv(path, split, set(load_banking77_labels(raw_dir)))
    expected = BANKING77_EXPECTED_ROWS[split]
    if len(examples) != expected:
        raise DataError(f"Expected {expected} rows in {path}, found {len(examples)}")
    return examples
