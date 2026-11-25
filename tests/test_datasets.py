from __future__ import annotations

import functools
import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from routelab.datasets import DataError, DataFile, download_files, read_examples_csv, verify_file


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def file_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Serve a local directory over HTTP so downloads can be tested offline."""
    root = tmp_path / "served"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *args, **kwargs: None  # keep test output quiet
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", root
    finally:
        server.shutdown()
        server.server_close()


def _publish(root: Path, repo_path: str, content: bytes) -> None:
    path = root / "owner/repo/abc123" / repo_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_verify_file_rejects_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "file.csv"
    path.write_bytes(b"hello")
    verify_file(path, _sha(b"hello"))
    with pytest.raises(DataError, match="SHA-256"):
        verify_file(path, _sha(b"different"))


def test_verify_file_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="missing"):
        verify_file(tmp_path / "nope.csv", _sha(b""))


def test_download_verifies_hash_and_skips_verified_files(file_server, tmp_path: Path) -> None:
    base_url, root = file_server
    content = b"text,category\nhello,card_arrival\n"
    _publish(root, "data/train.csv", content)
    item = DataFile("demo/train.csv", "owner/repo", "abc123", "data/train.csv", _sha(content), "CC-BY-4.0")
    raw_dir = tmp_path / "raw"

    [path] = download_files(raw_dir, [item], base_url=base_url)
    assert path.read_bytes() == content

    # A second call must not re-download: remove the served file to prove it.
    (root / "owner/repo/abc123/data/train.csv").unlink()
    [again] = download_files(raw_dir, [item], base_url=base_url)
    assert again == path


def test_download_rejects_unexpected_content(file_server, tmp_path: Path) -> None:
    base_url, root = file_server
    _publish(root, "data/train.csv", b"tampered")
    item = DataFile("demo/train.csv", "owner/repo", "abc123", "data/train.csv", _sha(b"original"), "x")
    raw_dir = tmp_path / "raw"
    with pytest.raises(DataError, match="SHA-256"):
        download_files(raw_dir, [item], base_url=base_url)
    assert not (raw_dir / "demo/train.csv").exists()
    assert list((raw_dir / "demo").glob(".download-*")) == []


def test_read_examples_csv_handles_quoted_newlines(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text('text,category\n"line one\nline two",card_arrival\nplain,change_pin\n', encoding="utf-8")
    examples = read_examples_csv(path, "train", {"card_arrival", "change_pin"})
    assert [e.text for e in examples] == ["line one\nline two", "plain"]
    assert [e.id for e in examples] == ["train-00000", "train-00001"]


def test_read_examples_csv_rejects_unknown_label(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("text,category\nhi,not_a_label\n", encoding="utf-8")
    with pytest.raises(DataError, match="Unknown label"):
        read_examples_csv(path, "train", {"card_arrival"})
