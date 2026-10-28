import hashlib

import httpx
import pytest

from primekg.fetch import ChecksumError, RemoteFile, download

PAYLOAD = b"node_index\tnode_name\n" * 1000


def remote(payload: bytes = PAYLOAD) -> RemoteFile:
    return RemoteFile("nodes.tab", 1, len(payload), hashlib.sha256(payload).hexdigest())


def client_for(payload: bytes, calls: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        rng = request.headers.get("range")
        if rng:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=payload[start:])
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_verifies_and_writes(tmp_path):
    calls = []
    path = download(remote(), tmp_path, client_for(PAYLOAD, calls), progress=False)
    assert path.read_bytes() == PAYLOAD
    assert not (tmp_path / "nodes.tab.part").exists()
    assert len(calls) == 1


def test_complete_file_is_not_downloaded_again(tmp_path):
    (tmp_path / "nodes.tab").write_bytes(PAYLOAD)
    calls = []
    download(remote(), tmp_path, client_for(PAYLOAD, calls), progress=False)
    assert calls == []


def test_resumes_partial_download(tmp_path):
    (tmp_path / "nodes.tab.part").write_bytes(PAYLOAD[:500])
    calls = []
    path = download(remote(), tmp_path, client_for(PAYLOAD, calls), progress=False)
    assert calls[0].headers["range"] == "bytes=500-"
    assert path.read_bytes() == PAYLOAD


def test_checksum_mismatch_raises_and_removes_partial(tmp_path):
    calls = []
    with pytest.raises(ChecksumError, match="doi.org/10.7910/DVN/IXA7BM"):
        download(remote(), tmp_path, client_for(PAYLOAD[:-1] + b"X", calls), progress=False)
    assert not (tmp_path / "nodes.tab.part").exists()
    assert not (tmp_path / "nodes.tab").exists()
