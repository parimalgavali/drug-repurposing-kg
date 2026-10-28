"""Download PrimeKG files from Harvard Dataverse with resume and checksum verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

DOI_PAGE = "https://doi.org/10.7910/DVN/IXA7BM"
DATAFILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
CHUNK = 1 << 20


@dataclass(frozen=True)
class RemoteFile:
    name: str
    file_id: int
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return DATAFILE_URL.format(file_id=self.file_id)


# PrimeKG version 2 (released 2022-05-02).
PRIMEKG = (
    RemoteFile(
        "kg.csv",
        6180620,
        981_751_236,
        "0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69",
    ),
    RemoteFile(
        "nodes.tab",
        6180617,
        8_893_757,
        "18ebcf24887e8529b95c7a7c96c81089cc7c5ba22d53a5c05d56480331961af8",
    ),
)
FEATURES = (
    RemoteFile(
        "drug_features.tab",
        6180619,
        10_030_011,
        "d9db67a3ae9e4e1bea6f50ac55c9e3152c30f9a128b45cec0d5b003fabdc30c6",
    ),
    RemoteFile(
        "disease_features.tab",
        6180618,
        113_534_270,
        "bdd18b39aae9c80e37c3716c2bfabaf854f6a5a7e44c8d2e5b082f0dd27124cb",
    ),
)
GROUPS = {"primekg": PRIMEKG, "features": FEATURES, "all": PRIMEKG + FEATURES}


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def is_complete(path: Path, remote: RemoteFile) -> bool:
    return (
        path.exists() and path.stat().st_size == remote.size and sha256_file(path) == remote.sha256
    )


def download(
    remote: RemoteFile, dest_dir: Path, client: httpx.Client, progress: bool = True
) -> Path:
    """Download `remote` into `dest_dir`, resuming a `.part` file; verify size and sha256."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / remote.name
    if is_complete(target, remote):
        return target

    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset >= remote.size:
        part.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    with client.stream("GET", remote.url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        if response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with (
            part.open(mode) as f,
            tqdm(
                total=remote.size,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=remote.name,
                disable=not progress,
            ) as bar,
        ):
            for chunk in response.iter_bytes(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))

    digest = sha256_file(part)
    if digest != remote.sha256:
        part.unlink()
        raise ChecksumError(
            f"{remote.name}: sha256 {digest} != expected {remote.sha256}. "
            f"The Dataverse release may have changed; check {DOI_PAGE}"
        )
    part.replace(target)
    return target


def fetch(group: str, dest_dir: Path, progress: bool = True) -> list[Path]:
    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0)) as client:
        return [download(r, dest_dir, client, progress) for r in GROUPS[group]]
