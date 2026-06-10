"""Dataset download utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tarfile
import zipfile

import requests


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    source_url: str | None
    target_dir: Path
    archive_name: str | None = None
    extract: bool = True
    timeout_seconds: int = 60


@dataclass(frozen=True, slots=True)
class DownloadResult:
    destination: Path
    downloaded: bool
    extracted: bool


class DatasetDownloader:
    def __init__(self, config: DownloadConfig) -> None:
        self.config = config

    def download(self) -> DownloadResult:
        self.config.target_dir.mkdir(parents=True, exist_ok=True)
        if self.config.source_url is None:
            return DownloadResult(self.config.target_dir, downloaded=False, extracted=False)

        source = self.config.source_url
        if source.startswith("file://"):
            local_path = Path(source.replace("file://", ""))
            destination = self.config.target_dir / local_path.name
            shutil.copy2(local_path, destination)
        elif Path(source).exists():
            local_path = Path(source)
            destination = self.config.target_dir / local_path.name
            shutil.copy2(local_path, destination)
        else:
            file_name = self.config.archive_name or Path(source).name or "dataset.bin"
            destination = self.config.target_dir / file_name
            response = requests.get(source, stream=True, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)

        extracted = False
        if self.config.extract:
            extracted = self._extract_if_archive(destination)

        return DownloadResult(destination=destination, downloaded=True, extracted=extracted)

    def _extract_if_archive(self, archive_path: Path) -> bool:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(self.config.target_dir)
            return True

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                archive.extractall(self.config.target_dir)
            return True

        return False