from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Iterator, TextIO
from zipfile import ZipFile

from training.data.contracts import CompetitionExample, DatasetSummary


TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
DEFAULT_ARCHIVE_NAME = "nvidia-nemotron-model-reasoning-challenge.zip"
_CSV_MEMBERS = {
    TRAIN_SPLIT: "train.csv",
    TEST_SPLIT: "test.csv",
}


@dataclass(slots=True)
class CompetitionDataSource:
    location: Path
    storage: str

    def member_name(self, split: str) -> str:
        try:
            return _CSV_MEMBERS[split]
        except KeyError as exc:
            raise ValueError(f"Unsupported competition split: {split}") from exc


def discover_competition_data(path: str | Path = "data/raw") -> CompetitionDataSource:
    location = Path(path)
    if location.is_file():
        if location.suffix.lower() != ".zip":
            raise FileNotFoundError(f"Expected a .zip archive, got: {location}")
        _validate_zip_members(location)
        return CompetitionDataSource(location=location, storage="zip")

    if not location.exists():
        raise FileNotFoundError(f"Competition data path not found: {location}")

    train_csv = location / "train.csv"
    test_csv = location / "test.csv"
    if train_csv.exists() and test_csv.exists():
        return CompetitionDataSource(location=location, storage="directory")

    archive = location / DEFAULT_ARCHIVE_NAME
    if archive.exists():
        _validate_zip_members(archive)
        return CompetitionDataSource(location=archive, storage="zip")

    zip_files = sorted(location.glob("*.zip"))
    if len(zip_files) == 1:
        _validate_zip_members(zip_files[0])
        return CompetitionDataSource(location=zip_files[0], storage="zip")

    raise FileNotFoundError(
        "Could not find Kaggle competition data. Expected train.csv/test.csv or a competition .zip archive."
    )


def load_competition_split(source: CompetitionDataSource | str | Path, split: str) -> list[CompetitionExample]:
    resolved_source = source if isinstance(source, CompetitionDataSource) else discover_competition_data(source)
    rows: list[CompetitionExample] = []

    with _open_text_member(resolved_source, resolved_source.member_name(split)) as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, split)

        for row in reader:
            rows.append(
                CompetitionExample(
                    example_id=(row.get("id") or "").strip(),
                    prompt=row["prompt"],
                    answer=row.get("answer") or None,
                    split=split,
                )
            )
    return rows


def summarize_competition_data(source: CompetitionDataSource | str | Path) -> dict[str, DatasetSummary]:
    resolved_source = source if isinstance(source, CompetitionDataSource) else discover_competition_data(source)
    summaries: dict[str, DatasetSummary] = {}

    for split, member_name in _CSV_MEMBERS.items():
        rows = load_competition_split(resolved_source, split)
        fields = ("id", "prompt") if split == TEST_SPLIT else ("id", "prompt", "answer")
        summaries[split] = DatasetSummary(
            name=f"competition_{split}",
            split=split,
            rows=len(rows),
            rows_with_answers=sum(1 for row in rows if row.answer),
            path=resolved_source.location,
            format="csv",
            fields=fields,
        )
    return summaries


@contextmanager
def _open_text_member(source: CompetitionDataSource, member_name: str) -> Iterator[TextIO]:
    if source.storage == "directory":
        with (source.location / member_name).open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle
        return

    if source.storage == "zip":
        with ZipFile(source.location) as archive:
            with archive.open(member_name) as raw_handle:
                with TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="") as handle:
                    yield handle
        return

    raise ValueError(f"Unsupported competition storage: {source.storage}")


def _validate_zip_members(path: Path) -> None:
    with ZipFile(path) as archive:
        members = set(archive.namelist())
    missing = [member for member in _CSV_MEMBERS.values() if member not in members]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Competition archive is missing required members: {missing_text}")


def _require_columns(fieldnames: list[str] | None, split: str) -> None:
    if fieldnames is None:
        raise ValueError(f"{split}.csv is empty")

    required = {"id", "prompt"}
    if split == TRAIN_SPLIT:
        required.add("answer")

    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{split}.csv is missing required columns: {', '.join(missing)}")
