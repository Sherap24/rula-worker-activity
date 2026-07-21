"""Conservative CML subject ID parsing from source file paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

CMU_PATTERN = re.compile(r"^[^/]+/(\d+)_\d+_\d+\.txt$", re.IGNORECASE)
HDM05_PATTERN = re.compile(r"^[^/]+/HDM_([a-z]+)_", re.IGNORECASE)
BERKELEY_PATTERN = re.compile(r"^[^/]+/skl_s(\d+)_", re.IGNORECASE)
SBU_PATTERN = re.compile(r"^[^/]+/(s\d+s\d+)_\d+\.txt$", re.IGNORECASE)
INLAB_NUMERIC_PATTERN = re.compile(r"^\d{1,3}$")


@dataclass(frozen=True)
class SubjectParseResult:
    subject_id: str | None
    subject_parse_status: str
    parser_name: str | None
    parsing_confidence: str | None
    source_sequence_id: str | None


def _parse_cmu(source_file: str) -> SubjectParseResult:
    match = CMU_PATTERN.match(source_file.strip())
    if not match:
        return SubjectParseResult(None, "unresolved_pattern", "cmu_subject", None, None)
    subject_num = match.group(1)
    seq = _cmu_sequence_id(source_file)
    return SubjectParseResult(
        f"CMU:{subject_num}",
        "parsed_verified_pattern",
        "cmu_subject",
        "high",
        seq,
    )


def _cmu_sequence_id(source_file: str) -> str | None:
    match = re.search(r"/(\d+)_(\d+)_\d+\.txt$", source_file)
    if match:
        return f"CMU:{match.group(1)}_{match.group(2)}"
    return None


def _parse_hdm05(source_file: str) -> SubjectParseResult:
    match = HDM05_PATTERN.match(source_file.strip())
    if not match:
        return SubjectParseResult(None, "unresolved_pattern", "hdm05_performer", None, None)
    code = match.group(1)
    return SubjectParseResult(
        f"HDM05:{code}",
        "parsed_consistent_pattern",
        "hdm05_performer",
        "medium",
        source_file,
    )


def _parse_berkeley(source_file: str) -> SubjectParseResult:
    match = BERKELEY_PATTERN.match(source_file.strip())
    if not match:
        return SubjectParseResult(None, "unresolved_pattern", "berkeley_subject", None, None)
    subject = match.group(1)
    return SubjectParseResult(
        f"Berkeley:s{subject}",
        "parsed_consistent_pattern",
        "berkeley_subject",
        "high",
        source_file,
    )


def _parse_sbu(source_file: str) -> SubjectParseResult:
    match = SBU_PATTERN.match(source_file.strip())
    if not match:
        return SubjectParseResult(None, "unresolved_pattern", "sbu_pair", None, None)
    pair = match.group(1)
    return SubjectParseResult(
        f"SBU:{pair}",
        "parsed_consistent_pattern",
        "sbu_interaction_pair",
        "medium",
        source_file,
    )


def _parse_inlab(source_file: str) -> SubjectParseResult:
    value = source_file.strip()
    if INLAB_NUMERIC_PATTERN.match(value):
        return SubjectParseResult(
            f"In-lab:{value.zfill(2)}",
            "parsed_consistent_pattern",
            "inlab_numeric_id",
            "medium",
            value,
        )
    return SubjectParseResult(None, "unresolved_pattern", "inlab_numeric", None, None)


PARSERS: dict[str, Callable[[str], SubjectParseResult]] = {
    "CMU": _parse_cmu,
    "HDM05": _parse_hdm05,
    "Berkeley": _parse_berkeley,
    "SBU": _parse_sbu,
    "In-lab experiment data": _parse_inlab,
}


def parse_subject_id(source_dataset: str | None, source_file: str | None) -> SubjectParseResult:
    if source_file is None or str(source_file).strip() == "":
        return SubjectParseResult(None, "missing_source_file", None, None, None)
    if source_dataset is None or str(source_dataset).strip() == "":
        return SubjectParseResult(None, "unresolved_pattern", None, None, None)

    parser = PARSERS.get(str(source_dataset).strip())
    if parser is None:
        return SubjectParseResult(None, "unresolved_pattern", None, None, None)
    return parser(str(source_file).strip())


def profile_source_path(
    source_dataset: str,
    source_file: str,
    path_depth: int,
    filename: str,
) -> dict[str, str | int]:
    return {
        "source_dataset": source_dataset,
        "source_file": source_file,
        "path_depth": path_depth,
        "filename": filename,
        "parser": PARSERS.get(source_dataset, lambda _: None).__name__ if source_dataset in PARSERS else "none",
    }
