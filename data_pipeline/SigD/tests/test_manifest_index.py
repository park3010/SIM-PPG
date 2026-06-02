from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root, load_pipeline_config  # noqa: E402
from manifest_index import ManifestIndex  # noqa: E402


def build_index() -> ManifestIndex:
    root = detect_project_root(Path(__file__).resolve().parents[3])
    return ManifestIndex(root, load_pipeline_config(root))


def test_common_available_count_and_array_indices() -> None:
    index = build_index()
    result = index.validate()
    assert result["passed"] is True
    assert result["common_available_count"] == 20974
    assert result["array_index_min"] == 0
    assert result["array_index_max"] == 20973
    assert len(index.by_array_index) == 20974


def test_subject_session_lookup_works() -> None:
    index = build_index()
    subject = index.subject_ids()[0]
    sessions = index.available_sessions(subject)
    assert sessions
    indices = index.array_indices_for_session(subject, sessions[0])
    assert indices
    row = index.get_metadata(indices[0])
    assert row["subject_id"] == subject
    assert row["session_id"] == sessions[0]


def test_morphology_validity_does_not_change_lookup_inclusion() -> None:
    index = build_index()
    ipa_invalid = [row for row in index.rows if not row["ipa_valid_mask"]]
    assert ipa_invalid
    assert ipa_invalid[0]["array_index"] in index.by_array_index
