from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_sigd_exhaustive_evaluation_protocol import run_audit  # noqa: E402
from build_sigd_exhaustive_evaluation_protocol import build_protocol, load_config, sampled_protocol_hashes  # noqa: E402


@pytest.fixture(scope="module")
def exhaustive_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = Path(__file__).resolve().parents[3]
    config = load_config(root, None)
    output_dir = tmp_path_factory.mktemp("exhaustive_eval_v2")
    config = {**config, "output_dir": str(output_dir)}
    before = sampled_protocol_hashes(root)
    summary = build_protocol(root, config, overwrite=True)
    audit = run_audit(root, config)
    after = sampled_protocol_hashes(root)
    return {"root": root, "config": config, "output_dir": output_dir, "summary": summary, "audit": audit, "before": before, "after": after}


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_rows(fixture, split: str, kind: str) -> list[dict[str, str]]:
    prefix = "validation" if split == "val" else split
    return read_csv(fixture["output_dir"] / f"{prefix}_{kind}_trials_exhaustive_k5m1_seed42.csv")


def genuine_rows(fixture, split: str) -> list[dict[str, str]]:
    prefix = "validation" if split == "val" else split
    return read_csv(fixture["output_dir"] / f"{prefix}_genuine_trials_k5m1_seed42.csv")


def test_exhaustive_val_impostor_count(exhaustive_fixture) -> None:
    assert len(split_rows(exhaustive_fixture, "val", "impostor")) == 112950


def test_exhaustive_test_impostor_count(exhaustive_fixture) -> None:
    assert len(split_rows(exhaustive_fixture, "test", "impostor")) == 107775


def test_every_val_probe_has_45_impostor_templates(exhaustive_fixture) -> None:
    counts = Counter(row["probe_window_index"] for row in split_rows(exhaustive_fixture, "val", "impostor"))
    assert set(counts.values()) == {45}
    assert len(counts) == 2510


def test_every_test_probe_has_45_impostor_templates(exhaustive_fixture) -> None:
    counts = Counter(row["probe_window_index"] for row in split_rows(exhaustive_fixture, "test", "impostor"))
    assert set(counts.values()) == {45}
    assert len(counts) == 2395


def test_no_same_subject_impostor_trial(exhaustive_fixture) -> None:
    for split in ("val", "test"):
        assert all(row["enroll_subject_id"] != row["probe_subject_id"] for row in split_rows(exhaustive_fixture, split, "impostor"))


def test_all_impostor_probes_are_later_session_only(exhaustive_fixture) -> None:
    from datetime import datetime

    for split in ("val", "test"):
        for row in split_rows(exhaustive_fixture, split, "impostor")[:1000]:
            probe = datetime.strptime(row["probe_session_id"], "%Y-%m-%d-%H-%M")
            ref = datetime.strptime(row["probe_reference_enrollment_session_id"], "%Y-%m-%d-%H-%M")
            assert probe > ref
            assert float(row["probe_time_gap_days"]) > 0


def test_no_duplicate_template_probe_combination(exhaustive_fixture) -> None:
    for split in ("val", "test"):
        combos = [(row["template_id"], row["probe_window_index"]) for row in split_rows(exhaustive_fixture, split, "impostor")]
        assert len(combos) == len(set(combos))


def test_gap_bucket_impostor_distribution_is_45x_probe_distribution(exhaustive_fixture) -> None:
    for split in ("val", "test"):
        genuine_counts = Counter(row["probe_time_gap_bucket"] for row in genuine_rows(exhaustive_fixture, split))
        impostor_counts = Counter(row["probe_time_gap_bucket"] for row in split_rows(exhaustive_fixture, split, "impostor"))
        assert impostor_counts == Counter({bucket: count * 45 for bucket, count in genuine_counts.items()})


def test_existing_sampled_protocol_files_not_overwritten(exhaustive_fixture) -> None:
    assert exhaustive_fixture["before"] == exhaustive_fixture["after"]
    assert exhaustive_fixture["summary"]["sampled_v2_files_unchanged"] is True
    assert exhaustive_fixture["audit"]["sampled_v2_files_unchanged"] is True


def test_morphology_validity_not_used_for_exhaustive_protocol(exhaustive_fixture) -> None:
    assert exhaustive_fixture["summary"]["morphology_validity_used_for_protocol"] is False
    assert exhaustive_fixture["audit"]["morphology_validity_used_for_protocol"] is False
