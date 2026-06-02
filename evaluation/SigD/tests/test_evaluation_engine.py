from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))
if "common" in sys.modules and str(SRC) not in str(getattr(sys.modules["common"], "__file__", "")):
    del sys.modules["common"]

from common import add_data_pipeline_src, detect_project_root, load_csv_rows, load_eval_config, load_yaml_config, resolve_from_root, sha256_file, write_json  # noqa: E402
from cosine_verifier import score_trials  # noqa: E402
from embedding_cache import collect_unique_array_indices, compute_embedding_cache, save_embedding_cache  # noqa: E402
from metrics import compute_split_metrics  # noqa: E402
from mock_encoder import DeterministicMockEncoder  # noqa: E402
from papagei_s_adapter import COMMON_INPUT_POLICY, PaPaGeiSFrozenAdapter  # noqa: E402
from reporting import write_score_csv  # noqa: E402
from run_papagei_s_frozen_baseline import (  # noqa: E402
    check_official_ready,
    ensure_scientific_result_root_available,
    official_source_provenance,
    write_skipped_manifest,
)
from setup_papagei_model_reference import combined_manifest, inspect_or_setup, safe_torch_load  # noqa: E402
from stratified_analysis import time_gap_metrics  # noqa: E402
from thresholds import compute_eer_threshold  # noqa: E402


def _load_real_context():
    root = detect_project_root(Path(__file__).resolve().parents[3])
    config = load_eval_config(root)
    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from transforms import PerWindowZScore

    dp_config_path = resolve_from_root(root, config["input"]["data_pipeline_config_path"])
    assert dp_config_path is not None
    dp_config = load_yaml_config(dp_config_path)
    index = ManifestIndex(root, dp_config)
    transform = PerWindowZScore()
    protocol_dir = resolve_from_root(root, dp_config["protocol"]["protocol_dir"])
    assert protocol_dir is not None
    templates = load_csv_rows(protocol_dir / "enrollment_templates_k5_seed42.csv")
    trials = load_csv_rows(protocol_dir / "verification_trials_k5m1_seed42.csv")
    return root, config, dp_config, index, transform, templates, trials


def _balanced_subset(rows: list[dict[str, str]], split: str, n_each: int = 2) -> list[dict[str, str]]:
    genuine = [row for row in rows if row["split"] == split and row["label"] == "1"][:n_each]
    impostor = [row for row in rows if row["split"] == split and row["label"] == "0"][:n_each]
    return genuine + impostor


def test_mock_encoder_end_to_end_score_generation() -> None:
    _, config, _, index, transform, templates, trials = _load_real_context()
    subset = _balanced_subset(trials, "val", 2)
    encoder = DeterministicMockEncoder()
    unique_indices = collect_unique_array_indices(templates, subset)
    embeddings = compute_embedding_cache(
        manifest_index=index,
        transform=transform,
        encoder=encoder,
        array_indices=unique_indices,
        batch_size=8,
        device=torch.device("cpu"),
    )
    scores = score_trials(
        trial_rows=subset,
        template_rows=templates,
        embedding_cache=embeddings,
        encoder_id=encoder.encoder_id,
        eps=float(config["template_aggregation"]["epsilon"]),
    )
    assert len(scores) == 4
    assert all(np.isfinite(float(row["score"])) for row in scores)
    assert all(-1.000001 <= float(row["score"]) <= 1.000001 for row in scores)


def test_val_test_score_outputs_are_separate(tmp_path: Path) -> None:
    _, _, _, _, _, _, trials = _load_real_context()
    val_rows = [{"trial_id": row["trial_id"], "split": "val", "score": 0.1} for row in _balanced_subset(trials, "val", 1)]
    test_rows = [{"trial_id": row["trial_id"], "split": "test", "score": 0.2} for row in _balanced_subset(trials, "test", 1)]
    write_score_csv(tmp_path / "validation_scores.csv", val_rows)
    write_score_csv(tmp_path / "test_scores.csv", test_rows)
    assert (tmp_path / "validation_scores.csv").exists()
    assert (tmp_path / "test_scores.csv").exists()
    assert (tmp_path / "validation_scores.csv").read_text() != (tmp_path / "test_scores.csv").read_text()


def test_validation_threshold_applied_to_test_without_test_selection() -> None:
    val_threshold = compute_eer_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["threshold"]
    test_metrics = compute_split_metrics(
        split="test",
        labels=[0, 1],
        scores=[0.3, 0.7],
        validation_eer_threshold=val_threshold,
    )
    assert test_metrics["validation_fixed_eer_threshold"]["threshold"] == val_threshold


def test_mock_run_manifest_scientific_reporting_false(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, {"scientific_reporting_allowed": False, "encoder_id": "deterministic_mock_encoder_engine_audit_only"})
    assert '"scientific_reporting_allowed": false' in path.read_text()


def test_embedding_cache_key_contains_protocol_fields(tmp_path: Path) -> None:
    embeddings = {0: np.zeros(4, dtype=np.float32)}
    manifest = tmp_path / "cache_manifest.json"
    save_embedding_cache(
        path=tmp_path / "cache.npz",
        manifest_path=manifest,
        embeddings=embeddings,
        metadata={
            "encoder_id": "mock",
            "input_protocol_id": "COMMON_PPG_10S_125HZ_V1",
            "protocol_id": "SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_V2",
            "normalization_policy": "per_window_zscore_common_dataloader",
        },
    )
    text = manifest.read_text()
    assert "encoder_id" in text
    assert "input_protocol_id" in text
    assert "protocol_id" in text


def test_official_script_skips_without_verified_checkpoint(tmp_path: Path) -> None:
    readiness = check_official_ready(tmp_path)
    assert readiness["ready"] is False
    manifest = write_skipped_manifest(tmp_path, {"evaluation_id": "PAPAGEI_S_FROZEN_COSINE_SIGD_V1"}, "missing")
    payload = manifest.read_text()
    assert "skipped_due_to_missing_verified_checkpoint" in payload
    assert '"scientific_reporting_allowed": false' in payload


def test_mock_time_gap_output_contains_two_threshold_names() -> None:
    rows = [
        {"label": 1, "score": 0.8, "probe_time_gap_bucket": "le_30d"},
        {"label": 0, "score": 0.2, "probe_time_gap_bucket": "le_30d"},
        {"label": 1, "score": 0.7, "probe_time_gap_bucket": "gt_365d"},
        {"label": 0, "score": 0.3, "probe_time_gap_bucket": "gt_365d"},
    ]
    buckets = ["le_30d", "gt_365d"]
    output = time_gap_metrics(rows, 0.5, buckets, "validation_eer_threshold")
    output += time_gap_metrics(rows, 0.6, buckets, "validation_far_1pct_threshold")
    assert {row["threshold_name"] for row in output} == {
        "validation_eer_threshold",
        "validation_far_1pct_threshold",
    }


def test_papagei_adapter_refuses_unverified_manifest(tmp_path: Path) -> None:
    config = {
        "encoder": {
            "official_source_dir": "evaluation/SigD/official_reference/PaPaGei_Model/source/papagei-foundation-model",
            "checkpoint_path": "evaluation/SigD/official_reference/PaPaGei_Model/weights/papagei_s.pt",
        }
    }
    with pytest.raises(RuntimeError):
        PaPaGeiSFrozenAdapter(tmp_path, config)


def test_papagei_adapter_uses_common_input_without_native_preprocessing() -> None:
    assert COMMON_INPUT_POLICY["input_setting"] == "common_input"
    assert COMMON_INPUT_POLICY["official_native_preprocessing_reapplied"] is False


def test_forward_output_tuple_contract_false_keeps_readiness_false() -> None:
    manifest = combined_manifest(
        source={"verified": True, "model_definition_present": True, "import_verification": {"loading_utility_imported": True}},
        checkpoint={"verified": True, "size_bytes": 123, "local_checkpoint_path": "x", "observed_md5": "m", "sha256": "s"},
        load_forward={
            "architecture_instantiated": True,
            "strict_checkpoint_loaded": True,
            "forward_output_is_tuple_or_list": False,
            "embedding_dim_verified": True,
            "frozen_verified": True,
            "forward_embedding_finite": True,
        },
        network_download_performed=False,
    )
    assert manifest["ready_for_scientific_frozen_baseline"] is False
    assert "PaPaGei-S forward output tuple/list contract" in manifest["missing_items"]


def test_official_source_provenance_and_adapter_hash_recorded(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "evaluation" / "SigD" / "metadata"
    src_dir = tmp_path / "evaluation" / "SigD" / "src"
    metadata_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    source_manifest = metadata_dir / "papagei_source_snapshot_manifest.json"
    adapter_source = src_dir / "papagei_s_adapter.py"
    write_json(
        source_manifest,
        {
            "source_repository": "https://github.com/Nokia-Bell-Labs/papagei-foundation-model",
            "git_commit_sha": "abc123",
            "archive_sha256": None,
        },
    )
    adapter_source.write_text("adapter-code\n", encoding="utf-8")
    provenance = official_source_provenance(tmp_path)
    assert provenance["official_source_repository"].endswith("papagei-foundation-model")
    assert provenance["official_source_git_commit_sha"] == "abc123"
    assert provenance["official_source_manifest_sha256"] == sha256_file(source_manifest)
    assert provenance["adapter_source_sha256"] == sha256_file(adapter_source)


def test_embedding_cache_key_reflects_source_and_adapter_hash(tmp_path: Path) -> None:
    embeddings = {0: np.zeros(4, dtype=np.float32)}
    manifest = tmp_path / "cache_manifest.json"
    metadata = {
        "encoder_id": "papagei_s_frozen_official_common_input_v1",
        "checkpoint_sha256": "checkpoint-a",
        "official_source_git_commit_sha": "source-a",
        "official_source_archive_sha256": None,
        "adapter_source_sha256": "adapter-a",
        "input_protocol_id": "COMMON_PPG_10S_125HZ_V1",
        "protocol_id": "SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_V2",
        "normalization_policy": "per_window_zscore_common_dataloader",
    }
    save_embedding_cache(path=tmp_path / "cache.npz", manifest_path=manifest, embeddings=embeddings, metadata=metadata)
    text = manifest.read_text()
    assert "official_source_git_commit_sha" in text
    assert "adapter_source_sha256" in text


def test_official_runner_refuses_existing_result_without_overwrite(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    result_root.mkdir()
    (result_root / "validation_scores.csv").write_text("exists\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pass --overwrite"):
        ensure_scientific_result_root_available(result_root, overwrite=False)


def test_safe_checkpoint_load_fallback_requires_verified_checksum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fake")
    calls = {"count": 0}

    def fake_load(path, map_location=None, weights_only=None):
        calls["count"] += 1
        if weights_only is True:
            raise TypeError("weights_only unsupported")
        return {"weight": torch.zeros(1)}

    monkeypatch.setattr("setup_papagei_model_reference.torch.load", fake_load)
    payload, method = safe_torch_load(checkpoint, checkpoint_verified=True)
    assert payload["weight"].shape == (1,)
    assert method == "fallback_full_load_after_md5_verification"
    assert calls["count"] == 2


def test_run_official_baseline_automatically_enables_verification() -> None:
    script = Path("evaluation/SigD/scripts/run_frozen_baseline_pipeline.sh").read_text(encoding="utf-8")
    assert 'if [[ "$RUN_OFFICIAL" == true ]]; then' in script
    assert "VERIFY_OFFICIAL=true" in script


def _minimal_eval_config() -> dict:
    return {
        "encoder": {
            "official_source_dir": "evaluation/SigD/official_reference/PaPaGei_Model/source/papagei-foundation-model",
            "checkpoint_path": "evaluation/SigD/official_reference/PaPaGei_Model/weights/papagei_s.pt",
        }
    }


def _write_ready_manifest(root: Path) -> Path:
    path = root / "evaluation" / "SigD" / "metadata" / "papagei_model_reference_manifest.json"
    write_json(
        path,
        {
            "ready_for_scientific_frozen_baseline": True,
            "official_source_verified": True,
            "checkpoint_verified": True,
            "pretrained_weights_verified": True,
            "architecture_verified": True,
            "loading_api_verified": True,
            "embedding_dim_verified": True,
            "missing_items": [],
        },
    )
    return path


def test_plain_inspection_does_not_downgrade_verified_readiness_manifest(tmp_path: Path) -> None:
    manifest = _write_ready_manifest(tmp_path)
    before = sha256_file(manifest)
    args = argparse.Namespace(download_official_assets=False, overwrite=False, verify=False)
    result = inspect_or_setup(tmp_path, _minimal_eval_config(), args)
    after = sha256_file(manifest)
    assert before == after
    assert result["combined"]["ready_for_scientific_frozen_baseline"] is True
    assert result["verified_readiness_manifest_preserved"]["value"] is True


def test_plain_inspection_writes_separate_local_inspection_manifest(tmp_path: Path) -> None:
    _write_ready_manifest(tmp_path)
    args = argparse.Namespace(download_official_assets=False, overwrite=False, verify=False)
    inspect_or_setup(tmp_path, _minimal_eval_config(), args)
    local_manifest = tmp_path / "evaluation" / "SigD" / "metadata" / "papagei_local_inspection_manifest.json"
    assert local_manifest.exists()
    payload = local_manifest.read_text(encoding="utf-8")
    assert "readiness_manifest_authoritative" in payload


def test_verified_baseline_path_remains_ready_after_pipeline_preflight(tmp_path: Path) -> None:
    manifest = _write_ready_manifest(tmp_path)
    args = argparse.Namespace(download_official_assets=False, overwrite=False, verify=False)
    inspect_or_setup(tmp_path, _minimal_eval_config(), args)
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["ready_for_scientific_frozen_baseline"] is True
    assert data["missing_items"] == []
