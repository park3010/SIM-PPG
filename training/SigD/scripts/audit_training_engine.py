#!/usr/bin/env python3
"""Audit E4/E5 PaPaGei-S projection-head adaptation correctness."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adaptation_evaluator import evaluate_validation_only  # noqa: E402
from backbone_feature_cache import (  # noqa: E402
    ensure_backbone_cache,
    load_backbone_cache,
    role_paths,
    verify_cache_payload,
)
from common import (  # noqa: E402
    DEFAULT_CS_CONFIG,
    DEFAULT_GENERIC_CONFIG,
    add_data_pipeline_src,
    detect_project_root,
    ensure_dir,
    load_training_config,
    load_yaml_config,
    resolve_from_root,
    set_random_seed,
    utc_now_iso,
    write_json,
)
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402
from positive_masks import build_positive_mask, validate_positive_mask  # noqa: E402
from supervised_contrastive_loss import supervised_contrastive_loss  # noqa: E402


def compare_configs(generic: dict[str, Any], cs: dict[str, Any]) -> dict[str, Any]:
    """Check E4/E5 configs differ only in allowed fields."""

    allowed = {
        ("experiment_id",),
        ("experiment_stage",),
        ("training", "sampler_mode"),
        ("training", "positive_mask_mode"),
        ("output", "result_root"),
        ("fairness", "compared_against"),
    }
    differences: list[str] = []

    def walk(a: Any, b: Any, path: tuple[str, ...] = ()) -> None:
        if path in allowed:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), (*path, str(key)))
        elif a != b:
            differences.append(".".join(path))

    walk(generic, cs)
    return {
        "passed": len(differences) == 0,
        "differences_outside_allowed_fields": differences,
        "allowed_differences": [".".join(path) for path in sorted(allowed)],
    }


def _build_train_components(root: Path, config: dict[str, Any]):
    add_data_pipeline_src(root)
    from collate import train_collate_fn
    from common_window_dataset import CommonPPGWindowDataset
    from manifest_index import ManifestIndex
    from session_aware_batch_sampler import SessionAwareBatchSampler
    from train_subject_pool import TrainSubjectPool
    from transforms import PerWindowZScore

    dp_path = resolve_from_root(root, config["input"]["common_data_pipeline_config"])
    assert dp_path is not None
    dp_config = load_yaml_config(dp_path)
    manifest_index = ManifestIndex(root, dp_config)
    transform = PerWindowZScore()
    train_pool = TrainSubjectPool(root, dp_config, manifest_index)
    dataset = CommonPPGWindowDataset(manifest_index, transform=transform, index_mode="array_index")
    sampler = SessionAwareBatchSampler(
        train_pool,
        mode=config["training"]["sampler_mode"],
        seed=int(config["seed"]),
        subjects_per_batch=int(config["training"]["subjects_per_batch"]),
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=train_collate_fn)
    return next(iter(loader)), train_pool


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def _max_parameter_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    deltas = []
    for name, parameter in module.named_parameters():
        if name in before:
            deltas.append(float((before[name] - parameter.detach().cpu()).abs().max().item()))
    return max(deltas) if deltas else 0.0


def audit_one_mode(root: Path, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Audit one adaptation mode with a real frozen backbone."""

    set_random_seed(int(config["seed"]))
    train_cache_manifest = ensure_backbone_cache(root=root, train_config=config, role="train", device=device)
    validation_cache_manifest = ensure_backbone_cache(root=root, train_config=config, role="validation_exhaustive", device=device)
    batch, train_pool = _build_train_components(root, config)
    model = PaPaGeiProjectionModel(root, config).to(device)
    train_indices, train_embeddings, train_manifest = load_backbone_cache(root, "train")
    train_position = {int(index): pos for pos, index in enumerate(train_indices.tolist())}
    backbone_before = _parameter_snapshot(model.backbone)
    projection_before = _parameter_snapshot(model.projection_head)
    waveforms = batch["waveforms"].to(device)
    batch_backbone = torch.from_numpy(
        __import__("numpy").stack(
            [train_embeddings[train_position[int(index)]] for index in batch["array_indices"].tolist()],
            axis=0,
        ).astype("float32")
    ).to(device)
    model.eval()
    with torch.inference_mode():
        direct_projection = model.encode(waveforms)
        cached_projection = model.project(batch_backbone)
    cached_equivalence_max_abs_diff = float((direct_projection - cached_projection).abs().max().detach().cpu().item())
    model.train(True)
    embeddings = model.project(batch_backbone)
    positive_mask = build_positive_mask(config["training"]["positive_mask_mode"], batch["subject_ids"], batch["session_ids"])
    mask_validation = validate_positive_mask(positive_mask)
    loss, diagnostics = supervised_contrastive_loss(
        embeddings,
        positive_mask.to(device),
        float(config["training"]["temperature"]),
        return_diagnostics=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    validation = evaluate_validation_only(
        root=root,
        train_config=config,
        model=model,
        device=device,
        max_trials=256,
    )
    train_verify = verify_cache_payload("train", train_indices, train_embeddings, train_manifest)
    train_cache_non_train_indices = [
        int(index)
        for index in train_indices.tolist()
        if train_pool.manifest_index.get_metadata(int(index))["subject_id"] not in train_pool.train_subject_set
    ]
    val_indices, val_embeddings, val_manifest = load_backbone_cache(root, "validation_exhaustive")
    val_verify = verify_cache_payload("validation_exhaustive", val_indices, val_embeddings, val_manifest)
    return {
        "experiment_id": config["experiment_id"],
        "sampler_mode": config["training"]["sampler_mode"],
        "positive_mask_mode": config["training"]["positive_mask_mode"],
        "embedding_shape": list(embeddings.shape),
        "loss_finite": torch.isfinite(loss).item(),
        "loss": float(loss.detach().cpu()),
        "positive_mask": mask_validation,
        "positive_pair_count": diagnostics["positive_pair_count"],
        "duplicate_array_index_count": len(batch["array_indices"].tolist()) - len(set(batch["array_indices"].tolist())),
        "cached_training_step_used": True,
        "cached_equivalence_max_abs_diff": cached_equivalence_max_abs_diff,
        "backbone_parameter_max_delta_after_step": _max_parameter_delta(backbone_before, model.backbone),
        "projection_parameter_max_delta_after_step": _max_parameter_delta(projection_before, model.projection_head),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "backbone_cache": {
            "train_cache_path": str(role_paths(root, "train")[0].relative_to(root)),
            "validation_cache_path": str(role_paths(root, "validation_exhaustive")[0].relative_to(root)),
            "train_cache_manifest": train_cache_manifest,
            "validation_cache_manifest": validation_cache_manifest,
            "train_cache_verification": train_verify,
            "validation_cache_verification": val_verify,
            "train_cache_non_train_index_count": len(train_cache_non_train_indices),
        },
        "validation_only_smoke": {
            "trial_count": validation["split_summary"]["trial_count"],
            "validation_exhaustive_eer": validation["validation_exhaustive_eer"],
            "test_data_read": validation["test_data_read"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--generic-config", default=None)
    parser.add_argument("--cs-config", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    generic = load_training_config(root, args.generic_config or root / DEFAULT_GENERIC_CONFIG)
    cs = load_training_config(root, args.cs_config or root / DEFAULT_CS_CONFIG)
    requested = args.device
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
    config_audit = compare_configs(generic, cs)
    generic_audit = audit_one_mode(root, copy.deepcopy(generic), device)
    cs_audit = audit_one_mode(root, copy.deepcopy(cs), device)
    errors: list[str] = []
    if not config_audit["passed"]:
        errors.append("fairness_config_mismatch")
    for name, payload, expected_positive_count in (
        ("generic", generic_audit, 3),
        ("cross_session", cs_audit, 2),
    ):
        if payload["embedding_shape"] != [32, 128]:
            errors.append(f"{name}_embedding_shape_mismatch")
        if not payload["loss_finite"]:
            errors.append(f"{name}_loss_nonfinite")
        if payload["duplicate_array_index_count"] != 0:
            errors.append(f"{name}_duplicate_array_index")
        if set(payload["positive_mask"]["positive_counts"]) != {expected_positive_count}:
            errors.append(f"{name}_positive_count_mismatch")
        if payload["backbone_parameter_max_delta_after_step"] != 0.0:
            errors.append(f"{name}_backbone_changed")
        if payload["projection_parameter_max_delta_after_step"] <= 0.0:
            errors.append(f"{name}_projection_not_updated")
        if payload["cached_equivalence_max_abs_diff"] > 1.0e-5:
            errors.append(f"{name}_cached_equivalence_failed")
        if payload["backbone_cache"]["train_cache_verification"]["array_index_count"] != 12219:
            errors.append(f"{name}_train_cache_count_mismatch")
        if payload["backbone_cache"]["train_cache_non_train_index_count"] != 0:
            errors.append(f"{name}_train_cache_leakage")
        if payload["backbone_cache"]["validation_cache_verification"]["array_index_count"] != 2740:
            errors.append(f"{name}_validation_cache_count_mismatch")
        if payload["validation_only_smoke"]["test_data_read"]:
            errors.append(f"{name}_validation_read_test_data")
    if generic_audit["trainable_parameter_count"] != cs_audit["trainable_parameter_count"]:
        errors.append("trainable_parameter_count_mismatch")
    summary = {
        "generated_datetime_utc": utc_now_iso(),
        "device": str(device),
        "config_comparison": config_audit,
        "generic_supcon": generic_audit,
        "cross_session_supcon": cs_audit,
        "passed": len(errors) == 0,
        "errors": errors,
    }
    path = root / "training" / "SigD" / "metadata" / "training_engine_audit_summary.json"
    write_json(path, summary)
    print(
        f"training_engine_audit_passed={summary['passed']} "
        f"generic_loss={generic_audit['loss']:.6f} cs_loss={cs_audit['loss']:.6f}"
    )
    if errors:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
