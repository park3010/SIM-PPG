"""Training loop for PaPaGei-S projection-head adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adaptation_evaluator import evaluate_validation_only
from backbone_feature_cache import (
    CachedBackboneEmbeddingDataset,
    cached_backbone_collate_fn,
    role_paths,
)
from checkpointing import EarlyStopping, save_projection_checkpoint
from common import add_data_pipeline_src, ensure_dir, load_yaml_config, resolve_from_root, set_random_seed, utc_now_iso, write_csv_rows, write_json
from papagei_projection_model import PaPaGeiProjectionModel
from objective_registry import compute_total_objective, lambda_align_from_config
from morphology_objective import compute_total_loss_with_morphology


class AdaptationTrainer:
    """Train only the projection head on top of frozen PaPaGei-S."""

    def __init__(
        self,
        root: Path,
        config: dict[str, Any],
        *,
        device: torch.device | None = None,
        model: PaPaGeiProjectionModel | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_random_seed(int(config["seed"]))
        self.dp_config, self.manifest_index, self.transform, self.train_pool, self.dataset = self._build_train_data()
        self.model = model or PaPaGeiProjectionModel(root, config)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        early_cfg = config["training"]["early_stopping"]
        self.early_stopping = EarlyStopping(
            patience=int(early_cfg["patience"]),
            min_delta=float(early_cfg["min_delta"]),
            mode=str(early_cfg["mode"]),
        )

    def _build_train_data(self):
        add_data_pipeline_src(self.root)
        from collate import train_collate_fn
        from common_window_dataset import CommonPPGWindowDataset
        from manifest_index import ManifestIndex
        from session_aware_batch_sampler import SessionAwareBatchSampler
        from train_subject_pool import TrainSubjectPool
        from transforms import PerWindowZScore

        dp_path = resolve_from_root(self.root, self.config["input"]["common_data_pipeline_config"])
        assert dp_path is not None
        dp_config = load_yaml_config(dp_path)
        manifest_index = ManifestIndex(self.root, dp_config)
        transform = PerWindowZScore(
            eps=float(dp_config["normalization"]["epsilon"]),
            output_channel_first=bool(dp_config["normalization"]["output_channel_first"]),
        )
        train_pool = TrainSubjectPool(self.root, dp_config, manifest_index)
        cache_path, cache_manifest_path = role_paths(self.root, "train")
        if cache_path.exists() and cache_manifest_path.exists():
            dataset = CachedBackboneEmbeddingDataset(self.root, "train", manifest_index, index_mode="array_index")
            self._collate_fn = cached_backbone_collate_fn
            self._uses_cached_backbone_embeddings = True
        else:
            dataset = CommonPPGWindowDataset(manifest_index, transform=transform, index_mode="array_index")
            self._collate_fn = train_collate_fn
            self._uses_cached_backbone_embeddings = False
        self._sampler_class = SessionAwareBatchSampler
        return dp_config, manifest_index, transform, train_pool, dataset

    def _batch_loader(self, epoch: int, num_batches: int | None = None) -> DataLoader:
        sampler = self._sampler_class(
            self.train_pool,
            mode=self.config["training"]["sampler_mode"],
            seed=int(self.config["seed"]),
            subjects_per_batch=int(self.config["training"]["subjects_per_batch"]),
            sessions_per_subject=2,
            windows_per_session=2,
            num_batches_per_epoch=int(num_batches or self.config["training"]["num_batches_per_epoch"]),
        )
        sampler.set_epoch(epoch)
        return DataLoader(self.dataset, batch_sampler=sampler, collate_fn=self._collate_fn)

    def train_one_epoch(self, epoch: int, num_batches: int | None = None) -> dict[str, Any]:
        """Train one epoch and return summary."""

        self.model.train(True)
        total_losses: list[float] = []
        supcon_losses: list[float] = []
        alignment_losses: list[float] = []
        svri_losses: list[float] = []
        sqi_losses: list[float] = []
        weighted_supcon_losses: list[float] = []
        unweighted_supcon_losses: list[float] = []
        sqi_weight_mins: list[float] = []
        sqi_weight_maxs: list[float] = []
        sqi_weight_means: list[float] = []
        sqi_weight_stds: list[float] = []
        svri_valid_counts: list[int] = []
        sqi_valid_counts: list[int] = []
        pos_sims: list[float] = []
        neg_sims: list[float] = []
        centroid_cosines: list[float] = []
        positive_pair_counts: list[int] = []
        for batch in self._batch_loader(epoch, num_batches):
            if "backbone_embeddings" in batch:
                embeddings = self.model.project(batch["backbone_embeddings"].to(self.device))
            else:
                waveforms = batch["waveforms"].to(self.device)
                embeddings = self.model.encode(waveforms)
            if morphology_enabled(self.config):
                predictions = self.model.predict_morphology(embeddings)
                loss, diagnostics = compute_total_loss_with_morphology(
                    embeddings,
                    batch["subject_ids"],
                    batch["session_ids"],
                    predictions,
                    batch,
                    self.config,
                )
            else:
                loss, diagnostics = compute_total_objective(
                    embeddings,
                    batch["subject_ids"],
                    batch["session_ids"],
                    self.config,
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                float(self.config["training"]["gradient_clip_norm"]),
            )
            self.optimizer.step()
            total_losses.append(float(diagnostics["total_loss"]))
            supcon_losses.append(float(diagnostics["supcon_loss"]))
            alignment_losses.append(float(diagnostics["alignment_loss"]))
            svri_losses.append(float(diagnostics.get("svri_loss", 0.0)))
            sqi_losses.append(float(diagnostics.get("sqi_loss", 0.0)))
            weighted_supcon_losses.append(float(diagnostics.get("weighted_supcon_loss", diagnostics["supcon_loss"])))
            unweighted_supcon_losses.append(float(diagnostics.get("unweighted_supcon_loss", diagnostics["supcon_loss"])))
            if diagnostics.get("sqi_weight_min") is not None:
                sqi_weight_mins.append(float(diagnostics["sqi_weight_min"]))
                sqi_weight_maxs.append(float(diagnostics["sqi_weight_max"]))
                sqi_weight_means.append(float(diagnostics["sqi_weight_mean"]))
                sqi_weight_stds.append(float(diagnostics["sqi_weight_std"]))
            svri_valid_counts.append(int(diagnostics.get("svri_valid_count", 0)))
            sqi_valid_counts.append(int(diagnostics.get("sqi_valid_count", 0)))
            pos_sims.append(float(diagnostics["mean_positive_similarity"]))
            if diagnostics["mean_negative_similarity"] is not None:
                neg_sims.append(float(diagnostics["mean_negative_similarity"]))
            if diagnostics.get("mean_centroid_cosine") is not None:
                centroid_cosines.append(float(diagnostics["mean_centroid_cosine"]))
            positive_pair_counts.append(int(diagnostics["positive_pair_count"]))
        return {
            "epoch": int(epoch),
            "mean_train_loss": float(np.mean(total_losses)),
            "train_total_loss": float(np.mean(total_losses)),
            "train_supcon_loss": float(np.mean(supcon_losses)),
            "train_weighted_supcon_loss": float(np.mean(weighted_supcon_losses)),
            "train_unweighted_supcon_loss": float(np.mean(unweighted_supcon_losses)),
            "train_alignment_loss": float(np.mean(alignment_losses)),
            "train_svri_loss": float(np.mean(svri_losses)),
            "train_sqi_loss": float(np.mean(sqi_losses)),
            "train_svri_valid_count": float(np.mean(svri_valid_counts)),
            "train_sqi_valid_count": float(np.mean(sqi_valid_counts)),
            "train_mean_positive_similarity": float(np.mean(pos_sims)),
            "train_mean_negative_similarity": float(np.mean(neg_sims)) if neg_sims else None,
            "train_mean_centroid_cosine": float(np.mean(centroid_cosines)) if centroid_cosines else None,
            "lambda_align": lambda_align_from_config(self.config),
            "lambda_svri": lambda_svri_from_config(self.config),
            "lambda_sqi": lambda_sqi_from_config(self.config),
            "sqi_weighting_mode": sqi_weighting_mode_from_config(self.config),
            "sqi_weight_min": float(np.mean(sqi_weight_mins)) if sqi_weight_mins else None,
            "sqi_weight_max": float(np.mean(sqi_weight_maxs)) if sqi_weight_maxs else None,
            "sqi_weight_mean": float(np.mean(sqi_weight_means)) if sqi_weight_means else None,
            "sqi_weight_std": float(np.mean(sqi_weight_stds)) if sqi_weight_stds else None,
            "mean_positive_pair_count": float(np.mean(positive_pair_counts)),
            "batch_count": len(total_losses),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "trainable_parameter_count": self.model.trainable_parameter_count(),
        }

    def fit(
        self,
        *,
        max_epochs: int | None = None,
        num_batches_per_epoch: int | None = None,
        smoke_test: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Run training; test split is never evaluated here."""

        result_root = resolve_training_result_root(self.root, self.config, smoke_test)
        protect_scientific_result_root(result_root, smoke_test=smoke_test, overwrite=overwrite)
        result_root = ensure_dir(result_root)
        checkpoints = ensure_dir(result_root / "checkpoints")
        max_epochs = int(max_epochs or self.config["training"]["max_epochs"])
        if smoke_test:
            max_epochs = min(max_epochs, 1)
            num_batches_per_epoch = int(num_batches_per_epoch or 2)
        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        best_epoch: int | None = None
        best_metric: float | None = None
        for epoch in range(max_epochs):
            train_summary = self.train_one_epoch(epoch, num_batches_per_epoch)
            train_rows.append(train_summary)
            if (epoch + 1) % int(self.config["training"]["validation_frequency_epochs"]) == 0:
                validation = evaluate_validation_only(
                    root=self.root,
                    train_config=self.config,
                    model=self.model,
                    device=self.device,
                    max_trials=512 if smoke_test else None,
                )
                metric = float(validation["validation_exhaustive_eer"])
                improved = self.early_stopping.update(metric, epoch)
                val_row = {
                    "epoch": epoch,
                    "validation_exhaustive_eer": metric,
                    "validation_tar_at_far_1pct": validation.get("validation_tar_at_far_1pct"),
                    "validation_trial_count": validation["split_summary"]["trial_count"],
                    "test_data_read": validation["test_data_read"],
                    "improved": improved,
                    "lambda_align": lambda_align_from_config(self.config),
                    "lambda_svri": lambda_svri_from_config(self.config),
                    "lambda_sqi": lambda_sqi_from_config(self.config),
                }
                val_rows.append(val_row)
                if improved:
                    best_epoch = epoch
                    best_metric = metric
                    save_projection_checkpoint(
                        checkpoints / "best_projection_head.pt",
                        self.model,
                        {"epoch": epoch, "validation_exhaustive_eer": metric, "experiment_id": self.config["experiment_id"]},
                        overwrite=True,
                    )
                if self.early_stopping.should_stop:
                    break
        save_projection_checkpoint(
            checkpoints / "last_projection_head.pt",
            self.model,
            {"epoch": train_rows[-1]["epoch"], "experiment_id": self.config["experiment_id"]},
            overwrite=True,
        )
        write_csv_rows(result_root / "train_history.csv", train_rows)
        write_csv_rows(result_root / "validation_history.csv", val_rows)
        manifest = {
            "experiment_id": self.config["experiment_id"],
            "experiment_stage": self.config["experiment_stage"],
            "seed": int(self.config["seed"]),
            "generated_datetime_utc": utc_now_iso(),
            "smoke_test": smoke_test,
            "run_mode": "smoke_test" if smoke_test else "full_training",
            "scientific_reporting_allowed": not smoke_test,
            "checkpoint_usable_for_final_evaluation": not smoke_test,
            "validation_subset_used": smoke_test,
            "best_epoch": best_epoch,
            "best_validation_exhaustive_eer": best_metric,
            "checkpoint_selection_metric": "validation_exhaustive_eer",
            "checkpoint_selection_uses_test": False,
            "test_data_read_during_training": False,
            "test_accessed_during_training": False,
            "final_test_evaluation_pending": True,
            "threshold_selection_during_training": "validation_eer_internal_for_checkpoint_selection_only",
            "cached_backbone_embeddings_used": bool(getattr(self, "_uses_cached_backbone_embeddings", False)),
            "lambda_align": lambda_align_from_config(self.config),
            "lambda_svri": lambda_svri_from_config(self.config),
            "lambda_sqi": lambda_sqi_from_config(self.config),
            "candidate_name": morphology_candidate_name_from_config(self.config),
            "morphology_heads_enabled": morphology_enabled(self.config),
            "morphology_targets": self.config.get("model", {}).get("morphology_heads", {}).get("targets", []),
            "use_ipa": bool(self.config.get("loss_components", {}).get("use_ipa", False)),
            "sqi_weighting_enabled": bool(self.config.get("loss_components", {}).get("sqi_weighting_enabled", False)),
            "sqi_weighting_mode": sqi_weighting_mode_from_config(self.config),
            "sqi_weighting_used_for_verification": False,
            "morphology_used_for_verification": False,
            "base_config_path": self.config.get("_runtime", {}).get("base_config_path"),
            "effective_result_root": _relative_to_root(self.root, result_root),
            "multi_seed_run": bool(self.config.get("_runtime", {}).get("multi_seed_run", False)),
            "fixed_e7a_candidate": bool(self.config.get("_runtime", {}).get("fixed_e7a_candidate", False)),
            "final_test_allowed_only_if_validation_beats_e7_a": bool(
                self.config.get("fairness", {}).get("final_test_allowed_only_if_validation_beats_e7_a", False)
            ),
            "loss_components": self.config.get("loss_components", {}),
            "post_e4_e5_policy_applies": bool(self.config.get("fairness", {}).get("post_e4_e5_policy_applies", False)),
            "model_metadata": self.model.get_model_metadata(),
            "early_stopping": self.early_stopping.state_dict(),
        }
        write_json(result_root / "manifest.json", manifest)
        return manifest


def resolve_training_result_root(root: Path, config: dict[str, Any], smoke_test: bool) -> Path:
    """Resolve canonical or smoke-specific result root."""

    canonical = resolve_from_root(root, config["output"]["result_root"])
    assert canonical is not None
    if config.get("_runtime", {}).get("exact_result_root", False):
        return canonical
    lambda_align = lambda_align_from_config(config)
    has_lambda_grid = bool(config.get("loss_components", {}).get("session_centroid_alignment_weight_candidates"))
    if has_lambda_grid and lambda_align > 0:
        canonical = canonical.parent / lambda_slug(lambda_align) / canonical.name
    candidate_name = morphology_candidate_name_from_config(config)
    if candidate_name:
        canonical = canonical.parent / candidate_name / canonical.name
    if not smoke_test:
        return canonical
    experiment_name = canonical.parent.name
    seed_dir = canonical.name
    if canonical.parent.name.startswith("lambda_"):
        experiment_name = canonical.parent.parent.name
        return root / "training" / "SigD" / "results" / "smoke_runs" / experiment_name / canonical.parent.name / seed_dir
    if candidate_name:
        experiment_name = canonical.parent.parent.name
        return root / "training" / "SigD" / "results" / "smoke_runs" / experiment_name / canonical.parent.name / seed_dir
    return root / "training" / "SigD" / "results" / "smoke_runs" / experiment_name / seed_dir


def lambda_slug(value: float) -> str:
    """Return a filesystem-safe lambda folder name."""

    return f"lambda_{float(value):.2f}".replace(".", "p")


def lambda_svri_from_config(config: dict[str, Any]) -> float:
    return float(config.get("loss_components", {}).get("lambda_svri", 0.0))


def lambda_sqi_from_config(config: dict[str, Any]) -> float:
    return float(config.get("loss_components", {}).get("lambda_sqi", 0.0))


def morphology_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("model", {}).get("morphology_heads", {}).get("enabled", False))


def morphology_candidate_name_from_config(config: dict[str, Any]) -> str | None:
    components = config.get("loss_components", {})
    if components.get("candidate_name"):
        return str(components["candidate_name"])
    if components.get("sqi_weighting_candidates"):
        return str(components.get("sqi_weighting_mode", "sqi_weighted"))
    if not components.get("morphology_loss_candidates"):
        return None
    return f"svri{lambda_svri_from_config(config):.2f}_sqi{lambda_sqi_from_config(config):.2f}".replace(".", "p")


def sqi_weighting_mode_from_config(config: dict[str, Any]) -> str | None:
    components = config.get("loss_components", {})
    if not components.get("sqi_weighting_enabled", False):
        return None
    return str(components.get("sqi_weighting_mode", ""))


def protect_scientific_result_root(result_root: Path, *, smoke_test: bool, overwrite: bool) -> None:
    """Refuse to overwrite full scientific training outputs without --overwrite."""

    if smoke_test:
        return
    protected = [
        result_root / "manifest.json",
        result_root / "checkpoints" / "best_projection_head.pt",
        result_root / "train_history.csv",
        result_root / "validation_history.csv",
    ]
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise RuntimeError(f"Scientific training output exists; pass --overwrite to replace: {existing[:3]}")


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
