# SigD Common Data Pipeline

## Purpose

This directory provides the model-independent Dataset, DataLoader, and dynamic training sampler layer for `COMMON_PPG_10S_125HZ_V1` and the canonical V2 verification protocol. It is shared by all common-input primary comparison models: PaPaGei-S frozen/adaptation, PaPaGei-P, Pulse-PPG common adapter, SIGMA-PPG common adapter, MTL-RAPID common setting, SigD CNN-LSTM, and SIM-PPG.

This layer does not implement encoders, projection heads, losses, optimizers, threshold selection, metric calculation, training loops, or native-input supplementary protocols.

## Read-Only Inputs

The pipeline reads these fixed assets:

```text
preprocessing/SigD/data/windows_10s/ppg_filtered_windows_10s_125hz.npy
preprocessing/SigD/metadata/final_common_preprocessing_snapshot_10s/
protocol/SigD/metadata/verification_trials_k5m1_seed42.csv
protocol/SigD/metadata/enrollment_templates_k5_seed42.csv
protocol/SigD/metadata/subject_split_seed42.csv
```

The common array is loaded with `numpy.load(..., mmap_mode="r")`. Preprocessing and protocol generation scripts are not called by this data pipeline.

## Evaluation Dataset

`VerificationTrialDataset` loads the canonical fixed trial list. Each item returns:

- `enrollment_windows`: `[5, 1, 1250]`
- `probe_window`: `[1, 1250]`
- label, trial type, subject/session ids, protocol id, and probe time-gap metadata

The dataset does not compute embeddings, template means, cosine similarity, thresholds, or metrics. Those belong to the later model/evaluator stage.

The dataset can load either the sampled V2 protocol or the final exhaustive evaluation V2 protocol from config. The exhaustive config uses the same common array, same per-window transform, and same enrollment template format, but reads validation/test trial CSVs from:

```text
protocol/SigD/metadata/exhaustive_eval_v2/
```

Training pool and dynamic sampler behavior remain tied to the train split and are not changed by the exhaustive evaluation config.

## Training Pool and Dynamic Sampler

`TrainSubjectPool` contains only train split subjects and common-input windows. It does not use the fixed evaluation trial CSV as the full training pair space.

`SessionAwareBatchSampler` supports:

- `same_subject_cross_session`: SIM-PPG / cross-session SupCon mode. Each batch selects 8 subjects, 2 distinct sessions per subject, and 2 windows per session for a 32-window batch.
- `same_subject_any_session`: Generic SupCon baseline mode with the same batch budget but without a distinct-session constraint. It samples distinct waveform windows within each selected subject and never duplicates the same common-array index inside the batch.

Training samplers must use train subjects only and must not inspect validation/test trials, scores, or threshold decisions.

When using the sampler as a PyTorch `batch_sampler`, initialize `CommonPPGWindowDataset` with `index_mode="array_index"` because the sampler yields common-array row indices directly.

Sampler mode and loss semantics are intentionally separate. `same_subject_cross_session` guarantees that a batch contains same-user, different-session samples, but the CS-SupCon objective must still define positives and negatives in the loss layer:

- Generic SupCon positive: same subject, different sample
- CS-SupCon positive: same subject, different session
- Negative: different subject

The sampler alone does not complete the CS-SupCon objective; it only provides the batch structure needed for a fair comparison.

## Normalization

Stored preprocessing arrays are not z-score normalized. `PerWindowZScore` applies the common per-window z-score transform at dataloader time for every primary comparison model. Native-input supplementary settings must use a separate pipeline.

## Morphology Annotations

SQI/skewness and sVRI are primary auxiliary target candidates. IPA is a masked optional auxiliary target and should only contribute where `ipa_valid_mask=True`.

Target masks are independent:

- `sqi_valid_mask`
- `svri_valid_mask`
- `ipa_valid_mask`

Morphology validity is not used for sample inclusion, protocol eligibility, or training sampler filtering.

Invalid auxiliary targets may contain `NaN` values and must be handled with their target-specific masks. Later morphology losses should index or select valid targets before reduction; a `NaN * 0` masked reduction is not safe. The model/loss layer should provide a `MaskedRegressionLoss` utility before SIM-PPG training begins.

## Audit

Run:

```bash
bash data_pipeline/SigD/scripts/run_sigd_data_pipeline_audit.sh
```

The audit checks array shape/dtype, finite values, manifest-index consistency, evaluation dataset lengths and tensor shapes, train-pool leakage, dynamic sampler structure, deterministic epoch behavior, and morphology mask availability.
