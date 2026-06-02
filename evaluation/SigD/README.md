# SigD Verification Evaluation Engine

## Purpose

This directory implements the common verification evaluation engine for the canonical SigD common-input protocol:

- input protocol: `COMMON_PPG_10S_125HZ_V1`
- protocol: `SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_V2`
- baseline target: `PaPaGei-S Frozen + Cosine Similarity`

The engine is model-facing infrastructure only. It does not implement fine-tuning, projection-head learning, contrastive loss, morphology loss, SQI weighting, optimizers, or training loops.

## Evaluation Flow

The fixed evaluation path is:

1. Load canonical `VerificationTrialDataset` from `data_pipeline/SigD`.
2. Apply the common per-window z-score transform.
3. Encode each unique common-array window once.
4. Build a K=5 enrollment template by L2-normalizing each enrollment embedding, averaging, and L2-normalizing the mean template.
5. L2-normalize the M=1 probe embedding.
6. Score with cosine similarity.
7. Select thresholds on validation only.
8. Apply validation-fixed thresholds to test.

The encoder receives tensors shaped `[B, 1, 1250]`. Trial items expose enrollment tensors `[5, 1, 1250]` and probe tensors `[1, 1250]`.

## Frozen Baseline Definition

The PaPaGei-S frozen baseline means:

- verified pretrained PaPaGei-S encoder weights
- no fine-tuning
- no trainable projection head
- no task-specific adaptation
- cosine similarity over K=5 mean enrollment templates and M=1 probes

Random initialization is forbidden. A random or unverified encoder result must not be saved or reported as `PaPaGei-S Frozen`.

## Scientific Reporting Guardrail

`MockEncoder` exists only to audit evaluator correctness. Its outputs are stored under:

```text
evaluation/SigD/results/mock_encoder_engine_audit/
```

Mock metrics are marked:

```text
scientific_reporting_allowed: false
```

The official PaPaGei-S baseline script checks `papagei_model_reference_manifest.json`. If the official source or pretrained checkpoint is missing or unverified, the baseline is skipped and no scientific result is produced.

## Official PaPaGei Assets

Official assets are stored as runtime references, not modified project code:

```text
evaluation/SigD/official_reference/PaPaGei_Model/source/papagei-foundation-model/
evaluation/SigD/official_reference/PaPaGei_Model/weights/papagei_s.pt
```

`setup_papagei_model_reference.py` has three modes:

- default inspection: local file checks only, no network access
- `--download-official-assets`: explicit user-triggered download of the official GitHub source snapshot and Zenodo `papagei_s.pt`
- `--verify`: source import, architecture instantiation, checkpoint MD5/SHA256 validation, strict checkpoint loading, and forward smoke test

The expected Zenodo MD5 for `papagei_s.pt` is:

```text
a4cdb32392e2a7b25999128af92813b5
```

Only after source verification, checkpoint verification, strict loading, frozen-parameter verification, and `[1, 512]` forward smoke output pass can `ready_for_scientific_frozen_baseline` become `true`.

## Common-Input Policy

This primary baseline uses common input, not PaPaGei native raw preprocessing:

```text
input_setting: common_input
official_native_preprocessing_reapplied: false
common_transform_source: data_pipeline/SigD/PerWindowZScore
native_input_supplementary_evaluation_pending: true
```

The adapter does not call `preprocess_one_ppg_signal`, segmentation, resampling, pyPPG filtering, or extra normalization. Native PaPaGei preprocessing belongs only in a future supplementary native-input setting.

## Metrics

The engine computes:

- ROC-AUC
- diagnostic EER
- validation-fixed FAR/FRR/TAR at the validation EER threshold
- TAR at validation FAR = 1%
- time-gap-stratified metrics
- subject-macro summaries
- session-pair macro score tables

Diagnostic test EER may be written as a score-distribution statistic, but operational accept/reject reporting must use thresholds selected from validation only. Test threshold tuning is prohibited.

## Fairness Notes

All primary comparison models should share:

- the common input array
- the canonical Protocol V2 trial list for sampled development runs
- the exhaustive Protocol V2 trial list for final validation/test reporting
- the same per-window z-score dataloader transform
- validation-only threshold selection

External pretraining resources must be reported. PaPaGei pretraining and SigD-Core both derive from clinical waveform sources, so source-level overlap limitations must be acknowledged when interpreting frozen foundation-model results.

Important limitation: PaPaGei pretraining includes MIMIC-III, while SigD-Core is reconstructed from MIMIC-III waveform records. Frozen absolute performance must therefore be interpreted with source-level overlap caution. Same-backbone PaPaGei-S adaptation ablations remain useful as method-level controlled comparisons because they share the same pretrained exposure.

## Environment Notes

Common preprocessing was already completed outside this evaluation stage. For official PaPaGei-S frozen inference, use a separate environment such as `papagei_eval` rather than changing the existing preprocessing environment.

Recommended setup:

- Python 3.10
- official PaPaGei repository requirements
- project dependencies needed for data loading: torch, numpy, pandas, pyyaml

The official adapter consumes already-normalized common-input tensors, but importing `ResNet1DMoE` may still require dependencies from the official repository.

## Current Scope Exclusions

This directory does not implement:

- trainable heads
- Generic SupCon
- cross-session SupCon
- session centroid alignment
- morphology preservation loss
- SQI quality weighting
- PaPaGei-P, Pulse-PPG, SIGMA-PPG, MTL-RAPID, or SIM-PPG adapters

## Commands

Mock-only evaluator audit:

```bash
bash evaluation/SigD/scripts/run_frozen_baseline_pipeline.sh --audit-only
```

Official PaPaGei-S baseline, after verified checkpoint setup:

```bash
bash evaluation/SigD/scripts/run_frozen_baseline_pipeline.sh \
  --download-official-assets \
  --verify-official-assets \
  --audit-only

bash evaluation/SigD/scripts/run_frozen_baseline_pipeline.sh --run-official-baseline
```

If the verified checkpoint is absent, the official path records a skip manifest instead of reporting a baseline score.

Final exhaustive PaPaGei-S frozen evaluation uses a separate config and result root:

```bash
bash protocol/SigD/scripts/run_sigd_exhaustive_evaluation_protocol.sh --overwrite

bash evaluation/SigD/scripts/run_frozen_baseline_pipeline.sh \
  --config evaluation/SigD/config/papagei_s_frozen_cosine_exhaustive_eval.yaml \
  --run-official-baseline
```

Sampled results remain useful for development and ablation iteration. Exhaustive results under `evaluation/SigD/results/papagei_s_frozen_cosine_exhaustive_eval/seed42/` are the final-reporting baseline path.
