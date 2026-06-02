# SigD Common 10s Verification Protocol

## Overview

This directory builds the canonical subject-disjoint split and verification trial protocol for `COMMON_PPG_10S_125HZ_V1`. The protocol is the shared common-input evaluation surface for SIM-PPG, PaPaGei-S adaptation objectives, PaPaGei-P, Pulse-PPG common adapter, SIGMA-PPG common adapter, MTL-RAPID common setting, and SigD CNN-LSTM baseline.

The preprocessing snapshot is read-only. Protocol eligibility uses only `common_input_available` from `preprocessing_manifest_10s.csv`; sVRI, SQI, IPA, or any morphology validity mask is never used to include or exclude protocol windows.

## Primary Protocol

- `protocol_id`: `SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_V2`
- `input_protocol_id`: `COMMON_PPG_10S_125HZ_V1`
- Split: subject-disjoint train/validation/test with seed 42
- Subject counts: train 139, validation 46, test 46
- Enrollment: each subject's earliest valid session
- Enrollment template size: `K=5`
- Genuine probe: same subject, later valid session windows only
- Impostor probe: different subject, probe subject's later valid session windows only
- Probe aggregation unit: `M=1`
- Impostor sampling: split-internal only
- Impostor ratio: train 1:1, validation 1:5, test 1:5 relative to genuine trials

Genuine and impostor trials share the same later-session probe condition. This avoids temporal-condition mismatch between genuine and impostor trials.

Validation threshold selection must use validation split only. Test split labels and scores must not be used for threshold tuning.

## Deprecated V1

The earlier V1 protocol is preserved under:

```text
protocol/SigD/metadata/deprecated_v1_unrestricted_impostor_probe/
```

V1 correctly implemented the subject-disjoint split and genuine later-session probes. However, its impostor probe pool sampled from all common-input windows inside each split, so the probe subject's earliest enrollment-session windows could appear as impostor probes. V1 is therefore deprecated for cross-session primary evaluation. V2 restricts impostor probes to later-session windows for the probe subject.

## Outputs

```text
protocol/SigD/metadata/subject_split_seed42.csv
protocol/SigD/metadata/subject_split_summary_seed42.json
protocol/SigD/metadata/enrollment_templates_k5_seed42.csv
protocol/SigD/metadata/genuine_trials_k5m1_seed42.csv
protocol/SigD/metadata/impostor_trials_k5m1_seed42.csv
protocol/SigD/metadata/verification_trials_k5m1_seed42.csv
protocol/SigD/metadata/protocol_summary_k5m1_seed42.json
```

`verification_trials_k5m1_seed42.csv` is the canonical window-level evaluation trial list for validation/test reporting and frozen baseline comparison.

## Exhaustive Evaluation V2

The sampled V2 protocol remains available for development, debugging, and ablation loops. Final scientific reporting should use the separate exhaustive evaluation protocol:

```text
SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_EXHAUSTIVE_EVAL_V2
```

It reuses the same subject split, earliest-session `K=5` enrollment templates, and genuine later-session probes. For validation and test only, every later-session probe window is paired with every other subject's enrollment template inside the same split. Genuine and impostor probes therefore keep the same later-session temporal condition.

Outputs are written under:

```text
protocol/SigD/metadata/exhaustive_eval_v2/
```

The sampled V2 files in `protocol/SigD/metadata/` are read-only inputs for this step and are not overwritten.

## Training Sampler vs Evaluation Trial List

The fixed verification trials are not intended to constrain all contrastive training pairs. SIM-PPG, Generic SupCon, PaPaGei-S/P adaptation, and related training code may use a separate session-aware dynamic sampler over the train split subject/window pool.

The training sampler must use train subjects only and must not inspect validation/test trials, scores, or threshold decisions. Fixed train verification trials may be used for development evaluation or sanity checks, but they are not the full definition of the contrastive training pair space.

## Evaluation Reporting

The default evaluation is window-level operational verification using the canonical trial list. Because subjects or sessions with many windows can dominate pooled metrics, final reporting should also include:

- session-pair macro metrics
- subject-macro metrics
- time-gap-stratified EER/TAR

Thresholds are selected on the validation split only. Test threshold tuning is prohibited.

## Running

```bash
bash protocol/SigD/scripts/run_sigd_protocol_pipeline.sh --overwrite
```

The script runs subject split generation, verification protocol construction, and leakage/protocol audit. Without `--overwrite`, existing protocol files are not replaced.

Final exhaustive validation/test protocol generation:

```bash
bash protocol/SigD/scripts/run_sigd_exhaustive_evaluation_protocol.sh --overwrite
```

This command does not rerun preprocessing, does not rebuild the sampled V2 split/protocol, and does not touch the train-time sampler.

## Audit

The audit checks:

- train/validation/test subject overlap is zero
- every trial references subjects inside its own split
- genuine trials use the same subject and a later session
- impostor trials use a different subject and the probe subject's later session
- `probe_time_gap_days` is present, positive, and bucketed consistently
- enrollment and probe array indices are inside the common array range
- enrollment templates have exactly `K=5` windows
- impostor:genuine ratios match the config
- morphology validity is not used for protocol eligibility

## Supplementary Native Setting

Model-specific native inputs or model-specific heads must use separate supplementary protocols. They must not overwrite or silently reinterpret this common trial list.
