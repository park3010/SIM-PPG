# SigD PaPaGei-S Projection Adaptation

This directory implements the first trainable adaptation layer for the common-input SigD-Core protocol.

## 연구 단계

- `E4`: PaPaGei-S + Generic SupCon
- `E5`: PaPaGei-S + Cross-Session SupCon
- `E6-Base`: Generic SupCon on cross-session batch, no alignment
- `E6-A`: CS-SupCon plus session-centroid alignment
- `E6-B`: Generic SupCon on cross-session batch plus session-centroid alignment
- `E7-A`: E4 branch plus sVRI/SQI morphology preservation
- `E7-B`: E6-Base branch plus sVRI/SQI morphology preservation

These stages start from the same verified PaPaGei-S checkpoint and keep the backbone frozen. Only the projection head is trainable.

## Controlled Comparison

E4 and E5 are designed as a strict pair-definition ablation:

- Same common input: `COMMON_PPG_10S_125HZ_V1`
- Same final protocol: `SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_EXHAUSTIVE_EVAL_V2`
- Same verified PaPaGei-S initialization
- Same frozen backbone
- Same projection head architecture
- Same trainable parameter budget
- Same optimizer, learning rate, epoch budget, and early stopping policy

For E4/E5, the intended difference is:

- E4 sampler/objective: same subject, different sample
- E5 sampler/objective: same subject, different session

E6 decomposes the E4/E5 gap:

| Model | Sampler | Positive Mask | Alignment |
| --- | --- | --- | --- |
| E4 Generic SupCon | `same_subject_any_session` | `same_subject_different_sample` | none |
| E5 CS-SupCon | `same_subject_cross_session` | `same_subject_different_session` | none |
| E6-Base | `same_subject_cross_session` | `same_subject_different_sample` | none |
| E6-A | `same_subject_cross_session` | `same_subject_different_session` | session centroid |
| E6-B | `same_subject_cross_session` | `same_subject_different_sample` | session centroid |

## Frozen Baseline 차이

The frozen baseline scores raw 512-d PaPaGei-S embeddings with cosine similarity. E4/E5 score learned 128-d projection embeddings on top of the frozen backbone. Therefore frozen-to-adapted improvement includes task-specific projection adaptation. The E4 versus E5 comparison is the controlled evidence for explicit cross-session positive definition.

## Observed E4/E5 Result

Final exhaustive test results after fixed E4/E5 configurations:

- Frozen PaPaGei-S: diagnostic EER `41.5866%`, TAR@validation FAR=1% `2.5470%`
- E4 Generic SupCon: diagnostic EER `34.8745%`, TAR@validation FAR=1% `6.3048%`
- E5 CS-SupCon: diagnostic EER `36.8611%`, TAR@validation FAR=1% `3.2150%`

Generic SupCon is currently the strongest adaptation baseline. Cross-session positives alone improve over the frozen baseline, but they do not outperform Generic SupCon in the fixed E5 setting.

## Why E6 Is Decomposed

E4 and E5 differ in both batch composition and positive-mask definition. E6-Base controls for sampler composition by using the cross-session batch with the generic positive mask. E6-A tests whether alignment helps under CS-only positives. E6-B tests whether alignment helps the stronger generic-positive branch on cross-session batches.

This prevents attributing all effects to the cross-session positive definition alone.

## Session Centroid Alignment

Session centroid alignment is computed in the learned 128-d projection space. For each subject in a cross-session batch, two session centroids are averaged from the two windows in each session, L2-normalized, and penalized with `1 - cosine(c_session1, c_session2)`.

Alignment is a session-invariance regularizer, not an identity classification loss. It is always combined with SupCon to avoid collapse. It is not applied to the frozen 512-d backbone representation.

## Smoke vs Scientific Outputs

Smoke runs are not scientific results and are written only under:

```text
training/SigD/results/smoke_runs/
```

The canonical experiment roots under `training/SigD/results/papagei_s_*` are reserved for full training. Existing smoke artifacts from earlier runs can be preserved manually:

```bash
mkdir -p training/SigD/results/smoke_runs

mv training/SigD/results/papagei_s_generic_supcon_head_only \
   training/SigD/results/smoke_runs/

mv training/SigD/results/papagei_s_cs_supcon_head_only \
   training/SigD/results/smoke_runs/
```

Smoke manifests set `scientific_reporting_allowed=false` and `checkpoint_usable_for_final_evaluation=false`.

## Frozen Backbone Cache

Because E4/E5 keep the PaPaGei-S backbone frozen and use no augmentation, the 512-d PaPaGei-S representation can be precomputed once for the shared common-input windows. The cache is a computational optimization only; it does not change the experiment definition.

Shared cache path:

```text
training/SigD/cache/papagei_s_frozen_common_input_v1/
```

Training uses train and validation caches. Test cache is only built/used by final evaluation. E4 and E5 share the same backbone cache; projection embeddings remain model-specific.

## Test Usage Policy

Training, early stopping, and checkpoint selection use validation exhaustive EER only. Test scores are generated only by `evaluate_adapted_model.py` after a checkpoint/configuration is fixed. Thresholds are selected on validation only and then applied to test.

## Post-E4/E5 Development Policy

E4/E5 final exhaustive test results were inspected after their configurations were fixed. From E6 onward, model development, lambda selection, and hyperparameter selection use exhaustive validation metrics only. Final exhaustive test evaluation is performed only after each reported configuration is frozen.

Any analysis motivated after observing test results must be labeled post-hoc or supplementary in the manifest and paper notes.

## Current Scope Exclusions

This stage does not implement morphology loss, SQI weighting, IPA loss, encoder fine-tuning, hard negative mining, augmentation, multi-seed training, or confidence intervals. E6 adds only session centroid alignment.

## Commands

Audit-only:

```bash
bash training/SigD/scripts/run_adaptation_pipeline.sh --audit-only
```

Build train/validation frozen backbone caches:

```bash
python training/SigD/scripts/build_frozen_backbone_cache.py \
  --build-train \
  --build-validation \
  --verify \
  --verbose
```

Smoke training:

```bash
bash training/SigD/scripts/run_adaptation_pipeline.sh --smoke-test --train-generic
bash training/SigD/scripts/run_adaptation_pipeline.sh --smoke-test --train-cs
```

E6 alignment audit and smoke training:

```bash
bash training/SigD/scripts/run_alignment_pipeline.sh --audit-only

bash training/SigD/scripts/run_alignment_pipeline.sh --smoke-test --train-e6-base

bash training/SigD/scripts/run_alignment_pipeline.sh \
  --smoke-test \
  --train-e6-a-grid \
  --lambda-align 0.10

bash training/SigD/scripts/run_alignment_pipeline.sh \
  --smoke-test \
  --train-e6-b-grid \
  --lambda-align 0.10
```

Full training is intentionally not launched by default:

```bash
bash training/SigD/scripts/run_adaptation_pipeline.sh --train-generic
bash training/SigD/scripts/run_adaptation_pipeline.sh --train-cs
```

Final evaluation after choosing a best checkpoint:

```bash
bash training/SigD/scripts/run_adaptation_pipeline.sh --evaluate-generic
bash training/SigD/scripts/run_adaptation_pipeline.sh --evaluate-cs
```

## E7 Morphology Preservation

E7 adds train-time auxiliary morphology preservation for `sVRI` and `SQI` only. The auxiliary heads are small independent regressors on the 128-d projected embedding:

```text
Linear(128, 64) -> ReLU -> Linear(64, 1)
```

The heads use masked MSE with `svri_valid_mask` and `sqi_valid_mask`. Invalid targets do not contribute to loss. IPA remains excluded from primary E7 because valid coverage is limited and mask-aware IPA needs a separate ablation. SQI weighting of contrastive loss is reserved for E8.

Morphology heads are train-time auxiliaries:

- Backbone remains frozen.
- Projection head remains trainable.
- Morphology heads are trainable.
- Verification still uses only the 128-d projection embedding and cosine score.
- Morphology predictions are not used for template aggregation, thresholding, or score computation.

E7-A tests morphology preservation on the current strongest E4 branch. E7-B tests the same morphology preservation on the E6-Base cross-session-batch branch. Candidate selection is validation-only; final test evaluation is deferred until a branch and candidate are frozen.

E7 smoke commands:

```bash
bash training/SigD/scripts/run_morphology_pipeline.sh --audit-only

bash training/SigD/scripts/run_morphology_pipeline.sh \
  --smoke-test \
  --train-e7-a \
  --candidate-name svri0p05_sqi0p05 \
  --lambda-svri 0.05 \
  --lambda-sqi 0.05

bash training/SigD/scripts/run_morphology_pipeline.sh \
  --smoke-test \
  --train-e7-b \
  --candidate-name svri0p05_sqi0p05 \
  --lambda-svri 0.05 \
  --lambda-sqi 0.05
```

## E8 SQI-Weighted SupCon

E8 tests SQI-weighted SupCon on top of the current E7-A candidate (`svri0p05_sqi0p05`). SQI weighting is a training-only anchor reliability weight for the contrastive loss. It does not weight sVRI/SQI morphology MSE, does not drop samples, does not change protocol eligibility, and is not used during validation/test scoring.

The verification path remains:

```text
cached 512-d PaPaGei-S backbone embedding -> 128-d projection embedding -> cosine score
```

Morphology heads and SQI weights are ignored for template aggregation, threshold selection, and test scoring. IPA remains disabled in E8.

E8 candidate selection is validation-only. If no E8 candidate improves over the locked E7-A validation EER, E8 final test evaluation is forbidden and E7-A remains the seed42 final model. E8 final test can be run only after the selector explicitly sets `final_e8_test_evaluation_allowed=true`.

E8 smoke commands:

```bash
bash training/SigD/scripts/run_sqi_weighting_pipeline.sh --audit-only

bash training/SigD/scripts/run_sqi_weighting_pipeline.sh \
  --smoke-test \
  --train-e8 \
  --candidate-name sqi_mild_linear \
  --sqi-weighting-mode mild_linear
```

## Final Multi-Seed Experiment

The final seed42 selection is fixed before multi-seed repetition:

- E4 Generic SupCon is the strongest trainable baseline.
- E7-A is the final SIM-PPG seed42 model: Generic SupCon plus sVRI/SQI morphology preservation.
- E7-A candidate is fixed to `svri0p05_sqi0p05` with `lambda_svri=0.05` and `lambda_sqi=0.05`.
- E8 is excluded from final multi-seed reporting because its validation-only selector did not beat E7-A.

The planned seeds are `42 52 123 777 2026`. Existing seed42 outputs are preserved and reused when valid. New seeds write to seed-specific roots:

```text
training/SigD/results/papagei_s_generic_supcon_head_only/seed52/
training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/svri0p05_sqi0p05/seed52/
```

No per-seed hyperparameter tuning is performed. Each seed trains on the train split, selects the best checkpoint by exhaustive validation EER, then runs final exhaustive test evaluation once with validation-selected thresholds.

Audit-only preflight:

```bash
bash training/SigD/scripts/run_multiseed_final_pipeline.sh --audit-only
```

Full run commands, when ready:

```bash
bash training/SigD/scripts/run_multiseed_final_pipeline.sh \
  --all \
  --seeds "42 52 123 777 2026" \
  --skip-existing

python training/SigD/scripts/summarize_multiseed_results.py \
  --seeds "42 52 123 777 2026"

python training/SigD/scripts/audit_multiseed_results.py \
  --seeds "42 52 123 777 2026"
```

The summary reports mean and standard deviation over seeds, plus paired per-seed deltas for E7-A versus E4. It does not perform significance testing.

## Next Steps

Run the final E4/E7-A multi-seed experiment, summarize paired seed deltas, then optionally add masked IPA or other supplementary ablations after the main comparison is locked.
