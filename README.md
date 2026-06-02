# SIM-PPG: Morphology-Preserving Adaptation for Cross-Session PPG Verification

This repository contains the implementation of **SIM-PPG**, a PaPaGei-S-based morphology-preserving adaptation framework for **cross-session photoplethysmography (PPG) user verification**.

The goal of this project is to study how a frozen PPG foundation encoder can be adapted for time-stable user verification under cross-session variability. Instead of comparing multiple foundation backbones, this work focuses on a **controlled same-backbone evaluation** using **PaPaGei-S** as the frozen encoder and analyzes the effect of downstream adaptation objectives.

## Overview

PPG-based user authentication is attractive because PPG signals can be collected from wearable and mobile devices. However, PPG signals are highly sensitive to session-dependent changes such as sensor placement, physiological condition, motion artifacts, and temporal drift. These factors make cross-session user verification challenging.

This project addresses the following question:

> Given a frozen PaPaGei-S PPG foundation encoder, which downstream adaptation strategy improves cross-session PPG verification stability?

We evaluate several adaptation strategies under a unified SigD-based verification protocol:

* frozen PaPaGei-S cosine baseline
* generic supervised contrastive adaptation
* cross-session supervised contrastive adaptation
* cross-session batch construction
* session centroid alignment
* sVRI/SQI morphology preservation
* SQI-weighted contrastive learning

The final selected model is:

```text
SIM-PPG / E7-A
= Frozen PaPaGei-S backbone
+ Generic supervised contrastive projection head
+ sVRI/SQI morphology preservation
```

## Scope of This Repository

This repository is released as a **code-only implementation**.

The following artifacts are intentionally **not included**:

* reconstructed SigD waveform `.npz` files
* preprocessed PPG `.npy` arrays
* generated metadata `.json` / `.csv`
* experiment logs
* trained projection checkpoints
* PaPaGei pretrained weights
* evaluation scores and result snapshots
* downloaded external repositories

Users should run the dataset reconstruction, preprocessing, protocol generation, training, and evaluation scripts locally to reproduce the artifacts.

## Repository Structure

```text
SIM_PPG/
├── dataset/
│   └── SigD/
│       ├── scripts/        # SigD source setup, annotation parsing, waveform reconstruction, audit
│       ├── tests/          # dataset reconstruction unit tests
│       ├── README.md
│       └── requirements.txt
│
├── preprocessing/
│   └── SigD/
│       ├── scripts/        # common 10-s / 125-Hz PPG preprocessing and audit
│       ├── src/            # signal processing, morphology target extraction, snapshot validation
│       ├── tests/
│       ├── README.md
│       └── requirements.txt
│
├── protocol/
│   └── SigD/
│       ├── scripts/        # subject split and verification trial protocol generation
│       ├── tests/
│       └── README.md
│
├── data_pipeline/
│   └── SigD/
│       ├── scripts/        # data pipeline audit
│       ├── src/            # dataset classes, manifest index, transforms, samplers, collate functions
│       ├── tests/
│       └── README.md
│
├── evaluation/
│   └── SigD/
│       ├── scripts/        # frozen PaPaGei-S baseline and evaluation engine
│       ├── src/            # encoder interface, cosine verifier, metrics, thresholds, reporting
│       ├── tests/
│       └── README.md
│
├── training/
│   └── SigD/
│       ├── scripts/        # adaptation training, evaluation, ablation, multi-seed summary/audit
│       ├── src/            # projection model, losses, objective registry, trainer, evaluators
│       ├── tests/
│       └── README.md
│
├── docs/
│   └── sim_ppg_architecture_v1.png
│
└── README.md
```

## Dataset

### SigD-Core

The main dataset used in this project is **SigD**, a cross-session PPG dataset for user authentication. Because this project does not directly use protected demographic or clinical metadata, we reconstruct a waveform-only public subset referred to as **SigD-Core**.

The reconstruction process follows the official SigD annotation source and retrieves PLETH waveform ranges from the MIMIC-III waveform database where available.

### Reconstruction Summary

The full reconstruction produced:

| Item                                    | Count |
| --------------------------------------- | ----: |
| Parsed subjects                         |   247 |
| Parsed sessions                         |   570 |
| Parsed raw ranges                       | 1,246 |
| Successfully reconstructed raw ranges   | 1,241 |
| Failed raw ranges                       |     5 |
| Available subjects after reconstruction |   247 |
| Available interval pairs                |   484 |

The project does not store or redistribute reconstructed waveform files in this repository.

## Preprocessing

A common standardized preprocessing pipeline was used for all primary comparison models.

### Common Input Protocol

```text
COMMON_PPG_10S_125HZ_V1
```

### Preprocessing Profile

```text
STANDARDIZED_PPG_FILTER_0P5_12HZ_10S_125HZ_V1
```

The preprocessing pipeline performs:

1. SigD reconstruction snapshot validation
2. PLETH waveform loading
3. signal filtering
4. segmentation into 10-second windows
5. resampling / standardization to 125 Hz
6. morphology target extraction
7. post-QC audit

The final common array used in the experiments had:

| Item                 |           Value |
| -------------------- | --------------: |
| Window length        |            10 s |
| Sampling rate        |          125 Hz |
| Samples per window   |           1,250 |
| Valid common windows |          20,974 |
| Array shape          | `(20974, 1250)` |
| dtype                |       `float32` |

Morphology targets include:

* sVRI
* SQI
* IPA

However, the final SIM-PPG model uses only **sVRI** and **SQI** as auxiliary morphology preservation targets. IPA was excluded from the primary objective because its valid coverage was limited and more sensitive to morphology detection failure.

## Verification Protocol

The final evaluation protocol is:

```text
SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_EXHAUSTIVE_EVAL_V2
```

The protocol uses:

* subject-disjoint train / validation / test split
* earliest session enrollment
* later-session probe windows
* K = 5 enrollment windows
* M = 1 probe window
* exhaustive impostor trials for validation and test
* validation-only threshold selection

### Split Summary

| Split      | Subjects |
| ---------- | -------: |
| Train      |      139 |
| Validation |       46 |
| Test       |       46 |

### Final Exhaustive Evaluation Protocol

| Split      | Enrollment templates | Genuine trials | Impostor trials | Total trials |
| ---------- | -------------------: | -------------: | --------------: | -----------: |
| Validation |                   46 |          2,510 |         112,950 |      115,460 |
| Test       |                   46 |          2,395 |         107,775 |      110,170 |

Thresholds are selected only on the validation split and then fixed for test evaluation.

## Model Design

### Frozen Backbone

All primary experiments use **PaPaGei-S** as a frozen PPG foundation encoder.

```text
Input PPG window: [1, 1250]
Frozen encoder: PaPaGei-S / ResNet1DMoE
Backbone embedding: 512-d
Projection head: 512 → 256 → 128
Verification score: cosine similarity
```

The PaPaGei-S backbone is not fine-tuned. Only the downstream projection head and optional auxiliary heads are trained.

## Compared Models

This study is a **same-backbone controlled ablation** over PaPaGei-S. It does not claim to compare against all PPG foundation models.

| Stage   | Model                                | Description                                                                |
| ------- | ------------------------------------ | -------------------------------------------------------------------------- |
| Frozen  | PaPaGei-S Frozen + Cosine            | Frozen foundation representation baseline                                  |
| E4      | Generic SupCon                       | Projection head trained with subject-level supervised contrastive learning |
| E5      | Cross-Session SupCon                 | Positive pairs restricted to different sessions                            |
| E6-Base | Generic SupCon + cross-session batch | Tests cross-session batch composition without alignment                    |
| E6-A    | CS-SupCon + centroid alignment       | Tests alignment on cross-session positive branch                           |
| E6-B    | Generic SupCon + centroid alignment  | Tests alignment on generic branch with cross-session batches               |
| E7-A    | SIM-PPG                              | Generic SupCon + sVRI/SQI morphology preservation                          |
| E7-B    | E6-Base + morphology preservation    | Tests morphology preservation on cross-session batch branch                |
| E8      | SQI-weighted SupCon                  | Tests SQI as contrastive sample weighting                                  |

The final proposed model is **E7-A**.

## Final Objective

The final SIM-PPG objective is:

```text
L_total = L_SupCon + 0.05 * L_sVRI + 0.05 * L_SQI
```

where:

* `L_SupCon` is generic supervised contrastive loss
* `L_sVRI` is masked MSE for sVRI prediction
* `L_SQI` is masked MSE for SQI prediction
* IPA is not used
* SQI weighting is not used
* morphology heads are used only during training
* verification uses only the 128-d projection embedding

## Experimental Results

### Seed42 Ablation Results

| Model             | ROC-AUC ↑ |  EER ↓ | TAR@FAR=1% ↑ | Notes                                       |
| ----------------- | --------: | -----: | -----------: | ------------------------------------------- |
| PaPaGei-S Frozen  |    0.6165 | 0.4159 |       0.0255 | Foundation-only baseline                    |
| E4 Generic SupCon |    0.7205 | 0.3487 |       0.0630 | Strong trainable baseline                   |
| E5 CS-SupCon      |    0.6795 | 0.3686 |       0.0322 | CS-only positive was weaker than E4         |
| E6-Base           |    0.6849 | 0.3686 |       0.0534 | Some time-gap advantages but weaker overall |
| E7-A SIM-PPG      |    0.7233 | 0.3453 |       0.0668 | Final seed42 best model                     |

E7-A improved over E4 on ROC-AUC, EER, and TAR@FAR=1% in seed42.

### E8 SQI Weighting

E8 tested SQI-weighted contrastive learning on top of E7-A.

| E8 Candidate                 | Validation EER ↓ |
| ---------------------------- | ---------------: |
| E7-A Reference               |           0.3543 |
| SQI rank-bottom20 downweight |           0.3582 |
| SQI mild linear              |           0.3614 |
| SQI clipped linear           |           0.3649 |
| SQI strong linear            |           0.3653 |

No E8 candidate improved over E7-A on validation. Therefore, E8 was not evaluated on the final test split and was excluded from the final model.

## Multi-Seed Final Results

Final multi-seed experiments were conducted for:

* E4 Generic SupCon
* E7-A SIM-PPG

Seeds:

```text
42, 52, 123, 777, 2026
```

### Mean ± Std

| Model             |       ROC-AUC ↑ |           EER ↓ |    TAR@FAR=1% ↑ |    FAR@FAR=1% ↓ |
| ----------------- | --------------: | --------------: | --------------: | --------------: |
| E4 Generic SupCon | 0.7142 ± 0.0214 | 0.3459 ± 0.0188 | 0.0673 ± 0.0052 | 0.0097 ± 0.0020 |
| E7-A SIM-PPG      | 0.7134 ± 0.0182 | 0.3438 ± 0.0181 | 0.0751 ± 0.0117 | 0.0102 ± 0.0017 |

E7-A achieved:

* lower mean EER than E4
* higher mean TAR@FAR=1%
* improved TAR@FAR=1% in 4 out of 5 seeds

### Paired Delta: E7-A minus E4

| Seed |  ΔEER ↓ |  ΔAUC ↑ | ΔTAR@FAR=1% ↑ |
| ---: | ------: | ------: | ------------: |
|   42 | -0.0034 | +0.0028 |       +0.0038 |
|   52 | -0.0134 | +0.0254 |       +0.0029 |
|  123 | -0.0037 | +0.0014 |       -0.0046 |
|  777 | +0.0047 | -0.0153 |       +0.0092 |
| 2026 | +0.0055 | -0.0184 |       +0.0276 |

E7-A does not dominate E4 across all metrics and all seeds, but it improves the low-FAR operating point in most seeds.

## Key Findings

1. **Frozen PaPaGei-S alone is insufficient for robust cross-session verification.**
   The frozen foundation embedding showed limited verification performance on SigD-Core.

2. **Generic SupCon adaptation is a strong baseline.**
   Training a projection head with generic subject-level supervised contrastive learning substantially improved performance over the frozen baseline.

3. **Cross-session-only positive construction was not sufficient.**
   E5 and related cross-session-only strategies did not outperform generic SupCon.

4. **Session centroid alignment was not selected as the final method.**
   Alignment helped some variants but did not outperform the best generic adaptation baseline.

5. **sVRI/SQI morphology preservation improved the final model.**
   Adding sVRI/SQI auxiliary preservation to Generic SupCon produced the best seed42 result and improved mean EER and TAR@FAR=1% across five seeds.

6. **SQI weighting was not beneficial beyond morphology preservation.**
   SQI was more useful as a morphology preservation target than as an anchor-weighting signal for contrastive learning.

## How to Run

This repository excludes runtime configs and generated artifacts if the code-only `.gitignore` policy is used. The following commands show the intended execution flow. Users may need to recreate local YAML configs or use their own experiment configuration files.

### 1. Dataset Reconstruction

```bash
bash dataset/SigD/scripts/run_sigd_core_pipeline.sh --full
```

### 2. Preprocessing

```bash
bash preprocessing/SigD/scripts/run_sigd_preprocessing_pipeline.sh --full --verify-all-npz-hashes
```

### 3. Protocol Generation

```bash
bash protocol/SigD/scripts/run_sigd_protocol_pipeline.sh --overwrite
bash protocol/SigD/scripts/run_sigd_exhaustive_evaluation_protocol.sh --overwrite
```

### 4. Data Pipeline Audit

```bash
bash data_pipeline/SigD/scripts/run_sigd_data_pipeline_audit.sh
```

### 5. Frozen Baseline

```bash
bash evaluation/SigD/scripts/run_frozen_baseline_pipeline.sh --audit-only
```

After obtaining and verifying PaPaGei-S official assets:

```bash
bash evaluation/SigD/scripts/run_frozen_baseline_pipeline.sh --verify-official-assets --run-official-baseline
```

### 6. Adaptation Training

```bash
bash training/SigD/scripts/run_adaptation_pipeline.sh --train-generic
bash training/SigD/scripts/run_adaptation_pipeline.sh --train-cs
```

### 7. Morphology Preservation

```bash
bash training/SigD/scripts/run_morphology_pipeline.sh \
  --train-e7-a \
  --candidate-name svri0p05_sqi0p05 \
  --lambda-svri 0.05 \
  --lambda-sqi 0.05
```

### 8. Multi-Seed Final Experiment

```bash
bash training/SigD/scripts/run_multiseed_final_pipeline.sh \
  --all \
  --seeds "42 52 123 777 2026" \
  --skip-existing \
  --device auto
```

### 9. Summarize Multi-Seed Results

```bash
python training/SigD/scripts/summarize_multiseed_results.py \
  --seeds "42 52 123 777 2026"
```

### 10. Strict Audit

```bash
python training/SigD/scripts/audit_multiseed_results.py \
  --seeds "42 52 123 777 2026"
```

Expected audit status:

```text
multiseed_audit_passed=True
strict_ready=True
missing_count=0
```

## Reproducibility Notes

This repository does not include:

* SigD reconstructed waveform files
* MIMIC waveform data
* PaPaGei pretrained checkpoints
* generated model checkpoints
* evaluation scores
* generated metadata

To reproduce results, users must:

1. obtain the required public waveform sources
2. reconstruct SigD-Core locally
3. run preprocessing and protocol generation
4. obtain PaPaGei-S official weights
5. run training and evaluation scripts
6. generate local metadata and result files

## Limitations

This work is intentionally scoped as a **PaPaGei-S-based controlled adaptation study**.

It does not directly compare against other PPG foundation backbones such as:

* PaPaGei-P
* Pulse-PPG
* SIGMA-PPG

These models differ in pretraining objective, input-length assumptions, native preprocessing, and feature extraction APIs. A fair multi-backbone comparison would require:

* model-specific common-input adapters
* unified 10-s / 125-Hz preprocessing
* shared embedding projection and verifier
* matched fine-tuning budgets
* separate native-input supplementary evaluations

Therefore, a broader multi-backbone PPG foundation model leaderboard is left as future work.

Additional limitations:

* PaPaGei pretraining includes MIMIC-III, while SigD-Core is derived from MIMIC-III waveform records. This creates a source-level overlap caveat.
* E7-A improves average EER and TAR@FAR=1%, but the improvement is modest.
* E7-A does not outperform E4 on every seed and every metric.
* The final model uses PaPaGei-S as a frozen encoder and does not evaluate encoder fine-tuning.

## Future Work

Future directions include:

1. evaluating SIM-PPG objectives on other PPG foundation backbones such as PaPaGei-P, Pulse-PPG, and SIGMA-PPG
2. testing native-input and common-input settings separately
3. extending evaluation to additional public PPG authentication datasets
4. exploring robust IPA-aware objectives with better morphology validity handling
5. studying lightweight on-device deployment of the final projection-based verifier

## Citation

A formal citation will be added after the corresponding paper is completed.

## License

Please check the licenses of all external dependencies and upstream model repositories before redistribution. This repository does not redistribute the SigD dataset, MIMIC waveform data, or PaPaGei pretrained weights.
