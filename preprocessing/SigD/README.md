# SigD-Core Common Standardized PPG Preprocessing

## 1. 개요

이 폴더는 `dataset/SigD/`에서 완료된 SigD-Core (waveform-only public reconstruction) snapshot을 read-only input으로 사용하여 common standardized PPG input을 생성하는 preprocessing 단계이다. 출력 protocol은 `COMMON_PPG_10S_125HZ_V1`이며, SIM-PPG뿐 아니라 primary comparison에 포함되는 PaPaGei-P, PaPaGei-S, Pulse-PPG common adapter, SIGMA-PPG common adapter, MTL-RAPID common setting, SigD CNN-LSTM baseline이 공유하는 공통 입력이다.

이번 단계는 filtered/resampled 10초 PPG window, provenance, quality statistics, morphology auxiliary annotation, post-preprocessing common-input availability audit까지만 담당한다. Subject-disjoint split, enrollment/probe trial list, quality weight, dataloader normalization, baseline/model/SIM-PPG 학습 코드는 아직 생성하지 않는다.

## 2. Input Snapshot

기본 입력은 `dataset/SigD/metadata/final_reconstruction_snapshot/`이다. 실행 전 `SHA256SUMS.txt`를 사용해 snapshot metadata hash를 검증하며, smoke run에서는 선택된 raw NPZ hash를 반드시 검증한다. Full preprocessing에서는 `--verify-all-npz-hashes`를 사용해야 한다.

`dataset/SigD/data/raw_ranges/`와 `dataset/SigD/metadata/final_reconstruction_snapshot/`은 read-only input이다. 이 preprocessing pipeline은 해당 파일을 수정하거나 덮어쓰지 않는다.

## 3. 왜 247명 전체를 preprocessing하는가

Raw-level 10초 protocol eligible subject는 231명이다. 그러나 preprocessing, interpolation, filtering, hard validity 이후 실제 common-input 가능 window와 cross-session 가능 subject 수가 달라질 수 있다. 따라서 extraction에 성공한 1,241개 raw range 전체를 처리하고, post-QC audit에서 eligible subject/session/pair 수를 다시 산출한다.

## 4. Common Standardized Preprocessing and Foundation-Model Compatibility

이 pipeline은 PaPaGei-S 전용 입력 생성기가 아니라 primary comparison model이 공유하는 common input protocol이다.

- `comparison_role`: `primary_common_input`
- `input_protocol_id`: `COMMON_PPG_10S_125HZ_V1`
- `preprocessing_profile`: `STANDARDIZED_PPG_FILTER_0P5_12HZ_10S_125HZ_V1`
- `native_or_common_input`: `common`
- `normalization_policy`: `deferred_per_window_zscore_common_dataloader`
- `native_input_outputs_generated`: `false`

공식 PaPaGei repository(`https://github.com/Nokia-Bell-Labs/papagei-foundation-model`)는 filter/morphology definition provenance와 PaPaGei adapter 구현을 위한 reference source로 유지한다. `setup_papagei_reference.py`는 `README.md`, `LICENSE`, `preprocessing/ppg.py`, `preprocessing/flatline.py`, `segmentations.py`, `morphology.py`, `dataset.py`를 runtime reference로 내려받고 hash/commit을 `metadata/papagei_reference_manifest.json`에 기록한다. 이 reference가 존재한다는 사실은 출력이 PaPaGei-S native input이라는 뜻이 아니다.

현재 common preprocessing filter는 PaPaGei가 사용하는 pyPPG preprocessing definition과 정합되도록 다음 parameter를 사용한다.

- backend: pyPPG
- bandpass: 0.5-12 Hz
- order: 4
- segmentation: 10초 non-overlap
- target fs: 125 Hz
- z-score normalization: 저장하지 않고 향후 common dataloader에서 per-window 적용

향후 Pulse-PPG, SIGMA-PPG 등 native adapter 구현이 필요하면 별도의 reference manifest를 추가할 수 있다. Primary fair comparison은 common input과 common verifier를 기준으로 하고, model-specific native input/head 결과는 supplementary native-setting에서만 분리 보고한다.

## 5. Window 처리 순서

1. `np.load(..., allow_pickle=False)`로 raw NPZ safe load
2. raw nonfinite/flatline/basic stats 계산
3. 허용 범위 내 NaN/Inf linear interpolation
4. standardized pyPPG filtering
5. raw_range 내부에서만 non-overlap 10초 window 생성
6. 필요한 경우 target 125 Hz로 resampling
7. raw/filtered quality statistics 계산
8. sVRI, skewness SQI, IPA와 validity mask 계산
9. common-input available window만 contiguous `.npy` array에 저장
10. manifest 및 post-QC audit 생성

Raw range끼리 이어붙이지 않으며, trailing remainder는 버린다.

## 6. Morphology Auxiliary Annotation

sVRI, SQI/skewness, IPA 및 valid mask는 계속 계산한다. 다만 이 값들은 SIM-PPG 또는 같은 auxiliary objective를 적용하는 모델에서만 사용하는 auxiliary annotation이다.

`common_input_available` 판단에는 `svri_valid_mask`, `sqi_valid_mask`, `ipa_valid_mask`를 사용하지 않는다. IPA 실패나 morphology target 일부 invalid는 common comparison pool에서 window를 제외하는 이유가 아니다. Manifest의 `aux_morphology_annotation_available`은 auxiliary target 사용 가능성만 나타내며, subject/session/pair protocol eligibility를 결정하지 않는다.

## 7. Quality Policy

Low SQI, high flatline ratio, IPA failure만으로 window를 제거하지 않는다. 명백히 처리 불가능한 경우만 `common_input_available=False`로 기록한다. Quality weight, percentile threshold, rank normalization은 split 생성 이후 train-side SQI statistics만 사용해 정의해야 하므로 이번 단계에서는 생성하지 않는다.

## 8. Output Array와 Manifest

Full output array:

```text
preprocessing/SigD/data/windows_10s/ppg_filtered_windows_10s_125hz.npy
```

Smoke output array:

```text
preprocessing/SigD/data/windows_10s_smoke/ppg_filtered_windows_10s_125hz.npy
```

Array는 `float32`, shape `(N_common_input_available_windows, 1250)`이다. 저장된 waveform은 filtered/resampled signal이며 z-score normalization은 적용하지 않는다. `metadata/preprocessing_manifest_10s*.csv`는 모든 candidate window row를 유지하고, array에 저장된 row만 `array_index`를 갖는다.

Manifest와 summary JSON에는 `input_protocol_id`, `comparison_role`, `preprocessing_profile`, `native_or_common_input`, `normalization_policy`를 기록한다.

## 9. Normalization Fairness Policy

저장 array에는 z-score normalization을 적용하지 않는다. 향후 common-input primary comparison에서는 모든 모델이 동일한 per-window z-score transform을 common dataloader에서 적용한다. 모델별 native normalization은 supplementary native-setting에서만 허용한다.

## 10. Smoke Run과 Full Run

Smoke run은 full output을 덮어쓰지 않는다.

```bash
bash preprocessing/SigD/scripts/run_sigd_preprocessing_pipeline.sh --verbose
```

Full preprocessing은 명시적으로 실행해야 한다.

```bash
bash preprocessing/SigD/scripts/run_sigd_preprocessing_pipeline.sh \
  --full \
  --verify-all-npz-hashes \
  --verbose
```

구현 완료 직후에는 smoke preprocessing까지만 실행한다.

## 11. Post-QC Eligibility

Post-QC audit은 common-input available window count를 기준으로 session별 `K=1/3/5` 조건을 평가한다. Interval pair에서는 common available window 수로 `K/M` 조건을 계산한다.

별도로 `svri_valid_windows`, `sqi_valid_windows`, `ipa_valid_windows`를 보고하지만, morphology validity로 subject/session/pair를 protocol에서 제외하지 않는다. 아직 실제 train/validation/test split이나 enrollment/probe trial list는 생성하지 않는다.

## 12. Morphology Diagnostic Pilot

IPA validity가 낮을 때는 full preprocessing 전에 diagnostic pilot을 실행한다. 이 pilot은 final reconstruction snapshot과 raw NPZ를 read-only로 사용하며, smoke/full output을 덮어쓰지 않고 `metadata/diagnostics/` 아래에 별도 CSV, summary JSON, waveform plot을 저장한다.

```bash
python preprocessing/SigD/scripts/diagnose_morphology_validity.py \
  --root . \
  --window-seconds 10 \
  --verbose
```

이 pilot은 sVRI/SQI/IPA target별 validity를 독립적으로 보고한다. `aux_morphology_any_available`은 하나 이상의 auxiliary target이 사용 가능함을, `aux_morphology_all_available`은 모든 auxiliary target이 사용 가능함을 뜻한다. 둘 다 common-input eligibility filter가 아니다.

## 13. 향후 단계

- subject-disjoint split
- earliest enrollment / later probe protocol
- common numpy memmap 기반 dataloader
- common dataloader-side per-window z-score normalization
- Pulse-PPG/SIGMA-PPG common adapter
- supplementary native-input adapter
- baseline model
- SIM-PPG training
- train-side quality-aware weighting

## 14. 인용

최종 논문/보고서에는 SigD paper, MIMIC-III Waveform Database Matched Subset, PhysioNet, PaPaGei paper, PaPaGei official repository, pyPPG citation/license를 구분해 인용한다.
