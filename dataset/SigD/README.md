# SigD-Core (waveform-only public reconstruction)

## 1. 개요

SigD-Core는 PPG 기반 cross-session user verification 연구를 위해 구축하는 공개 waveform-only reconstruction dataset이다. 공식 SigD GitHub annotation이 지정한 subject/session/time range를 사용하고, 공개 접근 가능한 MIMIC-III Waveform Database Matched Subset v1.0에서 해당 range의 `PLETH` waveform만 읽어 raw range 단위로 저장한다.

이 폴더는 향후 SIM-PPG: Session-Invariant Morphology-Preserving PPG Verification Network 연구의 첫 단계만 담당한다. 범위는 dataset acquisition, raw waveform reconstruction, audit까지이며, preprocessing, fixed-length window 생성, SQI, morphology target, split, enrollment/probe protocol, model training은 포함하지 않는다.

English scope: SigD-Core is a waveform-only public reconstruction dataset for cross-session PPG verification. It preserves raw annotated PLETH ranges and provenance only.

## 2. 연구 범위

본 구현은 demographic metadata를 사용하지 않는다. MIMIC-III Clinical Database에 접근하지 않으며, age, gender, diagnosis, medication, ICU stay metadata를 요청하거나 저장하지 않는다. 공식 SigD annotation과 공개 waveform source만 사용한다.

“This implementation reconstructs a waveform-only public subset for cross-session PPG verification experiments. It does not use restricted clinical metadata and therefore does not support demographic subgroup analysis or claim exact reproduction of the original demographic cohort reported in the SigD paper.”

본 연구의 1차 구현 및 핵심 실험은 공개 접근 가능한 MIMIC-III Waveform Database Matched Subset과 공식 SigD annotation만을 사용하여 재구성한 SigD-Core를 기반으로 수행한다. MIMIC-III Clinical Database 및 age/gender metadata는 사용하지 않으며, demographic subgroup analysis는 본 연구 범위에서 제외한다. 따라서 본 연구는 원 SigD 논문의 demographic cohort 재현을 주장하지 않고, waveform-only cross-session PPG verification 문제에 집중한다.

The primary implementation and core experiments of this study are based on SigD-Core, reconstructed solely from the publicly accessible MIMIC-III Waveform Database Matched Subset and the official SigD annotations. We do not use the MIMIC-III Clinical Database or any age/gender metadata, and demographic subgroup analysis is outside the scope of this study. Accordingly, we do not claim exact reproduction of the demographic cohort reported in the original SigD paper; instead, we focus on waveform-only cross-session PPG verification.

## 3. 데이터 출처

- SigD official GitHub repository: `https://github.com/NasTul/SigD`
  - `README.md`, `Extracted_signal_records.pl`, `GetMIMIC-IIIdata.ipynb`를 local runtime source로 사용한다.
- MIMIC-III Waveform Database Matched Subset v1.0: `https://physionet.org/content/mimic3wdb-matched/1.0/`
  - 공개 waveform source로만 사용하며, `PLETH` channel의 annotation range만 읽는다.
- WFDB Python documentation: `https://wfdb.readthedocs.io/en/latest/io.html`
  - remote header lookup과 range-limited waveform read API 참조에 사용한다.

## 4. 데이터셋 명칭과 용어

이 폴더에서 생성하는 dataset 이름은 `SigD-Core (waveform-only public reconstruction)`이다.

- `raw_range` 또는 `annotated_range`: 공식 SigD annotation이 지정한 원본 waveform 추출 구간이다. 길이는 수 초, 수 분 또는 그 이상일 수 있으며 이번 reconstruction의 저장 단위이다.
- `window`: 향후 preprocessing 단계에서 raw range를 filtering/resampling한 뒤 생성할 5초, 10초, 30초 고정 길이 모델 입력이다. 이번 단계에서는 생성하지 않는다.
- `segment`: raw range와 model window를 혼동시킬 수 있으므로 reconstruction 코드와 metadata 컬럼명에서는 사용하지 않는다. 출력 폴더도 `raw_ranges`를 사용한다.

## 5. 생성되는 폴더 구조

```text
dataset/SigD/
├── README.md
├── requirements.txt
├── config/
│   └── sigd_core.yaml
├── official/
│   ├── .gitkeep
│   └── NasTul_SigD/
│       ├── README.md
│       ├── Extracted_signal_records.pl
│       └── GetMIMIC-IIIdata.ipynb
├── scripts/
│   ├── setup_sigd_source.py
│   ├── parse_sigd_annotations.py
│   ├── reconstruct_sigd_core.py
│   ├── audit_sigd_core.py
│   └── run_sigd_core_pipeline.sh
├── data/
│   ├── raw_ranges/
│   └── failed_records/
├── metadata/
├── logs/
└── tests/
    └── test_annotation_parser.py
```

`official/NasTul_SigD/`, `data/raw_ranges/`, `data/failed_records/`, `logs/`, generated CSV/summary JSON은 runtime output으로 취급한다.

## 6. 설치 방법

Python 3.10 이상을 기준으로 한다. 이 pipeline에 필요한 dependency는 `dataset/SigD/requirements.txt`에 별도 문서화했다.

```bash
python -m pip install -r dataset/SigD/requirements.txt
```

현재 repository root에는 별도 `requirements.txt`, `environment.yml`, `pyproject.toml`, `setup.py`가 없으므로 root dependency 파일을 수정하지 않았다. `wfdb` 설치 과정에서 최신 `pandas`가 함께 올라가면 일부 기존 도구가 요구하는 `pandas<3` 제약과 충돌할 수 있어, SigD-Core 전용 requirements에서는 `pandas>=2.2.3,<3` 범위를 명시한다. 기존 프로젝트 환경에 이미 version pinning이 있는 경우에는 root 환경 정책을 우선하고, SigD-Core dependency는 별도 환경 또는 constraints로 조정한다.

## 7. 공식 source 확보 방법

```bash
python dataset/SigD/scripts/setup_sigd_source.py --root . --verbose
```

첫 실행에서는 공식 SigD GitHub repository에서 필요한 파일을 내려받고, 가능한 경우 git commit hash를 기록한다. 각 파일의 SHA256, size, retrieval datetime은 `metadata/source_manifest.json`에 저장된다. 이후 기본 실행에서는 source manifest가 lock file처럼 동작하여 파일 hash가 달라지면 중단한다.

명시적 갱신이 필요할 때만 다음 옵션을 사용한다.

```bash
python dataset/SigD/scripts/setup_sigd_source.py --root . --refresh-source --verbose
```

공식 source 파일 자체는 재배포 라이선스가 명확하지 않을 수 있으므로 project Git에 commit하지 않는다. 대신 URL, commit hash, file hash, public database identifier만 manifest에 기록한다.

## 8. Annotation parsing 실행 방법

```bash
python dataset/SigD/scripts/parse_sigd_annotations.py --root . --verbose
```

이 단계는 waveform을 다운로드하지 않는다. `metadata/source_manifest.json`의 hash와 local `Extracted_signal_records.pl` hash가 일치할 때만 pickle-like annotation object를 load한다.

출력:

- `metadata/sigd_annotation_manifest.csv`
- `metadata/annotation_summary.json`

Offset parser는 공백을 제거하여 `HH:MM:SS-HH:MM:SS` 형태로 정규화한다. Hour 값은 24를 초과할 수 있으며, minute/second는 0-59 범위를 검증한다. 종료 시간이 시작 시간보다 작거나 같으면 해당 raw range만 failure row로 기록하고 전체 parsing은 계속한다.

## 9. Dry-run, Header-check, Smoke extraction 차이

- Dry-run: remote PhysioNet 접근을 수행하지 않는다. candidate record path와 예상 output NPZ path만 생성한다.
- Header-check: waveform array를 저장하지 않고 WFDB header만 조회하여 candidate path resolution, fs, signal length, channel 정보 가능성을 확인한다.
- Smoke extraction: 소수 raw range에 대해서만 실제 `PLETH` waveform을 읽고 NPZ를 저장한다.

Dry-run 결과는 `metadata/sigd_dry_run_manifest.csv`, header-check 결과는 `metadata/sigd_header_check_manifest.csv`에 저장된다. 두 모드는 실제 extraction 결과인 `metadata/sigd_extraction_manifest.csv`를 수정하지 않는다.

예:

```bash
python dataset/SigD/scripts/reconstruct_sigd_core.py --root . --dry-run --limit-ranges 5 --verbose
python dataset/SigD/scripts/reconstruct_sigd_core.py --root . --header-check --limit-ranges 2 --verbose
python dataset/SigD/scripts/reconstruct_sigd_core.py --root . --limit-subjects 1 --limit-ranges 2 --verbose
```

## 10. 전체 reconstruction 실행 방법

기본 pipeline script는 full extraction을 자동 수행하지 않는다. 전체 reconstruction이 필요할 때만 명시적으로 `--full`을 사용한다.

```bash
bash dataset/SigD/scripts/run_sigd_core_pipeline.sh --full
```

또는 직접 실행한다.

```bash
python dataset/SigD/scripts/reconstruct_sigd_core.py \
    --root . \
    --resume \
    --verbose

python dataset/SigD/scripts/audit_sigd_core.py \
    --root . \
    --verbose
```

구현 원칙은 `range-limited PLETH extraction without persistent full-record storage`이다. 전체 MIMIC waveform database를 다운로드하지 않고, 전체 patient waveform record를 결과물로 저장하지 않는다. WFDB header/layout 확인을 위한 remote metadata 접근은 허용하지만, 영구 저장하는 신호 배열은 annotation이 지정한 raw range의 `PLETH` channel뿐이다.

실제 extraction 결과만 `metadata/sigd_extraction_manifest.csv`에 저장된다. 이 파일은 현재 dataset reconstruction 상태를 나타내는 누적 snapshot이며, limited smoke/retry 실행은 같은 `raw_range_id` row만 update하고 기존 다른 row를 삭제하지 않는다. 각 실제 extraction 실행의 처리 이력은 `metadata/extraction_history/` 아래 timestamped CSV로 별도 보존된다. 즉 actual manifest는 current-state snapshot이고, history CSV는 run-level log이다. 기존 NPZ가 있고 `--overwrite`가 아닌 경우 이전 success row와 hash가 맞으면 기존 통계를 복사하고, 이전 success 통계가 부족하면 `allow_pickle=False`로 NPZ를 안전하게 읽어 raw integrity 통계를 복원한다.

WFDB read는 먼저 `wfdb.rdsamp(..., channel_names=["PLETH"])`로 필요한 range/channel만 요청한다. multi-segment 또는 variable-layout record에서 더 안정적인 처리가 필요하면 `wfdb.rdrecord(..., m2s=True, force_channels=True)` fallback을 사용한다. row 단위 예외 처리를 적용하여 하나의 failure가 전체 pipeline을 중단하지 않게 한다.

## 11. Audit 실행 방법

```bash
python dataset/SigD/scripts/audit_sigd_core.py --root . --verbose
```

Audit은 raw reconstruction integrity와 향후 window 생성 가능성만 확인한다. 계산하는 값은 fs, duration, sample count, NaN/Inf count, raw flatline ratio, basic amplitude statistics, candidate 5/10/30초 non-overlapping window count estimate이다.

이번 단계에서는 SQI, sVRI, IPA, peak detection 기반 morphology, filtering quality, z-normalized statistics를 계산하지 않는다. Window count는 raw duration 기반 preliminary availability estimate이며, preprocessing 후 실제 usable window 수와 달라질 수 있다.

## 12. 출력 파일 설명

- `metadata/source_manifest.json`: official source URL, commit hash, file SHA256, database identifier를 저장하는 source lock file.
- `metadata/sigd_dry_run_manifest.csv`: dry-run 전용 manifest. Actual extraction manifest를 덮어쓰지 않는다.
- `metadata/sigd_header_check_manifest.csv`: header-check 전용 manifest. Actual extraction manifest를 덮어쓰지 않는다.
- `metadata/sigd_annotation_manifest.csv`: official annotation을 raw range 단위로 정규화한 manifest.
- `metadata/annotation_summary.json`: annotation 기준 subject/session/raw range 및 parsing 통계.
- `metadata/sigd_extraction_manifest.csv`: 실제 smoke/full reconstruction의 누적 current-state snapshot manifest.
- `metadata/extraction_history/*.csv`: 실제 extraction 실행별 timestamped run history manifest. 이번 실행에서 처리한 row만 포함한다.
- `data/raw_ranges/{subject_id}/{session_timestamp}/range_XXX.npz`: 성공한 raw PLETH range.
- `data/failed_records/failed_raw_ranges.csv`: row 단위 failure 목록.
- `metadata/sigd_core_dataset_audit.csv`: raw range-level audit와 future window estimate.
- `metadata/sigd_core_subject_summary.csv`: subject-level cross-session availability.
- `metadata/sigd_core_session_summary.csv`: session-level duration/window availability.
- `metadata/sigd_core_interval_pairs.csv`: same-subject successful session pair와 surrogate timestamp gap.
- `metadata/sigd_core_audit_summary.json`: 전체 audit summary와 limitations.

각 NPZ에는 `ppg`, `fs`, `raw_range_id`, `subject_id`, `session_timestamp`, `annotation_range_index_within_session`, `offset_start_seconds`, `offset_end_seconds`, `requested_duration_seconds`, `sampfrom`, `sampto`, `resolved_wfdb_record_name`, `channel_name`, `source_database`, `source_version`, `dataset_name`, `dataset_version`가 포함된다. `ppg`는 `float32` 1D array로 저장한다.

## 13. 향후 SIM-PPG preprocessing/model loader interface contract

이번 reconstruction 단계가 제공하는 항목:

- Raw PPG ranges: `data/raw_ranges/{subject_id}/{session_timestamp}/range_XXX.npz`
- Provenance key: `raw_range_id`
- Session/time-gap metadata: `subject_id`, `session_timestamp`, success/failure, raw duration, fs, interval pair summary
- Raw integrity metadata: NaN ratio, raw flatline ratio, basic amplitude statistics

향후 별도 preprocessing 단계에서 생성해야 하는 항목:

- `window_id`
- `parent_raw_range_id`
- filter configuration hash
- filtered waveform
- target resample frequency, 기본 후보 125 Hz
- 5초/10초/30초 non-overlapping windows
- window-wise Z-score normalized waveform
- SQI
- sVRI
- IPA
- IPA valid mask
- unusable window rejection result
- preprocessing manifest

향후 protocol/model 단계에서 생성해야 하는 항목:

- subject-disjoint train/validation/test split
- enrollment session/probe session mapping
- genuine/impostor trial list
- time-gap stratified evaluation bins
- quality-stratified evaluation bins
- model-ready dataloader
- SIM-PPG training code

이번 단계에서는 어떠한 split도 만들지 않는다.

## 14. 해석상 주의점 및 한계

- Session timestamp는 공개 waveform database의 surrogate timestamp이며, 실제 calendar date로 해석하지 않는다.
- 본 구현은 원 SigD 논문의 demographic cohort 완전 재현을 주장하지 않는다.
- ICU/clinical waveform 기반이므로 consumer smartwatch 환경 전체를 대표하지 않는다.
- Device/sensor condition 변화가 cross-session shift에 일부 포함될 수 있다.
- Raw-level window availability는 filtering, resampling, SQI rejection 이후 달라질 수 있다.

## 15. Git 관리 원칙

Commit 대상:

- `dataset/SigD/README.md`
- `dataset/SigD/config/*.yaml`
- `dataset/SigD/scripts/*.py`
- `dataset/SigD/scripts/*.sh`
- `dataset/SigD/tests/*.py`
- `dataset/SigD/requirements.txt`
- placeholder `.gitkeep`

기본적으로 commit하지 않는 runtime output:

- `dataset/SigD/official/NasTul_SigD/**`
- `dataset/SigD/data/raw_ranges/**`
- `dataset/SigD/data/failed_records/**`
- `dataset/SigD/logs/**`
- `dataset/SigD/metadata/*.csv`
- `dataset/SigD/metadata/*summary.json`

`metadata/source_manifest.json`은 public URL, public commit hash, file hash, database identifier만 포함하고 local absolute path나 사용자 계정 정보를 저장하지 않는다. 민감 정보가 없으면 선택적으로 commit할 수 있지만, commit 여부는 프로젝트 운영 정책에 맞춰 결정한다.

## 16. 인용 목록

향후 논문 또는 보고서에는 다음을 구분해 인용한다.

- Original SigD paper
- Official SigD GitHub repository: `https://github.com/NasTul/SigD`
- MIMIC-III Waveform Database Matched Subset v1.0
- PhysioNet
- 필요한 경우 MIMIC-III original database publication

본 연구에서는 MIMIC-III Clinical Database 자체를 사용하지 않았음을 별도로 명시한다.
