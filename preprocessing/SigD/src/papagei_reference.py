"""Constants describing the PaPaGei preprocessing reference contract."""

PAPAGEI_REFERENCE_FILES = [
    "README.md",
    "LICENSE",
    "preprocessing/ppg.py",
    "preprocessing/flatline.py",
    "segmentations.py",
    "morphology.py",
    "dataset.py",
]

PAPAGEI_REFERENCED_FUNCTIONS = {
    "filtering_function": "preprocess_one_ppg_signal",
    "morphology_functions": ["extract_svri", "skewness_sqi", "compute_ipa"],
    "dataset_normalization": "Normalize(method='z-score')",
}
