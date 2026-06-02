from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root, load_pipeline_config  # noqa: E402
from manifest_index import ManifestIndex  # noqa: E402
from session_aware_batch_sampler import SessionAwareBatchSampler  # noqa: E402
from train_subject_pool import TrainSubjectPool  # noqa: E402


class _SingleSessionManifestIndex:
    def __init__(self) -> None:
        self.rows: dict[int, dict[str, str | int]] = {}
        index = 0
        for subject_idx in range(8):
            subject_id = f"synthetic_{subject_idx:02d}"
            for _ in range(4):
                self.rows[index] = {
                    "array_index": index,
                    "subject_id": subject_id,
                    "session_id": "only_session",
                }
                index += 1

    def get_metadata(self, array_index: int) -> dict[str, str | int]:
        return self.rows[int(array_index)]


class _SingleSessionPool:
    def __init__(self) -> None:
        self.manifest_index = _SingleSessionManifestIndex()
        self.train_subject_ids = [f"synthetic_{idx:02d}" for idx in range(8)]
        self.train_subject_set = set(self.train_subject_ids)
        self.cross_session_subject_ids: list[str] = []
        self.subject_sessions = {subject: ["only_session"] for subject in self.train_subject_ids}
        self.session_indices = {}
        for subject_idx, subject in enumerate(self.train_subject_ids):
            start = subject_idx * 4
            self.session_indices[(subject, "only_session")] = list(range(start, start + 4))

    def sessions_for_subject(self, subject_id: str) -> list[str]:
        return list(self.subject_sessions[subject_id])

    def indices_for_session(self, subject_id: str, session_id: str) -> list[int]:
        return list(self.session_indices[(subject_id, session_id)])

    def indices_for_subject(self, subject_id: str) -> list[int]:
        indices: list[int] = []
        for session in self.sessions_for_subject(subject_id):
            indices.extend(self.indices_for_session(subject_id, session))
        return indices


def build_pool():
    root = detect_project_root(Path(__file__).resolve().parents[3])
    config = load_pipeline_config(root)
    index = ManifestIndex(root, config)
    pool = TrainSubjectPool(root, config, index)
    return config, pool


def metadata_for_batch(pool: TrainSubjectPool, batch: list[int]) -> list[dict]:
    return [pool.manifest_index.get_metadata(idx) for idx in batch]


def test_cross_session_batch_structure() -> None:
    config, pool = build_pool()
    sampler = SessionAwareBatchSampler(pool, mode="same_subject_cross_session", **{
        "seed": 42,
        "subjects_per_batch": 8,
        "sessions_per_subject": 2,
        "windows_per_session": 2,
        "num_batches_per_epoch": 1,
    })
    batch = next(iter(sampler))
    rows = metadata_for_batch(pool, batch)
    assert len(batch) == 32
    by_subject = defaultdict(list)
    by_subject_session = defaultdict(int)
    for row in rows:
        by_subject[row["subject_id"]].append(row["session_id"])
        by_subject_session[(row["subject_id"], row["session_id"])] += 1
        assert row["subject_id"] in pool.train_subject_set
    assert len(by_subject) == 8
    assert all(len(set(sessions)) == 2 for sessions in by_subject.values())
    assert all(count == 2 for count in by_subject_session.values())


def test_cross_session_batch_has_no_duplicate_array_indices() -> None:
    _, pool = build_pool()
    sampler = SessionAwareBatchSampler(
        pool,
        mode="same_subject_cross_session",
        seed=42,
        subjects_per_batch=8,
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    batch = next(iter(sampler))
    assert len(batch) == len(set(batch))


def test_any_session_batch_size_matches() -> None:
    _, pool = build_pool()
    sampler = SessionAwareBatchSampler(
        pool,
        mode="same_subject_any_session",
        seed=42,
        subjects_per_batch=8,
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    batch = next(iter(sampler))
    assert len(batch) == 32
    assert all(pool.manifest_index.get_metadata(idx)["subject_id"] in pool.train_subject_set for idx in batch)


def test_any_session_batch_has_no_duplicate_array_indices_per_subject() -> None:
    _, pool = build_pool()
    sampler = SessionAwareBatchSampler(
        pool,
        mode="same_subject_any_session",
        seed=42,
        subjects_per_batch=8,
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    batch = next(iter(sampler))
    rows = metadata_for_batch(pool, batch)
    by_subject = defaultdict(list)
    for index, row in zip(batch, rows):
        by_subject[row["subject_id"]].append(index)
    assert all(len(indices) == len(set(indices)) for indices in by_subject.values())


def test_any_session_each_subject_contributes_four_samples() -> None:
    _, pool = build_pool()
    sampler = SessionAwareBatchSampler(
        pool,
        mode="same_subject_any_session",
        seed=42,
        subjects_per_batch=8,
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    rows = metadata_for_batch(pool, next(iter(sampler)))
    by_subject = defaultdict(int)
    for row in rows:
        by_subject[row["subject_id"]] += 1
    assert len(by_subject) == 8
    assert all(count == 4 for count in by_subject.values())


def test_any_session_does_not_require_distinct_sessions() -> None:
    pool = _SingleSessionPool()
    sampler = SessionAwareBatchSampler(
        pool,  # type: ignore[arg-type]
        mode="same_subject_any_session",
        seed=42,
        subjects_per_batch=8,
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    batch = next(iter(sampler))
    rows = metadata_for_batch(pool, batch)  # type: ignore[arg-type]
    by_subject_sessions = defaultdict(set)
    by_subject_indices = defaultdict(list)
    for index, row in zip(batch, rows):
        by_subject_sessions[row["subject_id"]].add(row["session_id"])
        by_subject_indices[row["subject_id"]].append(index)
    assert len(batch) == 32
    assert all(len(sessions) == 1 for sessions in by_subject_sessions.values())
    assert all(len(indices) == 4 and len(indices) == len(set(indices)) for indices in by_subject_indices.values())


def test_seed_epoch_determinism() -> None:
    _, pool = build_pool()
    kwargs = {
        "mode": "same_subject_cross_session",
        "seed": 42,
        "subjects_per_batch": 8,
        "sessions_per_subject": 2,
        "windows_per_session": 2,
        "num_batches_per_epoch": 1,
    }
    a = SessionAwareBatchSampler(pool, **kwargs)
    b = SessionAwareBatchSampler(pool, **kwargs)
    a.set_epoch(0)
    b.set_epoch(0)
    assert next(iter(a)) == next(iter(b))
    c = SessionAwareBatchSampler(pool, **kwargs)
    c.set_epoch(1)
    assert next(iter(a)) != next(iter(c))


def test_different_epoch_changes_batch_for_both_modes() -> None:
    _, pool = build_pool()
    for mode in ("same_subject_cross_session", "same_subject_any_session"):
        kwargs = {
            "mode": mode,
            "seed": 42,
            "subjects_per_batch": 8,
            "sessions_per_subject": 2,
            "windows_per_session": 2,
            "num_batches_per_epoch": 1,
        }
        epoch0_a = SessionAwareBatchSampler(pool, **kwargs)
        epoch0_b = SessionAwareBatchSampler(pool, **kwargs)
        epoch1 = SessionAwareBatchSampler(pool, **kwargs)
        epoch0_a.set_epoch(0)
        epoch0_b.set_epoch(0)
        epoch1.set_epoch(1)
        assert next(iter(epoch0_a)) == next(iter(epoch0_b))
        assert next(iter(epoch0_a)) != next(iter(epoch1))
