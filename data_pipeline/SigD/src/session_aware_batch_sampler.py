"""Session-aware dynamic batch sampler for SigD train windows."""

from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import Sampler

from train_subject_pool import TrainSubjectPool


class SessionAwareBatchSampler(Sampler[list[int]]):
    """Yield batches of array indices using train subjects only."""

    def __init__(
        self,
        train_pool: TrainSubjectPool,
        *,
        mode: str = "same_subject_cross_session",
        seed: int = 42,
        subjects_per_batch: int = 8,
        sessions_per_subject: int = 2,
        windows_per_session: int = 2,
        num_batches_per_epoch: int = 100,
    ) -> None:
        if mode not in {"same_subject_cross_session", "same_subject_any_session"}:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        self.train_pool = train_pool
        self.mode = mode
        self.seed = int(seed)
        self.subjects_per_batch = int(subjects_per_batch)
        self.sessions_per_subject = int(sessions_per_subject)
        self.windows_per_session = int(windows_per_session)
        self.num_batches_per_epoch = int(num_batches_per_epoch)
        self.epoch = 0
        self.batch_size = self.subjects_per_batch * self.sessions_per_subject * self.windows_per_session
        self.last_batch_metadata: list[dict[str, str | int]] = []

    def set_epoch(self, epoch: int) -> None:
        """Set deterministic epoch state."""

        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1000003)
        for _ in range(self.num_batches_per_epoch):
            yield self._sample_batch(rng)

    def _sample_subjects(self, rng: random.Random) -> list[str]:
        candidates = (
            self.train_pool.cross_session_subject_ids
            if self.mode == "same_subject_cross_session"
            else self.train_pool.train_subject_ids
        )
        if len(candidates) < self.subjects_per_batch:
            raise RuntimeError("Not enough train subjects for batch sampling.")
        return rng.sample(candidates, self.subjects_per_batch)

    def _sample_distinct_from_list(self, rng: random.Random, values: list[int], count: int, context: str) -> list[int]:
        """Sample distinct array indices from a non-empty candidate list."""

        unique_values = sorted(set(values))
        if len(unique_values) < count:
            raise RuntimeError(f"Not enough distinct windows for {context}: {len(unique_values)} < {count}")
        return rng.sample(unique_values, count)

    def _metadata_for_index(self, array_index: int) -> dict[str, str | int]:
        """Return batch metadata for a sampled array index."""

        row = self.train_pool.manifest_index.get_metadata(array_index)
        return {
            "array_index": int(array_index),
            "subject_id": str(row["subject_id"]),
            "session_id": str(row["session_id"]),
        }

    def _sample_batch(self, rng: random.Random) -> list[int]:
        subjects = self._sample_subjects(rng)
        batch: list[int] = []
        metadata: list[dict[str, str | int]] = []
        for subject in subjects:
            sessions = self.train_pool.sessions_for_subject(subject)
            if self.mode == "same_subject_cross_session":
                if len(sessions) < self.sessions_per_subject:
                    raise RuntimeError(f"Subject lacks distinct sessions: {subject}")
                selected_sessions = rng.sample(sessions, self.sessions_per_subject)
                for session in selected_sessions:
                    indices = self.train_pool.indices_for_session(subject, session)
                    selected_indices = self._sample_distinct_from_list(
                        rng,
                        indices,
                        self.windows_per_session,
                        f"{subject}/{session}",
                    )
                    batch.extend(selected_indices)
                    metadata.extend(self._metadata_for_index(index) for index in selected_indices)
            else:
                windows_per_subject = self.sessions_per_subject * self.windows_per_session
                indices = self.train_pool.indices_for_subject(subject)
                selected_indices = self._sample_distinct_from_list(
                    rng,
                    indices,
                    windows_per_subject,
                    f"{subject}/any_session",
                )
                batch.extend(selected_indices)
                metadata.extend(self._metadata_for_index(index) for index in selected_indices)
        self.last_batch_metadata = metadata
        if len(batch) != self.batch_size:
            raise RuntimeError(f"Batch size mismatch: {len(batch)} != {self.batch_size}")
        return batch
