"""Session-centroid alignment regularizer for E6 SigD adaptation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import torch
from torch import nn


class SessionCentroidAlignmentLoss(nn.Module):
    """Align two session centroids for each subject in projected space."""

    def __init__(self, *, sessions_per_subject: int = 2, samples_per_session: int = 2, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.sessions_per_subject = int(sessions_per_subject)
        self.samples_per_session = int(samples_per_session)
        self.eps = float(eps)
        if self.sessions_per_subject != 2:
            raise ValueError("SessionCentroidAlignmentLoss currently expects exactly 2 sessions per subject.")
        if self.samples_per_session <= 0:
            raise ValueError("samples_per_session must be positive.")

    def forward(
        self,
        embeddings: torch.Tensor,
        subject_ids: Sequence[str],
        session_ids: Sequence[str],
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Return mean 1-cosine distance between per-subject session centroids."""

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [B, D].")
        if len(subject_ids) != embeddings.shape[0] or len(session_ids) != embeddings.shape[0]:
            raise ValueError("subject_ids/session_ids length must match batch size.")
        if not torch.isfinite(embeddings).all():
            raise ValueError("embeddings contain nonfinite values.")

        groups = self._group_indices(subject_ids, session_ids)
        losses: list[torch.Tensor] = []
        cosines: list[torch.Tensor] = []
        for subject_id, sessions in groups.items():
            if len(sessions) != self.sessions_per_subject:
                raise ValueError(f"Subject {subject_id} must have exactly {self.sessions_per_subject} sessions.")
            centroids: list[torch.Tensor] = []
            for session_id in sorted(sessions):
                indices = sessions[session_id]
                if len(indices) != self.samples_per_session:
                    raise ValueError(
                        f"Subject {subject_id} session {session_id} must have exactly "
                        f"{self.samples_per_session} samples."
                    )
                centroid = embeddings[indices].mean(dim=0)
                centroids.append(_safe_l2_normalize(centroid, self.eps))
            cosine = torch.sum(centroids[0] * centroids[1])
            if not torch.isfinite(cosine):
                raise RuntimeError("Session centroid cosine is nonfinite.")
            cosines.append(cosine)
            losses.append(1.0 - cosine)

        if not losses:
            raise ValueError("No subject centroid pairs were available.")
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Session centroid alignment loss is nonfinite.")
        if not return_diagnostics:
            return loss

        stacked = torch.stack(cosines)
        diagnostics = {
            "subject_count": len(groups),
            "centroid_pair_count": len(cosines),
            "mean_centroid_cosine": float(stacked.mean().detach().cpu()),
            "min_centroid_cosine": float(stacked.min().detach().cpu()),
            "max_centroid_cosine": float(stacked.max().detach().cpu()),
        }
        return loss, diagnostics

    @staticmethod
    def _group_indices(subject_ids: Sequence[str], session_ids: Sequence[str]) -> OrderedDict[str, OrderedDict[str, list[int]]]:
        groups: OrderedDict[str, OrderedDict[str, list[int]]] = OrderedDict()
        for index, (subject_id, session_id) in enumerate(zip(subject_ids, session_ids)):
            subject_key = str(subject_id)
            session_key = str(session_id)
            groups.setdefault(subject_key, OrderedDict()).setdefault(session_key, []).append(index)
        return groups


def _safe_l2_normalize(vector: torch.Tensor, eps: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, ord=2)
    if not torch.isfinite(norm):
        raise ValueError("Cannot normalize a nonfinite session centroid.")
    if float(norm.detach().cpu()) <= float(eps):
        raise ValueError("Cannot normalize zero or near-zero session centroid.")
    return vector / norm
