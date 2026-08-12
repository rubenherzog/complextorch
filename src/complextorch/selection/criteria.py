"""Shared information-criterion scoring and candidate ranking utilities.

Selection in ComplexTorch follows one common pattern: generate fitted candidates,
score them on a common criterion, then rank or select them. This module contains
model-family-agnostic information-criterion primitives, one common batched
Gaussian innovations likelihood, and small parameter-count helpers for VAR and
minimal innovations-form state-space candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import numpy as np
import torch

from ..linalg import stable_cholesky

LikelihoodScale = Literal["total", "mean"]
CriterionScale = Literal["total", "per_observation"]
InformationCriterion = Literal["aic", "bic", "hqc"]
LikelihoodReduction = Literal["none", "mean", "sum"]


def gaussian_log_likelihood(
    innovations: np.ndarray | torch.Tensor,
    covariance: np.ndarray | torch.Tensor,
    *,
    reduction: LikelihoodReduction = "sum",
) -> torch.Tensor:
    r"""Return Gaussian innovations log likelihood in batched Torch form.

    Parameters
    ----------
    innovations
        Prediction errors with shape ``(time, variables)`` or
        ``(batch, time, variables)``.
    covariance
        Innovation covariance with shape ``(variables, variables)`` or
        ``(batch, variables, variables)``. A singleton covariance batch is
        broadcast over independent trajectories.
    reduction
        ``"none"`` preserves the time axis, ``"mean"`` averages over time, and
        ``"sum"`` returns the total log likelihood per trajectory.

    Returns
    -------
    torch.Tensor
        A scalar for one unbatched trajectory, or one value per batch element
        for ``"mean"``/``"sum"``. ``"none"`` returns the corresponding time
        series of pointwise log-likelihood contributions.

    Notes
    -----
    For innovation :math:`e_t` and covariance :math:`V`, each contribution is

    .. math::

       \ell_t = -\frac{1}{2}\left(e_t^\top V^{-1}e_t
       + \log|V| + n\log(2\pi)\right).

    This is the same Gaussian innovations convention used by ComplexTorch's
    VAR and state-space predictive diagnostics. The implementation preserves
    dtype/device and never mixes trajectories.
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean' or 'sum'")
    errors = torch.as_tensor(innovations)
    if not errors.is_floating_point():
        errors = errors.to(torch.float64)
    single = errors.ndim == 2
    if single:
        errors = errors.unsqueeze(0)
    if errors.ndim != 3:
        raise ValueError(
            "innovations must have shape (time,n) or (batch,time,n)"
        )
    if errors.shape[1] < 1 or errors.shape[2] < 1:
        raise ValueError("innovations must contain time points and variables")
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("innovations must be finite")

    cov = torch.as_tensor(covariance, dtype=errors.dtype, device=errors.device)
    if cov.ndim == 2:
        cov = cov.unsqueeze(0)
    if cov.ndim != 3 or cov.shape[-2:] != (errors.shape[-1], errors.shape[-1]):
        raise ValueError("covariance has incompatible variable dimensions")
    if cov.shape[0] == 1 and errors.shape[0] > 1:
        cov = cov.expand(errors.shape[0], -1, -1)
    if cov.shape[0] != errors.shape[0]:
        raise ValueError("covariance batch dimension must match innovations")
    if not bool(torch.isfinite(cov).all()):
        raise ValueError("covariance must be finite")

    chol, _ = stable_cholesky(cov, jitter=1e-10)
    solved = torch.cholesky_solve(errors.unsqueeze(-1), chol[:, None]).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    values = -0.5 * (
        (errors * solved).sum(-1)
        + logdet[:, None]
        + errors.shape[-1] * np.log(2.0 * np.pi)
    )
    if reduction == "mean":
        values = values.mean(dim=1)
    elif reduction == "sum":
        values = values.sum(dim=1)
    return values[0] if single else values


def symmetric_covariance_parameter_count(n_variables: int) -> int:
    """Return the free-parameter count of a symmetric covariance matrix."""
    n = int(n_variables)
    if n < 1:
        raise ValueError("n_variables must be positive")
    return n * (n + 1) // 2


def var_parameter_count(
    n_variables: int,
    order: int,
    *,
    include_covariance: bool = True,
    include_intercept: bool = False,
) -> int:
    r"""Return the parameter count of a VAR(:math:`p`) likelihood model."""
    n = int(n_variables)
    p = int(order)
    if n < 1 or p < 1:
        raise ValueError("n_variables and order must be positive")
    count = n * n * p
    if include_intercept:
        count += n
    if include_covariance:
        count += symmetric_covariance_parameter_count(n)
    return count


def innovations_state_space_parameter_count(
    n_variables: int,
    n_states: int,
    *,
    include_covariance: bool = True,
    include_mean: bool = False,
    minimal: bool = True,
) -> int:
    r"""Return the identifiable parameter count of an innovations SSM.

    A raw innovations realization contributes :math:`r^2+2Nr` entries through
    ``A``, ``C`` and ``K``. For a minimal realization, similarity transforms of
    the latent state remove :math:`r^2` non-identifiable degrees of freedom,
    leaving :math:`2Nr` identifiable dynamic parameters.
    """
    n = int(n_variables)
    r = int(n_states)
    if n < 1 or r < 1:
        raise ValueError("n_variables and n_states must be positive")
    count = 2 * n * r if minimal else r * r + 2 * n * r
    if include_mean:
        count += n
    if include_covariance:
        count += symmetric_covariance_parameter_count(n)
    return count


@dataclass(frozen=True)
class InformationCriterionScores:
    """AIC, BIC, and HQC scores with a common output scale."""

    aic: np.ndarray
    bic: np.ndarray
    hqc: np.ndarray
    scale: CriterionScale


@dataclass(frozen=True)
class SelectionCandidate:
    """Likelihood and complexity metadata for one fitted candidate.

    Array-valued fields describe independent batched comparisons. Candidate
    fields broadcast over batch dimensions; candidates themselves are stacked
    on the final axis by :func:`select_by_information_criterion`.
    """

    name: str
    log_likelihood: float | np.ndarray
    n_parameters: int | np.ndarray
    n_observations: int | np.ndarray
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SelectionResult:
    """Ranked candidate scores for one chosen information criterion."""

    candidates: tuple[SelectionCandidate, ...]
    criterion: InformationCriterion
    scores: np.ndarray
    deltas: np.ndarray
    best_index: int | np.ndarray
    best_candidate_name: str | np.ndarray
    information_criteria: InformationCriterionScores

    def as_records(self) -> list[dict[str, object]]:
        """Return one record per candidate for tabular serialization."""
        records: list[dict[str, object]] = []
        for index, candidate in enumerate(self.candidates):
            record: dict[str, object] = {
                "name": candidate.name,
                "aic": self.information_criteria.aic[..., index],
                "bic": self.information_criteria.bic[..., index],
                "hqc": self.information_criteria.hqc[..., index],
                "score": self.scores[..., index],
                "delta": self.deltas[..., index],
                "n_parameters": candidate.n_parameters,
                "n_observations": candidate.n_observations,
            }
            if candidate.metadata:
                record.update(candidate.metadata)
            records.append(record)
        return records


def score_information_criteria(
    log_likelihood: float | np.ndarray,
    n_parameters: int | np.ndarray,
    n_observations: int | np.ndarray,
    *,
    likelihood: LikelihoodScale = "total",
    scale: CriterionScale = "total",
    hurvich_tsai: bool = False,
) -> InformationCriterionScores:
    r"""Compute AIC, BIC, and HQC without silently mixing likelihood scales.

    ``likelihood`` specifies whether ``log_likelihood`` is total or mean per
    effective observation. ``scale`` independently specifies whether returned
    criteria are total or per observation. The latter reproduces the historical
    MVGC-style convention used by :class:`VAROrderSelectionIC`.
    """
    if likelihood not in {"total", "mean"}:
        raise ValueError("likelihood must be 'total' or 'mean'")
    if scale not in {"total", "per_observation"}:
        raise ValueError("scale must be 'total' or 'per_observation'")

    loglik = np.asarray(log_likelihood, dtype=float)
    params = np.asarray(n_parameters, dtype=float)
    obs = np.asarray(n_observations, dtype=float)
    try:
        loglik, params, obs = np.broadcast_arrays(loglik, params, obs)
    except ValueError as exc:
        raise ValueError(
            "log_likelihood, n_parameters, and n_observations must broadcast"
        ) from exc
    if not np.all(np.isfinite(loglik)):
        raise ValueError("log_likelihood must be finite")
    if np.any(~np.isfinite(params)) or np.any(params < 0):
        raise ValueError("n_parameters must be finite and non-negative")
    if np.any(~np.isfinite(obs)) or np.any(obs <= 1):
        raise ValueError("n_observations must be finite and greater than one")

    total_loglik = loglik if likelihood == "total" else loglik * obs
    penalty_factor = np.ones_like(obs)
    if hurvich_tsai:
        denominator = obs - params - 1.0
        penalty_factor = obs / denominator

    aic = -2.0 * total_loglik + 2.0 * params * penalty_factor
    if hurvich_tsai:
        aic = np.where(obs - params - 1.0 <= 0.0, np.nan, aic)
    bic = -2.0 * total_loglik + params * np.log(obs)
    hqc = -2.0 * total_loglik + 2.0 * params * np.log(np.log(obs))

    if scale == "per_observation":
        aic = aic / obs
        bic = bic / obs
        hqc = hqc / obs
    return InformationCriterionScores(aic=aic, bic=bic, hqc=hqc, scale=scale)


def _stack_candidate_values(
    candidates: tuple[SelectionCandidate, ...], attribute: str
) -> np.ndarray:
    """Broadcast one candidate attribute and stack candidates on the last axis."""
    values = [
        np.asarray(getattr(candidate, attribute), dtype=float)
        for candidate in candidates
    ]
    try:
        broadcast = np.broadcast_arrays(*values)
    except ValueError as exc:
        raise ValueError(f"candidate {attribute} values must broadcast") from exc
    return np.stack(broadcast, axis=-1)


def _best_names(names: tuple[str, ...], indices: np.ndarray) -> str | np.ndarray:
    """Map scalar or batched minimizing indices back to candidate names."""
    if indices.shape == ():
        return names[int(indices)]
    return np.asarray(names, dtype=object)[indices]


def select_by_information_criterion(
    candidates: Iterable[SelectionCandidate],
    *,
    criterion: InformationCriterion = "bic",
    likelihood: LikelihoodScale = "total",
    scale: CriterionScale = "total",
    hurvich_tsai: bool = False,
) -> SelectionResult:
    """Score and rank arbitrary fitted candidates by one information criterion.

    The same function ranks VAR-vs-VAR, SSM-vs-SSM, VAR-vs-SSM, and batched
    mixtures. Within each independent comparison, all candidates must have been
    scored on exactly the same number of observed samples. This prevents silent
    comparison of likelihoods evaluated on different temporal windows.
    """
    if criterion not in {"aic", "bic", "hqc"}:
        raise ValueError("criterion must be 'aic', 'bic' or 'hqc'")
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("at least one candidate is required")
    names = tuple(candidate.name for candidate in candidate_tuple)
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")

    loglik = _stack_candidate_values(candidate_tuple, "log_likelihood")
    params = _stack_candidate_values(candidate_tuple, "n_parameters")
    obs = _stack_candidate_values(candidate_tuple, "n_observations")
    if not np.all(obs == obs[..., :1]):
        raise ValueError(
            "all candidates must use the same n_observations within each comparison"
        )
    all_scores = score_information_criteria(
        loglik,
        params,
        obs,
        likelihood=likelihood,
        scale=scale,
        hurvich_tsai=hurvich_tsai,
    )
    scores = getattr(all_scores, criterion)
    if np.any(np.all(np.isnan(scores), axis=-1)):
        raise ValueError("criterion is NaN for every candidate in a comparison")
    best_index = np.nanargmin(scores, axis=-1)
    deltas = scores - np.nanmin(scores, axis=-1, keepdims=True)
    best_array = np.asarray(best_index)
    return SelectionResult(
        candidates=candidate_tuple,
        criterion=criterion,
        scores=scores,
        deltas=deltas,
        best_index=best_index,
        best_candidate_name=_best_names(names, best_array),
        information_criteria=all_scores,
    )
