"""Shared information-criterion scoring and candidate ranking utilities.

Selection in ComplexTorch follows one common pattern: generate fitted candidates,
score them on a common criterion, then rank or select them.  This module contains
model-family-agnostic information-criterion primitives plus small parameter-count
helpers for VAR and minimal innovations-form state-space candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import numpy as np

LikelihoodScale = Literal["total", "mean"]
CriterionScale = Literal["total", "per_observation"]
InformationCriterion = Literal["aic", "bic", "hqc"]


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
    r"""Return the parameter count of a VAR(:math:`p`) likelihood model.

    The autoregressive coefficients contribute :math:`N^2p` parameters.  The
    innovation covariance and intercept are included only when requested, so
    callers can match the exact likelihood convention used for candidate fits.
    """
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

    An innovations realization with state dimension :math:`r` and observation
    dimension :math:`N` has raw dynamic matrices ``A``, ``C`` and ``K`` with
    :math:`r^2 + 2Nr` entries.  For a minimal realization, state-basis
    similarity transforms contribute :math:`r^2` non-identifiable degrees of
    freedom, leaving :math:`2Nr` identifiable dynamic parameters.  The
    innovation covariance and observation mean are included only when requested.

    ``minimal=False`` returns the raw matrix-entry count and is therefore not an
    identifiable model dimension; it is provided only for explicit diagnostics.
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

    Array-valued fields may describe a batch of independent comparisons.  All
    candidate fields are broadcast before candidates are stacked on the final
    axis by :func:`select_by_information_criterion`.
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
    best_candidate: str | np.ndarray
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

    Parameters
    ----------
    log_likelihood
        Gaussian log likelihood, either total or mean per effective observation
        according to ``likelihood``.
    n_parameters
        Number of free/identifiable parameters penalized by the criterion.
    n_observations
        Number of effective observations contributing to the likelihood.
    likelihood
        ``"total"`` for :math:`\log L` or ``"mean"`` for
        :math:`\ell=\log L/N`.
    scale
        ``"total"`` returns the standard total AIC/BIC/HQC formulas;
        ``"per_observation"`` divides every criterion by :math:`N`.  This latter
        convention matches the existing MVGC-style VAR order selectors.
    hurvich_tsai
        Apply the existing ComplexTorch/MVGC Hurvich--Tsai AIC penalty factor
        :math:`N/(N-k-1)`.  Invalid candidates with :math:`N-k-1\leq0` receive
        ``NaN`` AIC values.

    Notes
    -----
    For total log likelihood the uncorrected criteria are

    .. math::

       \mathrm{AIC}=-2\log L+2k,\quad
       \mathrm{BIC}=-2\log L+k\log N,\quad
       \mathrm{HQC}=-2\log L+2k\log\log N.

    ``likelihood`` controls the input convention; ``scale`` controls only the
    returned convention.  Therefore all four combinations are explicit and
    mathematically equivalent after the corresponding multiplication/division by
    :math:`N`.
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

    Candidate identity is model-family agnostic, so the same function ranks
    VAR-vs-VAR, SSM-vs-SSM, VAR-vs-SSM, or batched mixtures.  Comparisons are
    meaningful only when candidates describe likelihoods for the same observed
    data under the same likelihood convention.
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
        best_candidate=_best_names(names, best_array),
        information_criteria=all_scores,
    )
