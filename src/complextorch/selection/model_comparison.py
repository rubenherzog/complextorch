"""Generic likelihood-based model-comparison utilities.

The routines in this module are intentionally model-family agnostic: a
candidate is described only by its Gaussian log likelihood, number of
identifiable parameters and number of effective temporal observations.  Small
helpers provide the conventional parameter counts for VAR and minimal
innovations-form state-space models so the same criteria can be used for one
model family or across families.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


def symmetric_covariance_parameter_count(n_variables: int) -> int:
    """Return the number of free parameters in an SPD covariance matrix."""
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
    r"""Return the nominal identifiable parameter count for a VAR(p).

    The dynamic coefficients contribute :math:`N^2p` parameters. Optional
    intercept and innovation covariance terms can be included when they are part
    of the compared likelihood model.
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
    r"""Return parameter count for an innovations-form state-space model.

    For a minimal realization, similarity transforms of the latent state remove
    :math:`r^2` non-identifiable degrees of freedom. The identifiable dynamic
    part is therefore counted as :math:`2Nr` for ``C`` and ``K`` after quotienting
    out the state-basis freedom. Set ``minimal=False`` to count the raw entries
    of ``A``, ``C`` and ``K`` as :math:`r^2+2Nr`.
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
class InformationCriteria:
    """AIC, BIC and HQC values for one or more candidates."""

    aic: np.ndarray
    bic: np.ndarray
    hqc: np.ndarray


@dataclass(frozen=True)
class ModelSelectionCandidate:
    """Likelihood and complexity metadata for one fitted model candidate.

    Parameters
    ----------
    name
        Human-readable model label.
    log_likelihood
        Total Gaussian log likelihood. Arrays are allowed for batched
        comparisons and must broadcast with ``n_parameters`` and
        ``n_observations``.
    n_parameters
        Number of identifiable free parameters.
    n_observations
        Number of effective temporal observations contributing to the
        likelihood.
    metadata
        Optional auxiliary values such as family, order or state dimension.
    """

    name: str
    log_likelihood: float | np.ndarray
    n_parameters: int | np.ndarray
    n_observations: int | np.ndarray
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ModelComparisonResult:
    """Information-criterion comparison across candidate models."""

    candidates: tuple[ModelSelectionCandidate, ...]
    aic: np.ndarray
    bic: np.ndarray
    hqc: np.ndarray
    best_aic: str | np.ndarray
    best_bic: str | np.ndarray
    best_hqc: str | np.ndarray
    best_aic_index: int | np.ndarray
    best_bic_index: int | np.ndarray
    best_hqc_index: int | np.ndarray
    delta_aic: np.ndarray
    delta_bic: np.ndarray
    delta_hqc: np.ndarray

    def as_records(self) -> list[dict[str, object]]:
        """Return one record per candidate for tabular serialization."""
        records = []
        for index, candidate in enumerate(self.candidates):
            record = {
                "name": candidate.name,
                "aic": self.aic[..., index],
                "bic": self.bic[..., index],
                "hqc": self.hqc[..., index],
                "delta_aic": self.delta_aic[..., index],
                "delta_bic": self.delta_bic[..., index],
                "delta_hqc": self.delta_hqc[..., index],
                "n_parameters": candidate.n_parameters,
                "n_observations": candidate.n_observations,
            }
            if candidate.metadata:
                record.update(candidate.metadata)
            records.append(record)
        return records


def information_criteria(
    log_likelihood: float | np.ndarray,
    n_parameters: int | np.ndarray,
    n_observations: int | np.ndarray,
    *,
    average: bool = False,
) -> InformationCriteria:
    r"""Compute AIC, BIC and HQC from total log likelihood.

    The criteria are

    .. math::

       \mathrm{AIC}=-2\log L+2k,
       \quad
       \mathrm{BIC}=-2\log L+k\log n,
       \quad
       \mathrm{HQC}=-2\log L+2k\log\log n.

    Set ``average=True`` to divide the returned values by ``n_observations``;
    the minimizing candidate is unchanged for a common ``n``.
    """
    loglik = np.asarray(log_likelihood, dtype=float)
    params = np.asarray(n_parameters, dtype=float)
    obs = np.asarray(n_observations, dtype=float)
    if np.any(params < 0):
        raise ValueError("n_parameters must be non-negative")
    if np.any(obs <= 1):
        raise ValueError("n_observations must exceed one")
    neg2ll = -2.0 * loglik
    aic = neg2ll + 2.0 * params
    bic = neg2ll + params * np.log(obs)
    hqc = neg2ll + 2.0 * params * np.log(np.log(obs))
    if average:
        aic = aic / obs
        bic = bic / obs
        hqc = hqc / obs
    return InformationCriteria(aic=aic, bic=bic, hqc=hqc)


def information_criteria_from_negative_log_likelihood(
    negative_log_likelihood: float | np.ndarray,
    n_parameters: int | np.ndarray,
    n_observations: int | np.ndarray,
    *,
    mean: bool = True,
    average: bool = False,
) -> InformationCriteria:
    """Compute IC values from total or mean negative log likelihood."""
    nll = np.asarray(negative_log_likelihood, dtype=float)
    obs = np.asarray(n_observations, dtype=float)
    total_nll = nll * obs if mean else nll
    return information_criteria(
        -total_nll, n_parameters, obs, average=average
    )


def _best_names(names: tuple[str, ...], indices: np.ndarray) -> str | np.ndarray:
    if indices.shape == ():
        return names[int(indices)]
    lookup = np.asarray(names, dtype=object)
    return lookup[indices]


def _stack_candidate_values(
    candidates: tuple[ModelSelectionCandidate, ...], attribute: str
) -> np.ndarray:
    values = [np.asarray(getattr(candidate, attribute), dtype=float) for candidate in candidates]
    return np.stack(np.broadcast_arrays(*values), axis=-1)


def compare_model_candidates(
    candidates: Iterable[ModelSelectionCandidate],
    *,
    average: bool = False,
) -> ModelComparisonResult:
    """Compare any number of fitted model candidates by AIC, BIC and HQC."""
    candidate_tuple = tuple(candidates)
    if len(candidate_tuple) < 1:
        raise ValueError("at least one candidate is required")
    names = tuple(candidate.name for candidate in candidate_tuple)
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")

    loglik = _stack_candidate_values(candidate_tuple, "log_likelihood")
    params = _stack_candidate_values(candidate_tuple, "n_parameters")
    obs = _stack_candidate_values(candidate_tuple, "n_observations")
    criteria = information_criteria(loglik, params, obs, average=average)

    best_aic_index = np.nanargmin(criteria.aic, axis=-1)
    best_bic_index = np.nanargmin(criteria.bic, axis=-1)
    best_hqc_index = np.nanargmin(criteria.hqc, axis=-1)

    delta_aic = criteria.aic - np.nanmin(criteria.aic, axis=-1, keepdims=True)
    delta_bic = criteria.bic - np.nanmin(criteria.bic, axis=-1, keepdims=True)
    delta_hqc = criteria.hqc - np.nanmin(criteria.hqc, axis=-1, keepdims=True)

    return ModelComparisonResult(
        candidates=candidate_tuple,
        aic=criteria.aic,
        bic=criteria.bic,
        hqc=criteria.hqc,
        best_aic=_best_names(names, np.asarray(best_aic_index)),
        best_bic=_best_names(names, np.asarray(best_bic_index)),
        best_hqc=_best_names(names, np.asarray(best_hqc_index)),
        best_aic_index=best_aic_index,
        best_bic_index=best_bic_index,
        best_hqc_index=best_hqc_index,
        delta_aic=delta_aic,
        delta_bic=delta_bic,
        delta_hqc=delta_hqc,
    )
