from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "src/complextorch/state_space.py"
SELECTION = ROOT / "src/complextorch/selection.py"
INIT = ROOT / "src/complextorch/__init__.py"
PYPROJECT = ROOT / "pyproject.toml"
TESTS = ROOT / "tests/test_state_space_estimators_batch.py"

state = STATE.read_text()
state = state.replace(
    "from .representations import LinearDynamicalSystem\n",
    "from .representations import LinearDynamicalSystem\n"
    "from .control import InnovationsStateSpace\n"
    "from ._state_space_order import _block_hankel, _resolve_dtype\n",
)
start = state.index("class N4SID(BaseEstimator):")
new_tail = r'''class N4SID(BaseEstimator):
    r"""Estimate linear Gaussian state-space systems by compact N4SID.

    The estimator accepts one trajectory ``(time, variables)`` or batch-first
    observations ``(batch, time, variables)``. In ``mode="pooled"`` all
    trajectories identify one common system, while preserving trial boundaries
    when estimating state transitions. In ``mode="independent"`` one system is
    estimated per trajectory using batched Torch linear algebra.

    References
    ----------
    - Van Overschee, P. and De Moor, B. (1994). N4SID.
    """

    def __init__(
        self,
        n_states: int,
        block_rows: int = 10,
        ridge: float = 1e-8,
        *,
        mode: str = "pooled",
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
    ):
        """Initialize N4SID settings."""
        self.n_states = n_states
        self.block_rows = block_rows
        self.ridge = ridge
        self.mode = mode
        self.device = device
        self.dtype = dtype

    def fit(self, observations, y=None):
        """Fit one pooled system or one system per trajectory.

        Parameters
        ----------
        observations
            Array with shape ``(time, variables)`` or
            ``(batch, time, variables)``.
        y
            Unused scikit-learn compatibility target.
        """
        del y
        values, single = _normalise_ss_observations(
            observations, device=self.device, dtype=self.dtype
        )
        if self.n_states < 1 or self.block_rows < 1 or self.ridge < 0:
            raise ValueError("invalid N4SID settings")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        batch, n_times, n_variables = values.shape
        if n_times <= 2 * self.block_rows + 2:
            raise ValueError("not enough samples for requested block_rows")
        values = values - values.mean(dim=1, keepdim=True)
        past, future = _block_hankel(values, self.block_rows, self.block_rows)
        n_columns = past.shape[-1]

        if self.mode == "pooled":
            past_fit = past.permute(1, 0, 2).reshape(past.shape[1], -1).unsqueeze(0)
            future_fit = future.permute(1, 0, 2).reshape(future.shape[1], -1).unsqueeze(0)
        else:
            past_fit, future_fit = past, future

        identity = torch.eye(
            past_fit.shape[-2], dtype=values.dtype, device=values.device
        )
        gram = past_fit @ past_fit.transpose(-1, -2) + self.ridge * identity
        projection = (
            future_fit
            @ past_fit.transpose(-1, -2)
            @ torch.linalg.pinv(gram)
            @ past_fit
        )
        left, singular_values, _ = torch.linalg.svd(projection, full_matrices=False)
        if self.n_states > singular_values.shape[-1]:
            raise ValueError("n_states exceeds identifiable subspace rank")
        observability = left[..., : self.n_states] * torch.sqrt(
            singular_values[..., : self.n_states]
        ).unsqueeze(-2)

        if self.mode == "pooled":
            common_observability = observability.expand(batch, -1, -1)
            state_columns = torch.linalg.lstsq(
                common_observability, future
            ).solution
        else:
            state_columns = torch.linalg.lstsq(observability, future).solution
        states = state_columns.transpose(-1, -2)
        transition, observation, process_covariance, observation_covariance = (
            _fit_general_state_space_from_states(
                values,
                states,
                observation_start=self.block_rows,
                mode=self.mode,
                min_covar=max(self.ridge, 1e-12),
            )
        )
        state_covariance = (
            _pooled_covariance(states) if self.mode == "pooled" else _batch_covariance(states)
        )
        if single and self.mode == "independent":
            transition = transition[0]
            observation = observation[0]
            process_covariance = process_covariance[0]
            observation_covariance = observation_covariance[0]
            state_covariance = state_covariance[0]
        self.system_ = LinearDynamicalSystem(
            transition,
            observation,
            process_covariance,
            observation_covariance,
            state_covariance=state_covariance,
        )
        self.transition_ = transition
        self.observation_ = observation
        self.process_covariance_ = process_covariance
        self.observation_covariance_ = observation_covariance
        self.singular_values_ = singular_values.squeeze(0) if self.mode == "pooled" else singular_values
        self.states_ = states[0] if single else states
        self.n_states_ = self.n_states
        return self


class LarimoreStateSpace(BaseEstimator):
    r"""Estimate an innovations-form state-space model by Larimore CVA.

    The fitted model is

    .. math::

       z_{t+1}=Az_t+K\varepsilon_t,\qquad
       y_t=Cz_t+\varepsilon_t,\quad
       \varepsilon_t\sim\mathcal N(0,V).

    Batch semantics match :class:`N4SID`: pooled mode estimates one common
    system without connecting trial boundaries, while independent mode returns
    a batched collection of systems.

    References
    ----------
    - Larimore, W. E. (1990, 1996).
    - Bauer, D. (2001), for the associated order-selection criterion.
    - ComplexBox ``mvgc.ss.tsdata_to_ss``.
    """

    def __init__(
        self,
        n_states: int,
        past_horizon: int,
        *,
        future_horizon: int | None = None,
        ridge: float = 1e-12,
        covariance: str = "unbiased",
        mode: str = "pooled",
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
    ):
        """Initialize Larimore state-space identification settings."""
        self.n_states = n_states
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon
        self.ridge = ridge
        self.covariance = covariance
        self.mode = mode
        self.device = device
        self.dtype = dtype

    def fit(self, observations, y=None):
        """Estimate ``A``, ``C``, ``K`` and ``V`` from observations."""
        del y
        values, single = _normalise_ss_observations(
            observations, device=self.device, dtype=self.dtype
        )
        future_horizon = (
            self.past_horizon
            if self.future_horizon is None
            else int(self.future_horizon)
        )
        if self.n_states < 1 or self.past_horizon < 1 or future_horizon < 1:
            raise ValueError("state dimension and horizons must be positive")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        if self.covariance not in {"mle", "unbiased"}:
            raise ValueError("covariance must be 'mle' or 'unbiased'")
        if self.past_horizon + future_horizon > values.shape[1]:
            raise ValueError("past/future horizons are too large for the series")

        values = values - values.mean(dim=1, keepdim=True)
        past, future = _block_hankel(values, self.past_horizon, future_horizon)
        batch, _, n_columns = past.shape
        if self.mode == "pooled":
            past_fit = past.permute(1, 0, 2).reshape(past.shape[1], -1).unsqueeze(0)
            future_fit = future.permute(1, 0, 2).reshape(future.shape[1], -1).unsqueeze(0)
        else:
            past_fit, future_fit = past, future

        correlations, right_vectors, cholesky_past = _larimore_decomposition(
            past_fit, future_fit, ridge=self.ridge
        )
        if self.n_states > correlations.shape[-1]:
            raise ValueError("n_states exceeds identifiable subspace rank")

        if self.mode == "pooled":
            whitened_past = torch.linalg.solve_triangular(
                cholesky_past.transpose(-1, -2), past_fit, upper=True
            )
            flat_states = (
                correlations[..., : self.n_states].unsqueeze(-1)
                * right_vectors[..., : self.n_states, :]
            ) @ whitened_past
            states = flat_states.squeeze(0).T.reshape(batch, n_columns, self.n_states)
        else:
            whitened_past = torch.linalg.solve_triangular(
                cholesky_past.transpose(-1, -2), past, upper=True
            )
            state_columns = (
                correlations[..., : self.n_states].unsqueeze(-1)
                * right_vectors[..., : self.n_states, :]
            ) @ whitened_past
            states = state_columns.transpose(-1, -2)

        transition, observation, gain, innovation_covariance, innovations = (
            _fit_innovations_state_space_from_states(
                values,
                states,
                observation_start=self.past_horizon,
                mode=self.mode,
                covariance=self.covariance,
                min_covar=max(self.ridge, 1e-12),
            )
        )
        if single and self.mode == "independent":
            transition = transition[0]
            observation = observation[0]
            gain = gain[0]
            innovation_covariance = innovation_covariance[0]
        self.system_ = InnovationsStateSpace(
            transition, observation, gain, innovation_covariance
        )
        self.transition_ = transition
        self.observation_ = observation
        self.kalman_gain_ = gain
        self.innovation_covariance_ = innovation_covariance
        self.canonical_correlations_ = (
            correlations.squeeze(0) if self.mode == "pooled" else correlations
        )
        self.states_ = states[0] if single else states
        self.innovations_ = innovations[0] if single else innovations
        self.n_states_ = self.n_states
        return self


class LinearGaussianEM(BaseEstimator):
    r"""Refine linear Gaussian state-space systems by batched EM.

    ``mode="pooled"`` estimates one system from independent trajectories by
    summing sufficient statistics over batch and time. ``mode="independent"``
    estimates one system per trajectory. Trial boundaries are never used as
    state transitions.

    References
    ----------
    - Shumway, R. H. and Stoffer, D. S. (1982).
    """

    def __init__(
        self,
        system,
        n_iter: int = 20,
        min_covar: float = 1e-7,
        *,
        mode: str = "pooled",
    ):
        """Initialize EM refinement."""
        self.system = system
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.mode = mode

    def fit(self, observations, y=None):
        """Run EM on one trajectory or a batch of independent trajectories."""
        del y
        if self.n_iter < 1 or self.min_covar < 0:
            raise ValueError("invalid EM settings")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        values, single = _normalise_ss_observations(
            observations,
            device=self.system.transition.device,
            dtype=self.system.transition.dtype,
        )
        system = self.system
        total_history = []
        trajectory_history = []

        for _ in range(self.n_iter):
            smoothed = kalman_smoother(values, system)
            mean = smoothed.smoothed_mean
            covariance = smoothed.smoothed_covariance
            lag_covariance = smoothed.lag_covariance
            if mean.ndim == 2:
                mean = mean.unsqueeze(0)
                covariance = covariance.unsqueeze(0)
                lag_covariance = lag_covariance.unsqueeze(0)
            second = covariance + mean.unsqueeze(-1) * mean.unsqueeze(-2)
            cross = lag_covariance[:, 1:] + (
                mean[:, 1:].unsqueeze(-1) * mean[:, :-1].unsqueeze(-2)
            )

            if self.mode == "pooled":
                previous_second = second[:, :-1].sum(dim=(0, 1))
                next_second = second[:, 1:].sum(dim=(0, 1))
                cross_moment = cross.sum(dim=(0, 1))
                transition = cross_moment @ torch.linalg.pinv(previous_second)
                yz = torch.einsum("btm,btd->md", values, mean)
                observation = yz @ torch.linalg.pinv(second.sum(dim=(0, 1)))
                process_covariance = _em_process_covariance(
                    previous_second,
                    next_second,
                    cross_moment,
                    transition,
                    values.shape[0] * (values.shape[1] - 1),
                )
                residual = values - torch.einsum("md,btd->btm", observation, mean)
                observation_covariance = symmetrise(
                    (
                        torch.einsum("btm,btn->mn", residual, residual)
                        + torch.einsum(
                            "md,btdk,nk->mn", observation, covariance, observation
                        )
                    )
                    / (values.shape[0] * values.shape[1])
                )
                state_covariance = second.mean(dim=(0, 1))
            else:
                previous_second = second[:, :-1].sum(dim=1)
                next_second = second[:, 1:].sum(dim=1)
                cross_moment = cross.sum(dim=1)
                transition = cross_moment @ torch.linalg.pinv(previous_second)
                yz = torch.einsum("btm,btd->bmd", values, mean)
                observation = yz @ torch.linalg.pinv(second.sum(dim=1))
                process_covariance = _em_process_covariance(
                    previous_second,
                    next_second,
                    cross_moment,
                    transition,
                    values.shape[1] - 1,
                )
                residual = values - torch.einsum("bmd,btd->btm", observation, mean)
                observation_covariance = symmetrise(
                    (
                        torch.einsum("btm,btn->bmn", residual, residual)
                        + torch.einsum(
                            "bmd,btdk,bnk->bmn", observation, covariance, observation
                        )
                    )
                    / values.shape[1]
                )
                state_covariance = second.mean(dim=1)

            process_covariance = _covariance_floor(process_covariance, self.min_covar)
            observation_covariance = _covariance_floor(
                observation_covariance, self.min_covar
            )
            if single and self.mode == "independent":
                transition = transition[0]
                observation = observation[0]
                process_covariance = process_covariance[0]
                observation_covariance = observation_covariance[0]
                state_covariance = state_covariance[0]
            system = LinearDynamicalSystem(
                transition,
                observation,
                process_covariance,
                observation_covariance,
                state_covariance=state_covariance,
            )
            likelihood = smoothed.log_likelihood
            if likelihood.ndim == 0:
                likelihood = likelihood.unsqueeze(0)
            trajectory_history.append(likelihood.detach().clone())
            total_history.append(float(likelihood.sum()))

        self.system_ = system
        self.log_likelihood_history_ = (
            torch.stack(trajectory_history)
            if self.mode == "independent"
            else total_history
        )
        self.trajectory_log_likelihood_history_ = torch.stack(trajectory_history)
        return self


def _normalise_ss_observations(observations, *, device, dtype):
    """Normalize estimator input to batch-first floating-point observations."""
    source = torch.as_tensor(observations)
    target_device = source.device if device == "auto" else torch.device(device)
    values = source.to(device=target_device, dtype=_resolve_dtype(dtype))
    single = values.ndim == 2
    if single:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError("observations must have shape (time,n) or (batch,time,n)")
    if not torch.isfinite(values).all():
        raise ValueError("observations must be finite")
    return values, single


def _batch_covariance(samples):
    """Return one unbiased covariance matrix per batch."""
    centered = samples - samples.mean(dim=1, keepdim=True)
    denominator = max(1, samples.shape[1] - 1)
    return symmetrise(centered.transpose(-1, -2) @ centered / denominator)


def _pooled_covariance(samples):
    """Return covariance after pooling samples without creating transitions."""
    flat = samples.reshape(-1, samples.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    return symmetrise(centered.T @ centered / max(1, flat.shape[0] - 1))


def _fit_general_state_space_from_states(values, states, *, observation_start, mode, min_covar):
    """Estimate ``A,C,Q,R`` while respecting trajectory boundaries."""
    previous = states[:, :-1]
    following = states[:, 1:]
    observations = values[:, observation_start : observation_start + previous.shape[1]]
    if mode == "pooled":
        x0 = previous.reshape(-1, previous.shape[-1])
        x1 = following.reshape(-1, following.shape[-1])
        y0 = observations.reshape(-1, observations.shape[-1])
        transition = torch.linalg.lstsq(x0, x1).solution.T
        observation = torch.linalg.lstsq(x0, y0).solution.T
        process_residual = x1 - x0 @ transition.T
        observation_residual = y0 - x0 @ observation.T
        process_covariance = _covariance_floor(_batch_covariance(process_residual.unsqueeze(0))[0], min_covar)
        observation_covariance = _covariance_floor(_batch_covariance(observation_residual.unsqueeze(0))[0], min_covar)
    else:
        transition = torch.linalg.lstsq(previous, following).solution.transpose(-1, -2)
        observation = torch.linalg.lstsq(previous, observations).solution.transpose(-1, -2)
        process_residual = following - previous @ transition.transpose(-1, -2)
        observation_residual = observations - previous @ observation.transpose(-1, -2)
        process_covariance = _covariance_floor(_batch_covariance(process_residual), min_covar)
        observation_covariance = _covariance_floor(_batch_covariance(observation_residual), min_covar)
    return transition, observation, process_covariance, observation_covariance


def _larimore_decomposition(past, future, *, ridge):
    """Return Larimore canonical correlations, right vectors and past factor."""
    n_effective = past.shape[-1]
    covariance_past = past @ past.transpose(-1, -2) / n_effective
    covariance_future = future @ future.transpose(-1, -2) / n_effective
    cross_covariance = past @ future.transpose(-1, -2) / n_effective
    ip = torch.eye(covariance_past.shape[-1], dtype=past.dtype, device=past.device)
    iff = torch.eye(covariance_future.shape[-1], dtype=future.dtype, device=future.device)
    lp = torch.linalg.cholesky(covariance_past + ridge * ip)
    lf = torch.linalg.cholesky(covariance_future + ridge * iff)
    left = torch.linalg.solve_triangular(lf, cross_covariance.transpose(-1, -2), upper=False)
    whitened = torch.linalg.solve_triangular(
        lp, left.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    _, correlations, right = torch.linalg.svd(whitened, full_matrices=False)
    return correlations, right, lp


def _fit_innovations_state_space_from_states(values, states, *, observation_start, mode, covariance, min_covar):
    """Estimate ``A,C,K,V`` while excluding between-trial transitions."""
    previous = states[:, :-1]
    following = states[:, 1:]
    observations = values[:, observation_start : observation_start + previous.shape[1]]
    denominator_adjustment = 1 if covariance == "unbiased" else 0
    if mode == "pooled":
        x0 = previous.reshape(-1, previous.shape[-1])
        x1 = following.reshape(-1, following.shape[-1])
        y0 = observations.reshape(-1, observations.shape[-1])
        transition = torch.linalg.lstsq(x0, x1).solution.T
        observation = torch.linalg.lstsq(x0, y0).solution.T
        innovations_flat = y0 - x0 @ observation.T
        state_residual = x1 - x0 @ transition.T
        gain = torch.linalg.lstsq(innovations_flat, state_residual).solution.T
        denominator = max(1, innovations_flat.shape[0] - denominator_adjustment)
        innovation_covariance = symmetrise(innovations_flat.T @ innovations_flat / denominator)
        innovations = innovations_flat.reshape(values.shape[0], -1, values.shape[-1])
    else:
        transition = torch.linalg.lstsq(previous, following).solution.transpose(-1, -2)
        observation = torch.linalg.lstsq(previous, observations).solution.transpose(-1, -2)
        innovations = observations - previous @ observation.transpose(-1, -2)
        state_residual = following - previous @ transition.transpose(-1, -2)
        gain = torch.linalg.lstsq(innovations, state_residual).solution.transpose(-1, -2)
        denominator = max(1, innovations.shape[1] - denominator_adjustment)
        innovation_covariance = symmetrise(
            innovations.transpose(-1, -2) @ innovations / denominator
        )
    innovation_covariance = _covariance_floor(innovation_covariance, min_covar)
    return transition, observation, gain, innovation_covariance, innovations


def _em_process_covariance(previous, following, cross, transition, denominator):
    """Evaluate the closed-form EM process-noise covariance update."""
    return symmetrise(
        (
            following
            - transition @ cross.transpose(-1, -2)
            - cross @ transition.transpose(-1, -2)
            + transition @ previous @ transition.transpose(-1, -2)
        )
        / denominator
    )


def _covariance_floor(covariance, floor):
    """Symmetrise covariance and add a diagonal numerical floor."""
    identity = torch.eye(
        covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
    )
    return symmetrise(covariance) + floor * identity
'''
STATE.write_text(state[:start] + new_tail)

selection = SELECTION.read_text()
selection = selection.replace(
    "Reserved for a future full state-space estimator. It must remain\n        ``False`` because this class performs order selection only and does not\n        estimate :math:`A,C,K,V`.",
    "Whether to fit a :class:`complextorch.state_space.LarimoreStateSpace`\n        model at the selected order. Currently supported for pooled mode;\n        independent trajectories may select different dimensions.",
)
selection = selection.replace(
    "        if self.refit:\n            raise ValueError(\n                \"refit=True is unavailable until a full Larimore state-space \"\n                \"estimator is implemented\"\n            )\n\n",
    "        if self.refit and self.mode != \"pooled\":\n            raise ValueError(\n                \"refit=True currently requires mode='pooled'; independent \"\n                \"trajectories may select different state dimensions\"\n            )\n\n",
)
selection = selection.replace(
    "        self.n_effective_ = result.n_effective\n        return self\n",
    "        self.n_effective_ = result.n_effective\n        if self.refit:\n            from .state_space import LarimoreStateSpace\n\n            self.best_estimator_ = LarimoreStateSpace(\n                n_states=int(self.best_order_),\n                past_horizon=self.past_horizon,\n                future_horizon=self.future_horizon,\n                ridge=self.ridge,\n                mode='pooled',\n                device=self.device,\n                dtype=self.dtype,\n            ).fit(X)\n        return self\n",
)
SELECTION.write_text(selection)

init = INIT.read_text()
init = init.replace(
    "from .state_space import kalman_filter, kalman_smoother, N4SID, LinearGaussianEM",
    "from .state_space import kalman_filter, kalman_smoother, N4SID, LarimoreStateSpace, LinearGaussianEM",
)
init = init.replace(
    '"kalman_filter", "kalman_smoother", "N4SID", "LinearGaussianEM",',
    '"kalman_filter", "kalman_smoother", "N4SID", "LarimoreStateSpace", "LinearGaussianEM",',
)
init = init.replace('__version__ = "0.7.1"', '__version__ = "0.8.0"')
INIT.write_text(init)

pyproject = PYPROJECT.read_text().replace('version="0.7.1"', 'version="0.8.0"')
PYPROJECT.write_text(pyproject)

TESTS.write_text(r'''"""Batch contracts for state-space estimators."""
import torch

from complextorch import (
    LarimoreStateSpace,
    LinearGaussianEM,
    N4SID,
    StateSpaceOrderSelection,
)
from complextorch.representations import LinearDynamicalSystem


def _data(batch=3, time=120, variables=2):
    generator = torch.Generator().manual_seed(17)
    noise = torch.randn(batch, time, variables, generator=generator, dtype=torch.float64)
    values = torch.zeros_like(noise)
    for t in range(1, time):
        values[:, t] = 0.65 * values[:, t - 1] + noise[:, t]
    return values


def test_n4sid_pooled_and_independent_batch_contracts():
    x = _data()
    pooled = N4SID(2, block_rows=5, mode="pooled").fit(x)
    independent = N4SID(2, block_rows=5, mode="independent").fit(x)
    assert pooled.transition_.shape == (2, 2)
    assert independent.transition_.shape == (3, 2, 2)
    assert pooled.states_.shape[0] == 3
    assert independent.states_.shape[0] == 3


def test_larimore_pooled_and_independent_batch_contracts():
    x = _data()
    pooled = LarimoreStateSpace(2, 5, mode="pooled").fit(x)
    independent = LarimoreStateSpace(2, 5, mode="independent").fit(x)
    assert pooled.transition_.shape == (2, 2)
    assert pooled.kalman_gain_.shape == (2, 2)
    assert independent.transition_.shape == (3, 2, 2)
    assert independent.kalman_gain_.shape == (3, 2, 2)
    assert pooled.states_.shape[0] == 3


def test_state_space_selector_refits_larimore_in_pooled_mode():
    selector = StateSpaceOrderSelection(5, refit=True).fit(_data())
    assert isinstance(selector.best_estimator_, LarimoreStateSpace)
    assert selector.best_estimator_.n_states_ == int(selector.best_order_)


def test_linear_gaussian_em_accepts_batched_trajectories():
    x = _data(batch=2, time=60)
    system = LinearDynamicalSystem(
        transition=torch.eye(2, dtype=torch.float64) * 0.5,
        observation=torch.eye(2, dtype=torch.float64),
        process_covariance=torch.eye(2, dtype=torch.float64),
        observation_covariance=torch.eye(2, dtype=torch.float64),
        state_covariance=torch.eye(2, dtype=torch.float64),
    )
    pooled = LinearGaussianEM(system, n_iter=2, mode="pooled").fit(x)
    independent = LinearGaussianEM(system, n_iter=2, mode="independent").fit(x)
    assert pooled.system_.transition.shape == (2, 2)
    assert independent.system_.transition.shape == (2, 2, 2)
    assert pooled.trajectory_log_likelihood_history_.shape == (2, 2)
    assert independent.log_likelihood_history_.shape == (2, 2)


def test_pooled_estimators_do_not_create_between_trial_transitions():
    x = _data(batch=2, time=90)
    shifted = x.clone()
    shifted[1] += 1000.0
    first = LarimoreStateSpace(2, 5, mode="pooled").fit(x)
    second = LarimoreStateSpace(2, 5, mode="pooled").fit(shifted)
    torch.testing.assert_close(first.transition_, second.transition_, rtol=1e-5, atol=1e-5)
''')
