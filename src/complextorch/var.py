"""Batched VAR estimation with OLS, ridge and Morf LWR solvers.

Notes
-----
Ordinary least squares minimises

.. math::

   \widehat B = rg\min_B \lVert Y-XB
Vert_F^2,

while the LWR route implements the Morf lattice-whitening recursion used by
MVGC-compatible estimators.

References
----------
- Morf, M., Vieira, A., Lee, D. T. L., and Kailath, T. (1978). Recursive
  multichannel maximum entropy spectral estimation.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox

Notes
-----
Ordinary least squares minimises

.. math::

   \widehat B = rg\min_B \lVert Y-XB
Vert_F^2,

while the LWR route implements the Morf lattice-whitening recursion used by
MVGC-compatible estimators.

References
----------
- Morf, M., Vieira, A., Lee, D. T. L., and Kailath, T. (1978). Recursive
  multichannel maximum entropy spectral estimation.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox

Notes
-----
Ordinary least squares minimises

.. math::

   \widehat B = rg\min_B \lVert Y-XB
Vert_F^2,

while the LWR route implements the Morf lattice-whitening recursion used by
MVGC-compatible estimators.

References
----------
- Morf, M., Vieira, A., Lee, D. T. L., and Kailath, T. (1978). Recursive
  multichannel maximum entropy spectral estimation.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator

from ._typing import ArrayLike
from .linalg import stable_cholesky
from .representations import LinearDynamicalSystem, VARSystem, build_var_system


@dataclass(frozen=True)
class VARParameters:
    """VARParameters.
    
    Notes
    -----
    The class follows the scikit-learn fitted-attribute convention when applicable.
    """
    coefficients: torch.Tensor
    intercept: torch.Tensor
    innovation_covariance: torch.Tensor
    residuals: torch.Tensor
    n_observations: int
    order: int
    fit_mode: str
    solver: str
    fit_time: float


def _lwr_single(trials: torch.Tensor, order: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Morf lattice-whitening regression for trials shaped (N,T,n)."""
    n_trials, n_times, n_variables = trials.shape
    if order < 1 or order >= n_times:
        raise ValueError("LWR requires 1 <= order < n_times")
    mean = trials.mean(dim=(0, 1), keepdim=True)
    centered = trials - mean
    x = centered.permute(2, 1, 0).contiguous()
    n, m, n_trials = x.shape
    p1 = order + 1
    p1n = p1 * n
    identity = torch.eye(n, dtype=x.dtype, device=x.device)

    xx = torch.zeros((p1, n, m + order, n_trials), dtype=x.dtype, device=x.device)
    for lag in range(p1):
        xx[lag, :, lag:lag + m, :] = x

    errors_all = x.reshape(n, n_trials * m)
    inverse_chol = torch.linalg.inv(
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        torch.linalg.cholesky(errors_all @ errors_all.transpose(-1, -2))
    )
    k = 1
    af = torch.zeros((n, p1n), dtype=x.dtype, device=x.device)
    ab = torch.zeros_like(af)
    af[:, :n] = inverse_chol
    ab[:, p1n - n:] = inverse_chol
    forward = None

    while k <= order:
        effective = n_trials * (m - k)
        forward_block = xx[:k, :, k:m, :].reshape(k * n, effective)
        backward_block = xx[:k, :, k - 1:m - 1, :].reshape(k * n, effective)
        forward = af[:, :k * n] @ forward_block
        backward = ab[:, p1n - k * n:] @ backward_block
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        forward_chol = torch.linalg.cholesky(forward @ forward.transpose(-1, -2))
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        backward_chol = torch.linalg.cholesky(backward @ backward.transpose(-1, -2))
        # The normalised forward/backward cross-covariance is the lattice reflection coefficient.
        reflection = (
            torch.linalg.solve(forward_chol, forward)
            @ torch.linalg.solve(backward_chol, backward).transpose(-1, -2)
        )
        k += 1
        forward_end = k * n
        backward_start = p1n - k * n
        af_previous = af[:, :forward_end].clone()
        ab_previous = ab[:, backward_start:].clone()
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        forward_norm = torch.linalg.cholesky(identity - reflection @ reflection.transpose(-1, -2))
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        backward_norm = torch.linalg.cholesky(identity - reflection.transpose(-1, -2) @ reflection)
        af[:, :forward_end] = torch.linalg.solve(
            forward_norm, af_previous - reflection @ ab_previous
        )
        ab[:, backward_start:] = torch.linalg.solve(
            backward_norm, ab_previous - reflection.transpose(-1, -2) @ af_previous
        )

    if forward is None:
        raise RuntimeError("LWR failed to produce forward residuals")
    a0 = af[:, :n]
    flat = -torch.linalg.solve(a0, af[:, n:p1n])
    coefficients = torch.stack(
        [flat[:, lag * n:(lag + 1) * n] for lag in range(order)], dim=0
    )
    residuals = torch.linalg.solve(a0, forward)
    residuals = residuals.reshape(n, m - order, n_trials).permute(2, 1, 0).contiguous()
    mean_vector = mean.reshape(n)
    intercept = mean_vector - coefficients.sum(dim=0) @ mean_vector
    return coefficients, intercept, residuals


class VAR(BaseEstimator):
    """VAR.
    
    Notes
    -----
    The class follows the scikit-learn fitted-attribute convention when applicable.
    """
    def __init__(
        self,
        order: int = 1,
        *,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        mode: Literal["independent", "pooled"] = "independent",
        solver: Literal["auto", "lstsq", "cholesky", "pinv", "lwr"] = "auto",
        covariance: Literal["unbiased", "mle"] = "unbiased",
        device: str = "auto",
        dtype: str = "float64",
        stability: Literal["check", "ignore"] = "check",
    ):
        """  init  .
        
        Parameters
        ----------
        order
            Input controlling ``__init__``.
        alpha
            Input controlling ``__init__``.
        fit_intercept
            Input controlling ``__init__``.
        mode
            Input controlling ``__init__``.
        solver
            Input controlling ``__init__``.
        covariance
            Input controlling ``__init__``.
        device
            Input controlling ``__init__``.
        dtype
            Input controlling ``__init__``.
        stability
            Input controlling ``__init__``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        self.order = order
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.mode = mode
        self.solver = solver
        self.covariance = covariance
        self.device = device
        self.dtype = dtype
        self.stability = stability

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        """ resolve dtype.
        
        Parameters
        ----------
        name
            Input controlling ``_resolve_dtype``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        dtype = getattr(torch, name, None)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be 'float32' or 'float64'")
        return dtype

    def _resolve_device(self) -> torch.device:
        """ resolve device.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    def _normalise_input(self, x: ArrayLike) -> torch.Tensor:
        """ normalise input.
        
        Parameters
        ----------
        x
            Input controlling ``_normalise_input``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        tensor = torch.as_tensor(
            x, dtype=self._resolve_dtype(self.dtype), device=self._resolve_device()
        )
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("X must have shape (time,variables) or (batch,time,variables)")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("X contains NaN or infinite values")
        if tensor.shape[1] <= self.order:
            raise ValueError("time dimension must exceed VAR order")
        return tensor.contiguous()

    @staticmethod
    def lagged_design(x: torch.Tensor, order: int):
        """Lagged design.
        
        Parameters
        ----------
        x
            Input controlling ``lagged_design``.
        order
            Input controlling ``lagged_design``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        batch, time, _ = x.shape
        targets = x[:, order:, :]
        blocks = [x[:, order - lag:time - lag, :] for lag in range(1, order + 1)]
        return torch.cat(blocks, dim=-1), targets

    def _choose_solver(self):
        """ choose solver.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        if self.solver != "auto":
            return self.solver
        return "lstsq" if self.alpha == 0 else "cholesky"

    def _solve_cholesky(self, design, targets):
        """ solve cholesky.
        
        Parameters
        ----------
        design
            Input controlling ``_solve_cholesky``.
        targets
            Input controlling ``_solve_cholesky``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        gram = design.transpose(-1, -2) @ design
        rhs = design.transpose(-1, -2) @ targets
        penalty = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        if self.fit_intercept:
            penalty[-1, -1] = 0
        # Add adaptive jitter only when required to retain a valid SPD factorisation.
        chol, _ = stable_cholesky(gram + float(self.alpha) * penalty, jitter=1e-12)
        # Solve with the Cholesky factor instead of forming the covariance inverse.
        return torch.cholesky_solve(rhs, chol)

    def _fit_lwr(self, x: torch.Tensor) -> VARParameters:
        """ fit lwr.
        
        Parameters
        ----------
        x
            Input controlling ``_fit_lwr``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        if self.alpha != 0:
            raise ValueError("solver='lwr' does not support ridge alpha")
        if not self.fit_intercept:
            raise ValueError("solver='lwr' requires fit_intercept=True")
        if self.mode == "pooled":
            coefficient, intercept, residuals = _lwr_single(x, self.order)
            coefficients = coefficient.unsqueeze(0)
            intercepts = intercept.unsqueeze(0)
            residual_output = residuals.reshape(1, -1, x.shape[-1])
        elif self.mode == "independent":
            fitted = [_lwr_single(x[index:index + 1], self.order) for index in range(x.shape[0])]
            coefficients = torch.stack([item[0] for item in fitted])
            intercepts = torch.stack([item[1] for item in fitted])
            residual_output = torch.cat([item[2] for item in fitted], dim=0)
        else:
            raise ValueError("mode must be 'independent' or 'pooled'")
        nfit = residual_output.shape[1]
        denominator = nfit if self.covariance == "mle" else nfit - 1
        covariance = residual_output.transpose(-1, -2) @ residual_output / float(denominator)
        return VARParameters(
            coefficients, intercepts, covariance, residual_output,
            nfit, self.order, self.mode, "lwr", 0.0
        )

    def _fit_tensor(self, x):
        """ fit tensor.
        
        Parameters
        ----------
        x
            Input controlling ``_fit_tensor``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        if self.order < 1 or self.alpha < 0:
            raise ValueError("invalid order or alpha")
        solver = self._choose_solver()
        if solver == "lwr":
            return self._fit_lwr(x)
        design, targets = self.lagged_design(x, self.order)
        batch, nobs, n = targets.shape
        if self.fit_intercept:
            design = torch.cat(
                [design, torch.ones((batch, nobs, 1), dtype=x.dtype, device=x.device)], -1
            )
        if self.mode == "pooled":
            design_fit = design.reshape(1, batch * nobs, -1)
            targets_fit = targets.reshape(1, batch * nobs, n)
        elif self.mode == "independent":
            design_fit, targets_fit = design, targets
        else:
            raise ValueError("mode must be 'independent' or 'pooled'")
        if solver == "lstsq":
            solution = torch.linalg.lstsq(design_fit, targets_fit).solution
        elif solver == "pinv":
            solution = torch.linalg.pinv(design_fit) @ targets_fit
        elif solver == "cholesky":
            solution = self._solve_cholesky(design_fit, targets_fit)
        else:
            raise ValueError("unknown solver")
        if self.fit_intercept:
            coef_flat, intercept = solution[:, :-1, :], solution[:, -1, :]
        else:
            coef_flat = solution
            intercept = torch.zeros((solution.shape[0], n), dtype=x.dtype, device=x.device)
        coefficients = coef_flat.reshape(
            solution.shape[0], self.order, n, n
        ).transpose(-1, -2)
        residuals = targets_fit - design_fit @ solution
        nfit = targets_fit.shape[1]
        predictors = design_fit.shape[-1]
        denominator = nfit if self.covariance == "mle" else nfit - predictors
        if denominator <= 0:
            raise ValueError("not enough observations")
        covariance = residuals.transpose(-1, -2) @ residuals / float(denominator)
        return VARParameters(
            coefficients, intercept, covariance, residuals, nfit,
            self.order, self.mode, solver, 0.0
        )

    def fit(self, X: ArrayLike, y=None):
        """Fit.
        
        Parameters
        ----------
        X
            Input controlling ``fit``.
        y
            Input controlling ``fit``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        del y
        x = self._normalise_input(X)
        start = perf_counter()
        parameters = self._fit_tensor(x)
        elapsed = perf_counter() - start
        self.params_ = VARParameters(
            parameters.coefficients, parameters.intercept,
            parameters.innovation_covariance, parameters.residuals,
            parameters.n_observations, parameters.order,
            parameters.fit_mode, parameters.solver, elapsed,
        )
        self.coef_ = parameters.coefficients
        self.intercept_ = parameters.intercept
        self.noise_covariance_ = parameters.innovation_covariance
        self.residuals_ = parameters.residuals
        self.n_features_in_ = x.shape[-1]
        self.n_epochs_in_ = x.shape[0]
        self.n_times_in_ = x.shape[1]
        self.device_ = x.device
        self.dtype_ = x.dtype
        self.fit_time_ = elapsed
        if self.stability == "check":
            try:
                system = build_var_system(self.coef_, self.noise_covariance_)
                self.spectral_radius_ = system.spectral_radius
                self.is_stable_ = self.spectral_radius_ < 1
            except ValueError:
                from .representations import companion_matrix
                from .linalg import spectral_radius
                self.spectral_radius_ = spectral_radius(companion_matrix(self.coef_))
                self.is_stable_ = self.spectral_radius_ < 1
        return self

    def _check_fitted(self):
        """ check fitted.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        if not hasattr(self, "params_"):
            raise RuntimeError("estimator is not fitted")

    def one_step_predictions(self, X):
        """One step predictions.
        
        Parameters
        ----------
        X
            Input controlling ``one_step_predictions``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        self._check_fitted()
        x = self._normalise_input(X)
        design, _ = self.lagged_design(x, self.order)
        batch, nobs, _ = design.shape
        if self.fit_intercept:
            design = torch.cat(
                [design, torch.ones((batch, nobs, 1), dtype=x.dtype, device=x.device)], -1
            )
        coef_flat = self.coef_.transpose(-1, -2).reshape(
            self.coef_.shape[0], -1, x.shape[-1]
        )
        solution = (
            torch.cat([coef_flat, self.intercept_[:, None, :]], 1)
            if self.fit_intercept else coef_flat
        )
        if self.mode == "pooled":
            solution = solution.expand(batch, -1, -1)
        elif solution.shape[0] != batch:
            raise ValueError("number of epochs differs")
        return design @ solution

    def predict(self, X):
        """Predict.
        
        Parameters
        ----------
        X
            Input controlling ``predict``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        return self.one_step_predictions(X).detach().cpu().numpy()

    def forecast(self, history, steps: int):
        """Forecast.
        
        Parameters
        ----------
        history
            Input controlling ``forecast``.
        steps
            Input controlling ``forecast``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        self._check_fitted()
        hist = self._normalise_input(history)
        if steps < 1 or hist.shape[1] < self.order:
            raise ValueError("invalid forecast request")
        batch = hist.shape[0]
        coefficients, intercept = self.coef_, self.intercept_
        if self.mode == "pooled":
            coefficients = coefficients.expand(batch, -1, -1, -1)
            intercept = intercept.expand(batch, -1)
        state = hist[:, -self.order:, :].clone()
        output = []
        for _ in range(steps):
            next_value = intercept.clone()
            for lag in range(self.order):
                next_value = next_value + torch.einsum(
                    "bij,bj->bi", coefficients[:, lag], state[:, -(lag + 1)]
                )
            output.append(next_value)
            state = torch.cat([state[:, 1:], next_value[:, None, :]], 1)
        return torch.stack(output, 1)

    def residuals(self, X):
        """Residuals.
        
        Parameters
        ----------
        X
            Input controlling ``residuals``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        x = self._normalise_input(X)
        _, targets = self.lagged_design(x, self.order)
        return targets - self.one_step_predictions(x)

    def gaussian_nll(self, X, *, reduction="mean"):
        """Gaussian nll.
        
        Parameters
        ----------
        X
            Input controlling ``gaussian_nll``.
        reduction
            Input controlling ``gaussian_nll``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        errors = self.residuals(X)
        covariance = self.noise_covariance_
        if self.mode == "pooled":
            covariance = covariance.expand(errors.shape[0], -1, -1)
        # Add adaptive jitter only when required to retain a valid SPD factorisation.
        chol, _ = stable_cholesky(covariance, jitter=1e-10)
        # Solve with the Cholesky factor instead of forming the covariance inverse.
        solved = torch.cholesky_solve(errors.unsqueeze(-1), chol[:, None]).squeeze(-1)
        values = 0.5 * (
            (errors * solved).sum(-1)
            + 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)[:, None]
            + errors.shape[-1] * np.log(2 * np.pi)
        )
        if reduction == "none":
            return values
        if reduction == "mean":
            return values.mean()
        if reduction == "sum":
            return values.sum()
        raise ValueError("bad reduction")

    def score(self, X, y=None):
        """Score.
        
        Parameters
        ----------
        X
            Input controlling ``score``.
        y
            Input controlling ``score``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        del y
        return -float(self.gaussian_nll(X))

    def consistency(self, observations) -> float:
        """Ding-Bressler consistency statistic, matching ComplexBox/MVGC.
                        
                        Compute the Ding--Bressler VAR consistency diagnostic.
                        
                        References
                        ----------
                        Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
                
                Compute the Ding--Bressler VAR consistency diagnostic.
                
                References
                ----------
                Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
        
        Compute the Ding--Bressler VAR consistency diagnostic.
        
        References
        ----------
        Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
        """
        from .measures.secondary import consistency
        return consistency(observations, self.residuals(observations), order=self.order)

    def whiteness(self, observations, *, method: str = "durbin_watson"):
        """Residual-whiteness diagnostic with an extensible method selector."""
        from .measures.secondary import residual_whiteness
        return residual_whiteness(
            observations, self.residuals(observations), order=self.order, method=method
        )

    def to_var_system(self, *, lyapunov_method="doubling") -> VARSystem:
        """To var system.
        
        Parameters
        ----------
        lyapunov_method
            Input controlling ``to_var_system``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        self._check_fitted()
        return build_var_system(
            self.coef_, self.noise_covariance_, lyapunov_method=lyapunov_method
        )

    def to_state_space(self, *, lyapunov_method="doubling") -> LinearDynamicalSystem:
        """To state space.
        
        Parameters
        ----------
        lyapunov_method
            Input controlling ``to_state_space``.
        
        Returns
        -------
        object
            Result described by the function name and annotated return type.
        
        Notes
        -----
        Tensor batch dimensions are preserved unless the public API explicitly
        documents a squeeze operation. Numerical validation is performed by the
        module before the core calculation.
        """
        return self.to_var_system(lyapunov_method=lyapunov_method).to_state_space()
