r'''Normalize scientific module and function docstrings to canonical content.

The script replaces, rather than appends, scientific documentation. Running it
multiple times is therefore idempotent.
'''
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "complextorch"

MODULE_DOCS = {
    "adapters": r'''Axis-layout adapters between ComplexTorch and ComplexBox.

ComplexBox follows ``(variables, time, trials)`` while ComplexTorch follows
``(trials, time, variables)``. Coefficient tensors are analogously permuted
between ``(target, source, lag)`` and ``(batch, lag, target, source)``. These
operations never alter numerical values.

References
----------
- Barnett, L. and Seth, A. K. (2014). The MVGC multivariate Granger causality
  toolbox. *Journal of Neuroscience Methods*, 223, 50--68.
- ComplexBox: https://github.com/bmilinkovic/complexbox
''',
    "representations": r'''Canonical VAR and linear Gaussian state-space representations.

A VAR(p) process is

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + \varepsilon_t,
   \qquad \varepsilon_t \sim \mathcal N(0,\Sigma).

The companion representation embeds this recursion into a first-order state
transition and provides the bridge to state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
''',
    "linalg": r'''Numerically stable linear-algebra primitives for covariance models.

For a symmetric positive-definite matrix :math:`S=LL^\top`, log determinants
are evaluated as

.. math::

   \log\det S = 2\sum_i\log L_{ii},

and systems are solved through triangular factors rather than explicit matrix
inverses.

References
----------
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*.
- PyTorch linear algebra: https://pytorch.org/docs/stable/linalg.html
''',
    "simulate": r'''Simulation and random generation for stationary Gaussian VAR systems.

Trajectories obey

.. math::

   x_t=c+\sum_{k=1}^{p}A_kx_{t-k}+\varepsilon_t.

Automatic burn-in uses the companion spectral radius to reduce the influence
of initial conditions below a requested tolerance.

References
----------
- Lütkepohl, H. (2005), Chapters 2--3.
- ComplexBox: https://github.com/bmilinkovic/complexbox
''',
    "var": r'''Torch-first estimation and forecasting for Gaussian VAR models.

Ordinary least squares solves

.. math::

   \widehat B=\arg\min_B\lVert Y-XB\rVert_F^2,

while ``solver="lwr"`` implements the Morf lattice-whitening recursion used by
MVGC-compatible estimators.

References
----------
- Morf, M., Vieira, A., Lee, D. T. L., and Kailath, T. (1978). Recursive
  multichannel maximum entropy spectral estimation.
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
- ComplexBox: https://github.com/bmilinkovic/complexbox
''',
    "selection": r'''Temporal cross-validation and information-criterion VAR order selection.

For per-observation Gaussian log likelihood :math:`\ell`, parameter count
:math:`k`, and effective sample size :math:`N`,

.. math::

   \mathrm{AIC}=-2\ell+2k/N,
   \qquad
   \mathrm{BIC}=-2\ell+(k/N)\log N,

.. math::

   \mathrm{HQC}=-2\ell+2(k/N)\log\log N.

Inside temporal CV these criteria are diagnostics computed on each training
fold; only held-out NLL or RMSE determines ``best_order_``.

References
----------
- Akaike, H. (1974). A new look at statistical model identification.
- Schwarz, G. (1978). Estimating the dimension of a model.
- Hannan, E. J. and Quinn, B. G. (1979). Determination of autoregression order.
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
''',
    "control": r'''Control-theoretic transformations for linear Gaussian systems.

For

.. math::

   z_{t+1}=Az_t+w_t,\qquad y_t=Cz_t+v_t,

steady-state Kalman and innovations quantities are obtained from discrete
algebraic Riccati equations, including process--observation cross covariance
when required.

References
----------
- Kalman, R. E. (1960). A new approach to linear filtering and prediction.
- Anderson, B. D. O. and Moore, J. B. (1979). *Optimal Filtering*.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
''',
    "state_space": r'''Inference for linear Gaussian state-space systems.

The module provides Kalman filtering, Rauch--Tung--Striebel smoothing, N4SID
subspace identification, and expectation--maximisation refinement.

References
----------
- Kalman, R. E. (1960). Linear filtering and prediction.
- Rauch, H. E., Tung, F., and Striebel, C. T. (1965). Maximum-likelihood
  estimates of linear dynamic systems.
- Van Overschee, P. and De Moor, B. (1994). N4SID.
- Shumway, R. H. and Stoffer, D. S. (1982). EM for time-series smoothing.
''',
    "gaussian": r'''Gaussian information-theoretic measures from covariance matrices.

For :math:`X\in\mathbb R^d` with covariance :math:`\Sigma`,

.. math::

   H(X)=\tfrac12\log\left((2\pi e)^d\det\Sigma\right).

Mutual information and higher-order quantities are evaluated through stable
log-determinant identities.

References
----------
- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*.
- Rosas, F. E. et al. (2019). Quantifying high-order interdependencies via the
  O-information. *Physical Review E*, 100, 032305.
''',
    "dynamics": r'''Analytical dynamics, autocovariances and spectra for Gaussian models.

For a VAR transfer function :math:`H(f)=A(f)^{-1}`, the cross-spectrum is

.. math::

   S(f)=H(f)\Sigma H(f)^*.

References
----------
- Lütkepohl, H. (2005). Spectral representation of VAR processes.
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
''',
    "mvgc": r'''Temporal and spectral multivariate Granger causality.

Conditional time-domain GC is

.. math::

   F_{Y\to X\mid Z}
   =\log\frac{\det\Sigma^{R}_{XX}}{\det\Sigma_{XX}},

where the reduced covariance excludes the source history. Spectral GC uses
innovations-form transfer functions and integrates to the time-domain value.

References
----------
- Geweke, J. (1982). Measurement of linear dependence and feedback.
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
- Barnett, L. and Seth, A. K. (2015). State-space Granger causality.
- MVGC: https://github.com/lcbarnett/MVGC1
''',
    "mvgc_api": r'''Public dispatch layer for model-based and observation-based MVGC.

Canonical VAR/state-space inputs are routed to analytical primary measures;
finite observations are routed to explicitly empirical secondary estimators.

References
----------
- Barnett, L. and Seth, A. K. (2014, 2015).
''',
    "cmem": r'''Covariance-memory measures for stationary Gaussian dynamics.

The implementation combines present-time total correlation, innovation total
correlation, and lagged Gaussian covariance blocks to quantify memory totals,
curves and finite-lag decompositions.

References
----------
- Cover, T. M. and Thomas, J. A. (2006). Gaussian information identities.
- ComplexTorch methodological tests and model-backbone implementation.
''',
    "criticality": r'''Stability and criticality diagnostics for linear dynamics.

For companion spectral radius :math:`\rho<1`, the dominant discrete timescale is

.. math::

   \tau=-1/\log\rho.

References
----------
- Lütkepohl, H. (2005). Stability of VAR processes.
''',
    "emergence": r'''Gaussian predictive-emergence measures under linear projection.

The measures compare microscopic predictive information with predictive
information retained by a user-specified macroscopic projection, using exact
conditional covariances when a generative model is supplied.

References
----------
- Barnett, L. and Seth, A. K. (2023). Dynamical independence and emergent
  macroscopic processes.
''',
    "phid": r'''Gaussian integrated information decomposition (PhiID).

Time-delayed mutual information is decomposed into integrated information atoms
using the minimum-mutual-information redundancy prescription.

References
----------
- Mediano, P. A. M. et al. (2021). Integrated information decomposition.
- Reference implementation: https://github.com/Imperial-MIND-lab/integrated-info-decomp
''',
    "discrete": r'''Discrete information and sequence-complexity estimators.

Probabilities are empirical masses. Lempel--Ziv complexity counts novel phrases
in an incremental parsing of a finite symbol sequence.

References
----------
- Lempel, A. and Ziv, J. (1976). On the complexity of finite sequences.
- Cover, T. M. and Thomas, J. A. (2006).
''',
    "backbone": r'''Canonical analytical backbone shared by VAR and state-space models.

Models are mapped to common observation covariance, autocovariance, innovations
and spectral primitives. Measures consume these invariants rather than
reimplementing model-specific formulas.

References
----------
- Lütkepohl, H. (2005). Companion-form VAR representation.
- Barnett, L. and Seth, A. K. (2015). State-space Granger causality.
''',
    "primary": r'''Strict analytical measures computed from supplied generative models.

Primary functions do not refit observations. A shared context computes the
maximum required autocovariance lag and caches reusable model-derived
primitives.

References
----------
- Cover, T. M. and Thomas, J. A. (2006).
- Barnett, L. and Seth, A. K. (2014, 2015).
''',
    "secondary": r'''Empirical measures and sample-based estimation helpers.

Secondary functions operate on finite observations, discretised sequences or
sample covariances and therefore include sampling and fitting effects.

References
----------
- Barnett, L. and Seth, A. K. (2014). Empirical MVGC workflow.
''',
    "planner": r'''Planning and reuse of intermediate dynamical-measure calculations.

The planner resolves requested outputs and shares autocovariance, covariance and
spectral intermediates across compatible measures.

References
----------
- ComplexTorch canonical model-backbone design.
''',
    "registry": r'''Registry separating analytical primary and empirical secondary measures.

References
----------
- ComplexTorch public API design and model-comparison tests.
''',
    "_model_comparison": r'''Shared nested-model primitives for empirical Gaussian comparisons.

Reduced and full VAR models are fitted with consistent target/source ordering;
log-determinant ratios and conditional spectra are then reused by empirical
MVGC and related estimators.

References
----------
- Geweke, J. (1982).
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
''',
}

FUNCTION_DOCS = {
    "gaussian_entropy": r'''Compute Gaussian differential entropy.

.. math::

   H(X)=\tfrac12\log\left((2\pi e)^d\det\Sigma\right).

References
----------
- Cover and Thomas (2006).
''',
    "conditional_covariance": r'''Compute a Gaussian conditional covariance by Schur complement.

.. math::

   \Sigma_{X\mid Y}
   =\Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}.

References
----------
- Cover and Thomas (2006).
''',
    "gaussian_mutual_information": r'''Compute Gaussian mutual information from covariance blocks.

.. math::

   I(X;Y)=\tfrac12\log
   \frac{\det\Sigma_X\det\Sigma_Y}{\det\Sigma_{XY}}.

References
----------
- Cover and Thomas (2006).
''',
    "o_information": r'''Compute Gaussian O-information.

Positive values are redundancy-dominated and negative values are
synergy-dominated.

References
----------
- Rosas et al. (2019), *Physical Review E* 100, 032305.
''',
    "temporal_mvgc": r'''Compute conditional time-domain multivariate Granger causality.

.. math::

   F_{Y\to X\mid Z}
   =\log\frac{\det\Sigma^{R}_{XX}}{\det\Sigma_{XX}}.

References
----------
- Geweke (1982); Barnett and Seth (2014, 2015).
''',
    "spectral_mvgc": r'''Compute conditional spectral multivariate Granger causality.

The frequency-resolved decomposition is obtained from innovations-form transfer
functions and integrates to temporal GC.

References
----------
- Geweke (1982); Barnett and Seth (2014, 2015).
''',
    "integrate_spectral_mvgc": r'''Integrate one-sided spectral GC to the time-domain value.

For normalized :math:`f\in[0,1/2]`,

.. math::

   F=2\int_0^{1/2} f_{Y\to X}(\nu)\,d\nu.

References
----------
- Geweke (1982); Barnett and Seth (2014).
''',
    "solve_dare": r'''Solve the steady-state discrete algebraic Riccati equation.

References
----------
- Anderson and Moore (1979), *Optimal Filtering*.
''',
    "solve_generalized_dare": r'''Solve the generalized DARE with noise cross covariance.

References
----------
- Anderson and Moore (1979); Barnett and Seth (2015).
''',
    "innovations_form": r'''Convert a linear Gaussian model to steady-state innovations form.

References
----------
- Kalman (1960); Anderson and Moore (1979); Barnett and Seth (2015).
''',
    "var_to_innovations_state_space": r'''Convert a VAR(p) exactly to companion innovations state space.

References
----------
- Lütkepohl (2005); Barnett and Seth (2015).
''',
    "cross_spectral_density": r'''Compute cross-spectral density from a transfer function.

.. math::

   S(f)=H(f)\Sigma H(f)^*.

References
----------
- Lütkepohl (2005); Barnett and Seth (2014).
''',
    "automatic_burnin": r'''Choose burn-in length from the companion spectral radius.

The smallest integer :math:`T` satisfying :math:`\rho^T<\epsilon` is returned.

References
----------
- ComplexBox simulation convention; Lütkepohl (2005).
''',
    "consistency": r'''Compute the Ding--Bressler VAR consistency diagnostic.

References
----------
- Ding et al. (2000); Barnett and Seth (2014); ComplexBox.
''',
    "residual_whiteness": r'''Test residual serial correlation with the requested method.

The current Durbin--Watson route follows the ComplexBox-compatible
approximation while preserving an extensible ``method`` argument.

References
----------
- Durbin and Watson (1950, 1951); ComplexBox.
''',
    "mvgc_pvalue": r'''Compute asymptotic MVGC p-values using MVGC conventions.

References
----------
- Barnett and Seth (2014); MVGC repository.
''',
    "significance": r'''Apply uncorrected or Benjamini--Hochberg FDR testing.

References
----------
- Benjamini and Hochberg (1995).
''',
    "gaussian_phiid_atoms": r'''Compute Gaussian PhiID atoms under MMI redundancy.

References
----------
- Mediano et al. (2021).
''',
    "lempel_ziv_complexity": r'''Compute incremental Lempel--Ziv phrase complexity.

References
----------
- Lempel and Ziv (1976).
''',
}


def _doc_lines(text: str, indentation: str) -> list[str]:
    parts = text.strip().splitlines()
    return [f'{indentation}"""{parts[0]}\n'] + [
        f"{indentation}{line}\n" for line in parts[1:]
    ] + [f'{indentation}"""\n']


def process(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, list[str]]] = []
    module_doc = MODULE_DOCS.get(path.stem)
    if module_doc and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        first = tree.body[0]
        edits.append((first.lineno - 1, first.end_lineno, _doc_lines(module_doc, "")))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue
        canonical = FUNCTION_DOCS.get(node.name)
        if canonical is None:
            continue
        first = node.body[0]
        if ast.get_docstring(node, clean=False) is None:
            edits.append((first.lineno - 1, first.lineno - 1, _doc_lines(canonical, " " * (node.col_offset + 4))))
        else:
            edits.append((first.lineno - 1, first.end_lineno, _doc_lines(canonical, " " * (node.col_offset + 4))))
    for start, stop, replacement in sorted(edits, reverse=True):
        lines[start:stop] = replacement
    updated = "".join(lines)
    if updated != source:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = [path for path in sorted(SOURCE.rglob("*.py")) if process(path)]
    print(f"Normalized {len(changed)} files")
    for path in changed:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
