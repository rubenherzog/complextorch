"""Enrich ComplexTorch source files with API-ready scientific documentation.

This one-off maintenance script is intentionally conservative: it edits only
Python docstrings and inserts comments immediately above recognised numerical
operations. It does not reformat or unparse the source tree.
"""
from __future__ import annotations

import ast
from pathlib import Path
from textwrap import indent

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "complextorch"

REFERENCES = {
    "adapters": """
Notes
-----
These adapters only permute axes; they do not alter numerical values.  The
ComplexBox/MVGC convention is ``(variables, time, trials)`` whereas
ComplexTorch uses ``(trials, time, variables)``.

References
----------
- Barnett, L. and Seth, A. K. (2014). The MVGC multivariate Granger causality
  toolbox. *Journal of Neuroscience Methods*, 223, 50--68.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox
""",
    "representations": """
Notes
-----
A VAR(p) process is represented as

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + \varepsilon_t,
   \qquad \varepsilon_t \sim \mathcal N(0,\Sigma).

Its companion-form state transition is used to connect VAR and linear
state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
""",
    "linalg": """
Notes
-----
Positive-definite operations use Cholesky factorisation whenever possible.
For a symmetric positive-definite matrix :math:`S=LL^\top`,

.. math::

   \log\det S = 2\sum_i \log L_{ii}.

References
----------
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*.
- PyTorch linear algebra: https://pytorch.org/docs/stable/linalg.html
""",
    "simulate": """
Notes
-----
VAR simulations implement the recursion

.. math::

   x_t = c + \sum_{k=1}^{p} A_k x_{t-k} + \varepsilon_t.

The automatic burn-in uses the companion spectral radius to suppress initial
conditions below a prescribed tolerance.

References
----------
- Lütkepohl, H. (2005), Chapters 2--3.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox
""",
    "var": """
Notes
-----
Ordinary least squares minimises

.. math::

   \widehat B = \arg\min_B \lVert Y-XB\rVert_F^2,

while the LWR route implements the Morf lattice-whitening recursion used by
MVGC-compatible estimators.

References
----------
- Morf, M., Vieira, A., Lee, D. T. L., and Kailath, T. (1978). Recursive
  multichannel maximum entropy spectral estimation.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox
""",
    "selection": """
Notes
-----
The information criteria are evaluated from the fitted Gaussian innovation
covariance. In per-observation form,

.. math::

   \mathrm{AIC}=-2\ell+2k/N,\quad
   \mathrm{BIC}=-2\ell+(k/N)\log N,

.. math::

   \mathrm{HQC}=-2\ell+2(k/N)\log\log N.

Within temporal cross-validation these quantities are training-fold
diagnostics only; held-out NLL or RMSE determines model selection.

References
----------
- Akaike, H. (1974). A new look at the statistical model identification.
- Schwarz, G. (1978). Estimating the dimension of a model.
- Hannan, E. J. and Quinn, B. G. (1979). The determination of the order of an
  autoregression.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
""",
    "control": """
Notes
-----
State-space routines use steady-state Kalman filtering and discrete algebraic
Riccati equations. For

.. math::

   z_{t+1}=Az_t+w_t,\qquad y_t=Cz_t+v_t,

an innovations representation is obtained by solving the corresponding DARE
for the steady-state prediction covariance.

References
----------
- Kalman, R. E. (1960). A new approach to linear filtering and prediction.
- Anderson, B. D. O. and Moore, J. B. (1979). *Optimal Filtering*.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
  *Physical Review E*, 91, 040101.
""",
    "state_space": """
Notes
-----
The package combines Kalman filtering/smoothing, N4SID subspace
identification, and expectation--maximisation for linear Gaussian systems.

References
----------
- Kalman, R. E. (1960). A new approach to linear filtering and prediction.
- Rauch, H. E., Tung, F., and Striebel, C. T. (1965). Maximum likelihood
  estimates of linear dynamic systems.
- Van Overschee, P. and De Moor, B. (1994). N4SID: Subspace algorithms for the
  identification of combined deterministic-stochastic systems.
- Shumway, R. H. and Stoffer, D. S. (1982). An approach to time series
  smoothing and forecasting using the EM algorithm.
""",
    "gaussian": """
Notes
-----
For a :math:`d`-dimensional Gaussian variable with covariance :math:`\Sigma`,

.. math::

   H(X)=\tfrac12\log\left((2\pi e)^d\det\Sigma\right).

Mutual informations and multivariate information measures are evaluated from
log-determinant identities.

References
----------
- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*.
- Rosas, F. E. et al. (2019). Quantifying high-order interdependencies via the
  O-information. *Physical Review E*, 100, 032305.
""",
    "dynamics": """
Notes
-----
Autocovariances, transfer functions and spectra are derived analytically from
stationary VAR/state-space parameters. For a VAR transfer function
:math:`H(f)=A(f)^{-1}`, the spectrum is

.. math::

   S(f)=H(f)\Sigma H(f)^*.

References
----------
- Lütkepohl, H. (2005), spectral representation of VAR processes.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
""",
    "mvgc": """
Notes
-----
Conditional time-domain Granger causality is the log ratio of reduced and full
innovation generalised variances,

.. math::

   F_{Y\to X\mid Z}=\log\frac{\det\Sigma^{R}_{XX}}
                                {\det\Sigma_{XX}}.

Spectral GC is computed from innovations-form transfer functions and integrates
to the time-domain value under the Geweke decomposition.

References
----------
- Geweke, J. (1982). Measurement of linear dependence and feedback between
  multiple time series.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
- Barnett, L. and Seth, A. K. (2015), state-space Granger causality.
- MVGC repository: https://github.com/lcbarnett/MVGC1
""",
    "mvgc_api": """
Notes
-----
This compatibility layer dispatches canonical model inputs to analytical MVGC
and legacy observation inputs to explicitly empirical estimators.

References
----------
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
""",
    "cmem": """
Notes
-----
CMem quantities decompose covariance-memory effects using Gaussian total
correlation and lagged covariance blocks. All determinants are evaluated with
positive-definite linear algebra.

References
----------
- Cover, T. M. and Thomas, J. A. (2006), Gaussian information identities.
- ComplexTorch repository methodological notes and tests.
""",
    "criticality": """
Notes
-----
Stability diagnostics are derived from the companion eigenvalues. The dominant
timescale associated with spectral radius :math:`\rho<1` is

.. math::

   \tau=-1/\log\rho.

References
----------
- Lütkepohl, H. (2005), stability of VAR processes.
""",
    "emergence": """
Notes
-----
Emergence measures compare predictive information at microscopic and projected
macroscopic levels, using exact Gaussian conditional covariances whenever a
model representation is available.

References
----------
- Barnett, L. and Seth, A. K. (2023). Dynamical independence: discovering
  emergent macroscopic processes in complex dynamical systems.
""",
    "phid": """
Notes
-----
Gaussian PhiID decomposes time-delayed mutual information into integrated
information atoms. The implementation uses the minimum-mutual-information
redundancy prescription.

References
----------
- Mediano, P. A. M. et al. (2021). Towards an extended taxonomy of information
  dynamics via integrated information decomposition.
- dit PhiID-related implementations: https://github.com/Imperial-MIND-lab/integrated-info-decomp
""",
    "discrete": """
Notes
-----
Discrete estimators use empirical probability masses. Lempel--Ziv complexity
counts novel phrases in the incremental parsing of a finite symbol sequence.

References
----------
- Lempel, A. and Ziv, J. (1976). On the complexity of finite sequences.
- Cover, T. M. and Thomas, J. A. (2006).
""",
    "backbone": """
Notes
-----
The canonical backbone maps VAR and linear state-space models to shared
covariance, innovation, spectral and autocovariance primitives. Measures are
then computed from these invariants rather than duplicated by model class.

References
----------
- Barnett, L. and Seth, A. K. (2015), state-space Granger causality.
- Lütkepohl, H. (2005), companion-form VAR representations.
""",
    "primary": """
Notes
-----
Primary measures are analytical functions of a supplied generative model. They
must not refit observations internally. Shared contexts cache the maximum
required autocovariance lag and other model-derived primitives.

References
----------
- Barnett, L. and Seth, A. K. (2014, 2015).
- Cover, T. M. and Thomas, J. A. (2006).
""",
    "secondary": """
Notes
-----
Secondary measures estimate quantities from finite observations and therefore
include sampling, fitting and discretisation effects. They are kept separate
from analytical model-derived measures.

References
----------
- Barnett, L. and Seth, A. K. (2014), empirical MVGC workflow.
""",
    "planner": """
Notes
-----
The planner coordinates requested dynamical measures and reuses shared
intermediate quantities to avoid repeated covariance or spectral calculations.
""",
    "registry": """
Notes
-----
The registry declares which measures belong to the analytical primary tier and
which require observations or sample estimators.
""",
}

FUNCTION_DOCS = {
    "gaussian_entropy": """Compute differential entropy of a Gaussian covariance.

The implemented identity is

.. math:: H(X)=\tfrac12\log((2\pi e)^d\det\Sigma).

References
----------
Cover and Thomas (2006), *Elements of Information Theory*.
""",
    "gaussian_mutual_information": """Compute Gaussian mutual information from covariance blocks.

.. math:: I(X;Y)=\tfrac12\log\frac{\det\Sigma_X\det\Sigma_Y}{\det\Sigma_{XY}}.

References
----------
Cover and Thomas (2006).
""",
    "conditional_covariance": """Return the Gaussian conditional covariance via a Schur complement.

.. math:: \Sigma_{X\mid Y}=\Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}.

References
----------
Cover and Thomas (2006).
""",
    "o_information": """Compute Gaussian O-information.

Positive values indicate redundancy-dominated dependence and negative values
indicate synergy-dominated dependence.

References
----------
Rosas et al. (2019), *Physical Review E* 100, 032305.
""",
    "temporal_mvgc": """Compute conditional time-domain multivariate Granger causality.

.. math:: F_{Y\to X\mid Z}=\log(\det\Sigma^R_{XX}/\det\Sigma_{XX}).

References
----------
Geweke (1982); Barnett and Seth (2014, 2015).
""",
    "spectral_mvgc": """Compute conditional spectral multivariate Granger causality.

The frequency-resolved decomposition is obtained from innovations-form transfer
functions and integrates to temporal GC.

References
----------
Geweke (1982); Barnett and Seth (2014, 2015).
""",
    "integrate_spectral_mvgc": """Integrate one-sided spectral GC to its time-domain value.

For normalised frequencies :math:`f\in[0,1/2]`, the implementation evaluates
:math:`2\int_0^{1/2} f_{Y\to X}(\nu)\,d\nu`.

References
----------
Geweke (1982); Barnett and Seth (2014).
""",
    "solve_dare": """Solve a discrete algebraic Riccati equation for steady-state covariance.

References
----------
Anderson and Moore (1979), *Optimal Filtering*; SciPy/PyTorch-compatible control
implementations.
""",
    "solve_generalized_dare": """Solve the generalised DARE with process--observation cross covariance.

References
----------
Anderson and Moore (1979); Barnett and Seth (2015).
""",
    "innovations_form": """Convert a linear Gaussian state-space model to steady-state innovations form.

References
----------
Kalman (1960); Anderson and Moore (1979); Barnett and Seth (2015).
""",
    "var_to_innovations_state_space": """Convert a VAR(p) exactly to companion innovations state space.

References
----------
Lütkepohl (2005); Barnett and Seth (2015).
""",
    "autocovariances": """Compute stationary observation autocovariances.

For state transition :math:`A`, state covariance :math:`P` and observation
matrix :math:`C`, positive-lag covariances use :math:`C A^\tau P C^\top`.

References
----------
Lütkepohl (2005); Anderson and Moore (1979).
""",
    "cross_spectral_density": """Compute the cross-spectral density from a transfer function.

.. math:: S(f)=H(f)\Sigma H(f)^*.

References
----------
Lütkepohl (2005); Barnett and Seth (2014).
""",
    "random_stable_var": """Generate random VAR coefficients scaled to a target spectral radius.

References
----------
Lütkepohl (2005); ComplexBox repository.
""",
    "simulate_var": """Simulate one or more Gaussian VAR trajectories.

References
----------
Lütkepohl (2005); ComplexBox repository.
""",
    "automatic_burnin": """Choose burn-in length from the companion spectral radius.

The smallest integer :math:`T` satisfying :math:`\rho^T<\epsilon` is used.
""",
    "consistency": """Compute the Ding--Bressler VAR consistency diagnostic.

References
----------
Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
""",
    "residual_whiteness": """Test residual serial correlation with the requested method.

The current Durbin--Watson route follows the ComplexBox/MVGC-compatible
approximation while retaining an extensible ``method`` argument.
""",
    "mvgc_pvalue": """Compute asymptotic MVGC p-values using MVGC2 conventions.

References
----------
Barnett and Seth (2014); MVGC repository.
""",
    "significance": """Apply uncorrected or Benjamini--Hochberg FDR significance testing.

References
----------
Benjamini and Hochberg (1995).
""",
    "gaussian_phiid_atoms": """Compute Gaussian PhiID atoms under MMI redundancy.

References
----------
Mediano et al. (2021), integrated information decomposition.
""",
    "lempel_ziv_complexity": """Compute incremental Lempel--Ziv phrase complexity.

References
----------
Lempel and Ziv (1976).
""",
}

CRITICAL_PATTERNS = {
    "torch.linalg.cholesky": "# Cholesky factorisation preserves the SPD structure and avoids explicit inversion.",
    "stable_cholesky(": "# Add adaptive jitter only when required to retain a valid SPD factorisation.",
    "spd_logdet(": "# Evaluate log-determinants through an SPD-aware factorisation for numerical stability.",
    "solve_generalized_dare(": "# Marginal innovations require the steady-state generalised Riccati solution.",
    "torch.linalg.eigvals": "# Companion eigenvalues determine stationarity through their spectral radius.",
    "reflection =": "# The normalised forward/backward cross-covariance is the lattice reflection coefficient.",
    "torch.trapz(": "# One-sided spectral integration recovers the corresponding time-domain quantity.",
    "torch.trapezoid(": "# One-sided spectral integration recovers the corresponding time-domain quantity.",
    "torch.cholesky_solve": "# Solve with the Cholesky factor instead of forming the covariance inverse.",
}


def _module_key(path: Path) -> str:
    return path.stem


def _format_doc(text: str, indentation: str) -> list[str]:
    text = text.strip("\n")
    return [f'{indentation}"""{text.splitlines()[0]}\n'] + [
        f"{indentation}{line}\n" for line in text.splitlines()[1:]
    ] + [f'{indentation}"""\n']


def _generic_function_doc(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
    args = [name for name in args if name not in {"self", "cls"}]
    title = node.name.replace("_", " ").capitalize() + "."
    sections = [title]
    if args:
        sections.extend(["", "Parameters", "----------"])
        sections.extend(f"{name}\n    Input controlling ``{node.name}``." for name in args)
    sections.extend([
        "",
        "Returns",
        "-------",
        "object",
        "    Result described by the function name and annotated return type.",
        "",
        "Notes",
        "-----",
        "Tensor batch dimensions are preserved unless the public API explicitly",
        "documents a squeeze operation. Numerical validation is performed by the",
        "module before the core calculation.",
    ])
    return "\n".join(sections)


def _enrich_existing_doc(original: str, addition: str) -> str:
    if "References\n----------" in original and addition.strip() in original:
        return original
    return original.rstrip() + "\n\n" + addition.strip() + "\n"


def process_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, list[str]]] = []
    key = _module_key(path)
    module_note = REFERENCES.get(key)

    if module_note:
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
            node = tree.body[0]
            original = node.value.value
            replacement = _enrich_existing_doc(original, module_note)
            edits.append((node.lineno - 1, node.end_lineno, _format_doc(replacement, "")))
        else:
            edits.append((0, 0, _format_doc(f"{path.stem} numerical routines.\n\n{module_note}", "")))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if not body:
            continue
        existing = ast.get_docstring(node, clean=False)
        custom = FUNCTION_DOCS.get(node.name)
        if existing is not None and custom is None:
            continue
        if isinstance(node, ast.ClassDef):
            generated = custom or (
                f"{node.name.replace('_', ' ')}.\n\n"
                "Notes\n-----\n"
                "The class follows the scikit-learn fitted-attribute convention when applicable."
            )
        else:
            generated = custom or _generic_function_doc(node)
        indentation = " " * (node.col_offset + 4)
        first = body[0]
        if existing is not None:
            replacement = _enrich_existing_doc(existing, generated) if custom else existing
            edits.append((first.lineno - 1, first.end_lineno, _format_doc(replacement, indentation)))
        elif first.lineno > node.lineno:
            edits.append((first.lineno - 1, first.lineno - 1, _format_doc(generated, indentation)))

    for start, stop, replacement in sorted(edits, reverse=True):
        lines[start:stop] = replacement

    documented = "".join(lines)
    output: list[str] = []
    for line in documented.splitlines(keepends=True):
        stripped = line.lstrip()
        indentation = line[: len(line) - len(stripped)]
        for pattern, comment in CRITICAL_PATTERNS.items():
            if pattern in stripped and not (output and output[-1].strip() == comment):
                output.append(f"{indentation}{comment}\n")
                break
        output.append(line)
    documented = "".join(output)
    if documented != source:
        path.write_text(documented, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for path in sorted(SOURCE.rglob("*.py")):
        if process_file(path):
            changed.append(path.relative_to(ROOT).as_posix())
    print(f"Documented {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
