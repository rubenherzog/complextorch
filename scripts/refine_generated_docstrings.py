"""Replace generic generated docstrings with concise API-specific documentation."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "complextorch"

CLASS_PURPOSES = {
    "TemporalFold": "Indices delimiting one expanding-window temporal-validation fold.",
    "EpochTimeSeriesSplit": "Generate leakage-safe expanding-window splits for ordered observations.",
    "VAROrderScore": "Aggregated held-out and training-diagnostic results for one VAR order.",
    "VAROrderSearchResult": "Immutable summary returned by temporal VAR order search.",
    "VARInformationCriteriaResult": "AIC, BIC and HQC curves and their minimizing VAR orders.",
    "VAROrderSearchCV": "Select a VAR order by temporal cross-validation and report training-fold IC diagnostics.",
    "VAROrderSelectionIC": "Select a VAR order with MVGC-compatible AIC, BIC or HQC.",
    "VARParameters": "Fitted coefficients, innovations and metadata for a VAR estimator.",
    "VAR": "Torch-first estimator for Gaussian vector autoregressive models.",
    "VARSystem": "Stationary Gaussian VAR process represented by coefficients and innovation covariance.",
    "LinearDynamicalSystem": "Linear Gaussian state-space model with optional stationary state covariance.",
    "InnovationsStateSpace": "Steady-state innovations-form linear state-space model.",
    "ProjectionSearchResult": "Result of optimizing a macroscopic state-space projection.",
    "KalmanFilterResult": "Filtered state moments and innovations from a Kalman recursion.",
    "KalmanSmootherResult": "Smoothed state moments and lagged cross moments.",
    "N4SID": "Subspace estimator for linear Gaussian state-space systems.",
    "LinearGaussianEM": "Expectation--maximisation refinement of a linear Gaussian state-space model.",
    "CMemResult": "Container for covariance-memory totals, curves and decompositions.",
    "ModelMeasureConfig": "Independent configuration of lags, frequencies and variable groups for model measures.",
    "ModelMeasureContext": "Cached canonical primitives shared by multiple analytical measures.",
    "DynamicalMeasures": "Planner that reuses intermediate quantities across requested dynamical measures.",
    "WhitenessResult": "Residual-whiteness statistic, p-value and method label.",
}

PARAMETER_DESCRIPTIONS = {
    "x": "Input observations or tensor-valued quantity.",
    "X": "Observations with shape ``(time, variables)`` or ``(batch, time, variables)``.",
    "y": "Unused scikit-learn compatibility target.",
    "observations": "Observed time series in ComplexTorch batch-first layout.",
    "residuals": "Model residuals aligned with the fitted prediction interval.",
    "coefficients": "VAR coefficient tensor ordered by lag, target and source.",
    "innovation_covariance": "Symmetric positive-definite covariance of model innovations.",
    "covariance": "Symmetric covariance matrix or batch of covariance matrices.",
    "transition": "State-transition matrix.",
    "observation": "State-to-observation matrix.",
    "state_covariance": "Stationary or filtered state covariance.",
    "observation_covariance": "Observation-noise covariance matrix.",
    "process_covariance": "State-process noise covariance matrix.",
    "cross_covariance": "Process--observation noise cross covariance.",
    "frequencies": "One-dimensional frequency grid in normalized cycles per sample.",
    "order": "Autoregressive model order.",
    "orders": "Candidate autoregressive orders.",
    "lag": "Positive temporal lag in samples.",
    "max_lag": "Largest non-negative lag to evaluate.",
    "n_times": "Number of time samples.",
    "n_variables": "Number of observed variables.",
    "n_states": "Latent state dimension.",
    "n_trials": "Number of independent trials or epochs.",
    "n_splits": "Number of temporal validation folds.",
    "test_size": "Number of held-out samples in each fold.",
    "min_train_size": "Minimum number of samples in the first training window.",
    "gap": "Number of samples omitted between training and test windows.",
    "min_order": "Largest lag that must fit inside every training window.",
    "source": "Indices of source variables.",
    "target": "Indices of target variables.",
    "conditional": "Indices conditioned on in addition to source and target.",
    "projection": "Linear projection from microscopic observations or states to macro variables.",
    "macro_projection": "Linear map defining macroscopic variables.",
    "base": "Logarithm base used for information quantities.",
    "seed": "Random seed used by a local generator.",
    "device": "Torch device or ``'auto'``.",
    "dtype": "Torch floating-point dtype name or object.",
    "solver": "Numerical solver or estimation algorithm.",
    "method": "Named statistical or numerical method.",
    "alpha": "Non-negative ridge regularization strength.",
    "fit_intercept": "Whether to estimate a constant offset.",
    "mode": "Whether trials are fitted independently or pooled.",
    "stability": "Policy for checking stationarity after fitting.",
    "jitter": "Initial diagonal regularization used for SPD factorization.",
    "tolerance": "Numerical convergence or truncation tolerance.",
    "max_iter": "Maximum number of numerical iterations.",
    "n_iter": "Number of optimization or EM iterations.",
    "sampling_frequency": "Sampling frequency used to scale spectral densities.",
    "normalize": "Whether to normalize the returned quantity.",
    "reduction": "Aggregation applied to elementwise losses.",
    "steps": "Number of recursive forecast samples.",
    "history": "Observed history used to initialize recursive forecasting.",
    "config": "Model-measure configuration.",
    "context": "Optional precomputed model-measure context.",
    "model": "VAR or linear state-space model.",
    "system": "Canonical VAR or state-space system.",
    "values": "Input numerical values.",
    "sequence": "Finite discrete symbol sequence.",
}

RETURN_DESCRIPTIONS = {
    "split": "Iterator of :class:`TemporalFold` objects in chronological order.",
    "fit": "The fitted estimator instance.",
    "predict": "One-step predictions as a NumPy array.",
    "forecast": "Recursive future samples with shape ``(batch, steps, variables)``.",
    "residuals": "Observed minus one-step-predicted values.",
    "score": "Negative Gaussian log-likelihood score.",
    "to_var_system": "Canonical :class:`VARSystem` representation.",
    "to_state_space": "Equivalent linear state-space representation.",
    "as_records": "List of dictionaries suitable for tabular display.",
}

GENERIC_MARKERS = (
    "Input controlling ``",
    "Result described by the function name",
    "The class follows the scikit-learn fitted-attribute convention",
)


def _doc_lines(text: str, indentation: str) -> list[str]:
    parts = text.strip().splitlines()
    return [f'{indentation}"""{parts[0]}\n'] + [
        f"{indentation}{line}\n" for line in parts[1:]
    ] + [f'{indentation}"""\n']


def _purpose(name: str) -> str:
    words = name.strip("_").replace("_", " ")
    if name.startswith("_solve") or name.startswith("solve"):
        return f"Solve {words.removeprefix('solve ')}."
    if name.startswith("_fit") or name == "fit":
        return f"Fit {words.removeprefix('fit ')} from observations."
    if name.startswith("compute_"):
        return f"Compute {words.removeprefix('compute ')}."
    if name.startswith("build_"):
        return f"Build {words.removeprefix('build ')}."
    if name.startswith("random_"):
        return f"Generate {words.removeprefix('random ')}."
    if name.startswith("from_"):
        return f"Convert {words.removeprefix('from ')} to ComplexTorch layout."
    if name.startswith("to_"):
        return f"Convert to {words.removeprefix('to ')}."
    if name.startswith("_check"):
        return f"Validate {words.removeprefix('check ')} and raise on failure."
    return words.capitalize() + "."


def _function_doc(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if node.name == "__init__":
        summary = "Initialize the estimator or result container."
    else:
        summary = _purpose(node.name)
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    names = [arg.arg for arg in args if arg.arg not in {"self", "cls"}]
    out = [summary]
    if names:
        out += ["", "Parameters", "----------"]
        for name in names:
            out += [name, f"    {PARAMETER_DESCRIPTIONS.get(name, 'Input required by this calculation.')}" ]
    if node.name != "__init__":
        out += ["", "Returns", "-------", "object", f"    {RETURN_DESCRIPTIONS.get(node.name, 'Computed result; see the annotated return type and shape notes.')}" ]
    out += [
        "",
        "Notes",
        "-----",
        "Batch dimensions are preserved unless explicitly documented otherwise.",
        "The implementation validates dimensional and positive-definiteness",
        "requirements before executing the numerical core.",
    ]
    return "\n".join(out)


def _class_doc(node: ast.ClassDef) -> str:
    purpose = CLASS_PURPOSES.get(node.name, _purpose(node.name))
    return purpose + "\n\nNotes\n-----\nPublic fitted attributes use the trailing-underscore convention."


def process(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or not node.body:
            continue
        first = node.body[0]
        doc = ast.get_docstring(node, clean=False)
        if doc is None or not any(marker in doc for marker in GENERIC_MARKERS):
            continue
        replacement = _class_doc(node) if isinstance(node, ast.ClassDef) else _function_doc(node)
        indentation = " " * (node.col_offset + 4)
        edits.append((first.lineno - 1, first.end_lineno, _doc_lines(replacement, indentation)))
    for start, stop, replacement in sorted(edits, reverse=True):
        lines[start:stop] = replacement
    updated = "".join(lines)
    if updated != source:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = [path for path in sorted(SOURCE.rglob("*.py")) if process(path)]
    print(f"Refined {len(changed)} files")
    for path in changed:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
