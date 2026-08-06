"""Refactor state-space order selection into the unified selection API."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "complextorch"

old_path = SRC / "model_order.py"
internal_path = SRC / "_state_space_order.py"
text = old_path.read_text(encoding="utf-8")
text = text.replace("from sklearn.base import BaseEstimator\n\n", "")
text = text.replace("def bauer_svc(", "def _bauer_svc(")
text = text.replace("class LarimoreStateSpaceOrderResult:", "class _StateSpaceOrderComputation:")
text = text.replace("LarimoreStateSpaceOrderResult", "_StateSpaceOrderComputation")
text = text.replace("def larimore_state_space_order(", "def _larimore_state_space_order(")
text = text.replace("bauer_svc(", "_bauer_svc(")
marker = "\n\nclass LarimoreStateSpaceOrder(BaseEstimator):"
if marker not in text:
    raise RuntimeError("Could not locate legacy estimator class")
text = text.split(marker, 1)[0].rstrip() + "\n"
text = text.replace(
    "State-space model-order selection by Bauer SVC and Larimore CVA.",
    "Internal Bauer SVC and Larimore CVA primitives for model-order selection.",
)
internal_path.write_text(text, encoding="utf-8")
old_path.unlink()

selection_path = SRC / "selection.py"
selection = selection_path.read_text(encoding="utf-8")
selection = selection.replace(
    '"""Temporal cross-validation and information-criterion VAR order selection.',
    'r"""Unified VAR and state-space model-order selection.',
    1,
)
selection = selection.replace(
    "from .var import VAR\n",
    "from .var import VAR\nfrom ._state_space_order import _larimore_state_space_order\n",
    1,
)
append = r'''

@dataclass(frozen=True)
class StateSpaceOrderSelectionResult:
    r"""Immutable result of state-space latent-order selection.

    Attributes
    ----------
    best_order
        Selected latent state dimension. A scalar tensor in pooled mode and
        one value per trajectory in independent mode.
    orders
        Candidate state dimensions represented by the final criterion axis.
    criterion
        Criterion values for every candidate order.
    criterion_name
        Name of the order-selection criterion; currently ``"bauer"``.
    subspace_method
        Method used to obtain the canonical-correlation spectrum; currently
        ``"larimore"``.
    canonical_correlations
        Unnormalised Larimore canonical correlations.
    normalized_canonical_correlations
        Correlations divided by the leading value for visualization only.
    n_effective
        Effective number of past/future Hankel columns.
    """

    best_order: torch.Tensor
    orders: torch.Tensor
    criterion: torch.Tensor
    criterion_name: str
    subspace_method: str
    canonical_correlations: torch.Tensor
    normalized_canonical_correlations: torch.Tensor
    n_effective: torch.Tensor


class StateSpaceOrderSelection(BaseEstimator):
    r"""Select latent state dimension using Larimore CVA and Bauer SVC.

    This estimator is the state-space counterpart of
    :class:`VAROrderSelectionIC`. It selects the latent state dimension
    :math:`r`, not the VAR lag order :math:`p`, and therefore keeps the two
    model families in separate estimators while exposing them through the same
    ``complextorch.selection`` architecture.

    Larimore CVA constructs whitened past/future block-Hankel covariances and
    returns canonical correlations :math:`\rho_i`. Bauer's singular-value
    criterion then minimizes

    .. math::

       \operatorname{SVC}(r)
       = \rho_{r+1}^{2}
         + \frac{2 n_y r\log N_{\mathrm{eff}}}{N_{\mathrm{eff}}}.

    Parameters
    ----------
    past_horizon
        Number of block rows in the past Hankel matrix.
    future_horizon
        Number of block rows in the future Hankel matrix. Defaults to
        ``past_horizon``.
    min_order
        Smallest candidate state dimension. Defaults to the number of observed
        variables, matching the full-model MVGC convention.
    subspace_method
        Subspace spectrum estimator. Only ``"larimore"`` is currently
        implemented.
    criterion
        State-order criterion. Only ``"bauer"`` is currently implemented.
    mode
        ``"pooled"`` concatenates Hankel columns across trajectories;
        ``"independent"`` selects one order per trajectory.
    ridge
        Non-negative diagonal regularizer used only in Cholesky whitening.
    device, dtype
        Torch execution device and floating-point dtype.
    refit
        Reserved for a future full state-space estimator. It must remain
        ``False`` because this class performs order selection only and does not
        estimate :math:`A,C,K,V`.

    References
    ----------
    - Larimore, W. E. (1990, 1996), canonical variate analysis for system
      identification.
    - Bauer, D. (2001). Order estimation for subspace methods. *Automatica*,
      37(10), 1561--1573.
    - ComplexBox ``mvgc.modelorder.tsdata_to_ssmo``.
    """

    def __init__(
        self,
        past_horizon: int,
        *,
        future_horizon: int | None = None,
        min_order: int | None = None,
        subspace_method: Literal["larimore"] = "larimore",
        criterion: Literal["bauer"] = "bauer",
        mode: Literal["pooled", "independent"] = "pooled",
        ridge: float = 1e-12,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
        refit: bool = False,
    ):
        """Initialize state-space order-selection settings."""
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon
        self.min_order = min_order
        self.subspace_method = subspace_method
        self.criterion = criterion
        self.mode = mode
        self.ridge = ridge
        self.device = device
        self.dtype = dtype
        self.refit = refit

    def fit(self, X: ArrayLike, y=None):
        """Estimate the canonical spectrum and select latent state order.

        Parameters
        ----------
        X
            Observations in ``(time, variables)`` or ComplexTorch batch-first
            ``(batch, time, variables)`` layout.
        y
            Unused scikit-learn compatibility target.

        Returns
        -------
        StateSpaceOrderSelection
            Fitted selector exposing ``best_order_``, ``orders_``,
            ``criterion_`` and canonical-correlation diagnostics.
        """
        del y
        if self.subspace_method != "larimore":
            raise ValueError("subspace_method must be 'larimore'")
        if self.criterion != "bauer":
            raise ValueError("criterion must be 'bauer'")
        if self.refit:
            raise ValueError(
                "refit=True is unavailable until a full Larimore state-space "
                "estimator is implemented"
            )

        computation = _larimore_state_space_order(
            X,
            self.past_horizon,
            future_horizon=self.future_horizon,
            min_order=self.min_order,
            mode=self.mode,
            ridge=self.ridge,
            device=self.device,
            dtype=self.dtype,
        )
        result = StateSpaceOrderSelectionResult(
            best_order=computation.best_order,
            orders=computation.candidate_orders,
            criterion=computation.criterion,
            criterion_name=self.criterion,
            subspace_method=self.subspace_method,
            canonical_correlations=computation.canonical_correlations,
            normalized_canonical_correlations=(
                computation.normalized_canonical_correlations
            ),
            n_effective=computation.n_effective,
        )
        self.result_ = result
        self.best_order_ = result.best_order
        self.orders_ = result.orders
        self.criterion_ = result.criterion
        self.criterion_name_ = result.criterion_name
        self.subspace_method_ = result.subspace_method
        self.canonical_correlations_ = result.canonical_correlations
        self.normalized_canonical_correlations_ = (
            result.normalized_canonical_correlations
        )
        self.n_effective_ = result.n_effective
        return self
'''
if "class StateSpaceOrderSelection(BaseEstimator):" in selection:
    raise RuntimeError("StateSpaceOrderSelection already exists")
selection_path.write_text(selection.rstrip() + append + "\n", encoding="utf-8")

init_path = SRC / "__init__.py"
init = init_path.read_text(encoding="utf-8")
init = init.replace(
    "from .selection import EpochTimeSeriesSplit, VAROrderSearchCV, VAROrderSearchResult, VAROrderSelectionIC, VARInformationCriteriaResult",
    "from .selection import EpochTimeSeriesSplit, VAROrderSearchCV, VAROrderSearchResult, VAROrderSelectionIC, VARInformationCriteriaResult, StateSpaceOrderSelection, StateSpaceOrderSelectionResult",
)
init = init.replace(
    "from .model_order import LarimoreStateSpaceOrder, LarimoreStateSpaceOrderResult, bauer_svc, larimore_state_space_order\n",
    "",
)
init = init.replace(
    '    "LarimoreStateSpaceOrder", "LarimoreStateSpaceOrderResult", "bauer_svc", "larimore_state_space_order",\n',
    '    "StateSpaceOrderSelection", "StateSpaceOrderSelectionResult",\n',
)
init = init.replace('__version__ = "0.7.0"', '__version__ = "0.7.1"')
init_path.write_text(init, encoding="utf-8")

pyproject_path = ROOT / "pyproject.toml"
pyproject = pyproject_path.read_text(encoding="utf-8").replace(
    'version="0.7.0"', 'version="0.7.1"'
)
pyproject_path.write_text(pyproject, encoding="utf-8")

test_path = ROOT / "tests" / "test_state_space_model_order.py"
tests = test_path.read_text(encoding="utf-8")
tests = tests.replace(
    "from complextorch import (\n    LarimoreStateSpaceOrder,\n    bauer_svc,\n    larimore_state_space_order,\n)",
    "from complextorch import StateSpaceOrderSelection\nfrom complextorch._state_space_order import (\n    _bauer_svc,\n    _larimore_state_space_order,\n)",
)
tests = tests.replace("bauer_svc(", "_bauer_svc(")
tests = tests.replace("larimore_state_space_order(", "_larimore_state_space_order(")
tests = tests.replace("LarimoreStateSpaceOrder(", "StateSpaceOrderSelection(")
tests = tests.replace("candidate_orders_", "orders_")
test_path.write_text(tests, encoding="utf-8")

example_path = ROOT / "examples" / "state_space_order_selection.py"
example = example_path.read_text(encoding="utf-8")
example = example.replace("LarimoreStateSpaceOrder", "StateSpaceOrderSelection")
example = example.replace("candidate_orders_", "orders_")
example_path.write_text(example, encoding="utf-8")
