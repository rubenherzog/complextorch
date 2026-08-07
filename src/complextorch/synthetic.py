"""Canonical synthetic Gaussian VAR(1) systems for controlled parameter studies.

The public :func:`synthetic_var` entry point separates three ingredients:

1. a named transition topology with explicit structural parameters;
2. an exact target spectral radius controlling dynamical persistence;
3. an equicorrelated innovation covariance controlling instantaneous
   redundancy/independence/synergy.

All implementations are Torch-first and support broadcast batch dimensions for
continuous parameters. The returned object is the canonical :class:`VARSystem`,
so synthetic systems can be passed directly to the analytical measure API.

Matrix orientation follows the ComplexTorch VAR convention
``A[target, source]``: a nonzero entry ``A[i, j]`` denotes ``j -> i``.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from .linalg import spectral_radius
from .representations import VARSystem, build_var_system


SYNTHETIC_SYSTEMS: tuple[str, ...] = (
    "uncoupled",
    "directed_chain",
    "nonnormal_feedforward",
    "directed_ring",
    "frustrated_ring",
    "fully_connected",
    "modular",
    "planted_modular",
    "erdos_renyi",
    "random_directed",
    "hub_broadcast",
    "hub_convergent",
)

SYNTHETIC_SYSTEM_PARAMETERS: dict[str, tuple[str, ...]] = {
    "uncoupled": ("self_coupling",),
    "directed_chain": ("self_coupling", "coupling"),
    "nonnormal_feedforward": ("self_coupling", "coupling"),
    "directed_ring": ("self_coupling", "coupling"),
    "frustrated_ring": ("self_coupling", "coupling"),
    "fully_connected": ("self_coupling", "coupling"),
    "modular": (
        "n_modules",
        "self_coupling",
        "within_coupling",
        "between_coupling",
    ),
    "planted_modular": ("n_modules", "self_coupling", "within_coupling"),
    "erdos_renyi": ("density", "self_coupling", "coupling", "seed"),
    "random_directed": ("density", "self_coupling", "coupling", "seed"),
    "hub_broadcast": ("hub", "self_coupling", "coupling"),
    "hub_convergent": ("hub", "self_coupling", "coupling"),
}

_ALIASES = {
    "chain": "directed_chain",
    "feedforward": "nonnormal_feedforward",
    "ring": "directed_ring",
    "cycle": "directed_ring",
    "signed_ring": "frustrated_ring",
    "random_network": "random_directed",
    "star_broadcast": "hub_broadcast",
    "star_convergent": "hub_convergent",
}


def _canonical_name(system: str) -> str:
    """Resolve aliases and validate a synthetic topology name."""

    canonical = _ALIASES.get(system.lower(), system.lower())
    if canonical not in SYNTHETIC_SYSTEMS:
        choices = ", ".join(SYNTHETIC_SYSTEMS)
        raise ValueError(f"unknown synthetic system {system!r}; choose from {choices}")
    return canonical


def available_synthetic_systems() -> tuple[str, ...]:
    """Return the canonical synthetic VAR topology names."""

    return SYNTHETIC_SYSTEMS


def synthetic_system_parameters(system: str) -> tuple[str, ...]:
    """Return topology-specific parameter names for programmatic search spaces."""

    return SYNTHETIC_SYSTEM_PARAMETERS[_canonical_name(system)]


def _validate_n_variables(n_variables: int) -> None:
    """Validate the minimum variable count required by synthetic systems."""

    if n_variables < 2:
        raise ValueError("n_variables must be at least 2")


def _parameter(
    value: float | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert a scalar or tensor parameter to the requested dtype and device."""

    return torch.as_tensor(value, dtype=dtype, device=device)


def _weighted_topology(
    adjacency: torch.Tensor,
    *,
    self_coupling: float | torch.Tensor,
    coupling: float | torch.Tensor,
) -> torch.Tensor:
    """Combine self-coupling and off-diagonal adjacency with broadcasting."""

    self_tensor = _parameter(
        self_coupling, dtype=adjacency.dtype, device=adjacency.device
    )
    coupling_tensor = _parameter(
        coupling, dtype=adjacency.dtype, device=adjacency.device
    )
    self_tensor, coupling_tensor = torch.broadcast_tensors(
        self_tensor, coupling_tensor
    )
    eye = torch.eye(
        adjacency.shape[-1], dtype=adjacency.dtype, device=adjacency.device
    )
    return (
        self_tensor[..., None, None] * eye
        + coupling_tensor[..., None, None] * adjacency
    )


def _normalise_topology(matrix: torch.Tensor) -> torch.Tensor:
    """Normalize each transition template to exact spectral radius one."""

    radius = spectral_radius(matrix)
    if bool((radius <= torch.finfo(matrix.dtype).eps).any()):
        raise ValueError(
            "topology has zero spectral radius; add nonzero self dynamics or "
            "choose a recurrent topology"
        )
    return matrix / radius[..., None, None]


def _directed_chain_adjacency(
    n_variables: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build a unit-weight directed-chain adjacency in target/source form."""

    adjacency = torch.zeros(
        (n_variables, n_variables), dtype=dtype, device=device
    )
    index = torch.arange(n_variables - 1, device=device)
    adjacency[index + 1, index] = 1.0
    return adjacency


def _directed_ring_adjacency(
    n_variables: int,
    *,
    frustrated: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a directed-ring adjacency, optionally with one frustrated edge."""

    adjacency = _directed_chain_adjacency(
        n_variables, dtype=dtype, device=device
    )
    adjacency[0, n_variables - 1] = 1.0
    if frustrated:
        # One inhibitory edge preserves the unsigned graph while changing its
        # signed interaction structure. Source 0 -> target 1 is flipped.
        adjacency[1, 0] = -1.0
    return adjacency


def _uncoupled(
    n_variables: int,
    *,
    self_coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the uncoupled normalized self-dynamics topology."""

    self_tensor = _parameter(self_coupling, dtype=dtype, device=device)
    eye = torch.eye(n_variables, dtype=dtype, device=device)
    return _normalise_topology(self_tensor[..., None, None] * eye)


def _directed_chain(
    n_variables: int,
    *,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized directed-chain transition template."""

    adjacency = _directed_chain_adjacency(
        n_variables, dtype=dtype, device=device
    )
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _nonnormal_feedforward(
    n_variables: int,
    *,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    r"""Dense feed-forward family ``D + eta U`` with fixed eigenvalues.

    ``U`` is strictly lower triangular in the ComplexTorch target/source
    orientation, so increasing ``coupling`` changes non-normality and transient
    amplification without changing the diagonal eigenvalues before spectral
    normalization.
    """

    adjacency = torch.tril(
        torch.ones((n_variables, n_variables), dtype=dtype, device=device),
        diagonal=-1,
    )
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _directed_ring(
    n_variables: int,
    *,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized directed-ring transition template."""

    adjacency = _directed_ring_adjacency(
        n_variables, frustrated=False, dtype=dtype, device=device
    )
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _frustrated_ring(
    n_variables: int,
    *,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized signed ring with one frustrated interaction."""

    adjacency = _directed_ring_adjacency(
        n_variables, frustrated=True, dtype=dtype, device=device
    )
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _fully_connected(
    n_variables: int,
    *,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized symmetric all-to-all transition template."""

    eye = torch.eye(n_variables, dtype=dtype, device=device)
    adjacency = torch.ones_like(eye) - eye
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _module_labels(n_variables: int, n_modules: int, device: torch.device) -> torch.Tensor:
    """Assign variables deterministically to nearly equal contiguous modules."""

    if not 1 < n_modules <= n_variables:
        raise ValueError("n_modules must lie in [2, n_variables]")
    # Nearly equal deterministic groups; unlike divisibility-based layouts this
    # remains defined for every N.
    return torch.div(
        torch.arange(n_variables, device=device) * n_modules,
        n_variables,
        rounding_mode="floor",
    )


def _modular(
    n_variables: int,
    *,
    n_modules: int = 2,
    self_coupling: float | torch.Tensor = 1.0,
    within_coupling: float | torch.Tensor = 1.0,
    between_coupling: float | torch.Tensor = 0.05,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized dense weighted modular transition template."""

    labels = _module_labels(n_variables, n_modules, device)
    same = labels[:, None] == labels[None, :]
    eye_mask = torch.eye(n_variables, dtype=torch.bool, device=device)
    within = (same & ~eye_mask).to(dtype)
    between = (~same).to(dtype)
    self_tensor, within_tensor, between_tensor = torch.broadcast_tensors(
        _parameter(self_coupling, dtype=dtype, device=device),
        _parameter(within_coupling, dtype=dtype, device=device),
        _parameter(between_coupling, dtype=dtype, device=device),
    )
    eye = torch.eye(n_variables, dtype=dtype, device=device)
    matrix = (
        self_tensor[..., None, None] * eye
        + within_tensor[..., None, None] * within
        + between_tensor[..., None, None] * between
    )
    return _normalise_topology(matrix)


def _planted_modular(
    n_variables: int,
    *,
    n_modules: int = 2,
    self_coupling: float | torch.Tensor = 1.0,
    within_coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Block-diagonal symmetric modules with invariant module-average modes."""

    return _modular(
        n_variables,
        n_modules=n_modules,
        self_coupling=self_coupling,
        within_coupling=within_coupling,
        between_coupling=0.0,
        dtype=dtype,
        device=device,
    )


def _erdos_renyi(
    n_variables: int,
    *,
    density: float = 0.35,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    seed: int = 0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a reproducible directed Erdos-Renyi transition template."""

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0, 1]")
    generator = torch.Generator(device=device).manual_seed(seed)
    adjacency = (
        torch.rand(
            (n_variables, n_variables),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        < density
    ).to(dtype)
    adjacency.fill_diagonal_(0.0)
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _random_directed(
    n_variables: int,
    *,
    density: float = 0.35,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    seed: int = 0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a reproducible sparse directed Gaussian-weight transition template."""

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0, 1]")
    generator = torch.Generator(device=device).manual_seed(seed)
    mask = (
        torch.rand(
            (n_variables, n_variables),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        < density
    ).to(dtype)
    mask.fill_diagonal_(0.0)
    weights = torch.randn(
        (n_variables, n_variables),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    adjacency = mask * weights
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _hub(
    n_variables: int,
    *,
    direction: str,
    hub: int = 0,
    self_coupling: float | torch.Tensor = 1.0,
    coupling: float | torch.Tensor = 1.0,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized broadcast or convergent hub transition template."""

    if not 0 <= hub < n_variables:
        raise ValueError("hub must index one of the variables")
    adjacency = torch.zeros(
        (n_variables, n_variables), dtype=dtype, device=device
    )
    others = torch.arange(n_variables, device=device)
    others = others[others != hub]
    if direction == "broadcast":
        adjacency[others, hub] = 1.0
    elif direction == "convergent":
        adjacency[hub, others] = 1.0
    else:  # pragma: no cover - private guard
        raise RuntimeError("unknown hub direction")
    return _normalise_topology(
        _weighted_topology(
            adjacency, self_coupling=self_coupling, coupling=coupling
        )
    )


def _hub_broadcast(n_variables: int, **kwargs) -> torch.Tensor:
    """Build a hub topology whose hub broadcasts to all other variables."""

    return _hub(n_variables, direction="broadcast", **kwargs)


def _hub_convergent(n_variables: int, **kwargs) -> torch.Tensor:
    """Build a hub topology whose other variables converge onto the hub."""

    return _hub(n_variables, direction="convergent", **kwargs)


_TOPOLOGY_BUILDERS: dict[str, Callable[..., torch.Tensor]] = {
    "uncoupled": _uncoupled,
    "directed_chain": _directed_chain,
    "nonnormal_feedforward": _nonnormal_feedforward,
    "directed_ring": _directed_ring,
    "frustrated_ring": _frustrated_ring,
    "fully_connected": _fully_connected,
    "modular": _modular,
    "planted_modular": _planted_modular,
    "erdos_renyi": _erdos_renyi,
    "random_directed": _random_directed,
    "hub_broadcast": _hub_broadcast,
    "hub_convergent": _hub_convergent,
}


def synthetic_transition_matrix(
    system: str,
    n_variables: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    **system_parameters,
) -> torch.Tensor:
    """Construct a normalized transition template for a named synthetic system.

    Parameters
    ----------
    system
        Topology name from :func:`available_synthetic_systems`. Common aliases
        such as ``"ring"`` and ``"random_network"`` are accepted.
    n_variables
        Number of observed variables. Every canonical family supports arbitrary
        ``N >= 2``.
    dtype, device
        Torch dtype and device of the returned transition matrix.
    **system_parameters
        Parameters specific to the selected topology. Continuous coupling
        parameters may be tensors; their broadcast dimensions are retained.
        :func:`synthetic_system_parameters` exposes the accepted names.

    Returns
    -------
    torch.Tensor
        Transition template with shape ``(..., N, N)`` and exact spectral
        radius one in every batch element.

    Notes
    -----
    ``directed_chain``, ``nonnormal_feedforward`` and the two hub systems are
    acyclic off the diagonal. Their cross-variable adjacency is nilpotent, so
    nonzero self dynamics are required for a meaningful spectral-radius axis.
    With fixed nonzero self dynamics, their ``coupling`` parameter changes
    non-normality without changing their eigenvalues before normalization.
    """

    _validate_n_variables(n_variables)
    canonical = _canonical_name(system)
    dev = torch.device(device)
    try:
        return _TOPOLOGY_BUILDERS[canonical](
            n_variables,
            dtype=dtype,
            device=dev,
            **system_parameters,
        )
    except TypeError as exc:
        raise TypeError(
            f"invalid parameters for synthetic system {canonical!r}: {exc}"
        ) from exc


def planted_module_projection(
    n_variables: int,
    n_modules: int = 2,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return row-orthonormal module-average modes for ``planted_modular``.

    The result has shape ``(n_modules, n_variables)``. Modules are the same
    nearly equal deterministic partition used by the transition generator.
    """

    _validate_n_variables(n_variables)
    dev = torch.device(device)
    labels = _module_labels(n_variables, n_modules, dev)
    projection = torch.zeros((n_modules, n_variables), dtype=dtype, device=dev)
    for module in range(n_modules):
        members = labels == module
        projection[module, members] = 1.0 / torch.sqrt(
            members.sum().to(dtype=dtype)
        )
    return projection


def equicorrelated_innovation_covariance(
    n_variables: int,
    correlation: float | torch.Tensor = 0.0,
    *,
    variance: float | torch.Tensor = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    r"""Return the compound-symmetry innovation covariance.

    .. math::

       Q(r) = \sigma^2[(1-r)I + r\mathbf{1}\mathbf{1}^\top].

    The positive-definite domain is ``-1/(N-1) < r < 1``. ``r=0`` gives
    independent innovations; positive and negative values provide controlled
    redundant/common-mode and compensatory/synergy-favoring noise structure.

    Parameters may carry arbitrary broadcast batch dimensions.
    """

    _validate_n_variables(n_variables)
    dev = torch.device(device)
    corr, var = torch.broadcast_tensors(
        _parameter(correlation, dtype=dtype, device=dev),
        _parameter(variance, dtype=dtype, device=dev),
    )
    lower = -1.0 / (n_variables - 1)
    if bool(((corr <= lower) | (corr >= 1.0)).any()):
        raise ValueError(f"correlation must satisfy {lower:.6g} < r < 1")
    if bool((var <= 0).any()):
        raise ValueError("variance must be strictly positive")
    eye = torch.eye(n_variables, dtype=dtype, device=dev)
    ones = torch.ones((n_variables, n_variables), dtype=dtype, device=dev)
    return var[..., None, None] * (
        (1.0 - corr)[..., None, None] * eye
        + corr[..., None, None] * ones
    )


def synthetic_var(
    system: str,
    n_variables: int,
    *,
    spectral_radius_target: float | torch.Tensor = 0.8,
    noise_correlation: float | torch.Tensor = 0.0,
    noise_variance: float | torch.Tensor = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    **system_parameters,
) -> VARSystem:
    """Build a stationary synthetic Gaussian VAR(1) with a named topology.

    This is the common entry point intended for parameter sweeps and numerical
    optimization. Topology, spectral persistence and innovation structure are
    deliberately factored so they can be changed independently.

    Parameters
    ----------
    system
        Named topology. See :func:`available_synthetic_systems`.
    n_variables
        Number of observed variables.
    spectral_radius_target
        Desired VAR spectral radius ``rho``. Values must satisfy ``0 < rho < 1``.
        Tensors are supported and broadcast against topology/noise parameters.
    noise_correlation
        Equicorrelation parameter ``r`` of the innovation covariance, satisfying
        ``-1/(N-1) < r < 1``.
    noise_variance
        Common innovation variance. Must be positive.
    dtype, device
        Torch dtype and device. Dtype/device are preserved by the returned model.
    **system_parameters
        Parameters for the selected topology. For example ``coupling`` for a
        chain/ring, ``n_modules`` and ``between_coupling`` for a modular system,
        or ``density``/``seed`` for random systems.

    Returns
    -------
    VARSystem
        Canonical ComplexTorch VAR representation. Continuous parameter batch
        dimensions are broadcast together and flattened into ``batch_size``;
        coefficients therefore have shape ``(batch, 1, N, N)``.

    Examples
    --------
    A rho/r grid can be evaluated without Python loops::

        rho, r = torch.meshgrid(
            torch.linspace(0.1, 0.95, 20),
            torch.linspace(-0.2, 0.8, 30),
            indexing="ij",
        )
        model = synthetic_var(
            "directed_ring", 6,
            spectral_radius_target=rho,
            noise_correlation=r,
        )

    For the non-normality/noise plane, keep ``spectral_radius_target`` fixed and
    broadcast a tensor-valued ``coupling`` in ``nonnormal_feedforward``.
    """

    _validate_n_variables(n_variables)
    dev = torch.device(device)
    template = synthetic_transition_matrix(
        system,
        n_variables,
        dtype=dtype,
        device=dev,
        **system_parameters,
    )
    rho = _parameter(spectral_radius_target, dtype=dtype, device=dev)
    if bool(((rho <= 0.0) | (rho >= 1.0)).any()):
        raise ValueError("spectral_radius_target must lie strictly in (0, 1)")
    covariance = equicorrelated_innovation_covariance(
        n_variables,
        noise_correlation,
        variance=noise_variance,
        dtype=dtype,
        device=dev,
    )

    batch_shape = torch.broadcast_shapes(
        template.shape[:-2], rho.shape, covariance.shape[:-2]
    )
    template = template.expand(*batch_shape, n_variables, n_variables)
    rho = rho.expand(batch_shape)
    covariance = covariance.expand(*batch_shape, n_variables, n_variables)

    transition = rho[..., None, None] * template
    coefficients = transition.reshape(-1, 1, n_variables, n_variables)
    covariance = covariance.reshape(-1, n_variables, n_variables)
    return build_var_system(coefficients, covariance)
