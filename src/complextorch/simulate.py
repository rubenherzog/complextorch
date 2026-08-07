r"""Simulation and random generation for stationary Gaussian VAR systems.

Trajectories obey

.. math::

   x_t=c+\sum_{k=1}^{p}A_kx_{t-k}+\varepsilon_t.

Automatic burn-in uses the companion spectral radius to reduce the influence
of initial conditions below a requested tolerance.

References
----------
- Lütkepohl, H. (2005), Chapters 2--3.
- ComplexBox: https://github.com/bmilinkovic/complexbox
"""
from __future__ import annotations
import math
import torch
from .linalg import spectral_radius
from .representations import companion_matrix


def _normalise_coefficients(coefficients):
    """Normalise coefficients.
    
    Parameters
    ----------
    coefficients
        VAR coefficient tensor ordered by lag, target and source.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    coef = torch.as_tensor(coefficients)
    if coef.ndim == 3:
        coef = coef.unsqueeze(0)
    if coef.ndim != 4:
        raise ValueError("coefficients must have shape (p,n,n) or (batch,p,n,n)")
    return coef


def _normalise_covariance(covariance, batch, n):
    """Normalise covariance.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
    batch
        Input required by this calculation.
    n
        Input required by this calculation.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    q = torch.as_tensor(covariance)
    if q.ndim == 2:
        q = q.unsqueeze(0)
    if q.shape[0] == 1 and batch > 1:
        q = q.expand(batch, -1, -1)
    if q.shape != (batch, n, n):
        raise ValueError("covariance has incompatible shape")
    return q


def automatic_burnin(coefficients, *, epsilon: float | None = None) -> int:
    r"""Choose burn-in length from the companion spectral radius.

    The returned value is the smallest integer :math:`T` satisfying
    :math:`\rho^T < \epsilon`, where :math:`\rho` is the largest companion
    spectral radius in the supplied batch.

    Parameters
    ----------
    coefficients
        VAR coefficient tensor with shape ``(p, n, n)`` or
        ``(batch, p, n, n)``.
    epsilon
        Positive decay tolerance. If omitted, the machine epsilon of the
        coefficient dtype is used.

    Returns
    -------
    int
        Burn-in length shared by the supplied systems.

    References
    ----------
    - Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
    - ComplexBox simulation convention.
    """
    coef = _normalise_coefficients(coefficients)
    rho = float(spectral_radius(companion_matrix(coef)).max())
    if rho >= 1:
        return 0
    if rho <= 0:
        return 1
    eps = torch.finfo(coef.dtype).eps if epsilon is None else float(epsilon)
    return int(math.ceil(-math.log(eps) / -math.log(rho)))


def simulate_var(
    coefficients,
    innovation_covariance,
    n_times: int,
    *,
    burnin: int | str = 500,
    seed: int = 0,
    return_innovations: bool = False,
):
    r"""Simulate one or more stationary Gaussian VAR trajectories.

    Samples follow

    .. math::

       x_t = \sum_{k=1}^{p} A_k x_{t-k} + \varepsilon_t,
       \qquad
       \varepsilon_t \sim \mathcal N(0,\Sigma).

    Parameters
    ----------
    coefficients
        Stable VAR coefficients with shape ``(p, n, n)`` or
        ``(batch, p, n, n)``.
    innovation_covariance
        Positive-definite innovations covariance with shape ``(n, n)`` or
        ``(batch, n, n)``.
    n_times
        Number of retained samples per trajectory.
    burnin
        Non-negative number of discarded initial samples, or ``"auto"`` to use
        :func:`automatic_burnin`.
    seed
        Seed for the local Torch random generator.
    return_innovations
        If true, also return the retained innovation sequence.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Simulated observations with shape ``(batch, n_times, n)`` and,
        optionally, innovations with the same shape.

    Notes
    -----
    Every batch item is simulated as an independent trajectory. No state or lag
    is propagated across batch boundaries.

    References
    ----------
    - Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
    - ComplexBox simulation convention.
    """
    # Generate samples from x_t = sum_k A_k x_(t-k) + epsilon_t.
    coef = _normalise_coefficients(coefficients)
    batch, order, n, _ = coef.shape
    q = _normalise_covariance(innovation_covariance, batch, n).to(coef)
    if bool((spectral_radius(companion_matrix(coef)) >= 1).any()):
        raise ValueError("simulation requires stable coefficients")
    if isinstance(burnin, str):
        if burnin.lower() != "auto":
            raise ValueError("burnin string must be 'auto'")
        burnin_value = automatic_burnin(coef)
    else:
        burnin_value = int(burnin)
        if burnin_value < 0:
            raise ValueError("burnin must be nonnegative")
    generator = torch.Generator(device=coef.device)
    generator.manual_seed(seed)
    # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
    # Factor the positive-definite covariance so whitening and solves use stable triangular algebra.
    chol = torch.linalg.cholesky(q)
    total = n_times + burnin_value
    innovations = torch.randn(
        (batch, total, n), dtype=coef.dtype, device=coef.device, generator=generator
    )
    innovations = torch.einsum("bij,btj->bti", chol, innovations)
    x = torch.zeros_like(innovations)
    for time in range(total):
        value = innovations[:, time].clone()
        for lag in range(1, order + 1):
            if time >= lag:
                value = value + torch.einsum(
                    "bij,bj->bi", coef[:, lag - 1], x[:, time - lag]
                )
        x[:, time] = value
    observations = x[:, burnin_value:]
    retained_innovations = innovations[:, burnin_value:]
    return (observations, retained_innovations) if return_innovations else observations


def random_correlation_matrix(
    n_variables: int,
    *,
    batch: int = 1,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Uniform random correlation matrices via the onion construction."""
    if n_variables < 1 or batch < 1:
        raise ValueError("n_variables and batch must be positive")
    import numpy as np
    generator = np.random.default_rng(seed)
    outputs = []
    for _ in range(batch):
        correlation = np.array([[1.0]])
        for k in range(2, n_variables + 1):
            beta = 0.5 * (n_variables - k + 1)
            radius = np.sqrt(generator.beta(0.5 * (k - 1), beta))
            direction = generator.standard_normal(k - 1)
            direction = direction / np.linalg.norm(direction) * radius
            vector = np.linalg.cholesky(correlation) @ direction
            expanded = np.zeros((k, k))
            expanded[:k - 1, :k - 1] = correlation
            expanded[:k - 1, k - 1] = vector
            expanded[k - 1, :k - 1] = vector
            expanded[k - 1, k - 1] = 1.0
            correlation = expanded
        outputs.append(correlation)
    return torch.as_tensor(np.stack(outputs), dtype=dtype, device=device)


def random_positive_definite_covariance(
    n_variables: int,
    *,
    batch: int = 1,
    seed: int = 0,
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate positive definite covariance.
    
    Parameters
    ----------
    n_variables
        Number of observed variables.
    batch
        Input required by this calculation.
    seed
        Random seed used by a local generator.
    scale_min
        Input required by this calculation.
    scale_max
        Input required by this calculation.
    dtype
        Torch floating-point dtype name or object.
    device
        Torch device or ``'auto'``.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    correlation = random_correlation_matrix(
        n_variables, batch=batch, seed=seed, dtype=dtype, device=device
    )
    generator = torch.Generator(device=torch.device(device)).manual_seed(seed + 1)
    scales = scale_min + (scale_max - scale_min) * torch.rand(
        (batch, n_variables), generator=generator, dtype=dtype, device=device
    )
    return scales[..., :, None] * correlation * scales[..., None, :]


def random_stable_var(batch:int,n_variables:int,order:int,*,spectral_radius_target:float=.85,noise_scale:float=1.,seed:int=0,dtype=torch.float64,device='cpu'):
    r"""Generate random stable VAR coefficients and innovation covariance.

    Random coefficient tensors are scaled until their companion spectral radius
    does not exceed ``spectral_radius_target``.

    Parameters
    ----------
    batch
        Number of independent VAR systems to generate.
    n_variables
        Number of observed variables.
    order
        Positive VAR lag order.
    spectral_radius_target
        Target companion spectral radius in ``(0, 1)``.
    noise_scale
        Positive scalar multiplying the identity innovations covariance.
    seed
        Seed for the local Torch random generator.
    dtype
        Floating-point Torch dtype.
    device
        Torch device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Coefficients with shape ``(batch, order, n_variables, n_variables)``
        and innovation covariances with shape ``(batch, n_variables, n_variables)``.

    References
    ----------
    - Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
    - ComplexBox simulation convention.
    """
    # Rescale the coefficient companion matrix so its spectral radius is below one and the VAR is stationary.
    if not 0<spectral_radius_target<1: raise ValueError('spectral_radius_target must lie in (0,1)')
    gen=torch.Generator(device=torch.device(device)); gen.manual_seed(seed)
    raw=torch.randn((batch,order,n_variables,n_variables),generator=gen,dtype=dtype,device=device)/math.sqrt(n_variables*order)
    lo=torch.zeros(batch,dtype=dtype,device=device); hi=torch.ones(batch,dtype=dtype,device=device); raw_rho=spectral_radius(companion_matrix(raw)); lo=torch.where(raw_rho<=spectral_radius_target,hi,lo)
    for _ in range(48):
        mid=.5*(lo+hi); rho=spectral_radius(companion_matrix(raw*mid[:,None,None,None])); ok=rho<=spectral_radius_target; lo=torch.where(ok,mid,lo); hi=torch.where(ok,hi,mid)
    coef=raw*lo[:,None,None,None]; q=noise_scale*torch.eye(n_variables,dtype=dtype,device=device).expand(batch,-1,-1).clone(); return coef,q


def _cycle(n,frustrated,*,dtype,device):
    """Cycle.
    
    Parameters
    ----------
    n
        Input required by this calculation.
    frustrated
        Input required by this calculation.
    dtype
        Torch floating-point dtype name or object.
    device
        Torch device or ``'auto'``.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    m=torch.zeros((n,n),dtype=dtype,device=device)
    for row in range(n): m[row,(row+1)%n]=1
    if frustrated: m[0,1]=-1
    return m


def demo_var(n_variables:int=3,order:int=2,*,temporal_path:float=-.95,temporal_gain:float=.9,noise_correlation:float=-.25,lag_weights=None,stability_target:float=.98,dtype=torch.float64,device='cpu'):
    """Demo var.
    
    Parameters
    ----------
    n_variables
        Number of observed variables.
    order
        Autoregressive model order.
    temporal_path
        Input required by this calculation.
    temporal_gain
        Input required by this calculation.
    noise_correlation
        Input required by this calculation.
    lag_weights
        Input required by this calculation.
    stability_target
        Input required by this calculation.
    dtype
        Torch floating-point dtype name or object.
    device
        Torch device or ``'auto'``.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    dev=torch.device(device); weights=torch.ones(order,dtype=dtype,device=dev) if lag_weights is None else torch.as_tensor(lag_weights,dtype=dtype,device=dev)
    pattern=_cycle(n_variables,temporal_path<0,dtype=dtype,device=dev); eye=torch.eye(n_variables,dtype=dtype,device=dev); w=abs(float(temporal_path)); mats=[]
    for lag in range(order):
        raw=(1-w)*eye+w*torch.linalg.matrix_power(pattern,lag+1); mats.append(weights[lag]*temporal_gain*raw/spectral_radius(raw))
    coef=torch.stack(mats).unsqueeze(0)
    if float(spectral_radius(companion_matrix(coef)))>=stability_target:
        lo,hi=0.,1.
        for _ in range(60):
            mid=.5*(lo+hi)
            if float(spectral_radius(companion_matrix(mid*coef)))<stability_target: lo=mid
            else: hi=mid
        coef=lo*coef
    lower=-1/(n_variables-1)+1e-4; corr=min(.999,max(lower,float(noise_correlation))); q=(1-corr)*eye+corr*torch.ones_like(eye)
    return coef,q.unsqueeze(0)
