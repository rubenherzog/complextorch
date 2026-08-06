"""Deterministic and random stable VAR systems for validation.

Notes
-----
VAR simulations implement the recursion

.. math::

   x_t = c + \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t.

The automatic burn-in uses the companion spectral radius to suppress initial
conditions below a prescribed tolerance.

References
----------
- Lütkepohl, H. (2005), Chapters 2--3.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox

Notes
-----
VAR simulations implement the recursion

.. math::

   x_t = c + \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t.

The automatic burn-in uses the companion spectral radius to suppress initial
conditions below a prescribed tolerance.

References
----------
- Lütkepohl, H. (2005), Chapters 2--3.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox

Notes
-----
VAR simulations implement the recursion

.. math::

   x_t = c + \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t.

The automatic burn-in uses the companion spectral radius to suppress initial
conditions below a prescribed tolerance.

References
----------
- Lütkepohl, H. (2005), Chapters 2--3.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox
"""
from __future__ import annotations
import math
import torch
from .linalg import spectral_radius
from .representations import companion_matrix


def _normalise_coefficients(coefficients):
    """ normalise coefficients.
    
    Parameters
    ----------
    coefficients
        Input controlling ``_normalise_coefficients``.
    
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
    coef = torch.as_tensor(coefficients)
    if coef.ndim == 3:
        coef = coef.unsqueeze(0)
    if coef.ndim != 4:
        raise ValueError("coefficients must have shape (p,n,n) or (batch,p,n,n)")
    return coef


def _normalise_covariance(covariance, batch, n):
    """ normalise covariance.
    
    Parameters
    ----------
    covariance
        Input controlling ``_normalise_covariance``.
    batch
        Input controlling ``_normalise_covariance``.
    n
        Input controlling ``_normalise_covariance``.
    
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
    q = torch.as_tensor(covariance)
    if q.ndim == 2:
        q = q.unsqueeze(0)
    if q.shape[0] == 1 and batch > 1:
        q = q.expand(batch, -1, -1)
    if q.shape != (batch, n, n):
        raise ValueError("covariance has incompatible shape")
    return q


def automatic_burnin(coefficients, *, epsilon: float | None = None) -> int:
    """MVGC-style transient length ceil(-log(eps)/-log(rho)).
            
            Choose burn-in length from the companion spectral radius.
            
            The smallest integer :math:`T` satisfying :math:`
            ho^T<\epsilon` is used.
        
        Choose burn-in length from the companion spectral radius.
        
        The smallest integer :math:`T` satisfying :math:`
        ho^T<\epsilon` is used.
    
    Choose burn-in length from the companion spectral radius.
    
    The smallest integer :math:`T` satisfying :math:`
    ho^T<\epsilon` is used.
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
    """Simulate one or more Gaussian VAR trajectories.
            
            References
            ----------
            Lütkepohl (2005); ComplexBox repository.
        
        Simulate one or more Gaussian VAR trajectories.
        
        References
        ----------
        Lütkepohl (2005); ComplexBox repository.
    
    Simulate one or more Gaussian VAR trajectories.
    
    References
    ----------
    Lütkepohl (2005); ComplexBox repository.
    """
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
    """Random positive definite covariance.
    
    Parameters
    ----------
    n_variables
        Input controlling ``random_positive_definite_covariance``.
    batch
        Input controlling ``random_positive_definite_covariance``.
    seed
        Input controlling ``random_positive_definite_covariance``.
    scale_min
        Input controlling ``random_positive_definite_covariance``.
    scale_max
        Input controlling ``random_positive_definite_covariance``.
    dtype
        Input controlling ``random_positive_definite_covariance``.
    device
        Input controlling ``random_positive_definite_covariance``.
    
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
    correlation = random_correlation_matrix(
        n_variables, batch=batch, seed=seed, dtype=dtype, device=device
    )
    generator = torch.Generator(device=torch.device(device)).manual_seed(seed + 1)
    scales = scale_min + (scale_max - scale_min) * torch.rand(
        (batch, n_variables), generator=generator, dtype=dtype, device=device
    )
    return scales[..., :, None] * correlation * scales[..., None, :]


def random_stable_var(batch:int,n_variables:int,order:int,*,spectral_radius_target:float=.85,noise_scale:float=1.,seed:int=0,dtype=torch.float64,device='cpu'):
    """Generate random VAR coefficients scaled to a target spectral radius.
            
            References
            ----------
            Lütkepohl (2005); ComplexBox repository.
        
        Generate random VAR coefficients scaled to a target spectral radius.
        
        References
        ----------
        Lütkepohl (2005); ComplexBox repository.
    
    Generate random VAR coefficients scaled to a target spectral radius.
    
    References
    ----------
    Lütkepohl (2005); ComplexBox repository.
    """
    if not 0<spectral_radius_target<1: raise ValueError('spectral_radius_target must lie in (0,1)')
    gen=torch.Generator(device=torch.device(device)); gen.manual_seed(seed)
    raw=torch.randn((batch,order,n_variables,n_variables),generator=gen,dtype=dtype,device=device)/math.sqrt(n_variables*order)
    lo=torch.zeros(batch,dtype=dtype,device=device); hi=torch.ones(batch,dtype=dtype,device=device); raw_rho=spectral_radius(companion_matrix(raw)); lo=torch.where(raw_rho<=spectral_radius_target,hi,lo)
    for _ in range(48):
        mid=.5*(lo+hi); rho=spectral_radius(companion_matrix(raw*mid[:,None,None,None])); ok=rho<=spectral_radius_target; lo=torch.where(ok,mid,lo); hi=torch.where(ok,hi,mid)
    coef=raw*lo[:,None,None,None]; q=noise_scale*torch.eye(n_variables,dtype=dtype,device=device).expand(batch,-1,-1).clone(); return coef,q


def _cycle(n,frustrated,*,dtype,device):
    """ cycle.
    
    Parameters
    ----------
    n
        Input controlling ``_cycle``.
    frustrated
        Input controlling ``_cycle``.
    dtype
        Input controlling ``_cycle``.
    device
        Input controlling ``_cycle``.
    
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
    m=torch.zeros((n,n),dtype=dtype,device=device)
    for row in range(n): m[row,(row+1)%n]=1
    if frustrated: m[0,1]=-1
    return m


def demo_var(n_variables:int=3,order:int=2,*,temporal_path:float=-.95,temporal_gain:float=.9,noise_correlation:float=-.25,lag_weights=None,stability_target:float=.98,dtype=torch.float64,device='cpu'):
    """Demo var.
    
    Parameters
    ----------
    n_variables
        Input controlling ``demo_var``.
    order
        Input controlling ``demo_var``.
    temporal_path
        Input controlling ``demo_var``.
    temporal_gain
        Input controlling ``demo_var``.
    noise_correlation
        Input controlling ``demo_var``.
    lag_weights
        Input controlling ``demo_var``.
    stability_target
        Input controlling ``demo_var``.
    dtype
        Input controlling ``demo_var``.
    device
        Input controlling ``demo_var``.
    
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
