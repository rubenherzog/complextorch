"""Canonical dynamical-system representations shared by estimators and measures.

Notes
-----
A VAR(p) process is represented as

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t,
   \qquad 
arepsilon_t \sim \mathcal N(0,\Sigma).

Its companion-form state transition is used to connect VAR and linear
state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.

Notes
-----
A VAR(p) process is represented as

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t,
   \qquad 
arepsilon_t \sim \mathcal N(0,\Sigma).

Its companion-form state transition is used to connect VAR and linear
state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.

Notes
-----
A VAR(p) process is represented as

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t,
   \qquad 
arepsilon_t \sim \mathcal N(0,\Sigma).

Its companion-form state transition is used to connect VAR and linear
state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.

Notes
-----
A VAR(p) process is represented as

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t,
   \qquad 
arepsilon_t \sim \mathcal N(0,\Sigma).

Its companion-form state transition is used to connect VAR and linear
state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.

Notes
-----
A VAR(p) process is represented as

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t,
   \qquad 
arepsilon_t \sim \mathcal N(0,\Sigma).

Its companion-form state transition is used to connect VAR and linear
state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.

Notes
-----
A VAR(p) process is represented as

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + 
arepsilon_t,
   \qquad 
arepsilon_t \sim \mathcal N(0,\Sigma).

Its companion-form state transition is used to connect VAR and linear
state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .linalg import LyapunovInfo, solve_discrete_lyapunov, spectral_radius

@dataclass(frozen=True)
class LinearDynamicalSystem:
    """Linear Gaussian state-space model with optional stationary state covariance.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    transition: torch.Tensor
    observation: torch.Tensor
    process_covariance: torch.Tensor
    observation_covariance: torch.Tensor
    state_covariance: torch.Tensor | None = None
    sampling_frequency: float | None = None
    channel_names: tuple[str, ...] | None = None
    @property
    def spectral_radius(self) -> torch.Tensor:
        """Spectral radius.
        
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
        return spectral_radius(self.transition)

@dataclass(frozen=True)
class VARSystem:
    """Stationary Gaussian VAR process represented by coefficients and innovation covariance.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    coefficients: torch.Tensor
    innovation_covariance: torch.Tensor
    companion: torch.Tensor
    companion_noise_covariance: torch.Tensor
    state_covariance: torch.Tensor
    projection: torch.Tensor
    present_covariance: torch.Tensor
    spectral_radius: torch.Tensor
    lyapunov_info: LyapunovInfo
    @property
    def batch_size(self) -> int: return int(self.coefficients.shape[0])
    @property
    def order(self) -> int: return int(self.coefficients.shape[1])
    @property
    def n_variables(self) -> int: return int(self.coefficients.shape[2])
    def to_state_space(self) -> LinearDynamicalSystem:
        """Convert to state space.
        
        Returns
        -------
        object
            Equivalent linear state-space representation.
        
        Notes
        -----
        Batch dimensions are preserved unless explicitly documented otherwise.
        The implementation validates dimensional and positive-definiteness
        requirements before executing the numerical core.
        """
        batch = self.companion.shape[0]; n = self.n_variables
        zero = torch.zeros((batch,n,n),dtype=self.companion.dtype,device=self.companion.device)
        return LinearDynamicalSystem(self.companion,self.projection,self.companion_noise_covariance,zero,self.state_covariance)

def _normalise_coefficients(coefficients: torch.Tensor) -> tuple[torch.Tensor,bool]:
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
    coef=torch.as_tensor(coefficients); unbatched=coef.ndim==3
    if unbatched: coef=coef.unsqueeze(0)
    if coef.ndim!=4 or coef.shape[-1]!=coef.shape[-2]: raise ValueError("coefficients must have shape (p,n,n) or (batch,p,n,n)")
    return coef,unbatched

def companion_matrix(coefficients: torch.Tensor) -> torch.Tensor:
    """Companion matrix.
    
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
    coef,unbatched=_normalise_coefficients(coefficients); batch,order,n,_=coef.shape
    out=torch.zeros((batch,order*n,order*n),dtype=coef.dtype,device=coef.device)
    out[:,:n,:]=coef.permute(0,2,1,3).reshape(batch,n,order*n)
    if order>1: out[:,n:,:(order-1)*n]=torch.eye((order-1)*n,dtype=coef.dtype,device=coef.device)
    return out[0] if unbatched else out

def build_var_system(coefficients: torch.Tensor, innovation_covariance: torch.Tensor, *, lyapunov_method: str="doubling", rtol: float=1e-10, atol: float=1e-12) -> VARSystem:
    """Build var system.
    
    Parameters
    ----------
    coefficients
        VAR coefficient tensor ordered by lag, target and source.
    innovation_covariance
        Symmetric positive-definite covariance of model innovations.
    lyapunov_method
        Input required by this calculation.
    rtol
        Input required by this calculation.
    atol
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
    coef,_=_normalise_coefficients(coefficients)
    q=torch.as_tensor(innovation_covariance,dtype=coef.dtype,device=coef.device)
    if q.ndim==2: q=q.unsqueeze(0)
    if q.shape[0]==1 and coef.shape[0]>1: q=q.expand(coef.shape[0],-1,-1).contiguous()
    if q.ndim!=3 or q.shape[1:]!=coef.shape[2:] or q.shape[0]!=coef.shape[0]: raise ValueError("invalid innovation covariance shape")
    comp=companion_matrix(coef); batch,state_dim,_=comp.shape; n=coef.shape[-1]
    process=torch.zeros_like(comp); process[:,:n,:n]=q
    state_cov,info=solve_discrete_lyapunov(comp,process,method=lyapunov_method,rtol=rtol,atol=atol)
    projection=torch.zeros((batch,n,state_dim),dtype=coef.dtype,device=coef.device); projection[:,:,:n]=torch.eye(n,dtype=coef.dtype,device=coef.device)
    present=projection@state_cov@projection.transpose(-1,-2)
    return VARSystem(coef,q,comp,process,state_cov,projection,present,spectral_radius(comp),info)
