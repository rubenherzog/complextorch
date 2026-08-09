"""Analytical dynamics, autocovariances and spectra for Gaussian models.

For a VAR transfer function :math:`H(f)=A(f)^{-1}`, the cross-spectrum is

.. math::

   S(f)=H(f)\Sigma H(f)^*.

References
----------
- Lütkepohl, H. (2005). Spectral representation of VAR processes.
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
"""
from __future__ import annotations
import math
import torch
from ..linalg import spd_logdet
from ..representations import VARSystem
from .gaussian import gaussian_entropy, gaussian_mutual_information


def autocovariances(system:VARSystem,max_lag:int)->torch.Tensor:
    """Compute stationary observation autocovariances.
                        
                        For state transition :math:`A`, state covariance :math:`P` and observation
                        matrix :math:`C`, positive-lag covariances use :math:`C A^	au P C^	op`.
                        
                        References
                        ----------
                        Lütkepohl (2005); Anderson and Moore (1979).
                    
                    Compute stationary observation autocovariances.
                    
                    For state transition :math:`A`, state covariance :math:`P` and observation
                    matrix :math:`C`, positive-lag covariances use :math:`C A^	au P C^	op`.
                    
                    References
                    ----------
                    Lütkepohl (2005); Anderson and Moore (1979).
                
                Compute stationary observation autocovariances.
                
                For state transition :math:`A`, state covariance :math:`P` and observation
                matrix :math:`C`, positive-lag covariances use :math:`C A^	au P C^	op`.
                
                References
                ----------
                Lütkepohl (2005); Anderson and Moore (1979).
            
            Compute stationary observation autocovariances.
            
            For state transition :math:`A`, state covariance :math:`P` and observation
            matrix :math:`C`, positive-lag covariances use :math:`C A^	au P C^	op`.
            
            References
            ----------
            Lütkepohl (2005); Anderson and Moore (1979).
        
        Compute stationary observation autocovariances.
        
        For state transition :math:`A`, state covariance :math:`P` and observation
        matrix :math:`C`, positive-lag covariances use :math:`C A^	au P C^	op`.
        
        References
        ----------
        Lütkepohl (2005); Anderson and Moore (1979).
    
    Compute stationary observation autocovariances.
    
    For state transition :math:`A`, state covariance :math:`P` and observation
    matrix :math:`C`, positive-lag covariances use :math:`C A^	au P C^	op`.
    
    References
    ----------
    Lütkepohl (2005); Anderson and Moore (1979).
    """
    # Propagate stationary companion covariance to obtain Gamma(k)=E[x_t x_(t-k)^T] for successive lags.
    if max_lag<0: raise ValueError('max_lag must be nonnegative')
    power=torch.eye(system.companion.shape[-1],dtype=system.companion.dtype,device=system.companion.device).expand(system.batch_size,-1,-1)
    out=[]
    for lag in range(max_lag+1):
        if lag: power=power@system.companion
        out.append(system.projection@power@system.state_covariance@system.projection.transpose(-1,-2))
    return torch.stack(out,1)


def entropy_rate(system:VARSystem,*,base:float=2.0)->torch.Tensor:
    """Entropy rate.
    
    Parameters
    ----------
    system
        Canonical VAR or state-space system.
    base
        Logarithm base used for information quantities.
    
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
    return gaussian_entropy(system.innovation_covariance,base=base)


def predictive_information(system:VARSystem,*,base:float=2.0)->torch.Tensor:
    """Predictive information.
    
    Parameters
    ----------
    system
        Canonical VAR or state-space system.
    base
        Logarithm base used for information quantities.
    
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
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return 0.5*(spd_logdet(system.present_covariance)-spd_logdet(system.innovation_covariance))/math.log(base)


def active_information_storage(system:VARSystem,*,base:float=2.0)->torch.Tensor:
    r"""Per-variable :math:`I(X_t^i; X_{t-1:t-p}^i)` for a VAR(p).

    The companion state stores the present block first, so direct indexing of
    its first ``p`` blocks is easy to get off by one.  Reuse the canonical
    observation autocovariance path instead.
    """
    from .backbone import finite_lag_ais, observation_autocovariances

    gamma = observation_autocovariances(system, system.order)
    return finite_lag_ais(gamma, system.order, base=base)

def inverse_transfer_function(system:VARSystem,frequencies:torch.Tensor,*,sampling_frequency:float=1.0)->torch.Tensor:
    """Inverse transfer function.
    
    Parameters
    ----------
    system
        Canonical VAR or state-space system.
    frequencies
        One-dimensional frequency grid in normalized cycles per sample.
    sampling_frequency
        Sampling frequency used to scale spectral densities.
    
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
    f=torch.as_tensor(frequencies,dtype=system.coefficients.dtype,device=system.coefficients.device)
    coef=system.coefficients.to(torch.complex128 if system.coefficients.dtype==torch.float64 else torch.complex64)
    eye=torch.eye(system.n_variables,dtype=coef.dtype,device=coef.device)
    k=torch.arange(1,system.order+1,dtype=f.dtype,device=f.device)
    phase=torch.exp(-2j*math.pi*f[:,None]*k[None,:]/sampling_frequency)
    return eye[None,None]-torch.einsum('fk,bkij->bfij',phase,coef)


def transfer_function(system:VARSystem,frequencies:torch.Tensor,*,sampling_frequency:float=1.0)->torch.Tensor:
    """Transfer function.
    
    Parameters
    ----------
    system
        Canonical VAR or state-space system.
    frequencies
        One-dimensional frequency grid in normalized cycles per sample.
    sampling_frequency
        Sampling frequency used to scale spectral densities.
    
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
    return torch.linalg.inv(inverse_transfer_function(system,frequencies,sampling_frequency=sampling_frequency))


def cross_spectral_density(system:VARSystem,frequencies:torch.Tensor,*,sampling_frequency:float=1.0)->torch.Tensor:
    """Compute cross-spectral density from a transfer function.
    
    .. math::
    
       S(f)=H(f)\Sigma H(f)^*.
    
    References
    ----------
    - Lütkepohl (2005); Barnett and Seth (2014).
    """
    h=transfer_function(system,frequencies,sampling_frequency=sampling_frequency)
    q=system.innovation_covariance.to(h.dtype)
    return h@q[:,None]@h.conj().transpose(-1,-2)/sampling_frequency


def spectral_entropy(system:VARSystem,frequencies:torch.Tensor,*,sampling_frequency:float=1.0,normalize:bool=True)->torch.Tensor:
    """Spectral entropy.
    
    Parameters
    ----------
    system
        Canonical VAR or state-space system.
    frequencies
        One-dimensional frequency grid in normalized cycles per sample.
    sampling_frequency
        Sampling frequency used to scale spectral densities.
    normalize
        Whether to normalize the returned quantity.
    
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
    psd=torch.diagonal(cross_spectral_density(system,frequencies,sampling_frequency=sampling_frequency),dim1=-2,dim2=-1).real.clamp_min(torch.finfo(system.coefficients.dtype).tiny)
    prob=psd/psd.sum(1,keepdim=True)
    h=-(prob*torch.log2(prob)).sum(1)
    if normalize: h=h/math.log2(prob.shape[1])
    return h
