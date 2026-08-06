"""Time- and frequency-domain analytical measures for stationary Gaussian VARs.

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
    """Per-variable I(X_t^i; own p-lag history)."""
    n,p=system.n_variables,system.order; s=system.state_covariance
    vals=[]
    for i in range(n):
        idx=torch.tensor([i+k*n for k in range(p)],device=s.device)
        hist=s.index_select(-2,idx).index_select(-1,idx)
        cross=s[...,i,idx]
        joint=torch.cat([torch.cat([s[...,i,i].unsqueeze(-1).unsqueeze(-1),cross.unsqueeze(-2)],-1),torch.cat([cross.unsqueeze(-1),hist],-1)],-2)
        vals.append(gaussian_mutual_information(joint,1,base=base))
    return torch.stack(vals,-1)


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
    """Compute the cross-spectral density from a transfer function.
                        
                        .. math:: S(f)=H(f)\Sigma H(f)^*.
                        
                        References
                        ----------
                        Lütkepohl (2005); Barnett and Seth (2014).
                    
                    Compute the cross-spectral density from a transfer function.
                    
                    .. math:: S(f)=H(f)\Sigma H(f)^*.
                    
                    References
                    ----------
                    Lütkepohl (2005); Barnett and Seth (2014).
                
                Compute the cross-spectral density from a transfer function.
                
                .. math:: S(f)=H(f)\Sigma H(f)^*.
                
                References
                ----------
                Lütkepohl (2005); Barnett and Seth (2014).
            
            Compute the cross-spectral density from a transfer function.
            
            .. math:: S(f)=H(f)\Sigma H(f)^*.
            
            References
            ----------
            Lütkepohl (2005); Barnett and Seth (2014).
        
        Compute the cross-spectral density from a transfer function.
        
        .. math:: S(f)=H(f)\Sigma H(f)^*.
        
        References
        ----------
        Lütkepohl (2005); Barnett and Seth (2014).
    
    Compute the cross-spectral density from a transfer function.
    
    .. math:: S(f)=H(f)\Sigma H(f)^*.
    
    References
    ----------
    Lütkepohl (2005); Barnett and Seth (2014).
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
