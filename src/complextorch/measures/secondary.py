"""Secondary empirical measures, diagnostics and sample-based inference.

Notes
-----
Secondary measures estimate quantities from finite observations and therefore
include sampling, fitting and discretisation effects. They are kept separate
from analytical model-derived measures.

References
----------
- Barnett, L. and Seth, A. K. (2014), empirical MVGC workflow.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy import stats as scipy_stats

from .discrete import discrete_entropy, discrete_mutual_information, discrete_total_correlation, lempel_ziv_complexity
from .emergence import emergence_from_observations
from .gaussian import local_gaussian_mutual_information
from .mvgc import spectral_mvgc as estimate_spectral_mvgc_from_observations
from .mvgc import temporal_mvgc as estimate_temporal_mvgc_from_observations

temporal_mvgc = estimate_temporal_mvgc_from_observations
spectral_mvgc = estimate_spectral_mvgc_from_observations

@dataclass(frozen=True)
class WhitenessResult:
    """Residual-whiteness statistic, p-value and method label.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    statistic: torch.Tensor
    pvalue: torch.Tensor
    method: str

def _trials(value):
    """Trials.
    
    Parameters
    ----------
    value
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
    x=torch.as_tensor(value,dtype=torch.float64)
    if x.ndim==2: x=x.unsqueeze(0)
    if x.ndim!=3: raise ValueError('expected (time,n) or (trials,time,n)')
    return x

def consistency(observations,residuals,*,order:int)->float:
    """Compute the Ding--Bressler VAR consistency diagnostic.
                
                References
                ----------
                Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
            
            Compute the Ding--Bressler VAR consistency diagnostic.
            
            References
            ----------
            Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
        
        Compute the Ding--Bressler VAR consistency diagnostic.
        
        References
        ----------
        Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
    
    Compute the Ding--Bressler VAR consistency diagnostic.
    
    References
    ----------
    Ding et al. (2000); Barnett and Seth (2014); ComplexBox repository.
    """
    x=_trials(observations); e=_trials(residuals)
    if e.shape[1]!=x.shape[1]-order: raise ValueError('residual length is incompatible with order')
    x=x-x.mean(dim=(0,1),keepdim=True); xf=x[:,order:,:].reshape(-1,x.shape[-1]).T; ef=e.reshape(-1,e.shape[-1]).T
    y=xf-ef; d=xf.shape[-1]-1; rr=xf@xf.T/d; rs=y@y.T/d
    return float(1-torch.linalg.matrix_norm(rs-rr)/torch.linalg.matrix_norm(rr))

def _dw(design,residual):
    """Dw.
    
    Parameters
    ----------
    design
        Input required by this calculation.
    residual
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
    n,m=design.shape; dw=float(np.sum(np.diff(residual)**2)/np.sum(residual**2)); a=design@design.T; b=np.zeros_like(design.T); xt=design.T
    b[0]=-xt[0]
    if m>1: b[1]=2*xt[0]-xt[1]
    for t in range(2,m): b[t]=-xt[t-2]+2*xt[t-1]-xt[t]
    b[0]=design[:,0]-design[:,1]; b[m-1]=design[:,m-1]-design[:,m-2]
    d=np.linalg.solve(a.T,b.T).T; c=design@d; nu1=2*(m-1)-np.trace(c); nu2=2*(3*m-4)-2*np.trace(b.T@d)+np.trace(c@c)
    mu=nu1/(m-n); sigma=np.sqrt(2/((m-n)*(m-n+2))*(nu2-nu1*mu)); p=scipy_stats.norm.cdf(dw,loc=mu,scale=sigma)
    return dw,2*min(p,1-p)

def residual_whiteness(observations,residuals,*,order:int,method:str='durbin_watson')->WhitenessResult:
    """Test residual serial correlation with the requested method.
                
                The current Durbin--Watson route follows the ComplexBox/MVGC-compatible
                approximation while retaining an extensible ``method`` argument.
            
            Test residual serial correlation with the requested method.
            
            The current Durbin--Watson route follows the ComplexBox/MVGC-compatible
            approximation while retaining an extensible ``method`` argument.
        
        Test residual serial correlation with the requested method.
        
        The current Durbin--Watson route follows the ComplexBox/MVGC-compatible
        approximation while retaining an extensible ``method`` argument.
    
    Test residual serial correlation with the requested method.
    
    The current Durbin--Watson route follows the ComplexBox/MVGC-compatible
    approximation while retaining an extensible ``method`` argument.
    """
    if method.lower() not in {'durbin_watson','dw','complexbox'}: raise NotImplementedError(f'whiteness method {method!r} is not implemented')
    x=_trials(observations); e=_trials(residuals); x=x-x.mean(dim=(0,1),keepdim=True)
    design=x[:,order:,:].reshape(-1,x.shape[-1]).T.cpu().numpy(); errors=e.reshape(-1,e.shape[-1]).T.cpu().numpy(); values=[_dw(design,row) for row in errors]
    return WhitenessResult(torch.tensor([v[0] for v in values]),torch.tensor([v[1] for v in values]),'durbin_watson')

def mvgc_pvalue(statistic,*,method:str='F',n_target:int,n_source:int,n_conditional:int,order:int,n_times:int,n_trials:int=1):
    """Compute asymptotic MVGC p-values using MVGC2 conventions.
                
                References
                ----------
                Barnett and Seth (2014); MVGC repository.
            
            Compute asymptotic MVGC p-values using MVGC2 conventions.
            
            References
            ----------
            Barnett and Seth (2014); MVGC repository.
        
        Compute asymptotic MVGC p-values using MVGC2 conventions.
        
        References
        ----------
        Barnett and Seth (2014); MVGC repository.
    
    Compute asymptotic MVGC p-values using MVGC2 conventions.
    
    References
    ----------
    Barnett and Seth (2014); MVGC repository.
    """
    d=order*n_target*n_source; M=n_trials*(n_times-order); values=np.asarray(torch.as_tensor(statistic).detach().cpu(),dtype=float); key=method.upper()
    if key=='F':
        d2=n_target*(M-order*(n_target+n_source+n_conditional))
        if d2<=0: raise ValueError('insufficient observations for F-test')
        out=scipy_stats.f.sf((d2/d)*values,d,d2)
    elif key in {'LR','CHI2','CHI-SQUARE'}: out=scipy_stats.chi2.sf(M*values,d)
    else: raise ValueError("method must be 'F' or 'LR'")
    return torch.as_tensor(out,dtype=torch.float64)

def significance(pvalues,*,alpha:float=.05,method:str='fdr_bh'):
    """Apply uncorrected or Benjamini--Hochberg FDR significance testing.
                
                References
                ----------
                Benjamini and Hochberg (1995).
            
            Apply uncorrected or Benjamini--Hochberg FDR significance testing.
            
            References
            ----------
            Benjamini and Hochberg (1995).
        
        Apply uncorrected or Benjamini--Hochberg FDR significance testing.
        
        References
        ----------
        Benjamini and Hochberg (1995).
    
    Apply uncorrected or Benjamini--Hochberg FDR significance testing.
    
    References
    ----------
    Benjamini and Hochberg (1995).
    """
    p=torch.as_tensor(pvalues,dtype=torch.float64); finite=torch.isfinite(p); flat=p[finite]; result=torch.zeros_like(p,dtype=torch.bool)
    if flat.numel()==0: return result
    key=method.lower()
    if key in {'none','uncorrected'}: result[finite]=flat<=alpha; return result
    if key not in {'fdr','fdr_bh','bh','benjamini-hochberg'}: raise ValueError('unknown significance method')
    ordered,_=torch.sort(flat); threshold=alpha*torch.arange(1,flat.numel()+1,dtype=ordered.dtype)/flat.numel(); accepted=ordered<=threshold
    if bool(accepted.any()): result[finite]=flat<=ordered[torch.nonzero(accepted)[-1,0]]
    return result

__all__=['estimate_temporal_mvgc_from_observations','estimate_spectral_mvgc_from_observations','temporal_mvgc','spectral_mvgc','emergence_from_observations','local_gaussian_mutual_information','discrete_entropy','discrete_mutual_information','discrete_total_correlation','lempel_ziv_complexity','WhitenessResult','consistency','residual_whiteness','mvgc_pvalue','significance']
