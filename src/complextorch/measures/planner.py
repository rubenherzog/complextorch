"""Planning and reuse of intermediate dynamical-measure calculations.

The planner resolves requested outputs and shares autocovariance, covariance and
spectral intermediates across compatible measures.

References
----------
- ComplexTorch canonical model-backbone design.
"""
from __future__ import annotations
from collections.abc import Iterable
import torch
from ..representations import VARSystem
from .cmem import compute_cmem
from .criticality import covariance_amplification,dominant_timescale,stability_margin
from .dynamics import autocovariances,entropy_rate,predictive_information,active_information_storage,transfer_function,inverse_transfer_function,cross_spectral_density,spectral_entropy
from .emergence import emergence_measures

class DynamicalMeasures:
    """Planner that reuses intermediate quantities across requested dynamical measures.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    AVAILABLE={'spectral_radius','stability_margin','dominant_timescale','covariance_amplification','stationary_covariance','autocovariances','entropy_rate','predictive_information','active_information_storage','transfer_function','inverse_transfer_function','cross_spectral_density','spectral_entropy','psi','delta','gamma','cmem3_total','cmem1_total','cmem3_lag','cmem3_curve','cmem1_curve','tc_innovation','tc_present'}
    def __init__(self,measures:Iterable[str],*,tau_max:int=10,max_lag:int=10,frequencies:torch.Tensor|None=None,sampling_frequency:float=1.0,macro_projection:torch.Tensor|None=None):
        """Initialize the estimator or result container.
        
        Parameters
        ----------
        measures
            Input required by this calculation.
        tau_max
            Input required by this calculation.
        max_lag
            Largest non-negative lag to evaluate.
        frequencies
            One-dimensional frequency grid in normalized cycles per sample.
        sampling_frequency
            Sampling frequency used to scale spectral densities.
        macro_projection
            Linear map defining macroscopic variables.
        
        Notes
        -----
        Batch dimensions are preserved unless explicitly documented otherwise.
        The implementation validates dimensional and positive-definiteness
        requirements before executing the numerical core.
        """
        self.measures=tuple(measures); self.tau_max=tau_max; self.max_lag=max_lag; self.frequencies=frequencies; self.sampling_frequency=sampling_frequency; self.macro_projection=macro_projection
        unknown=set(self.measures)-self.AVAILABLE
        if unknown: raise ValueError(f'unknown measures: {sorted(unknown)}')
    def __call__(self,system:VARSystem)->dict[str,torch.Tensor]:
        """Call.
        
        Parameters
        ----------
        system
            Canonical VAR or state-space system.
        
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
        result={}; cmem_names={'cmem3_total','cmem1_total','cmem3_lag','cmem3_curve','cmem1_curve','tc_innovation','tc_present'}; cmem=compute_cmem(system,self.tau_max) if cmem_names&set(self.measures) else None
        spectral_names={'transfer_function','inverse_transfer_function','cross_spectral_density','spectral_entropy'}
        if spectral_names&set(self.measures) and self.frequencies is None: raise ValueError('frequencies are required for spectral measures')
        emerg=None
        if {'psi','delta','gamma'}&set(self.measures):
            if self.macro_projection is None: raise ValueError('macro_projection is required for emergence measures')
            emerg=emergence_measures(system,self.macro_projection)
        for name in self.measures:
            if name=='spectral_radius': value=system.spectral_radius
            elif name=='stability_margin': value=stability_margin(system)
            elif name=='dominant_timescale': value=dominant_timescale(system,sampling_interval=1/self.sampling_frequency)
            elif name=='covariance_amplification': value=covariance_amplification(system)
            elif name=='stationary_covariance': value=system.present_covariance
            elif name=='autocovariances': value=autocovariances(system,self.max_lag)
            elif name=='entropy_rate': value=entropy_rate(system)
            elif name=='predictive_information': value=predictive_information(system)
            elif name=='active_information_storage': value=active_information_storage(system)
            elif name=='transfer_function': value=transfer_function(system,self.frequencies,sampling_frequency=self.sampling_frequency)
            elif name=='inverse_transfer_function': value=inverse_transfer_function(system,self.frequencies,sampling_frequency=self.sampling_frequency)
            elif name=='cross_spectral_density': value=cross_spectral_density(system,self.frequencies,sampling_frequency=self.sampling_frequency)
            elif name=='spectral_entropy': value=spectral_entropy(system,self.frequencies,sampling_frequency=self.sampling_frequency)
            elif name in {'psi','delta','gamma'}: value=emerg[name]
            else: value=getattr(cmem,name)
            result[name]=value
        return result
