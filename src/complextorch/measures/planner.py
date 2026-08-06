"""Dependency-aware measure planner with shared intermediate caching.

Notes
-----
The planner coordinates requested dynamical measures and reuses shared
intermediate quantities to avoid repeated covariance or spectral calculations.

Notes
-----
The planner coordinates requested dynamical measures and reuses shared
intermediate quantities to avoid repeated covariance or spectral calculations.
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
    """DynamicalMeasures.
    
    Notes
    -----
    The class follows the scikit-learn fitted-attribute convention when applicable.
    """
    AVAILABLE={'spectral_radius','stability_margin','dominant_timescale','covariance_amplification','stationary_covariance','autocovariances','entropy_rate','predictive_information','active_information_storage','transfer_function','inverse_transfer_function','cross_spectral_density','spectral_entropy','psi','delta','gamma','cmem3_total','cmem1_total','cmem3_lag','cmem3_curve','cmem1_curve','tc_innovation','tc_present'}
    def __init__(self,measures:Iterable[str],*,tau_max:int=10,max_lag:int=10,frequencies:torch.Tensor|None=None,sampling_frequency:float=1.0,macro_projection:torch.Tensor|None=None):
        """  init  .
        
        Parameters
        ----------
        measures
            Input controlling ``__init__``.
        tau_max
            Input controlling ``__init__``.
        max_lag
            Input controlling ``__init__``.
        frequencies
            Input controlling ``__init__``.
        sampling_frequency
            Input controlling ``__init__``.
        macro_projection
            Input controlling ``__init__``.
        
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
        self.measures=tuple(measures); self.tau_max=tau_max; self.max_lag=max_lag; self.frequencies=frequencies; self.sampling_frequency=sampling_frequency; self.macro_projection=macro_projection
        unknown=set(self.measures)-self.AVAILABLE
        if unknown: raise ValueError(f'unknown measures: {sorted(unknown)}')
    def __call__(self,system:VARSystem)->dict[str,torch.Tensor]:
        """  call  .
        
        Parameters
        ----------
        system
            Input controlling ``__call__``.
        
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
