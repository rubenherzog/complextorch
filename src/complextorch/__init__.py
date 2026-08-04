"""Torch-first batched linear-dynamics inference and complexity measures."""
from .adapters import from_complexbox_timeseries, from_complexbox_var, to_complexbox_timeseries, to_complexbox_var
from .representations import LinearDynamicalSystem, VARSystem, build_var_system, companion_matrix
from .selection import EpochTimeSeriesSplit, VAROrderSearchCV, VAROrderSearchResult
from .simulate import demo_var, random_stable_var, simulate_var
from .var import VAR, VARParameters
__all__=["VAR","VARParameters","VARSystem","LinearDynamicalSystem","build_var_system","companion_matrix","EpochTimeSeriesSplit","VAROrderSearchCV","VAROrderSearchResult","simulate_var","random_stable_var","demo_var","from_complexbox_timeseries","to_complexbox_timeseries","from_complexbox_var","to_complexbox_var"]
__version__="0.1.0"
