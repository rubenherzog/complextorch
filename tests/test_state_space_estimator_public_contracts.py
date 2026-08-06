"""Public API contracts shared by state-space estimators."""

import inspect

from sklearn.base import BaseEstimator

from complextorch import LarimoreStateSpace, LinearGaussianEM, N4SID


def test_state_space_estimators_follow_sklearn_fit_contract():
    for estimator in (N4SID, LarimoreStateSpace, LinearGaussianEM):
        assert issubclass(estimator, BaseEstimator)
        signature = inspect.signature(estimator.fit)
        assert "y" in signature.parameters


def test_state_space_estimators_are_publicly_exported():
    assert N4SID.__module__ == "complextorch.state_space"
    assert LarimoreStateSpace.__module__ == "complextorch.state_space"
    assert LinearGaussianEM.__module__ == "complextorch.state_space"
