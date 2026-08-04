import numpy as np
import pytest
import torch
from sklearn.base import clone
from complextorch.simulate import demo_var,random_stable_var,simulate_var
from complextorch.var import VAR

def test_sklearn_clone_and_parameters():
    estimator=VAR(order=3,alpha=.2,mode='pooled'); cloned=clone(estimator)
    assert cloned.get_params()['order']==3 and cloned.get_params()['alpha']==.2 and cloned.get_params()['mode']=='pooled'

def test_batched_independent_matches_single_epoch_loop():
    coef,q=random_stable_var(5,4,2,seed=10); x=simulate_var(coef,q,500,burnin=400,seed=11)
    batched=VAR(order=2,fit_intercept=False,stability='ignore').fit(x)
    loop=torch.stack([VAR(order=2,fit_intercept=False,stability='ignore').fit(x[e]).coef_[0] for e in range(x.shape[0])])
    torch.testing.assert_close(batched.coef_,loop,rtol=1e-10,atol=1e-10)

def test_pooled_returns_one_model():
    coef,q=random_stable_var(1,3,2,seed=2); x=simulate_var(coef.expand(6,-1,-1,-1),q.expand(6,-1,-1),350,seed=3)
    fit=VAR(order=2,mode='pooled',fit_intercept=False,stability='ignore').fit(x)
    assert fit.coef_.shape==(1,2,3,3); assert fit.forecast(x,4).shape==(6,4,3)

def test_statsmodels_ols_parity():
    statsmodels=pytest.importorskip('statsmodels.tsa.api'); coef,q=demo_var(order=2); x=simulate_var(coef,q,700,burnin=500,seed=4)[0]
    ours=VAR(order=2,fit_intercept=True,stability='ignore').fit(x); reference=statsmodels.VAR(x.numpy()).fit(2,trend='c')
    np.testing.assert_allclose(ours.coef_[0].numpy(),reference.coefs,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(ours.intercept_[0].numpy(),reference.intercept,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(ours.noise_covariance_[0].numpy(),reference.sigma_u,rtol=1e-11,atol=1e-11)

def test_fit_recovers_long_simulation():
    coef,q=demo_var(order=2,stability_target=.8); x=simulate_var(coef,q,8000,burnin=1000,seed=12)
    fit=VAR(order=2,fit_intercept=False,stability='ignore').fit(x); torch.testing.assert_close(fit.coef_,coef,rtol=0.,atol=.04)
