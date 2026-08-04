import torch
from complextorch.selection import EpochTimeSeriesSplit,VAROrderSearchCV
from complextorch.simulate import simulate_var

def test_epoch_split_is_temporal_and_disjoint():
    folds=list(EpochTimeSeriesSplit(n_splits=3,test_size=20,min_train_size=100,gap=5).split(200,min_order=4))
    assert len(folds)==3
    for fold in folds:
        assert fold.train_stop<fold.test_start<fold.test_stop and fold.test_start-fold.train_stop==5

def test_order_search_prefers_true_second_lag():
    coef=torch.zeros((8,2,2,2),dtype=torch.float64); coef[:,0]=torch.tensor([[.05,0.],[0.,.05]],dtype=torch.float64); coef[:,1]=torch.tensor([[.72,.08],[-.05,.68]],dtype=torch.float64)
    q=.15*torch.eye(2,dtype=torch.float64).expand(8,-1,-1).clone(); x=simulate_var(coef,q,700,burnin=700,seed=33)
    search=VAROrderSearchCV([1,2,3,4],cv=EpochTimeSeriesSplit(n_splits=4,test_size=60,min_train_size=380),scoring='rmse',selection_rule='best',fit_intercept=False,dtype='float64').fit(x)
    assert search.best_order_==2 and search.best_estimator_.coef_.shape==(8,2,2,2)
