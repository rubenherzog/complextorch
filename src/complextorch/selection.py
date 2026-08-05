"""Temporal validation and MVGC-compatible information-criterion order selection."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Literal
import numpy as np
import torch
from sklearn.base import BaseEstimator
from ._typing import ArrayLike
from .linalg import stable_cholesky, spd_logdet
from .var import VAR

@dataclass(frozen=True)
class TemporalFold:
    train_stop:int
    test_start:int
    test_stop:int

class EpochTimeSeriesSplit:
    def __init__(self,n_splits:int=5,*,test_size:int|None=None,min_train_size:int|None=None,gap:int=0): self.n_splits=n_splits; self.test_size=test_size; self.min_train_size=min_train_size; self.gap=gap
    def split(self,n_times:int,*,min_order:int=1):
        if self.n_splits<1 or self.gap<0: raise ValueError('invalid split settings')
        test_size=self.test_size or max(1,n_times//(self.n_splits+2)); min_train=self.min_train_size or max(min_order+5,n_times-self.n_splits*test_size-self.gap)
        if min_train+self.gap+self.n_splits*test_size>n_times: raise ValueError('requested folds do not fit')
        for fold in range(self.n_splits):
            train_stop=min_train+fold*test_size; test_start=train_stop+self.gap; yield TemporalFold(train_stop,test_start,test_start+test_size)

@dataclass(frozen=True)
class VAROrderScore:
    order:int
    mean_score:float
    standard_error:float
    fold_scores:tuple[float,...]
    failed_folds:int

@dataclass(frozen=True)
class VAROrderSearchResult:
    best_order:int
    scores:tuple[VAROrderScore,...]
    scoring:str
    selection_rule:str
    def as_records(self): return [dict(order=s.order,mean_score=s.mean_score,standard_error=s.standard_error,fold_scores=s.fold_scores,failed_folds=s.failed_folds) for s in self.scores]

class VAROrderSearchCV:
    def __init__(self,orders:Iterable[int],*,cv:EpochTimeSeriesSplit|None=None,scoring:Literal['nll','rmse']='nll',selection_rule:Literal['best','one_se']='one_se',alpha:float=0.,fit_intercept:bool=True,mode:Literal['independent','pooled']='independent',solver:str='auto',device:str='auto',dtype:str='float64',prediction_mode:Literal['rolling','recursive']='rolling',refit:bool=True):
        self.orders=tuple(int(x) for x in orders); self.cv=cv or EpochTimeSeriesSplit(); self.scoring=scoring; self.selection_rule=selection_rule; self.alpha=alpha; self.fit_intercept=fit_intercept; self.mode=mode; self.solver=solver; self.device=device; self.dtype=dtype; self.prediction_mode=prediction_mode; self.refit=refit
    @staticmethod
    def _normalise(x:ArrayLike):
        t=torch.as_tensor(x)
        if t.ndim==2: t=t.unsqueeze(0)
        if t.ndim!=3: raise ValueError('X must have shape (time,n) or (batch,time,n)')
        return t
    @staticmethod
    def _forecast_nll(errors,covariance):
        if covariance.shape[0]==1 and errors.shape[0]>1: covariance=covariance.expand(errors.shape[0],-1,-1)
        chol,_=stable_cholesky(covariance,jitter=1e-10); solved=torch.cholesky_solve(errors.unsqueeze(-1),chol[:,None]).squeeze(-1)
        return .5*((errors*solved).sum(-1)+2*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)[:,None]+errors.shape[-1]*np.log(2*np.pi))
    def _score_fold(self,data,order,fold):
        estimator=VAR(order=order,alpha=self.alpha,fit_intercept=self.fit_intercept,mode=self.mode,solver=self.solver,device=self.device,dtype=self.dtype,stability='ignore').fit(data[:,:fold.train_stop])
        if self.prediction_mode=='rolling': prediction=estimator.one_step_predictions(data[:,:fold.test_stop])[:,fold.test_start-order:fold.test_stop-order]
        else:
            forecast=estimator.forecast(data[:,:fold.train_stop],fold.test_stop-fold.train_stop); start=fold.test_start-fold.train_stop; prediction=forecast[:,start:start+fold.test_stop-fold.test_start]
        errors=data[:,fold.test_start:fold.test_stop].to(prediction)-prediction
        if self.scoring=='rmse': return float(torch.sqrt(torch.mean(errors.square())))
        if self.scoring=='nll': return float(self._forecast_nll(errors,estimator.noise_covariance_).mean())
        raise ValueError('unknown scoring')
    def fit(self,X:ArrayLike,y=None):
        del y
        if not self.orders or min(self.orders)<1: raise ValueError('orders must be positive')
        data=self._normalise(X); folds=tuple(self.cv.split(data.shape[1],min_order=max(self.orders))); results=[]
        for order in sorted(set(self.orders)):
            scores=[]; failed=0
            for fold in folds:
                try: scores.append(self._score_fold(data,order,fold))
                except (RuntimeError,ValueError,torch.linalg.LinAlgError): scores.append(float('inf')); failed+=1
            finite=np.asarray([x for x in scores if np.isfinite(x)]); mean=float(np.mean(finite)) if finite.size else float('inf'); se=float(np.std(finite,ddof=1)/sqrt(finite.size)) if finite.size>1 else 0.; results.append(VAROrderScore(order,mean,se,tuple(scores),failed))
        finite=[r for r in results if np.isfinite(r.mean_score)]
        if not finite: raise RuntimeError('all candidate orders failed')
        optimum=min(finite,key=lambda x:x.mean_score)
        if self.selection_rule=='best': best=optimum.order
        elif self.selection_rule=='one_se': best=min(r.order for r in finite if r.mean_score<=optimum.mean_score+optimum.standard_error)
        else: raise ValueError('unknown selection rule')
        self.result_=VAROrderSearchResult(best,tuple(results),self.scoring,self.selection_rule); self.best_order_=best; self.cv_results_=self.result_.as_records()
        if self.refit: self.best_estimator_=VAR(order=best,alpha=self.alpha,fit_intercept=self.fit_intercept,mode=self.mode,solver=self.solver,device=self.device,dtype=self.dtype).fit(data)
        return self

@dataclass(frozen=True)
class VARInformationCriteriaResult:
    p_aic:int; p_bic:int; p_hqc:int
    aic:np.ndarray; bic:np.ndarray; hqc:np.ndarray; loglik:np.ndarray
    orders:tuple[int,...]

class VAROrderSelectionIC(BaseEstimator):
    """MVGC2-compatible in-sample AIC/BIC/HQC order selection."""
    def __init__(self,orders:Iterable[int]=range(1,11),*,solver:str='lwr',hurvich_tsai:bool=False,device:str='auto',dtype:str='float64',refit:str|None='hqc'):
        self.orders=tuple(int(x) for x in orders); self.solver=solver; self.hurvich_tsai=hurvich_tsai; self.device=device; self.dtype=dtype; self.refit=refit
    def fit(self,X:ArrayLike,y=None):
        del y
        if not self.orders or min(self.orders)<1: raise ValueError('orders must be positive')
        data=torch.as_tensor(X)
        if data.ndim==2: data=data.unsqueeze(0)
        if data.ndim!=3: raise ValueError('X must have shape (time,n) or (trials,time,n)')
        n_trials,n_times,n=data.shape; loglik=[]; k=[]; meff=[]; fitted={}
        for order in self.orders:
            est=VAR(order=order,solver=self.solver,covariance='unbiased',mode='pooled' if n_trials>1 else 'independent',device=self.device,dtype=self.dtype,stability='ignore').fit(data); fitted[order]=est
            cov=est.noise_covariance_[0]; loglik.append(-.5*(n*np.log(2*np.pi)+float(spd_logdet(cov))+n)); k.append(order*n*n); meff.append(n_trials*(n_times-order))
        L=np.asarray(loglik); K=np.asarray(k,dtype=float)/np.asarray(meff,dtype=float); m=np.asarray(meff,dtype=float)
        if self.hurvich_tsai:
            fac=m/(m-np.asarray(k)-1); aic=-2*L+2*K*fac; aic=np.where(fac<=0,np.nan,aic)
        else: aic=-2*L+2*K
        bic=-2*L+K*np.log(m); hqc=-2*L+2*K*np.log(np.log(m))
        result=VARInformationCriteriaResult(self.orders[int(np.nanargmin(aic))],self.orders[int(np.nanargmin(bic))],self.orders[int(np.nanargmin(hqc))],aic,bic,hqc,L,self.orders)
        self.result_=result; self.p_aic_=result.p_aic; self.p_bic_=result.p_bic; self.p_hqc_=result.p_hqc; self.aic_=aic; self.bic_=bic; self.hqc_=hqc; self.loglik_=L
        if self.refit is not None:
            key=self.refit.lower()
            if key not in {'aic','bic','hqc'}: raise ValueError("refit must be None, 'aic', 'bic' or 'hqc'")
            self.best_order_=getattr(result,f'p_{key}'); self.best_estimator_=fitted[self.best_order_]
        return self
