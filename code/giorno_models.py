"""Giorno--Tsallis HMM-6 / blind-model reproducibility core.

6 free baseline parameters: p00, p11, nu0, nu1, s0, s1.
State-dependent centered Student t emissions have *scale*, not standard deviation, s.
Time t conditional state mixture is filtered with the full observed-data likelihood.
No proprietary indicators, confidential formulas, or external market data are used.
"""
from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import t as student_t
from numba import njit


@njit(cache=True)
def _forward(loge0, loge1, p00, p11, init0):
    """Exact two-state observed log likelihood; emissions are log densities."""
    n = len(loge0)
    preds = np.empty(n, np.float64)
    filters = np.empty(n, np.float64)
    pred0 = init0
    total = 0.0
    for i in range(n):
        preds[i] = pred0
        a = np.log(max(1e-300, pred0)) + loge0[i]
        b = np.log(max(1e-300, 1.0-pred0)) + loge1[i]
        peak = max(a, b)
        lt = peak + np.log(np.exp(a-peak) + np.exp(b-peak))
        total += lt
        post0 = np.exp(a-lt)
        filters[i] = post0
        pred0 = post0*p00 + (1.0-post0)*(1.0-p11)
    return total, preds, filters


def hmm_stationary_pi0(p00,p11):
    return (1.0-p11)/(2.0-p00-p11)


def log_emission(y, nu, scale):
    return (gammaln((nu+1.)/2.) - gammaln(nu/2.)
            -0.5*np.log(nu*np.pi) - np.log(scale)
            -(nu+1.)/2.*np.log1p((y/scale)**2/nu))


def hmm_forward(y, params, init_pi0=None):
    p00,p11,nu0,nu1,s0,s1 = (float(p) for p in params)
    if init_pi0 is None: init_pi0 = hmm_stationary_pi0(p00,p11)
    return _forward(log_emission(y,nu0,s0), log_emission(y,nu1,s1),p00,p11,float(init_pi0))


def hmm_test_predict(y_train, y_test, params):
    """Test one-step predictive log densities using previously observed test returns only."""
    p00,p11,nu0,nu1,s0,s1 = [float(v) for v in params]
    _,_,f = hmm_forward(y_train, params)
    state_pred0 = f[-1]*p00+(1-f[-1])*(1-p11)
    l0=log_emission(y_test,nu0,s0)
    l1=log_emission(y_test,nu1,s1)
    n=len(y_test)
    logs=np.empty(n)
    pred0=np.empty(n)
    filt0=np.empty(n)
    for t in range(n):
        pred0[t]=state_pred0
        aa=np.log(max(state_pred0,1e-300))+l0[t]
        bb=np.log(max(1-state_pred0,1e-300))+l1[t]
        v=np.logaddexp(aa,bb)
        logs[t]=v
        filt0[t]=np.exp(aa-v)
        state_pred0=filt0[t]*p00+(1-filt0[t])*(1-p11)
    return logs,pred0,filt0


def make_scenario_table():
    # Label 0 is the normal/light-tail or low-scale state by construction.
    stdscale = lambda nu: float(np.sqrt((nu-2)/nu))
    return {
       'calibration_anchor':dict(p00=.96,p11=.94,nu0=12.,nu1=4.5,s0=.50,s1=1.80,identification='identifiable',note='Strong identifiable state differences in both scale and tails; reference recovery case.'),
       'tail_only': dict(p00=.975,p11=.85,nu0=24.,nu1=4.6,s0=stdscale(24.),s1=stdscale(4.6), identification='identifiable',note='Equal state variances=1, zero means. HMM persistence only in tail shape.'),
       'scale_only':dict(p00=.965,p11=.82,nu0=8.,nu1=8.,s0=.65,s1=1.65,identification='identifiable',note='Same degrees of freedom; regime-dependent scale/variance.'),
       'full_hmm6':dict(p00=.97,p11=.86,nu0=18.,nu1=5.2,s0=.70,s1=1.45,identification='identifiable',note='Joint persistent differences in scale and tail shape.'),
       'weak_tail':dict(p00=.975,p11=.85,nu0=11.,nu1=8.,s0=stdscale(11.),s1=stdscale(8.),identification='weak',note='Small tail contrast; expected weak practical identification.'),
       'weak_persistence':dict(p00=.72,p11=.47,nu0=24.,nu1=4.6,s0=stdscale(24.),s1=stdscale(4.6),identification='identifiable',note='Weak positive serial state correlation: rho=0.19.'),
       'iid_mixture':dict(p00=.82,p11=.18,nu0=24.,nu1=4.6,s0=stdscale(24.),s1=stdscale(4.6),identification='identifiable',note='HMM representation of iid tail mixture: rho=0.'),
       'single_student_t':dict(p00=.975,p11=.85,nu0=7.5,nu1=7.5,s0=stdscale(7.5),s1=stdscale(7.5),identification='unidentified',note='Identical states: transition parameters unidentifiable; do NOT claim p recovery.'),
    }


def simulate_hmm6(n, burn, pars, seed):
    """Simulate exact centered two-state Student-t HMM, no AR/GARCH/jumps."""
    rng=np.random.default_rng(int(seed))
    p00,p11,nu0,nu1,s0,s1=[float(pars[k]) for k in ('p00','p11','nu0','nu1','s0','s1')]
    total=int(n+burn)
    st=np.empty(total,dtype=np.int8)
    u=rng.random(total)
    st[0]=0 if u[0]<hmm_stationary_pi0(p00,p11) else 1
    for t in range(1,total):
        st[t]= 0 if (u[t] < (p00 if st[t-1]==0 else 1-p11)) else 1
    z = rng.standard_t(np.where(st==0, nu0, nu1)) * np.where(st==0,s0,s1)
    st=st[burn:]
    z=z[burn:]
    par=[p00,p11,nu0,nu1,s0,s1]
    _,pi0_filtered,post=hmm_forward(z,par)
    v0=s0*s0*nu0/(nu0-2.)
    v1=s1*s1*nu1/(nu1-2.)
    # True predictive variance given observed past, not the unobserved true state.
    true_pred_var=pi0_filtered*v0+(1.-pi0_filtered)*v1
    df=pd.DataFrame({'t':np.arange(len(z)), 'return':z, 'true_state':st,
                     'true_state0_prob_pred':pi0_filtered,
                     'true_state0_prob_filtered':post,
                     'true_pred_cond_variance':true_pred_var,
                     'true_emission_variance':np.where(st==0,v0,v1),
                     'true_abs_gt_2':(np.abs(z)>2).astype(np.int8)})
    return df


def _hmm_candidate_starts(y,starts,seed):
    rng=np.random.default_rng(seed)
    sd=float(np.std(y))
    sd=max(sd,.15)
    res=[]
    # all starts are data-based or random, never use known simulation truth
    for k in range(starts):
        if k==0:
            v=[.93,.83, 18.,5., sd*.72,sd*1.30]
        elif k==1:
            v=[.80,.65, 8.,19., sd*1.3,sd*.7]
        else:
            v=[rng.uniform(.55,.98),rng.uniform(.45,.95),
               rng.uniform(5,28),rng.uniform(3.2,16),
               sd*rng.uniform(.5,1.4),sd*rng.uniform(.6,1.7)]
        res.append(np.array(v,dtype=float))
    return res


def _ordered_hmm_params(pars):
    p00,p11,n0,n1,s0,s1=np.asarray(pars,dtype=float)
    if (n0 < n1-1e-5) or (abs(n0-n1)<1e-5 and s0>s1):
        return np.array([p11,p00,n1,n0,s1,s0]),True
    return np.array(pars,dtype=float),False


def fit_hmm6(y,starts=4,maxiter=260,seed=123):
    """Unrestricted six-parameter two-state Student t HMM (zero state means)."""
    y=np.asarray(y,dtype=float)
    sd=max(float(np.std(y)),.1)
    bounds=[(.02,.998),(.02,.998),(2.1,100.),(2.1,100.),(sd*.08,sd*5.),(sd*.08,sd*5.)]
    def nll(p):
        try:
            out=hmm_forward(y,p)[0]
            return -out if np.isfinite(out) else 1e100
        except (ValueError, FloatingPointError): return 1e100
    winner=None
    history=[]
    for i,x0 in enumerate(_hmm_candidate_starts(y,starts,seed)):
        x0=np.clip(x0,np.array(bounds)[:,0]+1e-7,np.array(bounds)[:,1]-1e-7)
        fit=minimize(nll,x0,method='L-BFGS-B',bounds=bounds, options={'maxiter':maxiter,'ftol':1e-9})
        history.append(dict(start=i,success=bool(fit.success),message=str(fit.message),nll=float(fit.fun),nit=int(fit.nit)))
        if winner is None or fit.fun<winner.fun: winner=fit
    pars,swapped=_ordered_hmm_params(winner.x)
    nll_best=nll(pars)
    return {'params':dict(zip(('p00','p11','nu0','nu1','s0','s1'),map(float,pars))),
            'loglik_train':float(-nll_best), 'success':bool(winner.success),
            'swapped_labels':bool(swapped), 'boundary_flags':{
                k: bool(abs(pars[j]-lo)<.00025 or abs(pars[j]-hi)<.00025)
                for j,(k,(lo,hi)) in enumerate(zip(('p00','p11','nu0','nu1','s0','s1'),bounds))},
            'optimizer_starts':history}


def load_manifest(p):
    return pd.read_csv(p)


def ensure_dir(p):
    p=Path(p);p.mkdir(parents=True,exist_ok=True);return p


def save_json(path, data):
    def natural(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError(str(type(v)))
    with open(path,'w',encoding='utf8') as f:
        json.dump(data,f,indent=2,default=natural)


def date_time_info():
    return '2026-10-28'
