#!/usr/bin/env python3
"""Giorno--Tsallis: chronological one-step VaR/ES benchmark.
Single centered Student-t vs two-state centered Student-t HMM, both gap aware.
NO Gamma jumps, no trading strategy, no claims about omitted-jump causal costs.
Inputs: WIN MT5 TSV or U.S. CSV with datetime and Close.
Outputs: 1-step forecasts, coverage, pinball losses, and full provenance.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import t as student_t, chi2

KEYS_H=['p00','p11','nu0','nu1','s0','s1','log_overnight0','log_overnight1']

def load_prices(path, fmt):
    sep='\t' if fmt=='win' else ','
    df=pd.read_csv(path,sep=sep)
    if fmt=='win':
        times=pd.to_datetime(df['<DATE>'].astype(str)+' '+df['<TIME>'].astype(str),format='%Y.%m.%d %H:%M:%S',errors='coerce')
        prices=pd.to_numeric(df['<CLOSE>'],errors='coerce')
    else:
        times=pd.to_datetime(df['datetime'],errors='coerce')
        prices=pd.to_numeric(df['Close'],errors='coerce')
    f=pd.DataFrame({'time':times,'price':prices}).dropna()
    f=f.loc[f.price>0].sort_values('time',kind='stable')
    duplicates=int(f.time.duplicated().sum())
    f=f.drop_duplicates('time',keep='last')
    dt=f.time.diff().dt.total_seconds().div(60).to_numpy()[1:]
    on=(f.time.dt.date.to_numpy()[1:]!=f.time.dt.date.to_numpy()[:-1])
    values=100*np.diff(np.log(f.price.to_numpy(dtype=float)))
    mask=np.isfinite(values)&(dt>0)
    y=values[mask]; gap=on[mask]; times=f.time.to_numpy()[1:][mask]; mins=dt[mask]
    if len(y)<100: raise ValueError('Too few returns after cleaning')
    audit={'raw_rows':int(len(df)),'duplicate_timestamps':duplicates,'returns':int(len(y)),
           'overnights':int(gap.sum()),'same_day_nonminute':int(np.sum((mins!=1)&~gap)),
           'nonminute_total':int(np.sum(mins!=1)),'first_return':str(times[0]),'last_return':str(times[-1]),
           'return_units':'100*diff(log(Close)); values in percentage-point log returns',
           'timestamp_timezone':'As supplied; not independently verified'}
    return y,gap,times,audit,mins

def log_emission(y,nu,s):
    z=y/s
    return gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(nu*np.pi)-np.log(s)-0.5*(nu+1)*np.log1p(z*z/nu)

def student_scales(gap,x):return x[1]*np.exp(x[2]*gap)

def hmm_scales(gap,x):return x[4]*np.exp(x[6]*gap),x[5]*np.exp(x[7]*gap)

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(f): return f
        return wrapper

@njit(cache=True)
def fast_hmm_ll(l0,l1,p00,p11):
    pi0=(1.0-p11)/(2.0-p00-p11)
    a=pi0; b=1.0-pi0; total=0.0
    for i in range(len(l0)):
        m=max(l0[i],l1[i]); aa=a*np.exp(l0[i]-m);bb=b*np.exp(l1[i]-m)
        z=aa+bb
        if z<=0.0: return -1e90
        a=aa/z;b=bb/z;total+=m+np.log(z)
        na=a*p00+b*(1.0-p11)
        nb=a*(1.0-p00)+b*p11
        a=na;b=nb
    return total

def hmm_forward(log0,log1,p00,p11,return_last=False):
    pi0=(1-p11)/(2-p00-p11)
    pred=np.array([pi0,1-pi0]); ll=0.
    P=np.array([[p00,1-p00],[1-p11,p11]])
    for a,b in zip(log0,log1):
        z=np.log(np.maximum(pred,1e-300))+[a,b]
        logz=logsumexp(z); ll+=logz
        post=np.exp(z-logz);pred=post@P
    return (ll,pred) if return_last else ll

def fit_student(y,gap):
    x0=np.array([5.,max(np.std(y)*.65,.002),np.log(3.)])
    bounds=[(2.025,80),(.0005,2.),(0,np.log(300))]
    def objective(x):return -float(np.mean(log_emission(y,x[0],student_scales(gap,x))))
    starts=[x0,np.array([9.,x0[1],np.log(7.)]),np.array([3.3,x0[1]*.7,np.log(12.)])]
    results=[minimize(objective,x,method='L-BFGS-B',bounds=bounds,options={'maxiter':400,'ftol':1e-12}) for x in starts]
    best=min(results,key=lambda x:x.fun)
    return best,{'nstarts':len(results),'start_spread_ll':float(len(y)*(max(z.fun for z in results)-best.fun)),
                 'all_success':[bool(z.success) for z in results],'success':bool(best.success),'grad_inf':float(np.max(np.abs(best.jac)))}

def fit_hmm(y,gap):
    med=max(float(np.median(np.abs(y))/.6745),.003)
    lower=np.array([.70,.70,2.025,2.025,.0005,.0005,0,0]);upper=np.array([.9999,.9999,80,80,2,2,np.log(300),np.log(300)])
    def objective(x):
        s0,s1=hmm_scales(gap,x)
        return -fast_hmm_ll(log_emission(y,x[2],s0),log_emission(y,x[3],s1),x[0],x[1])/len(y)
    starts=[np.array([.991,.990,nu0,nu1,med*a,med*b,np.log(8),np.log(8)]) for nu0,nu1,a,b in
            [(4.2,7.0,.65,1.7),(7.0,4.2,.65,1.7),(3.5,8.5,.80,2.0),(8.5,3.5,.80,2.0)]]
    results=[]
    for x in starts:
        x=np.clip(x,lower+1e-7,upper-1e-7)
        results.append(minimize(objective,x,method='L-BFGS-B',bounds=list(zip(lower,upper)),
                                options={'maxiter':350,'ftol':2e-11,'eps':1e-6,'maxls':40}))
    best=min(results,key=lambda z:z.fun)
    diag={'nstarts':len(results),'start_spread_ll':float(len(y)*(max(z.fun for z in results)-best.fun)),
          'all_success':[bool(z.success) for z in results],'success':bool(best.success),
          'grad_inf':float(np.max(np.abs(best.jac))),
          'boundary_proximity':{k:bool((best.x[i]-lower[i])/(upper[i]-lower[i])<.0005 or (upper[i]-best.x[i])/(upper[i]-lower[i])<.0005) for i,k in enumerate(KEYS_H)}}
    return best,diag

def mixture_quantile(alpha,w,nu,s):
    lo=min(student_t.ppf(alpha/4,nu[0])*s[0],student_t.ppf(alpha/4,nu[1])*s[1])
    hi=0.
    f=lambda x:w*student_t.cdf(x/s[0],nu[0])+(1-w)*student_t.cdf(x/s[1],nu[1])-alpha
    if f(lo)>0: lo*=2
    return brentq(f,lo,hi,xtol=1e-11)

def t_lower_es(nu,s,q):
    z=q/s
    density=student_t.pdf(z,nu)
    # E[Y 1{Y <= q}], centered Student-t scale parameter s; nu>1
    return -s*(nu+z*z)/(nu-1)*density

def pinball(y,q,alpha):return (alpha-(y<q))*(y-q)

def hac_mean_se(z,lags):
    z=np.asarray(z,dtype=float);z=z[np.isfinite(z)];n=len(z)
    if n<3:return np.nan,np.nan
    zz=z-z.mean();gamma0=float(np.dot(zz,zz)/n);v=gamma0
    for lag in range(1,min(int(lags),n-1)+1):
        cov=float(np.dot(zz[lag:],zz[:-lag])/n)
        v+=2*(1-lag/(lags+1))*cov
    return float(z.mean()),float(np.sqrt(max(v,0)/n))

def kupiec(breaches,alpha):
    n=len(breaches);k=int(np.sum(breaches));rate=k/n
    p=min(max(rate,1e-15),1-1e-15)
    ll0=k*np.log(alpha)+(n-k)*np.log(1-alpha)
    ll1=k*np.log(p)+(n-k)*np.log(1-p)
    return {'count':k,'n':n,'observed_rate':rate,'target_rate':alpha,'LR_uc':float(2*(ll1-ll0)),
            'p_chi2_reference':float(chi2.sf(max(0,2*(ll1-ll0)),1))}

def run(args):
    if not (0.5<=args.train_fraction<.95):raise ValueError('train_fraction must be in [0.5,.95)')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    y,gap,timestamps,audit,minutes=load_prices(args.csv,args.format)
    if args.max_rows: y,gap,timestamps,minutes=y[-args.max_rows:],gap[-args.max_rows:],timestamps[-args.max_rows:],minutes[-args.max_rows:]
    n=len(y); split=int(n*args.train_fraction)
    if split<100 or split==n: raise ValueError('Invalid train/test split')
    print(f'{args.market}: {n} returns; train={split}, test={n-split}, overnights={gap.sum()}',flush=True)
    print('Fitting Student-t (with overnight multiplier) ...',flush=True)
    fres,fd=fit_student(y[:split],gap[:split]);x=fres.x
    print('Fitting gap-aware HMM (multi-start; can take time on long series) ...',flush=True)
    hres,hd=fit_hmm(y[:split],gap[:split]);h=hres.x
    if not fres.success or not hres.success:warnings.warn('An optimizer did not report successful convergence: results require audit')
    # initialize predictive state from last TRAIN observation, respecting causal filtering.
    a,b=hmm_scales(gap[:split],h)
    # Vectorized emissions + fast compiled forward score for training, followed by
    # a single pass to recover the predictive state. The latter is O(N) once.
    _,prior=hmm_forward(log_emission(y[:split],h[2],a),log_emission(y[:split],h[3],b),h[0],h[1],True)
    P=np.array([[h[0],1-h[0]],[1-h[1],h[1]]])
    alphas=sorted(set(args.alphas),reverse=True)
    columns={'timestamp':pd.to_datetime(timestamps[split:]),'return_pct_log':y[split:],
             'overnight':gap[split:],'elapsed_minutes':minutes[split:]}
    for name in ('student_t','hmm6'):
        for alpha in alphas:
            tag=str(round((1-alpha)*100,3)).replace('.','p')
            columns[f'{name}_VaR_{tag}_pct']=np.empty(n-split)
            columns[f'{name}_ES_{tag}_pct']=np.empty(n-split)
            columns[f'{name}_pinball_{tag}']=np.empty(n-split)
            columns[f'{name}_breach_{tag}']=np.empty(n-split,dtype=bool)
    weights=np.empty(n-split)
    logscore={'student_t':0.,'hmm6':0.}
    for i,t in enumerate(range(split,n)):
        yt=float(y[t]);on=int(gap[t]);weights[i]=prior[0]
        ss=float(student_scales(on,x));
        s0,s1=map(float,hmm_scales(on,h)); sigmas=[s0,s1]; nus=[h[2],h[3]]
        l0=float(log_emission(yt,h[2],s0));l1=float(log_emission(yt,h[3],s1))
        logscore['student_t']+=float(log_emission(yt,x[0],ss))
        ll=logsumexp([np.log(prior[0])+l0,np.log(prior[1])+l1]);logscore['hmm6']+=float(ll)
        for alpha in alphas:
            tag=str(round((1-alpha)*100,3)).replace('.','p')
            q1=ss*student_t.ppf(alpha,x[0]);q2=mixture_quantile(alpha,prior[0],nus,sigmas)
            es1=float(t_lower_es(x[0],ss,q1)/alpha)
            es2=float((prior[0]*t_lower_es(nus[0],s0,q2)+prior[1]*t_lower_es(nus[1],s1,q2))/alpha)
            for name,q,es in [('student_t',q1,es1),('hmm6',q2,es2)]:
                columns[f'{name}_VaR_{tag}_pct'][i]=-q
                columns[f'{name}_ES_{tag}_pct'][i]=-es
                columns[f'{name}_pinball_{tag}'][i]=pinball(yt,q,alpha)
                columns[f'{name}_breach_{tag}'][i]=(yt<q)
        post=np.exp([np.log(prior[0])+l0-ll,np.log(prior[1])+l1-ll]);prior=post@P
        if args.progress and (i+1)%args.progress==0:print(f'Forecasts {i+1}/{n-split}',flush=True)
    columns['hmm_predicted_prob_state0']=weights
    forecasts=pd.DataFrame(columns)
    forecasts.to_csv(out/'var_forecasts.csv',index=False)
    rows=[]
    for alpha in alphas:
        tag=str(round((1-alpha)*100,3)).replace('.','p')
        loss1=forecasts[f'student_t_pinball_{tag}'].to_numpy();loss2=forecasts[f'hmm6_pinball_{tag}'].to_numpy()
        diff=loss1-loss2
        m,se=hac_mean_se(diff,args.hac_lags)
        for sample_name,sel in [('all',np.ones(len(forecasts),dtype=bool)),('within_session',~gap[split:]),('overnight',gap[split:])]:
            if not sel.any():continue
            for name in ['student_t','hmm6']:
                vaar=forecasts[f'{name}_VaR_{tag}_pct'].to_numpy()[sel];es=forecasts[f'{name}_ES_{tag}_pct'].to_numpy()[sel]
                breach=forecasts[f'{name}_breach_{tag}'].to_numpy()[sel]
                row={'market':args.market,'sample':sample_name,'alpha':alpha,'confidence_pct':100*(1-alpha),
                     'model':name,'n':int(sel.sum()),'mean_VaR_pct_log':float(vaar.mean()),
                     'mean_ES_pct_log':float(es.mean()),'mean_pinball_loss':float(forecasts[f'{name}_pinball_{tag}'].to_numpy()[sel].mean()),
                     **kupiec(breach,alpha)}
                if sample_name=='all':row.update({'student_t_minus_hmm6_pinball':m,'difference_HAC_se':se,
                                                  'difference_HAC_z':m/se if se and se>0 else np.nan})
                rows.append(row)
    pd.DataFrame(rows).to_csv(out/'var_comparison.csv',index=False)
    config={'market_label':args.market,'input_file':str(Path(args.csv).resolve()),'instrument_identity':'USER MUST VERIFY; filename is not proof of ticker',
            'data_audit':audit,'analysis_n':n,'split_index':split,'train_end':str(timestamps[split-1]),'test_start':str(timestamps[split]),
            'models':{'student_t_gap_aware_3param':{'nu':float(x[0]),'s':float(x[1]),'overnight_multiplier':float(np.exp(x[2])),'fit':fd,'train_loglik':float(-fres.fun*split)},
                      'hmm_gap_aware_8param':{**{k:float(v) for k,v in zip(KEYS_H,h)},'fit':hd,'train_loglik':float(-hres.fun*split)}},
            'test_mean_log_density':{k:v/(n-split) for k,v in logscore.items()},
            'forecast':'one observed price-to-price return ahead (NOT fixed 1 elapsed minute across gaps)',
            'predictive_hmm':'posterior after previous return multiplied by transition matrix; next observation not used',
            'gap_policy':'overnight if calendar date changes; irregular same-day intervals retain regular scale; flags supplied on target interval',
            'inference':'HAC approximate for unconditional pinball differential; examine overnight cluster and bootstrap robustness',
            'alpha_levels':alphas,'scenario':'no Gamma jumps; observational M1 vs M2 benchmark only',
            'software_note':'SciPy L-BFGS-B multi-start, no global optimality guarantee; audit start_spread_ll and boundary_proximity'}
    (out/'run_metadata.json').write_text(json.dumps(config,indent=2,allow_nan=False),encoding='utf-8')
    print('SAVED:',out/'var_forecasts.csv',out/'var_comparison.csv',out/'run_metadata.json',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv',required=True);p.add_argument('--format',choices=['win','us'],required=True)
    p.add_argument('--market',required=True,help='Verified instrument label e.g. WIN or SPY; do not infer from filename')
    p.add_argument('--out',required=True);p.add_argument('--train-fraction',type=float,default=.8)
    p.add_argument('--max-rows',type=int,default=0,help='Optional LAST N returns for a speed-only pilot; affects estimates')
    p.add_argument('--alphas',type=float,nargs='+',default=[.05,.01,.005]);p.add_argument('--hac-lags',type=int,default=60)
    p.add_argument('--progress',type=int,default=5000)
    a=p.parse_args()
    if any(not 0<v<.5 for v in a.alphas):p.error('All lower-tail alpha levels must be in (0,0.5)')
    run(a)
if __name__=='__main__':main()
