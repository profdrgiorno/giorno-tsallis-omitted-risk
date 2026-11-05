"""Check Lemma B3 numerically: E[d xi / d beta] = 0 for every parameter (location, scale, nu), while
E[d a / d mu] != 0, on a long clean HMM path with a misspecified static Student-t fit."""
import numpy as np, sys
sys.path.insert(0,'/home/claude/paperB_work')
import hmm9_srj as H, localpower_check as L
rng=np.random.default_rng(5); x=L.hmm_path(300_000,rng); loc,sc,nu=H.fit_t(x)
Xi,kap=L.xi_matrix(x,loc,sc,nu)
def A(l,s,v): return np.array([H._afun(x,l,s,v,th,True) for th in H.GRID])
def S(l,s,v): return H._smu(x,l,s,v)
def xi(l,s,v): return A(l,s,v)-kap[:,None]*S(l,s,v)[None,:]
out=[]
for name,d in (('mu',(1e-3,0,0)),('scale',(0,1e-3,0)),('nu',(0,0,1e-2))):
    p=np.array([loc,sc,nu]); dp=np.array(d)
    dxi=(xi(*(p+dp))-xi(*(p-dp)))/(2*max(d)); da=(A(*(p+dp))-A(*(p-dp)))/(2*max(d))
    print(f"{name:6s} E[d xi]: "+" ".join(f"{m:+.4f}({s_:.4f})" for m,s_ in zip(dxi.mean(1),dxi.std(1)/np.sqrt(x.size))))
    print(f"{'':6s} E[d a ]: "+" ".join(f"{m:+.4f}({s_:.4f})" for m,s_ in zip(da.mean(1),da.std(1)/np.sqrt(x.size))))
print('pseudo-true',loc,sc,nu,'kappa',kap.round(4)); print('max|a| over sample',np.abs(A(loc,sc,nu)).max(1).round(3))
