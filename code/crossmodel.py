"""Cross-model check of the general omitted-jump displacement formula
   d beta*_M / d eps |0 = H_M^{-1} E_0[S_M W],   W = sum_t sum_j gamma_tj (h_j(y_t)/f_j(y_t) - 1)
DGP: full_hmm6 centred Student-t HMM; jumps Y = X - B*G, B~Bern(eps), G=theta*Z, Z~Gamma(2,1).
'Simulated' derivative: pseudo-true fits on one long path with common random numbers over an eps grid.
'Predicted': H from the long eps=0 path; E[S W]/T from N independent short paths (score of model M at its eps=0 pseudo-true value)."""
import os
HERE=os.path.dirname(os.path.abspath(__file__))
import ctypes, json, sys, numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, roots_genlaguerre, gammaln
from scipy.stats import t as tdist
dp=ctypes.POINTER(ctypes.c_double); ip=ctypes.POINTER(ctypes.c_int)
H6=ctypes.CDLL(os.path.join(HERE,'mhmm.so')); H6.mhmm_ll.restype=ctypes.c_double; H6.mhmm_ll.argtypes=[dp,ip,ctypes.c_int,dp,dp]
AP=ctypes.CDLL(os.path.join(HERE,'aparch.so')); AP.aparch_ll.restype=ctypes.c_double; AP.aparch_ll.argtypes=[dp,ctypes.c_int,dp,ctypes.c_double]
P0=np.array([.97,.86,18.,5.2,.7,1.45]); K=2.0
def sim_hmm(T,rng,N=None):
    p00,p11,n0,n1,s0,s1=P0; pi0=(1-p11)/(2-p00-p11)
    shape=(T,) if N is None else (N,T)
    k=np.zeros(shape,int); u=rng.random(shape)
    if N is None:
        k[0]=u[0]>pi0
        for t in range(1,T): k[t]=(u[t]>p00) if k[t-1]==0 else (u[t]>1-p11)
    else:
        k[:,0]=u[:,0]>pi0
        for t in range(1,T): k[:,t]=np.where(k[:,t-1]==0,u[:,t]>p00,u[:,t]>1-p11)
    return np.where(k==0,rng.standard_t(n0,shape)*s0,rng.standard_t(n1,shape)*s1)
# ---------------- models in natural coordinates
class StudentT:
    name='Student-t'; names=['mu','s','nu']
    def ll(self,b,y): return float(np.sum(tdist.logpdf((y-b[0])/b[1],b[2])-np.log(b[1])))
    def start(self,y): return np.array([np.median(y),0.7*y.std(),5.0])
    def to_free(self,b): return np.array([b[0],np.log(b[1]),logit((b[2]-2.05)/97.95)])
    def from_free(self,q): return np.array([q[0],np.exp(q[1]),2.05+97.95*expit(q[2])])
class Aparch:
    name='APARCH(1,1)-t'; names=['mu','omega','alpha','gamma','beta','delta','nu']
    def ll(self,b,y):
        if b[1]<=0 or not(0<=b[2]<1) or not(-1<b[3]<1) or not(0<=b[4]<1) or b[5]<=0.3 or b[6]<=2.05: return -1e300
        yy=np.ascontiguousarray(y); bb=np.ascontiguousarray(b,dtype=float)
        return AP.aparch_ll(yy.ctypes.data_as(dp),len(yy),bb.ctypes.data_as(dp),float(self.h0))
    def start(self,y): return np.array([0.,.2,.14,0.,.68,1.6,4.5])
    def to_free(self,b): return np.array([b[0],np.log(b[1]),logit(b[2]),np.arctanh(b[3]),logit(b[4]),logit((b[5]-.3)/3.7),logit((b[6]-2.05)/97.95)])
    def from_free(self,q): return np.array([q[0],np.exp(q[1]),expit(q[2]),np.tanh(q[3]),expit(q[4]),.3+3.7*expit(q[5]),2.05+97.95*expit(q[6])])
class HMM6:
    name='HMM-6'; names=['p00','p11','nu0','nu1','s0','s1']
    def ll(self,b,y):
        if not(0<b[0]<1 and 0<b[1]<1 and b[2]>2 and b[3]>2 and b[4]>0 and b[5]>0): return -1e300
        yy=np.ascontiguousarray(y); o=np.zeros(len(yy),np.int32); bb=np.ascontiguousarray(np.r_[b,1.,1.])
        return H6.mhmm_ll(yy.ctypes.data_as(dp),o.ctypes.data_as(ip),len(yy),bb.ctypes.data_as(dp),None)
    def start(self,y): return P0.copy()
    def to_free(self,b): return np.array([logit(b[0]),logit(b[1]),np.log(b[2]-2.05),np.log(b[3]-2.05),np.log(b[4]),np.log(b[5])])
    def from_free(self,q): return np.array([expit(q[0]),expit(q[1]),2.05+np.exp(q[2]),2.05+np.exp(q[3]),np.exp(q[4]),np.exp(q[5])])
def num_grad(f,b,rel=1e-4):
    g=np.zeros(len(b))
    for i in range(len(b)):
        h=rel*max(abs(b[i]),1e-2); e=np.zeros(len(b)); e[i]=h; g[i]=(f(b+e)-f(b-e))/(2*h)
    return g
def num_hess(f,b,rel=1e-3):
    n=len(b); Hm=np.zeros((n,n)); hs=[rel*max(abs(x),1e-2) for x in b]
    for i in range(n):
        for j in range(i,n):
            ei=np.zeros(n); ej=np.zeros(n); ei[i]=hs[i]; ej[j]=hs[j]
            v=(f(b+ei+ej)-f(b+ei-ej)-f(b-ei+ej)+f(b-ei-ej))/(4*hs[i]*hs[j]); Hm[i,j]=Hm[j,i]=v
    return Hm
def fit(model,y,b0=None,newton=4):
    f=lambda q:-model.ll(model.from_free(q),y)/len(y)
    q0=model.to_free(model.start(y) if b0 is None else b0)
    r=minimize(f,q0,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-15,'gtol':1e-10})
    b=model.from_free(r.x)
    g=lambda bb: model.ll(bb,y)/len(y)
    for _ in range(newton):   # Newton polish in natural coordinates
        G=num_grad(g,b); Hm=num_hess(g,b)
        step=np.linalg.solve(Hm,G); b=b-step
        if np.max(np.abs(step))<1e-9: break
    return b
# ---------------- jump-incidence score W under the true HMM (full-sequence smoothing)
def f_t(y,nu,s): return tdist.pdf(y/s,nu)/s
XN,WN=roots_genlaguerre(48,K-1); WN=WN/np.exp(gammaln(K))
def W_paths(Y,theta):
    p00,p11,n0,n1,s0,s1=P0; P=np.array([[p00,1-p00],[1-p11,p11]]); pi=np.array([1-p11,1-p00])/(2-p00-p11)
    E=np.stack([f_t(Y,n0,s0),f_t(Y,n1,s1)],-1); N,T,_=E.shape
    al=np.zeros((N,T,2)); be=np.ones((N,T,2))
    a=pi*E[:,0]; a/=a.sum(1,keepdims=True); al[:,0]=a
    for t in range(1,T): a=(a@P)*E[:,t]; a/=a.sum(1,keepdims=True); al[:,t]=a
    for t in range(T-2,-1,-1): b=(E[:,t+1]*be[:,t+1])@P.T; b/=b.sum(1,keepdims=True); be[:,t]=b
    g=al*be; g/=g.sum(2,keepdims=True)
    W=np.zeros(N)
    for j,(nu,s) in enumerate(((n0,s0),(n1,s1))):
        h=sum(w*f_t(Y+theta*x,nu,s) for x,w in zip(XN,WN))
        W+=(g[:,:,j]*(h/f_t(Y,nu,s)-1)).sum(1)
    return W
if __name__=='__main__':
    T=int(sys.argv[1]) if len(sys.argv)>1 else 200000; theta=float(sys.argv[2]) if len(sys.argv)>2 else 0.9
    Ns=int(sys.argv[3]) if len(sys.argv)>3 else 4000; Ts=2000
    rng=np.random.default_rng(20261031); X=sim_hmm(T,rng); Bu=rng.random(T); Z=rng.gamma(K,1.0,T)
    eps_grid=[0.0,0.0025,0.005,0.01]
    res={}
    models=[StudentT(),Aparch(),HMM6()]
    # Single-impulse representation (Theorem 1): E0[S W]/T = E_{t,G}[ S(X^{[t,G]}) - S(X) ]  per observation.
    # A shock at t changes only the predictive terms u >= t, so the score difference is computed on a window
    # [t-L0, t+L1] (identical burn-in terms cancel exactly).
    Nimp=int(sys.argv[3]) if len(sys.argv)>3 else 20000; L0,L1=400,400
    rI=np.random.default_rng(99); ts=rI.integers(L0,T-L1,Nimp); gs=rI.gamma(K,1.0,Nimp)*theta
    for M in models:
        if isinstance(M,Aparch): M.h0=float(X.var())
        fits=[]; bb=None
        for e in eps_grid:
            y=X-(Bu<e)*Z*theta
            bb=fit(M,y,b0=bb); fits.append(bb)
        fits=np.array(fits); E=np.array(eps_grid)
        sim_d=np.array([np.polyfit(E,fits[:,i],2)[1] for i in range(fits.shape[1])])
        b0=fits[0]; g=lambda v: M.ll(v,X)/len(X); Hbar=-num_hess(g,b0)
        D=np.zeros((Nimp,len(b0)))
        for i,(t,gv) in enumerate(zip(ts,gs)):
            if isinstance(M,StudentT):
                w0=X[t:t+1]; w1=w0-gv
            else:
                w0=X[t-L0:t+L1]; w1=w0.copy(); w1[L0]-=gv
            D[i]=num_grad(lambda v: M.ll(v,w1)-M.ll(v,w0),b0)
        bbar=D.mean(0); bse=D.std(0)/np.sqrt(Nimp)
        Hinv=np.linalg.inv(Hbar); pred=Hinv@bbar; pse=np.sqrt((Hinv**2)@(bse**2))
        res[M.name]=dict(params=M.names,b0=b0.tolist(),fits=fits.tolist(),sim_deriv=sim_d.tolist(),pred=pred.tolist(),pred_se=pse.tolist(),bbar=bbar.tolist(),bbar_se=bse.tolist())
        print(M.name,flush=True)
        for i,nm in enumerate(M.names): print(f'  {nm:6s} b0={b0[i]: .4f}  simulated d/deps={sim_d[i]: .4f}  predicted={pred[i]: .4f} (s.e. {pse[i]:.4f})',flush=True)
    json.dump(res,open(f'crossmodel_theta{theta}.json','w'),indent=1)
