import numpy as np, json, sys
sys.path.insert(0,'/home/claude/paperB_work'); import hmm9_srj as H, localpower_check as L
r=json.load(open('lp_c4.json')); loc,sc,nu=r['pseudo_true']; kap=np.array(r['kappa'])
rng=np.random.default_rng(778); x=L.hmm_path(400_000,rng); G=0.9*rng.gamma(2.0,1.0,x.size)
Xi,_=L.xi_matrix(x,loc,sc,nu,kappa=kap); Xj,_=L.xi_matrix(x-G,loc,sc,nu,kappa=kap)
lam=(Xj-Xi).mean(1); Om=(Xi**2).mean(1); M2=(Xj**2).mean(1); eps=4/np.sqrt(20000)
print('lam',lam.round(3)); print('Omega',Om.round(3)); print('E xi(X-G)^2',M2.round(2))
print('first-order pred',(4*lam/np.sqrt(Om)).round(3))
print('denominator-adjusted pred',(4*lam/np.sqrt((1-eps)*Om+eps*M2)).round(3))
print('sim',np.round(r['sim_meanZ'],3),'null-bias-corrected sim',np.round(np.array(r['sim_meanZ'])-np.array(json.load(open('lp_c0.json'))['sim_meanZ']),3))
