#include <math.h>
/* APARCH(1,1) with unit-variance Student-t innovations, as in the repo (typ 4):
   q_t = omega + alpha(|e|-gamma e)^delta + beta h^{delta/2},  h_t = q^{2/delta}.  p={mu,omega,alpha,gamma,beta,delta,nu} */
double aparch_ll(const double*y,int n,const double*p,double h0){
  double mu=p[0],om=p[1],al=p[2],ga=p[3],be=p[4],de=p[5],nu=p[6];
  double c=lgamma(0.5*(nu+1))-lgamma(0.5*nu)-0.5*log(M_PI*(nu-2)),h=h0,ep=0,ll=0;
  for(int t=0;t<n;t++){
    if(t>0){double q=om+al*pow(fabs(ep)-ga*ep,de)+be*pow(h,0.5*de); if(q<1e-20)q=1e-20; h=pow(q,2.0/de);}
    if(!isfinite(h)||h>1e10) return -1e300;
    double e=y[t]-mu; ll+=c-0.5*log(h)-0.5*(nu+1)*log1p(e*e/((nu-2)*h)); ep=e;
  }
  return ll;
}
