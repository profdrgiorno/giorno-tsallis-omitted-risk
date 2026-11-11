#include <math.h>
/* centred Student-t log density with scale s */
static double lt(double y,double nu,double s){double z=y/s;return lgamma(0.5*(nu+1))-lgamma(0.5*nu)-0.5*log(M_PI*nu)-log(s)-0.5*(nu+1)*log1p(z*z/nu);}
/* marked two-state HMM: p = {p00,p11,nu0,nu1,s0,s1,m0,m1}; o[t]=1 overnight (scale s_j*m_j) */
double mhmm_ll(const double*y,const int*o,int n,const double*p,double*pred0){
  double p00=p[0],p11=p[1],pi0=(1-p11)/(2-p00-p11),post0=pi0,ll=0;
  for(int t=0;t<n;t++){
    double q0=(t==0)?pi0:post0*p00+(1-post0)*(1-p11);
    if(pred0)pred0[t]=q0;
    double s0=p[4]*(o[t]?p[6]:1.0),s1=p[5]*(o[t]?p[7]:1.0);
    double a=log(q0)+lt(y[t],p[2],s0),b=log1p(-q0)+lt(y[t],p[3],s1),c=a>b?a:b;
    double l=c+log(exp(a-c)+exp(b-c)); post0=exp(a-l); ll+=l;
  }
  return ll;
}
