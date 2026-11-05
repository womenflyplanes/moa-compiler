/* Generated from typed MoA ψ DNF → summary rule → γ ONF. */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
static inline size_t gamma2(size_t i,size_t j,size_t w){return i*w+j;}
void generated_attention(const double *Q,const double *K,const double *V,
                         double *O,size_t n,size_t m,size_t d,size_t dv){
 if(!n||!m||!d||!dv)abort();
 const double scale=1.0/sqrt((double)d);
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<n;i++){
  double *u=(double*)calloc(dv,sizeof(double)); if(!u)abort();
  double a=-INFINITY,z=0;
  for(size_t k=0;k<m;k++){
   double score=0;
   for(size_t j=0;j<d;j++)score+=Q[gamma2(i,j,d)]*K[gamma2(k,j,d)];
   score*=scale;
   double next=fmax(a,score);
   double old=(z==0)?0:exp(a-next),fresh=exp(score-next);
   z=old*z+fresh;
   for(size_t v=0;v<dv;v++)u[v]=old*u[v]+fresh*V[gamma2(k,v,dv)];
   a=next;
  }
  for(size_t v=0;v<dv;v++)O[gamma2(i,v,dv)]=u[v]/z;
  free(u);
 }
}
