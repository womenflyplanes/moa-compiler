/* Generated from typed MoA ψ DNF → stable summary → γ ONF → Ω ownership. */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
static inline size_t gamma2(size_t i,size_t j,size_t w){return i*w+j;}
void generated_attention(const double *Q,const double *K,const double *V,
                         double *O,size_t n,size_t m,size_t d,size_t dv){
 if(n!=8 || m!=8 || d!=4 || dv!=4)abort();
 const double scale=1.0/sqrt((double)d);
 const size_t owner_begin[] = {0,2,4,6};
 const size_t owner_end[] = {2,4,6,8};
 /* The OpenACC data region transfers inputs and outputs; ownership is disjoint. */
 #pragma acc data copyin(Q[0:n*d],K[0:m*d],V[0:m*dv]) copyout(O[0:n*dv])
 {
 #pragma acc parallel loop gang copyin(owner_begin[0:4],owner_end[0:4])
 for(size_t owner=0;owner<4;owner++){
  #pragma acc loop seq
  for(size_t i=owner_begin[owner];i<owner_end[owner];i++){
   double u[4];
   for(size_t v=0;v<dv;v++)u[v]=0.0;
   double a=-INFINITY,z=0.0;
   for(size_t k=0;k<m;k++){
    double score=0.0;
    for(size_t j=0;j<d;j++)score+=Q[gamma2(i,j,d)]*K[gamma2(k,j,d)];
    score*=scale;
    double next=fmax(a,score);
    double old=(z==0.0)?0.0:exp(a-next),fresh=exp(score-next);
    z=old*z+fresh;
    for(size_t v=0;v<dv;v++)u[v]=old*u[v]+fresh*V[gamma2(k,v,dv)];
    a=next;
   }
   for(size_t v=0;v<dv;v++)O[gamma2(i,v,dv)]=u[v]/z;
  }
 }
 }
}
