/* Gate 09: sequential rank-by-rank simulation of MPI collectives.
   NOT a genuine MPI implementation, parallel run, or network prediction. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#ifndef N
#define N 4
#endif
#ifndef M
#define M 4
#endif
#ifndef D
#define D 8
#endif
#ifndef DV
#define DV 8
#endif
#ifndef P
#define P 6
#endif
static int current_rank=0;
static double assembled[N*DV];
static int visits[P];
int MPI_Comm_rank(MPI_Comm comm,int *rank){(void)comm;*rank=current_rank;return MPI_SUCCESS;}
int MPI_Comm_size(MPI_Comm comm,int *p){(void)comm;*p=P;return MPI_SUCCESS;}
int MPI_Allgatherv(const void *send,int n,int sendtype,void *recv,const int *counts,const int *displs,int recvtype,MPI_Comm comm){
 (void)recv;(void)comm;
 if(sendtype!=MPI_DOUBLE||recvtype!=MPI_DOUBLE||n!=counts[current_rank])return 17;
 if(displs[current_rank]<0||displs[current_rank]+n>N*DV)return 18;
 memcpy(assembled+displs[current_rank],send,(size_t)n*sizeof(double));visits[current_rank]++;
 return MPI_SUCCESS;
}
int generated_attention_mpi(const double*,const double*,const double*,double*,size_t,size_t,size_t,size_t,MPI_Comm);
static void reference(const double *Q,const double *K,const double *V,double *O){
 for(int i=0;i<N;i++){
  double scores[M],a=-INFINITY,z=0;
  for(int k=0;k<M;k++){
   double dot=0;for(int j=0;j<D;j++)dot+=Q[i*D+j]*K[k*D+j];
   scores[k]=dot/sqrt((double)D);if(scores[k]>a)a=scores[k];
  }
  for(int k=0;k<M;k++)z+=exp(scores[k]-a);
  for(int v=0;v<DV;v++){
   double u=0;for(int k=0;k<M;k++)u+=exp(scores[k]-a)*V[k*DV+v];
   O[i*DV+v]=u/z;
  }
 }
}
int main(void){
 double Q[N*D],K[M*D],V[M*DV],expected[N*DV],unused[N*DV];
 double worst=0;
 for(int t=0;t<25;t++){
  for(int i=0;i<N*D;i++)Q[i]=sin((i+1)*.17+t*.013)*(t%2?1:7);
  for(int i=0;i<M*D;i++)K[i]=cos((i+3)*.13-t*.019)*(t%2?1:7);
  for(int i=0;i<M*DV;i++)V[i]=sin((i+5)*.11+t*.023);
  reference(Q,K,V,expected);
  for(int i=0;i<N*DV;i++)assembled[i]=NAN;
  memset(visits,0,sizeof visits);
  for(int r=0;r<P;r++){
   current_rank=r;
   if(generated_attention_mpi(Q,K,V,unused,N,M,D,DV,0))return 2;
  }
  for(int r=0;r<P;r++)if(visits[r]!=1)return 3;
  for(int i=0;i<N*DV;i++){
   double e=fabs(assembled[i]-expected[i]);
   if(!isfinite(assembled[i])||e>1e-11){fprintf(stderr,"FAIL trial=%d index=%d err=%g\n",t,i,e);return 4;}
   if(e>worst)worst=e;
  }
 }
 printf("PASS emulated MPI rank ownership and output assembly: N=%d M=%d D=%d DV=%d P=%d trials=25 maxerr=%.17g\n",N,M,D,DV,P,worst);
 return 0;
}
