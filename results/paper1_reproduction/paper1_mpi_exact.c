/* MoA ψ DNF -> stable summary -> γ ONF -> Ω MPI ownership. */
/* Contract: each rank holds identical complete read-only Q,K,V. */
/* Complete O via audited staged point-to-point Omega transfers. */
#include <mpi.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
static inline size_t gamma2(size_t i,size_t j,size_t w){return i*w+j;}
int generated_attention_mpi(const double *Q,const double *K,const double *V,
 double *O,size_t n,size_t m,size_t d,size_t dv,MPI_Comm comm){
 int rank=0,p=0;
 if(MPI_Comm_rank(comm,&rank)!=MPI_SUCCESS)return 1;
 if(MPI_Comm_size(comm,&p)!=MPI_SUCCESS)return 2;
 if(p!=4 || n!=4 || m!=5 || d!=4 || dv!=3)return 3;
 const int counts[] = {3,3,3,3};
 const int displs[] = {0,3,6,9};
 const size_t owner_begin[] = {0,1,2,3};
 const size_t owner_end[] = {1,2,3,4};
 const double scale=1.0/sqrt((double)d);
 double *local=(double*)malloc((size_t)(counts[rank]?counts[rank]:1)*sizeof(double));
 if(!local)return 4;
 for(size_t i=owner_begin[rank];i<owner_end[rank];i++){
  double *u=(double*)calloc(dv,sizeof(double));if(!u){free(local);return 5;}
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
  for(size_t v=0;v<dv;v++)local[(i-owner_begin[rank])*dv+v]=u[v]/z;
  free(u);
 }

 /* Copy the locally owned gamma interval into final output storage. */
 memcpy(O+displs[rank],local,(size_t)counts[rank]*sizeof(double));
 free(local);

 /* Each row is {owner_rank,source_rank,destination_rank,count}. */
 static const int transfers[][4] = {
  {0,0,1,3},
  {0,0,2,3},
  {1,1,0,3},
  {1,1,3,3},
  {2,2,0,3},
  {3,3,1,3},
  {0,1,3,3},
  {1,0,2,3},
  {2,0,1,3},
  {2,2,3,3},
  {3,1,0,3},
  {3,3,2,3}
 };
 static const int stage_offsets[] = {0,6,12};

 for(int stage=0;stage<2;stage++){
   int first=stage_offsets[stage];
   int last=stage_offsets[stage+1];
   int request_capacity=2*(last-first);
   MPI_Request *requests=(MPI_Request*)malloc(
       (size_t)(request_capacity?request_capacity:1)*sizeof(MPI_Request));
   if(!requests)return 7;
   int used=0;

   /* Post receives before sends. */
   for(int k=first;k<last;k++){
     int owner=transfers[k][0];
     int sender=transfers[k][1];
     int receiver=transfers[k][2];
     int count=transfers[k][3];
     if(rank==receiver){
       if(MPI_Irecv(O+displs[owner],count,MPI_DOUBLE,
                    sender,owner,comm,&requests[used++])!=MPI_SUCCESS){
         free(requests);return 8;
       }
     }
   }

   for(int k=first;k<last;k++){
     int owner=transfers[k][0];
     int sender=transfers[k][1];
     int receiver=transfers[k][2];
     int count=transfers[k][3];
     if(rank==sender){
       if(MPI_Isend(O+displs[owner],count,MPI_DOUBLE,
                    receiver,owner,comm,&requests[used++])!=MPI_SUCCESS){
         free(requests);return 9;
       }
     }
   }

   if(MPI_Waitall(used,requests,MPI_STATUSES_IGNORE)!=MPI_SUCCESS){
     free(requests);return 10;
   }
   free(requests);
 }

 return 0;
}
