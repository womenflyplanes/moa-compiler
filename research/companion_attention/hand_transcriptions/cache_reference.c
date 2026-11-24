/* New hand C ONF of Paper III Listing 5, lifted over independent cache groups.
 * G=1 is the published rank-one key/value case. N<C, coherent resident buffers. */
int hand_append_openmp(float *K,float *V,float *key,float *value,int G,int C,int N,int D,int E) {
 #pragma omp parallel for schedule(static)
 for(int g=0;g<G;g++) {
  for(int j=0;j<D;j++) K[(g*C+N)*D+j]=key[g*D+j];
  for(int d=0;d<E;d++) V[(g*C+N)*E+d]=value[g*E+d];
 }
 return N+1;
}
int hand_append_openacc(float *K,float *V,float *key,float *value,int G,int C,int N,int D,int E) {
 #pragma acc parallel loop gang present(K,V,key,value)
 for(int g=0;g<G;g++) {
  for(int j=0;j<D;j++) K[(g*C+N)*D+j]=key[g*D+j];
  for(int d=0;d<E;d++) V[(g*C+N)*E+d]=value[g*E+d];
 }
 return N+1;
}
