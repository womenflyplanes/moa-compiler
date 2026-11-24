/* New hand ONF transcription from Paper III Listing 7 and equations (1)-(3),(10).
 * This is a derived C baseline for the paper's Python algorithm, not an archived
 * C implementation. H%G==0; positive extents; N<=MAX_N. */
void hand_grouped_openmp(float *Q,float *K,float *V,float *Out,int H,int G,int N,int D,int E) {
 float scale=1.0f/sqrtf((float)D);
 #pragma omp parallel for schedule(static)
 for(int h=0;h<H;h++) {
  int group=h/(H/G);
  float row[MAX_N];
  float m=-1e38f;
  for(int k=0;k<N;k++) {
   float dot=0.0f;
   for(int j=0;j<D;j++) dot+=Q[h*D+j]*K[(group*N+k)*D+j];
   row[k]=scale*dot;
   if(row[k]>m)m=row[k];
  }
  float z=0.0f;
  for(int k=0;k<N;k++){row[k]=expf(row[k]-m);z+=row[k];}
  for(int d=0;d<E;d++) {
   float total=0.0f;
   for(int k=0;k<N;k++) total+=(row[k]/z)*V[(group*N+k)*E+d];
   Out[h*E+d]=total;
  }
 }
}
void hand_grouped_openacc(float *Q,float *K,float *V,float *Out,int H,int G,int N,int D,int E) {
 float scale=1.0f/sqrtf((float)D);
 #pragma acc parallel loop gang
 for(int h=0;h<H;h++) {
  int group=h/(H/G);
  float row[MAX_N];
  float m=-1e38f;
  for(int k=0;k<N;k++) {
   float dot=0.0f;
   for(int j=0;j<D;j++) dot+=Q[h*D+j]*K[(group*N+k)*D+j];
   row[k]=scale*dot;
   if(row[k]>m)m=row[k];
  }
  float z=0.0f;
  for(int k=0;k<N;k++){row[k]=expf(row[k]-m);z+=row[k];}
  for(int d=0;d<E;d++) {
   float total=0.0f;
   for(int k=0;k<N;k++) total+=(row[k]/z)*V[(group*N+k)*E+d];
   Out[h*E+d]=total;
  }
 }
}
