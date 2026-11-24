/* Manual transcription of arXiv:2607.19456 Listing 3, printed page 8.
 * Formatting/comments removed; operators, pragmas, precision and order retained. */
void moa_decode_acc(const float * restrict q,const float * restrict K,const float * restrict V,
                    float * restrict out,int n,int dk,int dv) {
 const float sc=1.0f/sqrtf((float)dk);
 float *s=malloc(n*sizeof(float));
 float *e=malloc(n*sizeof(float));
 float m=-1e38f;
 #pragma acc data copyin(q[0:dk],K[0:n*dk],V[0:n*dv]) copyout(out[0:dv]) create(s[0:n],e[0:n])
 {
  #pragma acc parallel loop gang reduction(max:m)
  for(int l=0;l<n;l++) {
   float dot=0.0f;
   #pragma acc loop vector reduction(+:dot)
   for(int j=0;j<dk;j++) dot+=q[j]*K[l*dk+j];
   s[l]=sc*dot;
   if(s[l]>m)m=s[l];
  }
  float Z=0.0f;
  #pragma acc parallel loop gang reduction(+:Z)
  for(int l=0;l<n;l++){e[l]=expf(s[l]-m);Z+=e[l];}
  float inv_Z=1.0f/Z;
  #pragma acc parallel loop gang
  for(int d=0;d<dv;d++) out[d]=0.0f;
  #pragma acc parallel loop gang
  for(int l=0;l<n;l++) {
   float scaled_weight=e[l]*inv_Z;
   #pragma acc loop vector
   for(int d=0;d<dv;d++) {
    #pragma acc atomic
    out[d]+=scaled_weight*V[l*dv+d];
   }
  }
 }
 free(s);free(e);
}
