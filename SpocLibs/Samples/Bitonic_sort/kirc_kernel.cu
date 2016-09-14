__device__ float spoc_fadd ( float a, float b ) { return (a + b);}
__device__ float spoc_fminus ( float a, float b ) { return (a - b);}
__device__ float spoc_fmul ( float a, float b ) { return (a * b);}
__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}
__device__ int logical_and (int a, int b ) { return (a & b);}
__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}
__device__ int spoc_xor (int a, int b ) { return (a^b);}


#ifdef __cplusplus
extern "C" {
#endif

__global__ void spoc_dummy (  float* v, int j, int k ) {
  int i;
  int ixj;
  float temp;
  i = (threadIdx.x + (blockDim.x * blockIdx.x)) ;
  ixj = spoc_xor (i,j)  ;
  temp = 0.f ;
  if (ixj >= i){
    if (logical_and (i,k)  == 0){
      if (v[i] > v[ixj]){
        temp = v[ixj] ;
        v[ixj] = v[i]; ;
        v[i] = temp;;
      }      ;
    }
    else{
      if (v[i] < v[ixj]){
        temp = v[ixj] ;
        v[ixj] = v[i]; ;
        v[i] = temp;;
      }      ;
    }
    ;
  }  
  
  
  
}
#ifdef __cplusplus
}
#endif