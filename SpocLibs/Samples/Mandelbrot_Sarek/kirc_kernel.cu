__device__ float spoc_fadd ( float a, float b ) { return (a + b);}
__device__ float spoc_fminus ( float a, float b ) { return (a - b);}
__device__ float spoc_fmul ( float a, float b ) { return (a * b);}
__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}
__device__ int logical_and (int a, int b ) { return (a & b);}
__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}
__device__ int spoc_xor (int a, int b ) { return (a^b);}


__device__ float spoc_fun__2  ( float x ){return (x * x);
  }
__device__ float spoc_fun__1  ( float x, float y ){return (spoc_fun__2 (x)  + spoc_fun__2 (y) );
  }
#ifdef __cplusplus
extern "C" {
#endif

__global__ void spoc_dummy (  int* img, int shiftx, int shifty, float zoom ) {
  int y;
  int x;
  int x0;
  int y0;
  int cpt;
  float x1;
  float y1;
  float x2;
  float y2;
  float a;
  float b;
  float norm;
  y = (threadIdx.y + (blockIdx.y * blockDim.y)) ;
  x = (threadIdx.x + (blockIdx.x * blockDim.x)) ;
  if (y >= 1000 || x >= 1000){
    return  ;
  }   ;
  x0 = (x + shiftx) ;
  y0 = (y + shifty) ;
  cpt = 0 ;
  x1 = 0.f ;
  y1 = 0.f ;
  x2 = 0.f ;
  y2 = 0.f ;
  a = ((4.f * (((float) (x0)  / (float) (1000) ) / zoom)) - 2.f) ;
  b = ((4.f * (((float) (y0)  / (float) (1000) ) / zoom)) - 2.f) ;
  norm = spoc_fun__1 (x1,y1)  ;
  while (cpt < 512 && norm <= 4.f){
    cpt = (cpt + 1) ;
    x2 = (((x1 * x1) - (y1 * y1)) + a) ;
    y2 = (((2.f * x1) * y1) + b) ;
    x1 = x2 ;
    y1 = y2 ;
    norm = ((x1 * x1) + (y1 * y1));} ;
  img[((y * 1000) + x)] = cpt;
  
  
  
  
  
  
  
  
  
  
  
  
}
#ifdef __cplusplus
}
#endif