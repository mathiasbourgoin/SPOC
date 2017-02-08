#ifdef __cplusplus
extern "C" {
  #endif

  struct point{
    float x;
    float y;
  };
  
  
  __global__ void pi(const struct point* A, int* res, const int nbPoint, const float ray){
    const int idx = 32*blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nbPoint-32*blockDim.x)
      #pragma unroll 16
      for (int j = 0; j < 32; j++) {
	int i = idx + blockDim.x * j;
	res[i] = (A[i].x*A[i].x + A[i].y*A[i].y <= ray);
      }
  }

  
  struct point2{
    double x;
    double y;
  };

  
__global__ void pi_double(const struct point2* A, int* res, const int nbPoint, const float ray){
    const int idx = 32*blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nbPoint-32*blockDim.x)
      if (idx < (int)(nbPoint-32*blockDim.x))
	#pragma unroll 16
	for (int j = 0; j < 32; j++) {
	  int i = idx + blockDim.x * j;
	  res[i] = (A[i].x*A[i].x + A[i].y*A[i].y <= (double)ray);
	}
  }
  
  
  #ifdef __cplusplus
}
#endif
