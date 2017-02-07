#ifdef __cplusplus
extern "C" {
#endif

__global__ void vec_add(float *A, float *B, float* C,
				int size)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;

  if(index<size)
    C[index] = A[index] + B[index];
}




__global__ void vec_add_double(double *A, double *B, double* C,
				int size)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;

  if(index<size)
    C[index] = A[index] + B[index];
}

#ifdef __cplusplus
}
#endif
