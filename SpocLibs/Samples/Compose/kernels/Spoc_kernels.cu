/******************************************************************************
 * © Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is allow GPU programming
 * with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
 *
 * NOTE:  This file contains source code provided by NVIDIA Corporation.
*******************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

/****** Single precision *****/
__global__ void vec_add(const float* A, const float* B, float* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		;
	C[i] = A[i] + B[i];
}

__global__ void vec_mult(const float* A, const float* B, float* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		;
	C[i] = A[i] * B[i];
}

__global__ void vec_div(const float* A, const float* B, float* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		;
	C[i] = A[i] / B[i];
}

__global__ void vec_sub(const float* A, const float* B, float* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		;
	C[i] = A[i] - B[i];
}

__global__ void vec_fma(const float* A, const float* B, float* C, float* D,
		int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		;
	D[i] = A[i] + B[i] * C[i];
}

/****** Double precision *****/
__global__ void vec_add_64(const double* A, const double* B, double* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		;
	C[i] = A[i] + B[i];
}


__global__ void sum(
           int * vec1,
           int * result,
           int* tmp1,
const int count)
{
  //parallel reduction on global memory:
  int n = count/2;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n)
      tmp1[tid] =  vec1[tid] + vec1[tid+n];
	__syncthreads();

	for (unsigned int stride = n/2; stride > 0; stride /= 2)
  {
    if (tid < stride)
      tmp1[tid] += tmp1[tid+stride];
    __syncthreads();
  }
  if (tid == 0)
    *result = tmp1[0];
}


__global__ void spoc_max(const double* input, double* output, const int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > 0) return;
	double r = fabs(input[0]);
	for (int j = 1; j < size; j++)
	{
		if (r < fabs(input[j]))
			r = fabs(input[j]);
	}
	output[0] = r;
}

__global__ void int_bubble_filter(
   	 int* input,
	const int* vec1,
	int* output,
   	const int count)
{
	int i;
	int k = 1;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid <= count/2)
	{
		output[tid*2] = vec1[tid*2];
		output[tid*2+1] = vec1[tid*2+1];
	    //barrier(CLK_GLOBAL_MEM_FENCE);

		for (int n = 0; n < count*2; n++)
		{
			k = (k)?0:1;
			i = (tid*2) + k;
			if( i+1 < count)
			{
				if ((!input[i]) && (input[i+1]))
				{
					input[i] = 1;
					input[i+1] = 0;
					output[i] = output[i+1];
					output[i+1] = 0;
				}
				else
				{
					if (!input[i])
						output[i] = 0;
					if (!input[i+1])
						output[i+1] = 0;
				}
			}
	    		__syncthreads();
		}
	}
}


#ifdef __cplusplus
}
#endif
