/******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
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
*******************************************************************************/
__kernel void vec_add(__global const float * a, __global const float * b, __global float * c, int N)
{
    int nIndex = get_global_id(0);
    if (nIndex >= N)
          return;
    c[nIndex] = a[nIndex] + b[nIndex];
}

__kernel void vec_sub(__global const float * a, __global const float * b, __global float * c, int N)
{
    int nIndex = get_global_id(0);
    if (nIndex >= N)
          return;
    c[nIndex] = a[nIndex] - b[nIndex];
}
__kernel void vec_div(__global const float * a, __global const float * b, __global float * c, int N)
{
    int nIndex = get_global_id(0);
    if (nIndex >= N)
          return;
    c[nIndex] = a[nIndex] / b[nIndex];
}
__kernel void vec_mult(__global const float * a, __global const float * b, __global float * c, int N)
{
    int nIndex = get_global_id(0);
    if (nIndex >= N)
          return;
    c[nIndex] = a[nIndex] * b[nIndex];
}
__kernel void vec_fma(__global const float * a, __global const float * b, __global const float * c, __global float *d,  int N)
{
    int nIndex = get_global_id(0);
    if (nIndex >= N)
          return;
    d[nIndex] = a[nIndex] * b[nIndex] + c[nIndex];
}


__kernel void sum(
          __global int * vec1,
          __global int * result,
		__global int* tmp1,
__global int count)
{
  //parallel reduction on global memory:
  int n = count/2;
	if (get_global_id(0) < n)
      tmp1[get_global_id(0)] =  vec1[get_global_id(0)] + vec1[get_global_id(0)+n];
    barrier(CLK_GLOBAL_MEM_FENCE);


	for (unsigned int stride = n/2; stride > 0; stride /= 2)
  {
    if (get_global_id(0) < stride)
      tmp1[get_global_id(0)] += tmp1[get_global_id(0)+stride];
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  if (get_global_id(0) == 0)
    *result = tmp1[0];
}

//#define NB_THREADS 256
__kernel void spoc_max(__global const double* input, __global double* output, const int size) 
{
	int i = get_global_id(0);
	if (i > 1) return;
	const int NB_THREADS = 256;
	__local double r[NB_THREADS]; 
	double res = 0;
	r[i] = input[(size/NB_THREADS * i)];
	#pragma unroll 8
	for (int j = (size/NB_THREADS * i) ; j < (size/NB_THREADS * (i +1)); j++)
	{
		if (r[i] < fabs(input[j]))
			r[i] = fabs(input[j]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	res = r[0];
	if (i == 0)
		for (int j = 1; j < NB_THREADS; j++)
		{
		if (res < (r[j]))
			res = (r[j]);
	}
	output[0] = res;
	
	
}	

__kernel void int_bubble_filter(
   	__global int* input,
	__constant int* vec1,
	__global int* output,
   	const int count)
{
	int i;
	int k = 1;
int nb_threads = get_global_size(0);
	if (get_global_id(0) <= count/2)
	{
		output[get_global_id(0)*2] = vec1[get_global_id(0)*2];
		output[get_global_id(0)*2+1] = vec1[get_global_id(0)*2+1];
	    //barrier(CLK_GLOBAL_MEM_FENCE);

		for (int n = 0; n < count*2; n++)
		{
			k = (k)?0:1;
			i = (get_global_id(0)*2) + k;
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
	    		barrier(CLK_GLOBAL_MEM_FENCE);
		}
	}
}
