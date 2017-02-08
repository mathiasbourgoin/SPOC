
struct point{
  float x;
  float y;
};


__kernel void pi(__global struct point* A, __global int* res, const int nbPoint, const float ray){
  const int idx = 32*get_local_size(0) * get_group_id (0) + get_local_id (0);
  int dim = get_local_size (0);
  if (idx < (int)(nbPoint-32*dim))
    #pragma unroll 16
    for (int j = 0; j < 32; j++) {
      int i = idx + dim * j;
      res[i] = (A[i].x*A[i].x + A[i].y*A[i].y <= ray);
    }
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable


struct point2{
  double x;
  double y;
};

__kernel void pi_double(__global struct point2* A, __global int* res, const int nbPoint, const float ray){
  const int idx = 32*get_local_size(0) * get_group_id (0) + get_local_id (0);
  int dim = get_local_size (0);
  if (idx < (int)(nbPoint-32*dim))
    #pragma unroll 16
    for (int j = 0; j < 32; j++) {
      int i = idx + dim * j;
      res[i] = (A[i].x*A[i].x + A[i].y*A[i].y <= (double) ray);
    }
}
#endif
