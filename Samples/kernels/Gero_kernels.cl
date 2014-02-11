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