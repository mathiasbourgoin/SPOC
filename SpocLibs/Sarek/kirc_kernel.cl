float spoc_fadd ( float a, float b );
float spoc_fminus ( float a, float b );
float spoc_fmul ( float a, float b );
float spoc_fdiv ( float a, float b );
float spoc_fadd ( float a, float b ) { return (a + b);}
float spoc_fminus ( float a, float b ) { return (a - b);}
float spoc_fmul ( float a, float b ) { return (a * b);}
float spoc_fdiv ( float a, float b ) { return (a / b);}
__kernel void spoc_dummy ( __global int* spoc_var0, __global int* spoc_var1, __global int* spoc_var3 ) 
{
spoc_var3[get_global_id(0)] = spoc_var0[get_global_id (0)] * spoc_var1[get_global_id (0)];
}