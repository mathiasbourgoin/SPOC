float spoc_fadd ( float a, float b );
float spoc_fminus ( float a, float b );
float spoc_fmul ( float a, float b );
float spoc_fdiv ( float a, float b );
int logical_and (int, int);
int spoc_powint (int, int);
int spoc_xor (int, int);
float spoc_fadd ( float a, float b ) { return (a + b);}
float spoc_fminus ( float a, float b ) { return (a - b);}
float spoc_fmul ( float a, float b ) { return (a * b);}
float spoc_fdiv ( float a, float b ) { return (a / b);}
int logical_and (int a, int b ) { return (a & b);}
int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}
int spoc_xor (int a, int b ) { return (a^b);}
/************* CUSTOM TYPES ******************/


/************* FUNCTION PROTOTYPES ******************/
/************* FUNCTION DEFINITIONS ******************/
__kernel void spoc_dummy ( __global int* m_in, __global int* m_out, int n, int m ) {
  int y;
  int x;
  int tmp;
  y = ((get_local_id (1)) + ((get_group_id (1)) * (get_local_size (1)))) ;
  x = ((get_local_id (0)) + ((get_group_id (0)) * (get_local_size (0)))) ;
  if (x < m && y < n){
    tmp = 0 ;
    for (int j = 0; j <= y; j++){
      for (int i = 0; i <= x; i++){
        tmp = (tmp + m_in[((j * m) + i)]);};} ;
    m_out[((y * m) + x)] = (tmp % 2);;
  }  
  
  
  
}