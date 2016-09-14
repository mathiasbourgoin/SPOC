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
__kernel void spoc_dummy ( __global float* v, int j, int k ) {
  int i;
  int ixj;
  float temp;
  i = (get_local_id (0)) + (get_local_size (0)) * (get_group_id (0)) ;
  ixj = spoc_xor (i,j)  ;
  temp = 0.f ;
  if (ixj >= i){
    if (logical_and (i,k)  == 0){
      if (v[i] > v[ixj]){
        temp = v[ixj] ;
        v[ixj] = v[i]; ;
        v[i] = temp;
      }      ;
    }
    else{
      if (v[i] < v[ixj]){
        temp = v[ixj] ;
        v[ixj] = v[i]; ;
        v[i] = temp;
      }      ;
    }
    
  }  
  
  
  
}
