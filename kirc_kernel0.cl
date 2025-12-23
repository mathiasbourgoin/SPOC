#define SAREK_VEC_LENGTH(A) sarek_## A ##_length
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
void spoc_barrier ( ) { barrier(CLK_LOCAL_MEM_FENCE);}
/************* CUSTOM TYPES ******************/
struct Test_registered_type_point_sarek {
  float x;
  float y;
};struct Test_registered_type_point_sarek build_Test_registered_type_point_sarek(float x, float y) {
  struct Test_registered_type_point_sarek res;
  res.x = x;
  res.y = y;
  return res;
}struct Test_registered_variant_color_sarek_Red {
  int Test_registered_variant_color_sarek_Red_t;
};struct Test_registered_variant_color_sarek_Value {
  float Test_registered_variant_color_sarek_Value_t;
};union Test_registered_variant_color_sarek_union {
  struct Test_registered_variant_color_sarek_Red Test_registered_variant_color_sarek_Red;
  struct Test_registered_variant_color_sarek_Value Test_registered_variant_color_sarek_Value;
};struct Test_registered_variant_color_sarek {
  int Test_registered_variant_color_sarek_tag;
  union Test_registered_variant_color_sarek_union Test_registered_variant_color_sarek_union;
};struct Test_registered_variant_color_sarek build_Test_registered_variant_color_Red() {
  struct Test_registered_variant_color_sarek res;
  res.Test_registered_variant_color_sarek_tag = 0;
  /* no payload */
  return res;
}struct Test_registered_variant_color_sarek build_Test_registered_variant_color_Value(float v) {
  struct Test_registered_variant_color_sarek res;
  res.Test_registered_variant_color_sarek_tag = 1;
  res.Test_registered_variant_color_sarek_union.Test_registered_variant_color_sarek_Value.Test_registered_variant_color_sarek_Value_t = v;
  return res;
}struct Registered_defs_vec2_sarek {
  float x;
  float y;
};struct Registered_defs_vec2_sarek build_Registered_defs_vec2_sarek(float x, float y) {
  struct Registered_defs_vec2_sarek res;
  res.x = x;
  res.y = y;
  return res;
}struct Geometry_lib_point_sarek {
  float x;
  float y;
};struct Geometry_lib_point_sarek build_Geometry_lib_point_sarek(float x, float y) {
  struct Geometry_lib_point_sarek res;
  res.x = x;
  res.y = y;
  return res;
}

/************* FUNCTION PROTOTYPES ******************/
 float distance  ( struct Geometry_lib_point_sarek p1, struct Geometry_lib_point_sarek p2 );
/************* FUNCTION DEFINITIONS ******************/

 float distance  ( struct Geometry_lib_point_sarek p1, struct Geometry_lib_point_sarek p2 ){
float dx ;
    dx = (p1.x - p2.x) ;
    float dy ;
    dy = (p1.y - p2.y) ;
    return sqrt (((dx * dx) + (dy * dy))) ;}
__kernel void spoc_dummy ( __global struct Geometry_lib_point_sarek* a, int sarek_a_length, __global struct Geometry_lib_point_sarek* b, int sarek_b_length, __global float* out, int sarek_out_length, const int n ) {
  bool spoc_prof_cond;
   ;
  int tid ;
  tid = (get_local_id(0) + (get_local_size(0) * get_group_id(0))) ;
  spoc_prof_cond  = (tid < n);
  if ( spoc_prof_cond){
        out[tid] = distance (a[tid], b[tid]) ;;
  }  }