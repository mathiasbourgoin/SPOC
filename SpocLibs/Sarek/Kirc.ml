(******************************************************************************
   - * Mathias Bourgoin, Université Pierre et Marie Curie (2012)
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
 *******************************************************************************)
open Spoc
open Kernel

let debug = true

let profile_default () =
  match Sys.getenv_opt "SAREK_PROFILE" with
  | Some v -> (
      match String.lowercase_ascii v with
      | "1" | "true" | "yes" -> true
      | _ -> false)
  | None -> false

let idkern = ref 0

open Kirc_Ast
module Kirc_OpenCL = Gen.Generator (Kirc_OpenCL)
module Kirc_Cuda = Gen.Generator (Kirc_Cuda)
module Kirc_Profile = Gen.Generator (Profile)

type float64 = float

type float32 = float

type extension = ExFloat32 | ExFloat64

type ('a, 'b, 'c) kirc_kernel = {
  ml_kern : 'a;
  body : Kirc_Ast.k_ext;
  ret_val : Kirc_Ast.k_ext * ('b, 'c) Vector.kind;
  extensions : extension array;
}

type ('a, 'b, 'c, 'd) kirc_function = {
  fun_name : string;
  ml_fun : 'a;
  funbody : Kirc_Ast.k_ext;
  fun_ret : Kirc_Ast.k_ext * ('b, 'c) Vector.kind;
  fastflow_acc : 'd;
  fun_extensions : extension array;
}

type ('a, 'b, 'c, 'd, 'e) sarek_kernel =
  ('a, 'b) spoc_kernel * ('c, 'd, 'e) kirc_kernel

let constructors = ref []

let register_constructor_string s = constructors := s :: !constructors

let opencl_head =
  "#define SAREK_VEC_LENGTH(A) sarek_## A ##_length\n"
  ^ "float spoc_fadd ( float a, float b );\n"
  ^ "float spoc_fminus ( float a, float b );\n"
  ^ "float spoc_fmul ( float a, float b );\n"
  ^ "float spoc_fdiv ( float a, float b );\n" ^ "int logical_and (int, int);\n"
  ^ "int spoc_powint (int, int);\n" ^ "int spoc_xor (int, int);\n"
  ^ "float spoc_fadd ( float a, float b ) { return (a + b);}\n"
  ^ "float spoc_fminus ( float a, float b ) { return (a - b);}\n"
  ^ "float spoc_fmul ( float a, float b ) { return (a * b);}\n"
  ^ "float spoc_fdiv ( float a, float b ) { return (a / b);}\n"
  ^ "int logical_and (int a, int b ) { return (a & b);}\n"
  ^ "int spoc_powint (int a, int b ) { return ((int) pow (((float) a), \
     ((float) b)));}\n" ^ "int spoc_xor (int a, int b ) { return (a^b);}\n"
  ^ "void spoc_barrier ( ) { barrier(CLK_LOCAL_MEM_FENCE);}\n"

let opencl_common_profile =
  "\n/*********** PROFILER FUNCTIONS **************/\n"
  ^ "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n"
  ^ "void spoc_atomic_add(__global ulong *a, ulong b){ atom_add(a, (ulong)b);}\n"
  ^ "\n\
    \     void* memory_analysis(__global ulong *profile_counters, __global \
     void* mp, int store, int load){\n\
    \    if (store) spoc_atomic_add(profile_counters+0, 1ULL);\n\
    \    if (load) spoc_atomic_add(profile_counters+1, 1ULL);\n\
    \    return mp;\n\
     }" ^ "\n"

let opencl_profile_head =
  opencl_common_profile
  ^ "\n\
     void branch_analysis(__global ulong *profile_counters, int eval, int \
     counters){\n\
    \  \n\
    \  unsigned int threadIdxInGroup = \n\
    \                get_local_id(2)*get_local_size(0)*get_local_size(1) + \n\
    \                get_local_id(1)*get_local_size(0) + get_local_id(0);\n\
    \  \n\
    \  \n\
    \  //Get count of 1) active work items  in this workgroup\n\
    \  //2) work items that will take the branch\n\
    \  //3) work items that do not take the branch.\n\
    \  __local ulong numActive[1]; numActive[0] = 0;\n\
    \  __local ulong numTaken[1];  numTaken[0] = 0;\n\
    \  __local ulong numNotTaken[1]; numNotTaken[0] = 0;\n\
    \  __local unsigned int lig[1]; lig[0]  = 0;\n\
    \  barrier(CLK_LOCAL_MEM_FENCE);\n\n\
    \  atomic_inc(numActive);\n\
    \  atom_max(lig, threadIdxInGroup);\n\
    \  \n\
    \  if (eval) atomic_inc(numTaken);\n\
    \  if (!eval) atomic_inc(numNotTaken);\n\
    \  \n\
    \  barrier(CLK_LOCAL_MEM_FENCE);\n\
    \  \n\
    \  // The last active work item in each group gets to write results.\n\
    \  if (lig[0] == threadIdxInGroup) {\n\
    \    spoc_atomic_add(profile_counters+4, (ulong)1); //\n\
    \    spoc_atomic_add(profile_counters+counters+1, numActive[0]);\n\
    \    spoc_atomic_add(profile_counters+counters+2, numTaken[0]);\n\
    \    spoc_atomic_add(profile_counters+counters+3, numNotTaken[0]);\n\
    \    if (numTaken[0] != numActive[0] && numNotTaken[0] != numActive[0]) {\n\
    \      // If threads go different ways, note it.\n\
    \      spoc_atomic_add(profile_counters+5, (ulong)1);\n\
    \    }\n\
    \  }\n\
     }\n"
  ^ "void while_analysis(__global ulong *profile_counters, int eval){\n\
    \  /* unsigned int threadIdxInGroup = \n\
    \                get_local_id(2)*get_local_size(0)*get_local_size(1) + \n\
    \                get_local_id(1)*get_local_size(0) + get_local_id(0);\n\n\
    \  //Get count of 1) active work items  in this workgroup\n\
    \  //2) work items that will take the branch\n\
    \  //3) work items that do not take the branch.\n\
    \  __local ulong numActive[1]; numActive[0] = 0;\n\
    \  __local unsigned int numTaken;  numTaken = 0;\n\
    \  __local unsigned int numNotTaken; numNotTaken = 0;\n\
    \  __local unsigned int lig; lig  = 0;\n\
    \  barrier(CLK_LOCAL_MEM_FENCE);\n\n\
    \  atomic_inc(numActive);\n\
    \  atom_max(&lig, threadIdxInGroup);\n\
    \  \n\
    \  if (eval) atom_inc(&numTaken);\n\
    \  if (!eval) atom_inc(&numNotTaken);\n\
    \  \n\
    \  barrier(CLK_LOCAL_MEM_FENCE);\n\
    \  \n\
    \  // The last active work item in each group gets to write results.\n\
    \  if (lig == threadIdxInGroup) {\n\
    \    spoc_atomic_add(profile_counters+4, (ulong)1); //\n\
    \    if (numTaken != numActive[0] && numNotTaken != numActive[0]) {\n\
    \      // If threads go different ways, note it.\n\
    \      spoc_atomic_add(profile_counters+5, (ulong)1);\n\
    \    }\n\
    \  } */               \n\
     }\n\n"

let opencl_profile_head_cpu =
  opencl_common_profile
  ^ "\n\
     void branch_analysis(__global ulong *profile_counters, int eval, int \
     counters){\n\
    \  \n\
    \  unsigned int threadIdxInGroup = \n\
    \                get_local_id(2)*get_local_size(0)*get_local_size(1) + \n\
    \                get_local_id(1)*get_local_size(0) + get_local_id(0);\n\
    \  \n\
    \  spoc_atomic_add(profile_counters+4, (ul;ong)1); //\n\
    \  spoc_atomic_add(profile_counters+counters+1, 1);\n\
    \  if (eval) \n\
    \    spoc_atomic_add(profile_counters+counters+2, 1);\n\
    \  if (!eval)\n\
    \    spoc_atomic_add(profile_counters+counters+3, 1);\n\
    \ }\n"
  ^ "void while_analysis(__global ulong  *profile_counters, int eval){\n}\n\n"

let opencl_float64 =
  "#ifndef __FLOAT64_EXTENSION__ \n" ^ "#define __FLOAT64_EXTENSION__ \n"
  ^ "#if defined(cl_khr_fp64)  // Khronos extension available?\n"
  ^ "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  ^ "#elif defined(cl_amd_fp64)  // AMD extension available?\n"
  ^ "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n" ^ "#endif\n"
  ^ "double spoc_dadd ( double a, double b );\n"
  ^ "double spoc_dminus ( double a, double b );\n"
  ^ "double spoc_dmul ( double a, double b );\n"
  ^ "double spoc_ddiv ( double a, double b );\n"
  ^ "double spoc_dadd ( double a, double b ) { return (a + b);}\n"
  ^ "double spoc_dminus ( double a, double b ) { return (a - b);}\n"
  ^ "double spoc_dmul ( double a, double b ) { return (a * b);}\n"
  ^ "double spoc_ddiv ( double a, double b ) { return (a / b);}\n" ^ "#endif\n"

let cuda_float64 =
  "#ifndef __FLOAT64_EXTENSION__ \n" ^ "#define __FLOAT64_EXTENSION__ \n"
  ^ "__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"
  ^ "__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"
  ^ "__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"
  ^ "__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"
  ^ "#endif\n"

let cuda_head =
  "#define SAREK_VEC_LENGTH(a) sarek_ ## a ## _length\n"
  ^ "#define FULL_MASK 0xffffffff\n"
  ^ "__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"
  ^ "__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"
  ^ "__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"
  ^ "__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"
  ^ "__device__ int logical_and (int a, int b ) { return (a & b);}\n"
  ^ "__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) \
     a), ((double) b)));}\n"
  ^ "__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"

let cuda_profile_head =
  "\n/*********** PROFILER FUNCTIONS **************/\n"
  ^ "__device__ void spoc_atomic_add(unsigned long long int *a, unsigned int \
     b){ atomicAdd(a, b);}\n"
  ^ "__device__ __forceinline__ unsigned int get_laneid(void) {\n\
    \    unsigned int laneid;\n\
    \    asm volatile (\"mov.u32 %0, %laneid;\" : \"=r\"(laneid));\n\
    \    return laneid;\n\
    \  }\n"
  ^ "__device__ void branch_analysis(unsigned long long int *profile_counters, \
     int eval, int counters){\n\
    \  \n\
    \  int threadIdxInWarp =  get_laneid();//threadIdx.x & (warpSize-1);\n\
    \  \n\
    \  //Get count of 1) active threads  in this warp\n\
    \  //2) threads that will take the branch\n\
    \  //3) threads that do not take the branch.\n\
    \  unsigned int active = __ballot(1);\n\
    \  int taken = __ballot(eval);\n\
    \  int ntaken = __ballot(!eval);\n\
    \  int numActive = __popc(active);\n\
    \  int numTaken = __popc(taken), numNotTaken = __popc(ntaken);\n\n\
    \  // The first active thread in each warp gets to write results.\n\
    \  if ((__ffs(active)-1) == threadIdxInWarp) {\n\
    \    atomicAdd(profile_counters+4, 1ULL); //\n\
    \    atomicAdd(profile_counters+counters+1, numActive);\n\
    \    atomicAdd(profile_counters+counters+2, numTaken);\n\
    \    atomicAdd(profile_counters+counters+3, numNotTaken);\n\
    \    if (numTaken != numActive && numNotTaken != numActive) {\n\
    \      // If threads go different ways, note it.\n\
    \      atomicAdd(profile_counters+5, 1ULL);\n\
    \    }\n\
    \  }\n\
     }\n"
  ^ "__device__ void while_analysis(unsigned long long int *profile_counters, \
     int eval){\n\n\
    \  int threadIdxInWarp =  get_laneid();//threadIdx.x & (warpSize-1);\n\n\
    \  //Get count of 1) active threads  in this \
     warp                                  //2) threads that will take the \
     branch                                          //3) threads that do not \
     take the \
     branch.                                                                   \n\
    \  int active = __ballot(1);\n\
    \  int taken = __ballot(eval);\n\
    \  int ntaken = __ballot(!eval);\n\
    \  int numActive = __popc(active);\n\
    \  int numTaken = __popc(taken), numNotTaken = __popc(ntaken);\n\n\
    \  // The first active thread in each warp gets to write \
     results.                                              \n\
    \  if ((__ffs(active)-1) == threadIdxInWarp) {\n\
    \    atomicAdd(profile_counters+4, 1ULL); \
     //                                                                   \n\
    \    if (numTaken != numActive && numNotTaken != numActive) {\n\
    \      // If threads go different ways, note \
     it.                                                               \n\
    \      atomicAdd(profile_counters+5, 1ULL);\n\
    \    }\n\
    \  }\n\
     }\n\n"
  ^ "\n\
     #include<cuda.h>\n\
     template <typename T>\n\
     __device__ T __broadcast(T t, int fromWhom)\n\
     {\n\
    \  union {\n\
    \    int32_t shflVals[sizeof(T)];\n\
    \    T t;\n\
    \  } p;\n\
    \  \n\
    \  p.t = t;\n\
    \    #pragma unroll\n\
    \  for (int i = 0; i < sizeof(T); i++) {\n\
    \    int32_t shfl = (int32_t)p.shflVals[i];\n\
    \    p.shflVals[i] = __shfl(shfl, fromWhom);\n\
    \  }\n\
    \  return p.t;\n\
     }\n\n\
     /// The number of bits we need to shift off to get the cache line address.\n\
     #define LINE_BITS   5\n\n\
     template<typename T>\n\
     __device__ T* memory_analysis(unsigned long long int *profile_counters, \
     T* mp, int store, int load){\n\n\
    \  int threadIdxInWarp =  get_laneid();//threadIdx.x & (warpSize-1);\n\
    \  intptr_t addrAsInt = (intptr_t) mp;\n\n\
    \  if (__isGlobal(mp)){\n\
    \    if (store) atomicAdd(profile_counters+0, 1ULL);\n\
    \    if (load) atomicAdd(profile_counters+1, 1ULL);\n\n\
    \    unsigned unique = 0; // Num unique lines per warp.\n\n\
    \    // Shift off the offset bits into the cache line.\n\
    \    intptr_t lineAddr = addrAsInt >> LINE_BITS;\n\n\
    \    int workset = __ballot(1);\n\
    \    int firstActive = __ffs(workset)-1;\n\
    \    int numActive = __popc(workset);\n\
    \    while (workset) {\n\
    \      // Elect a leader, get its cache line, see who matches it.\n\
    \      int leader = __ffs(workset) - 1;\n\
    \      intptr_t leadersAddr = __broadcast(lineAddr, leader);\n\
    \      int notMatchesLeader = __ballot(leadersAddr != lineAddr);\n\n\
    \      // We have accounted for all values that match the leader’s.\n\
    \      // Let’s remove them all from the workset.\n\
    \      workset = workset & notMatchesLeader;\n\
    \      unique++;\n\
    \    }\n\n\
    \    if (firstActive == threadIdxInWarp && unique ) {\n\
    \      atomicAdd(profile_counters+6, 1ULL);\n\
    \    }\n\
    \    \n\
    \  }\n\
    \  return mp;\n\n\
     }\n" ^ "\n"

let eint32 = EInt32

let eint64 = EInt64

let efloat32 = EFloat32

let efloat64 = EFloat64

let global = Global

let local = LocalSpace

let shared = Shared

let new_var i = IdName ("spoc_var" ^ string_of_int i)

let new_array n l t m = Arr (n, l, t, m)

let var i s = IntId (s, i)

(*("spoc_var"^(string_of_int i)), i)*)
let spoc_gen_kernel args body = Kern (args, body)

let spoc_fun_kernel _a _b = ()

let global_fun a =
  GlobalFun
    ( a.funbody,
      (match snd a.fun_ret with
      | Vector.Int32 _ -> "int"
      | Vector.Float32 _ -> "float"
      | Vector.Custom _ -> (
          match fst a.fun_ret with
          | CustomVar (s, _, _) -> "struct " ^ s ^ "_sarek"
          | _ -> assert false)
      | _ -> "void"),
      a.fun_name )

let seq a b = Seq (a, b)

let app a b = App (a, b)

let spoc_unit () = Unit

let spoc_int a = Int a

let global_int_var a = GInt a

let global_float_var a = GFloat a

let global_float64_var a = GFloat64 a

let spoc_int32 a = Int (Int32.to_int a)

let spoc_float f = Float f

let spoc_double d = Double d

let spoc_int_id a = Int a

(*IntId (a,-1)*)
let spoc_float_id a = Float a

let spoc_plus a b = Plus (a, b)

let spoc_plus_float a b = Plusf (a, b)

let spoc_min a b = Min (a, b)

let spoc_min_float a b = Minf (a, b)

let spoc_mul a b = Mul (a, b)

let spoc_mul_float a b = Mulf (a, b)

let spoc_div a b = Div (a, b)

let spoc_div_float a b = Divf (a, b)

let spoc_mod a b = Mod (a, b)

let spoc_ife a b c = Ife (a, b, c)

let spoc_if a b = If (a, b)

let spoc_match s e l = Match (s, e, l)

let spoc_case i o e : case = (i, o, e)

let spoc_do a b c d = DoLoop (a, b, c, d)

let spoc_while a b = While (a, b)

let params l = Params l

let spoc_id _i = Id ""

let spoc_constr t c params = Constr (t, c, params)

let spoc_record t params = Record (t, params)

let spoc_return k = Return k

let concat a b = Concat (a, b)

let empty_arg () = Empty

let new_int_var ?(mutable_ = true) i s = IntVar (i, s, mutable_)

let new_float_var ?(mutable_ = true) i s = FloatVar (i, s, mutable_)

let new_float64_var ?(mutable_ = true) i s = DoubleVar (i, s, mutable_)

let new_double_var ?(mutable_ = true) i s = DoubleVar (i, s, mutable_)

let new_unit_var ?(mutable_ = true) i s = UnitVar (i, s, mutable_)

let new_custom_var n v s = Custom (n, v, s)

(* <--- *)

let new_int_vec_var v s = VecVar (Int 0, v, s)

let new_float_vec_var v s = VecVar (Float 0., v, s)

let new_double_vec_var v s = VecVar (Double 0., v, s)

let new_custom_vec_var n v s = VecVar (Custom (n, 0, s), v, s)

(* <--- *)

let int_vect i = IntVect i

let spoc_rec_get r id = RecGet (r, id)

let spoc_rec_set r v = RecSet (r, v)

let set_vect_var vecacc value = SetV (vecacc, value)

let set_arr_var arracc value = SetV (arracc, value)

let intrinsics a b = Intrinsics (a, b)

let spoc_local_env local_var b = Local (local_var, b)

let spoc_set name value = Set (name, value)

let spoc_declare name = Decl name

let spoc_local_var a = a

let spoc_acc a b = Acc (a, b)

let int_var i = i

let int32_var i = i

let float_var f = f

let double_var d = CastDoubleVar (d, "")

let equals a b = EqBool (a, b)

let equals_custom s v1 v2 = EqCustom (s, v1, v2)

let equals32 a b = EqBool (a, b)

let equals64 a b = EqBool (a, b)

let equalsF a b = EqBool (a, b)

let equalsF64 a b = EqBool (a, b)

let b_or a b = Or (a, b)

let b_and a b = And (a, b)

let b_not a = Not a

let lt a b = LtBool (a, b)

let lt32 a b = LtBool (a, b)

let lt64 a b = LtBool (a, b)

let ltF a b = LtBool (a, b)

let ltF64 a b = LtBool (a, b)

let gt a b = GtBool (a, b)

let gt32 a b = GtBool (a, b)

let gt64 a b = GtBool (a, b)

let gtF a b = GtBool (a, b)

let gtF64 a b = GtBool (a, b)

let lte a b = LtEBool (a, b)

let lte32 a b = LtEBool (a, b)

let lte64 a b = LtEBool (a, b)

let lteF a b = LtEBool (a, b)

let lteF64 a b = LtEBool (a, b)

let gte a b = GtEBool (a, b)

let gte32 a b = GtEBool (a, b)

let gte64 a b = GtEBool (a, b)

let gteF a b = GtEBool (a, b)

let gteF64 a b = GtEBool (a, b)

let get_vec a b = IntVecAcc (a, b)

let get_arr a b = IntVecAcc (a, b)

let return_unit () = Unit

let return_int i s = IntVar (i, s, true)

let return_float f s = FloatVar (f, s, true)

let return_double d s = DoubleVar (d, s, true)

let return_bool b s = BoolVar (b, s, true)

let return_custom n sn s = CustomVar (n, sn, s)

let spoc_native f = Native f

let pragma l e = Pragma (l, e)

let map f a b = Map (f, a, b)

let print_ast = Kirc_Ast.print_ast

let debug_print (ker : ('a, 'b, 'c, 'd, 'e) sarek_kernel) =
  let _, k = ker in
  let _k1, k2, _k3 = (k.ml_kern, k.body, k.ret_val) in
  print_ast k2

let rewrite ker =
  let b = ref false in
  let rec aux kern =
    match kern with
    | Native _ -> kern
    | Pragma (opts, k) -> Pragma (opts, aux k)
    | Block b -> Block (aux b)
    | Kern (k1, k2) -> Kern (aux k1, aux k2)
    | Params k -> Params (aux k)
    | Plus (k1, k2) -> Plus (aux k1, aux k2)
    | Plusf (k1, k2) -> (
        match (k1, k2) with
        | Float f1, Float f2 ->
            b := true ;
            Float (f1 +. f2)
        | _ -> Plusf (aux k1, aux k2))
    | Min (k1, k2) -> Min (aux k1, aux k2)
    | Minf (k1, k2) -> (
        match (k1, k2) with
        | Float f1, Float f2 ->
            b := true ;
            Float (f1 +. f2)
        | _ -> Minf (aux k1, aux k2))
    | Mul (k1, k2) -> Mul (aux k1, aux k2)
    | Mulf (k1, k2) -> (
        match (k1, k2) with
        | Float f1, Float f2 ->
            b := true ;
            Float (f1 +. f2)
        | _ -> Mulf (aux k1, aux k2))
    | Div (k1, k2) -> Div (aux k1, aux k2)
    | Divf (k1, k2) -> (
        match (k1, k2) with
        | Float f1, Float f2 ->
            b := true ;
            Float (f1 +. f2)
        | _ -> Divf (aux k1, aux k2))
    | Mod (k1, k2) -> Mod (aux k1, aux k2)
    | Id _ -> kern
    | IdName _ -> kern
    | IntVar _ -> kern
    | FloatVar _ -> kern
    | UnitVar _ -> Seq (kern, kern)
    | CastDoubleVar _ -> kern
    | DoubleVar _ -> kern
    | BoolVar _ -> kern
    | VecVar (k, idx, s) -> VecVar (aux k, idx, s)
    | Concat (k1, k2) -> Concat (aux k1, aux k2)
    | Constr (t, c, l) -> Constr (t, c, List.map aux l)
    | Record (t, l) -> Record (t, List.map aux l)
    | RecGet (r, s) -> RecGet (aux r, s)
    | RecSet (r, v) -> RecSet (aux r, aux v)
    | Empty -> kern
    | Seq (k1, Unit) -> aux k1
    | Seq (k1, k2) -> Seq (aux k1, aux k2)
    | Return k -> (
        match k with
        | Return k ->
            b := true ;
            aux (Return k)
        | Acc _ | Set _ -> aux k
        | Ife (k1, k2, k3) ->
            b := true ;
            Ife (aux k1, aux (Return k2), aux (Return k3))
        | If (k1, k2) ->
            b := true ;
            If (aux k1, aux (Return k2))
        | DoLoop (k1, k2, k3, k4) ->
            b := true ;
            DoLoop (aux k1, aux k2, aux k3, aux (Return k4))
        | While (k1, k2) ->
            b := true ;
            While (aux k1, aux (Return k2))
        | Seq (k1, k2) ->
            b := true ;
            Seq (aux k1, aux (Return k2))
        | Match (s, a, bb) ->
            b := true ;
            Match
              ( s,
                aux a,
                Array.map (fun (i, ofid, e) -> (i, ofid, aux (Return e))) bb )
        | _ -> Return (aux k))
    | Acc (k1, k2) -> (
        match k2 with
        | Ife (k1', k2', k3') ->
            b := true ;
            Ife (aux k1', aux (Acc (k1, k2')), aux (Acc (k1, k3')))
        | If (k1', k2') ->
            b := true ;
            If (aux k1', aux (Acc (k1, k2')))
        | DoLoop (k1', k2', k3', k4') ->
            b := true ;
            DoLoop (aux k1', aux k2', aux k3', aux (Acc (k1, k4')))
        | While (k1', k2') ->
            b := true ;
            While (aux k1', aux (Acc (k1, k2')))
        | Seq (k1', k2') ->
            b := true ;
            Seq (aux k1', aux (Acc (k1, k2')))
        | Match (s, a, bb) ->
            b := true ;
            Match
              ( s,
                aux a,
                Array.map (fun (i, ofid, e) -> (i, ofid, aux (Acc (k1, e)))) bb
              )
        | Return _ -> assert false
        | _ -> Acc (aux k1, aux k2))
    | Set (k1, k2) -> aux (Acc (k1, k2))
    | Decl k1 -> aux k1
    | SetV (k1, k2) -> (
        match k2 with
        | Seq (k3, k4) -> Seq (k3, SetV (aux k1, aux k4))
        | Ife (k3, k4, k5) ->
            b := true ;
            Ife (aux k3, SetV (aux k1, aux k4), SetV (aux k1, k5))
        | Match (s, a, bb) ->
            b := true ;
            Match
              ( s,
                aux a,
                Array.map
                  (fun (i, ofid, e) -> (i, ofid, SetV (aux k1, aux e)))
                  bb )
        | _ -> SetV (aux k1, aux k2))
    | SetLocalVar (k1, k2, k3) -> SetLocalVar (aux k1, aux k2, aux k3)
    | Intrinsics _ -> kern
    | IntId _ -> kern
    | Int _ -> kern
    | GInt _ -> kern
    | GFloat _ -> kern
    | GFloat64 _ -> kern
    | Float _ -> kern
    | Double _ -> kern
    | Custom _ -> kern
    | IntVecAcc (k1, k2) -> (
        match k2 with
        | Seq (k3, k4) -> Seq (k3, IntVecAcc (aux k1, aux k4))
        | _ -> IntVecAcc (aux k1, aux k2))
    | Local (k1, k2) -> Local (aux k1, aux k2)
    | Ife (k1, k2, k3) -> Ife (aux k1, aux k2, aux k3)
    | If (k1, k2) -> If (aux k1, aux k2)
    | Not k -> Not (aux k)
    | Or (k1, k2) -> Or (aux k1, aux k2)
    | And (k1, k2) -> And (aux k1, aux k2)
    | EqBool (k1, k2) -> EqBool (aux k1, aux k2)
    | EqCustom (n, k1, k2) -> EqCustom (n, aux k1, aux k2)
    | LtBool (k1, k2) -> LtBool (aux k1, aux k2)
    | GtBool (k1, k2) -> GtBool (aux k1, aux k2)
    | LtEBool (k1, k2) -> LtEBool (aux k1, aux k2)
    | GtEBool (k1, k2) -> GtEBool (aux k1, aux k2)
    | DoLoop (k1, k2, k3, k4) -> DoLoop (aux k1, aux k2, aux k3, aux k4)
    | Arr (l, t, s, m) -> Arr (l, t, s, m)
    | While (k1, k2) -> While (aux k1, aux k2)
    | App (a, b) -> App (aux a, Array.map aux b)
    | GlobalFun (a, b, n) -> GlobalFun (aux a, b, n)
    | Unit -> kern
    | Match (s, a, b) ->
        Match (s, aux a, Array.map (fun (i, ofid, e) -> (i, ofid, aux e)) b)
    | CustomVar _ -> kern
    | Map (a, b, c) -> Map (aux a, aux b, aux c)
  in
  let kern = ref (aux ker) in
  while !b do
    b := false ;
    kern := aux !kern
  done ;
  !kern

let return_v = ref ("", "")

let save file string =
  ignore (Sys.command ("rm -f " ^ file)) ;
  let channel = open_out file in
  output_string channel string ;
  close_out channel

let load_file f =
  let ic = open_in f in
  let n = in_channel_length ic in
  let s = Bytes.make n ' ' in
  really_input ic s 0 n ;
  close_in ic ;
  s

let opencl_source ?profile:(prof = profile_default ()) ?return:(r = false)
    (ker : ('a, 'b, 'c, 'd, 'e) sarek_kernel) dev =
  let kir, k = ker in
  let k1, k2, k3 = (k.ml_kern, k.body, k.ret_val) in
  return_v := ("", "") ;
  let k' =
    ( Kirc_Cuda.parse ~profile:prof 0 (fst k3) dev,
      match fst k3 with
      | IntVar (_i, s, _) | FloatVar (_i, s, _) | DoubleVar (_i, s, _) ->
          s ^ " = "
      | Unit -> ""
      | SetV _ -> ""
      | IntVecAcc _ -> ""
      | VecVar _ -> ""
      | _ ->
          debug_print
            ( kir,
              {
                ml_kern = k1;
                body = fst k3;
                ret_val = k3;
                extensions = k.extensions;
              } ) ;
          Stdlib.flush stdout ;
          assert false )
  in
  if r then (
    Kirc_Cuda.return_v := k' ;
    Kirc_OpenCL.return_v := k') ;
  ignore (Kirc_OpenCL.get_profile_counter ()) ;
  let opencl_head =
    Array.fold_left
      (fun header extension ->
        match extension with
        | ExFloat32 -> header
        | ExFloat64 -> opencl_float64 ^ header)
      opencl_head
      k.extensions
  in
  let src = Kirc_OpenCL.parse ~profile:prof 0 (rewrite k2) dev in
  let global_funs =
    ref "/************* FUNCTION DEFINITIONS ******************/\n"
  in
  Hashtbl.iter
    (fun _ a -> global_funs := !global_funs ^ "\n" ^ fst a ^ "\n")
    Kirc_OpenCL.global_funs ;
  let constructors =
    "/************* CUSTOM TYPES ******************/\n"
    ^ List.fold_left (fun a b -> b ^ a) "\n\n" !constructors
  in
  let protos =
    "/************* FUNCTION PROTOTYPES ******************/\n"
    ^ List.fold_left (fun a b -> b ^ ";\n" ^ a) "" !Kirc_OpenCL.protos
  in
  (if prof then "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
   else "")
  ^ opencl_head
  ^ (if prof then
       match dev.Devices.specific_info with
       | Devices.OpenCLInfo
           {Devices.device_type = Devices.CL_DEVICE_TYPE_CPU; _} ->
           opencl_profile_head_cpu
       | _ -> opencl_profile_head
     else "")
  ^ constructors ^ protos ^ !global_funs ^ src

(*external print_source : string -> unit = "kernel_source"*)

let gen_profile ker dev =
  let _kir, k = ker in
  let _k1, _k2, _k3 = (k.ml_kern, k.body, k.ret_val) in
  return_v := ("", "") ;
  (* let k' =
   *   ( Kirc_Profile.parse 0 (fst k3) dev
   *   , match fst k3 with
   *     | IntVar (i, s) | FloatVar (i, s) | DoubleVar (i, s) ->
   *         s (\*"sspoc_var"^(string_of_int i)^*\) ^ " = "
   *     | Unit -> ""
   *     | SetV _ -> ""
   *     | IntVecAcc _ -> ""
   *     | VecVar _ -> ""
   *     | _ ->
   *         debug_print
   *           ( kir
   *           , { ml_kern= k1
   *             ; body= fst k3
   *             ; ret_val= k3
   *             ; extensions= k.extensions } ) ;
   *         Stdlib.flush stdout ;
   *         assert false ) *)
  (* in *)
  let profile_source = Kirc_Profile.parse 0 _k2 dev in
  Printf.printf "%s" profile_source

(* external from SPOC*)
external nvrtc_ptx : string -> string array -> string = "spoc_nvrtc_ptx"

let gen ?keep_temp:(kt = false) ?profile:(prof = profile_default ()) ?return:(r = false)
    ?only:o ?nvrtc_options:(nvopt = [||])
    (ker : ('a, 'b, 'c, 'd, 'e) sarek_kernel) dev =
  let kir, k = ker in
  let k1, k2, k3 = (k.ml_kern, k.body, k.ret_val) in
  return_v := ("", "") ;
  let k' =
    ( Kirc_Cuda.parse ~profile:prof 0 (fst k3) dev,
      match fst k3 with
      | IntVar (_i, s, _) | FloatVar (_i, s, _) | DoubleVar (_i, s, _) ->
          s (*"sspoc_var"^(string_of_int i)^*) ^ " = "
      | Unit -> ""
      | SetV _ -> ""
      | IntVecAcc _ -> ""
      | VecVar _ -> ""
      | _ ->
          debug_print
            ( kir,
              {
                ml_kern = k1;
                body = fst k3;
                ret_val = k3;
                extensions = k.extensions;
              } ) ;
          Stdlib.flush stdout ;
          assert false )
  in
  if r then (
    Kirc_Cuda.return_v := k' ;
    Kirc_OpenCL.return_v := k') ;
  let gen_cuda ?opts:(_s = "") () =
    let cuda_head =
      Array.fold_left
        (fun header extension ->
          match extension with
          | ExFloat32 -> header
          | ExFloat64 -> cuda_float64 ^ header)
        cuda_head
        k.extensions
    in
    let src = Kirc_Cuda.parse ~profile:prof 0 (rewrite k2) dev in
    let global_funs = ref "" in
    Hashtbl.iter
      (fun _ a -> global_funs := !global_funs ^ fst a ^ "\n")
      Kirc_Cuda.global_funs ;
    let i = ref 0 in
    let constructors =
      List.fold_left
        (fun a b ->
          incr i ;
          (if !i mod 3 = 0 then " " else "__device__ ") ^ b ^ a)
        "\n\n"
        !constructors
    in
    let protos =
      "/************* FUNCTION PROTOTYPES ******************/\n"
      ^ List.fold_left (fun a b -> b ^ ";\n" ^ a) "" !Kirc_Cuda.protos
    in
    if debug then
      save
        ("kirc_kernel" ^ string_of_int !idkern ^ ".cu")
        (cuda_head
        ^ (if prof then cuda_profile_head else "")
        ^ constructors ^ protos ^ !global_funs ^ src) ;
    (*ignore(Sys.command ("nvcc -g -G "^ s ^" "^"-arch=sm_30 -m64  -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"));*)
    let genopt =
      match dev.Devices.specific_info with
      | Devices.CudaInfo cu ->
          let computecap = (cu.Devices.major * 10) + cu.Devices.minor in
          [|
            (if computecap < 35 then failwith "CUDA device too old for this XXX"
             else if computecap < 35 then "--gpu-architecture=compute_30"
             else if computecap < 50 then "--gpu-architecture=compute_35"
             else if computecap < 52 then "--gpu-architecture=compute_50"
             else if computecap < 53 then "--gpu-architecture=compute_52"
             else if computecap < 60 then "--gpu-architecture=compute_53"
             else if computecap < 61 then "--gpu-architecture=compute_60"
             else if computecap < 62 then "--gpu-architecture=compute_61"
             else if computecap < 70 then "--gpu-architecture=compute_30"
             else if computecap < 72 then "--gpu-architecture=compute_70"
             else if computecap < 75 then "--gpu-architecture=compute_72"
             else if computecap = 75 then "--gpu-architecture=compute_75"
             else if computecap = 80 then "--gpu-architecture=compute_80"
             else if computecap = 86 then "--gpu-architecture=compute_86"
             else "--gpu-architecture=compute_35");
          |]
      | _ -> [||]
    in
    let nvrtc_options = Array.append nvopt genopt in
    let s =
      nvrtc_ptx
        (cuda_head
        ^ (if prof then cuda_profile_head else "")
        ^ constructors ^ !global_funs ^ src)
        nvrtc_options
    in
    save ("kirc_kernel" ^ string_of_int !idkern ^ ".ptx") s ;
    (*let s = (load_file "kirc_kernel.ptx") in*)
    kir#set_cuda_sources s ;
    if not kt then
      ignore
        (Sys.command
           ("rm kirc_kernel" ^ string_of_int !idkern ^ ".cu kirc_kernel"
          ^ string_of_int !idkern ^ ".ptx")) ;
    incr idkern
  and gen_opencl () =
    let opencl_head =
      Array.fold_left
        (fun header extension ->
          match extension with
          | ExFloat32 -> header
          | ExFloat64 -> opencl_float64 ^ header)
        opencl_head
        k.extensions
    in
    let src = Kirc_OpenCL.parse ~profile:prof 0 (rewrite k2) dev in
    let global_funs =
      ref "/************* FUNCTION DEFINITIONS ******************/\n"
    in
    Hashtbl.iter
      (fun _ a -> global_funs := !global_funs ^ "\n" ^ fst a ^ "\n")
      Kirc_OpenCL.global_funs ;
    let constructors =
      "/************* CUSTOM TYPES ******************/\n"
      ^ List.fold_left (fun a b -> b ^ a) "\n\n" !constructors
    in
    let protos =
      "/************* FUNCTION PROTOTYPES ******************/\n"
      ^ List.fold_left (fun a b -> b ^ ";\n" ^ a) "" !Kirc_OpenCL.protos
    in
    let clkernel =
      (if prof then
         "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
       else "")
      ^ opencl_head
      ^ (if prof then
           match dev.Devices.specific_info with
           | Devices.OpenCLInfo
               {Devices.device_type = Devices.CL_DEVICE_TYPE_CPU; _} ->
               opencl_profile_head_cpu
           | _ -> opencl_profile_head
         else "")
      ^ constructors ^ protos ^ !global_funs ^ src
    in
    save ("kirc_kernel" ^ string_of_int !idkern ^ ".cl") clkernel ;
    kir#set_opencl_sources clkernel ;
    if not kt then
      ignore (Sys.command ("rm kirc_kernel" ^ string_of_int !idkern ^ ".cl"))
  in
  (match o with
  | None -> (
      match dev.Devices.specific_info with
      | Devices.OpenCLInfo _ ->
          ignore (Kirc_OpenCL.get_profile_counter ()) ;
          gen_opencl ()
      | _ ->
          ignore (Kirc_OpenCL.get_profile_counter ()) ;
          gen_cuda ())
  | Some d -> (
      match d with
      | Devices.Both ->
          ignore (Kirc_Cuda.get_profile_counter ()) ;
          gen_cuda () ;
          ignore (Kirc_OpenCL.get_profile_counter ()) ;
          gen_opencl ()
      | Devices.Cuda ->
          ignore (Kirc_Cuda.get_profile_counter ()) ;
          gen_cuda ()
      | Devices.OpenCL ->
          ignore (Kirc_OpenCL.get_profile_counter ()) ;
          gen_opencl ())) ;
  kir#reset_binaries () ;
  ignore (kir#compile dev) ;
  (kir, k)

let arg_of_vec v =
  match Vector.kind v with
  | Vector.Int32 _ -> Kernel.VInt32 v
  | Vector.Float32 _ -> Kernel.VFloat32 v
  | Vector.Int64 _ -> Kernel.VInt64 v
  | _ -> assert false

let run ?recompile:(r = false) (ker : ('a, 'b, 'c, 'd, 'e) sarek_kernel) a
    (block, grid) _q dev =
  let kir, k = ker in
  (match dev.Devices.specific_info with
  | Devices.CudaInfo _ -> (
      if r then ignore (gen ~only:Devices.Cuda (kir, k) dev)
      else
        match kir#get_cuda_sources () with
        | [] -> ignore (gen ~only:Devices.Cuda (kir, k) dev)
        | _ -> ())
  | Devices.OpenCLInfo _ -> (
      if r then ignore (gen ~only:Devices.OpenCL (kir, k) dev)
      else
        match kir#get_opencl_sources () with
        | [] -> ignore (gen ~only:Devices.OpenCL (kir, k) dev)
        | _ -> ())) ;
  let args = kir#args_to_list a in
  let offset = ref 0 in
  kir#compile ~debug:true dev ;
  let bin = Hashtbl.find (kir#get_binaries ()) dev in
  let nvec = ref 0 in
  Array.iter
    (fun a ->
      match a with
      | VChar _v
      | VFloat32 _v
      | VFloat64 _v
      | VInt32 _v
      | VInt64 _v
      | VComplex32 _v
      | VCustom _v ->
          incr nvec
      | _ -> ())
    args ;
  match dev.Devices.specific_info with
  | Devices.CudaInfo _cI ->
      let extra = Kernel.Cuda.cuda_create_extra (Array.length args + !nvec) in
      (*Kernel.Cuda.cuda_load_arg offset extra dev bin 0 (arg_of_vec profiler_counters);*)
      let idx = ref 0 in
      Array.iter
        (fun a ->
          match a with
          | VChar v
          | VFloat32 v
          | VFloat64 v
          | VInt32 v
          | VInt64 v
          | VComplex32 v
          | VCustom v ->
              Kernel.Cuda.cuda_load_arg offset extra dev bin !idx a ;
              Kernel.Cuda.cuda_load_arg
                offset
                extra
                dev
                bin
                (!idx + 1)
                (Kernel.Int32 (Vector.length v)) ;
              idx := !idx + 2
          | _ ->
              Kernel.Cuda.cuda_load_arg offset extra dev bin idx a ;
              incr idx)
        args ;
      Kernel.Cuda.cuda_launch_grid
        offset
        bin
        grid
        block
        extra
        dev.Devices.general_info
        0
  | Devices.OpenCLInfo _ ->
      (*Kernel.OpenCL.opencl_load_arg offset dev bin 0 (arg_of_vec profiler_counters);*)
      let idx = ref 0 in
      Array.iter
        (fun a ->
          match a with
          | VChar v
          | VFloat32 v
          | VFloat64 v
          | VInt32 v
          | VInt64 v
          | VComplex32 v
          | VCustom v ->
              Kernel.OpenCL.opencl_load_arg offset dev bin !idx a ;
              Kernel.OpenCL.opencl_load_arg
                offset
                dev
                bin
                (!idx + 1)
                (Kernel.Int32 (Vector.length v)) ;
              idx := !idx + 2
          | _ ->
              Kernel.OpenCL.opencl_load_arg offset dev bin !idx a ;
              incr idx)
        args ;
      (*Array.iteri (fun i a -> Kernel.OpenCL.opencl_load_arg offset dev bin (i) a) args;*)
      Kernel.OpenCL.opencl_launch_grid bin grid block dev.Devices.general_info 0

let profile_run ?recompile:(r = true) (ker : ('a, 'b, 'c, 'd, 'e) sarek_kernel)
    a b _q dev =
  let kir, k = ker in
  (match dev.Devices.specific_info with
  | Devices.CudaInfo _ -> (
      if r then ignore (gen ~profile:true ~only:Devices.Cuda (kir, k) dev)
      else
        match kir#get_cuda_sources () with
        | [] -> ignore (gen ~profile:true ~only:Devices.Cuda (kir, k) dev)
        | _ -> ())
  | Devices.OpenCLInfo _ -> (
      if r then ignore (gen ~profile:true ~only:Devices.OpenCL (kir, k) dev)
      else
        match kir#get_opencl_sources () with
        | [] -> ignore (gen ~profile:true ~only:Devices.OpenCL (kir, k) dev)
        | _ -> ())) ;
  (*kir#run a b q dev;*)
  let nCounter =
    !(match dev.Devices.specific_info with
     | Devices.CudaInfo _ -> Kirc_Cuda.profiler_counter
     | Devices.OpenCLInfo _ -> Kirc_OpenCL.profiler_counter)
  in
  (*Printf.printf "Number of counters : %d\n%!" nCounter;*)
  let profiler_counters = Vector.create Vector.int64 nCounter in
  for i = 0 to nCounter - 1 do
    Mem.set profiler_counters i 0L
  done ;
  (let args = kir#args_to_list a in
   let offset = ref 0 in
   kir#compile ~debug:true dev ;
   let block, grid = b in
   let bin = Hashtbl.find (kir#get_binaries ()) dev in
   match dev.Devices.specific_info with
   | Devices.CudaInfo _cI ->
       let extra = Kernel.Cuda.cuda_create_extra (Array.length args + 1) in
       Kernel.Cuda.cuda_load_arg
         offset
         extra
         dev
         bin
         0
         (arg_of_vec profiler_counters) ;
       Array.iteri
         (fun i a ->
           match a with
           | VChar _ | VFloat32 _ | VFloat64 _ | VInt32 _ | VInt64 _
           | VComplex32 _ | VCustom _ ->
               Kernel.Cuda.cuda_load_arg offset extra dev bin i a
           | _ -> Kernel.Cuda.cuda_load_arg offset extra dev bin i a)
         args ;
       Kernel.Cuda.cuda_launch_grid
         offset
         bin
         grid
         block
         extra
         dev.Devices.general_info
         0
   | Devices.OpenCLInfo _ ->
       Kernel.OpenCL.opencl_load_arg
         offset
         dev
         bin
         0
         (arg_of_vec profiler_counters) ;
       Array.iteri
         (fun i a -> Kernel.OpenCL.opencl_load_arg offset dev bin i a)
         args ;
       Kernel.OpenCL.opencl_launch_grid
         bin
         grid
         block
         dev.Devices.general_info
         0) ;
  Devices.flush dev () ;
  if not !Mem.auto then Mem.to_cpu profiler_counters () ;
  (*Spoc.Tools.iter (fun a -> Printf.printf "%Ld " a) profiler_counters;*)
  Gen.profile_vect := profiler_counters ;
  gen_profile ker dev

let compile_kernel_to_files s (ker : ('a, 'b, 'c, 'd, 'e) sarek_kernel) dev =
  let kir, k = ker in
  let k1, k2, k3 = (k.ml_kern, k.body, k.ret_val) in
  return_v := ("", "") ;
  let k' =
    ( (Kirc_Cuda.parse 0 (fst k3)) dev,
      match fst k3 with
      | IntVar (_i, s, _) | FloatVar (_i, s, _) | DoubleVar (_i, s, _) ->
          s
          ^
          (*"spoc_var"^(string_of_int i)^*)
          " = "
      | Unit -> ""
      | SetV _ -> ""
      | IntVecAcc _ -> ""
      | VecVar _ -> ""
      | _ ->
          debug_print
            ( kir,
              {
                ml_kern = k1;
                body = fst k3;
                ret_val = k3;
                extensions = k.extensions;
              } ) ;
          Stdlib.flush stdout ;
          assert false )
  in
  Kirc_Cuda.return_v := k' ;
  Kirc_OpenCL.return_v := k' ;
  let cuda_head =
    Array.fold_left
      (fun header extension ->
        match extension with
        | ExFloat32 -> header
        | ExFloat64 -> cuda_float64 ^ header)
      cuda_head
      k.extensions
  in
  let opencl_head =
    Array.fold_left
      (fun header extension ->
        match extension with
        | ExFloat32 -> header
        | ExFloat64 -> opencl_float64 ^ header)
      opencl_head
      k.extensions
  in
  save (s ^ ".cu") (cuda_head ^ Kirc_Cuda.parse 0 (rewrite k2) dev) ;
  save (s ^ ".cl") (opencl_head ^ Kirc_OpenCL.parse 0 (rewrite k2) dev)

module Std = struct
  let thread_idx_x = 1l

  let thread_idx_y = 1l

  let thread_idx_z = 1l

  let block_idx_x = 1l

  let block_idx_y = 1l

  let block_idx_z = 1l

  let block_dim_x = 1l

  let block_dim_y = 1l

  let block_dim_z = 1l

  let grid_dim_x = 1l

  let grid_dim_y = 1l

  let grid_dim_z = 1l

  let global_thread_id = 0l

  let return () = ()

  let float64 i = float (Int32.to_int i)

  let float i = float (Int32.to_int i)

  let int_of_float64 f = Int32.of_int (int_of_float f)

  let int_of_float f = Int32.of_int (int_of_float f)

  let block_barrier () = ()

  let make_shared i = Array.make (Int32.to_int i) 0l

  let make_local i = Array.make (Int32.to_int i) 0l

  let map f a b =
    assert (Vector.length a = Vector.length b) ;
    for i = 0 to Vector.length a do
      Mem.set b i (f (Mem.get a i))
    done

  let reduce f a b =
    let rec aux acc i =
      if Vector.length a < i then aux (f acc (Mem.get a i)) (i + 1) else acc
    in
    Mem.set b 0 (aux (Mem.get a 0) 1)
end

module Sarek_vector = struct
  let length v = Int32.of_int (Vector.length v)
end

module Math = struct
  let pow a b = Int32.of_float (Float.pow (Int32.to_float a) (Int32.to_float b))

  let logical_and a b = Int32.logand a b

  let xor a b = Int32.logxor a b

  module Float32 = struct
    let add = ( +. )

    let minus = ( -. )

    let mul = ( *. )

    let div = ( /. )

    let pow = ( ** )

    let sqrt = sqrt

    let rsqrt = sqrt

    (* todo*)
    let exp = exp

    let log = log

    let log10 = log10

    let expm1 = expm1

    let log1p = log1p

    let acos = acos

    let cos = cos

    let cosh = cosh

    let asin = asin

    let sin = sin

    let sinh = sinh

    let tan = tan

    let tanh = tanh

    let atan = atan

    let atan2 = atan2

    let hypot = hypot

    let ceil = ceil

    let floor = floor

    let abs_float = abs_float

    let copysign = copysign

    let modf = modf

    let zero = 0.

    let one = 1.

    let make_shared i = Array.make (Int32.to_int i) 0.

    let make_local i = Array.make (Int32.to_int i) 0.
  end

  module Float64 = struct
    let add = ( +. )

    let minus = ( -. )

    let mul = ( *. )

    let div = ( /. )

    let pow = ( ** )

    let sqrt = sqrt

    let rsqrt = sqrt

    (* todo*)
    let exp = exp

    let log = log

    let log10 = log10

    let expm1 = expm1

    let log1p = log1p

    let acos = acos

    let cos = cos

    let cosh = cosh

    let asin = asin

    let sin = sin

    let sinh = sinh

    let tan = tan

    let tanh = tanh

    let atan = atan

    let atan2 = atan2

    let hypot = hypot

    let ceil = ceil

    let floor = floor

    let abs_float = abs_float

    let copysign = copysign

    let modf = modf

    let zero = 0.

    let one = 1.

    let of_float32 f = f

    let of_float f = f

    let to_float32 f = f

    let make_shared i = Array.make (Int32.to_int i) 0.

    let make_local i = Array.make (Int32.to_int i) 0.
  end
end
