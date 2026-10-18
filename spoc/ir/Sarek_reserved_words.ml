(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Words are stored as space-separated text, one block per origin, so that
   adding a word is a one-token change and a reader can compare a block with
   the specification it was taken from. *)

let words text =
  String.split_on_char
    ' '
    (String.map (fun c -> if c = '\n' then ' ' else c) text)
  |> List.filter (fun w -> w <> "")

let set_of text =
  let table = Hashtbl.create 512 in
  List.iter (fun w -> Hashtbl.replace table w ()) (words text) ;
  fun name -> Hashtbl.mem table name

let is_identifier_char c =
  (c >= 'a' && c <= 'z')
  || (c >= 'A' && c <= 'Z')
  || (c >= '0' && c <= '9')
  || c = '_'

let invalid_identifier name =
  name = ""
  || (name.[0] >= '0' && name.[0] <= '9')
  || not (String.for_all is_identifier_char name)

let has_prefix prefixes name =
  List.exists (fun p -> String.starts_with ~prefix:p name) prefixes

(* [base] followed by one of [suffixes]: [float4], [uchar16], [half2x2]. *)
let is_suffixed ~bases ~suffixes name =
  List.exists
    (fun base ->
      String.starts_with ~prefix:base name
      &&
      let n = String.length base in
      List.mem (String.sub name n (String.length name - n)) suffixes)
    bases

(* ISO C11 and C++20 keywords, alternative tokens, the standard library names
   every generated source can see, and the C types of <stdint.h>. Metal is C++14
   and CUDA/HIP/OpenCL C compile as C/C++ with these in scope. *)
let c_keywords =
  set_of
    "alignas alignof and and_eq asm auto bitand bitor bool break case catch \
     char char8_t char16_t char32_t class compl concept const consteval \
     constexpr constinit const_cast continue co_await co_return co_yield \
     decltype default delete do double dynamic_cast else enum explicit export \
     extern false final float for friend goto if inline int long mutable \
     namespace new noexcept not not_eq nullptr operator or or_eq override \
     private protected public register reinterpret_cast requires restrict \
     return short signed sizeof static static_assert static_cast struct switch \
     template this thread_local throw true try typedef typeid typename union \
     unsigned using virtual void volatile wchar_t while xor xor_eq _Bool \
     _Complex _Imaginary _Atomic _Generic _Noreturn _Static_assert \
     _Thread_local NULL EOF errno main int8_t int16_t int32_t int64_t uint8_t \
     uint16_t uint32_t uint64_t size_t ptrdiff_t intptr_t uintptr_t ssize_t \
     off_t"

(* The <math.h> and <stdlib.h> functions a generated expression calls by their
   plain name: a local that hides one breaks every later call in its scope.
   The single-precision [f] variants are derived below. *)
let math_names =
  "sqrt cbrt exp exp2 expm1 log log2 log10 log1p pow sin cos tan asin acos \
   atan atan2 sinh cosh tanh asinh acosh atanh floor ceil round trunc rint \
   nearbyint fmod remainder fma fmin fmax fdim hypot erf erfc lgamma tgamma \
   ldexp frexp modf copysign nextafter fabs rsqrt sincos exp10 sinpi cospi \
   fmaf"

let c_functions =
  let derived = List.map (fun w -> w ^ "f") (words math_names) in
  set_of
    (String.concat
       " "
       (words math_names @ derived
       @ words
           "abs labs llabs min max div ldiv isnan isinf isfinite signbit \
            printf puts malloc free memcpy memset assert"))

let c_family name = c_keywords name || c_functions name

(* CUDA and HIP: built-in variables, vector and half types, atomics. *)
let cuda_specific =
  set_of
    "threadIdx blockIdx blockDim gridDim warpSize dim3 half half2 bfloat16 \
     bfloat162 uint uchar ushort ulong longlong ulonglong atomicAdd atomicSub \
     atomicExch atomicMin atomicMax atomicInc atomicDec atomicCAS atomicAnd \
     atomicOr atomicXor hipThreadIdx_x hipThreadIdx_y hipThreadIdx_z \
     hipBlockIdx_x hipBlockIdx_y hipBlockIdx_z hipBlockDim_x hipBlockDim_y \
     hipBlockDim_z hipGridDim_x hipGridDim_y hipGridDim_z"

let cuda_vector =
  is_suffixed
    ~bases:
      [
        "char";
        "uchar";
        "short";
        "ushort";
        "int";
        "uint";
        "long";
        "ulong";
        "longlong";
        "ulonglong";
        "float";
        "double";
      ]
    ~suffixes:["1"; "2"; "3"; "4"]

(* OpenCL C 1.2/2.0: address-space and function qualifiers, scalar and image
   types, work-item and synchronisation built-ins, the gentype built-in
   functions. *)
let opencl_specific =
  set_of
    "kernel global local constant private generic read_only write_only \
     read_write half uchar ushort uint ulong bool image1d_t image2d_t \
     image3d_t image1d_array_t image2d_array_t image1d_buffer_t sampler_t \
     event_t queue_t ndrange_t clk_event_t reserve_id_t pipe get_global_id \
     get_local_id get_group_id get_global_size get_local_size get_num_groups \
     get_work_dim get_global_offset barrier mem_fence read_mem_fence \
     write_mem_fence clamp mix step smoothstep sign select length distance \
     normalize dot cross mad mad24 mul24 mul_hi rotate popcount clz ctz hadd \
     rhadd upsample abs_diff add_sat sub_sat mad_sat any all bitselect isequal \
     isnotequal isgreater isgreaterequal isless islessequal islessgreater \
     isordered isunordered fract degrees radians powr pown rootn atan2pi \
     asinpi acospi atanpi tanpi fast_length fast_distance fast_normalize"

let opencl_prefixes =
  [
    "convert_";
    "as_";
    "native_";
    "half_";
    "vload";
    "vstore";
    "atomic_";
    "atom_";
    "get_";
    "sub_group_";
    "work_group_";
    "async_";
    "cl_";
    "intel_";
  ]

let opencl_vector =
  is_suffixed
    ~bases:
      [
        "char";
        "uchar";
        "short";
        "ushort";
        "int";
        "uint";
        "long";
        "ulong";
        "float";
        "double";
        "half";
      ]
    ~suffixes:["2"; "3"; "4"; "8"; "16"]

(* Metal Shading Language 3: address spaces and function qualifiers, the types
   of the metal namespace (which [using namespace metal] brings in scope), the
   common and geometric functions. *)
let metal_specific =
  set_of
    "kernel vertex fragment device constant thread threadgroup \
     threadgroup_imageblock object_data ray_data mesh object payload half \
     uchar ushort uint ulong bfloat metal simd access sampler atomic \
     memory_order texture1d texture2d texture3d texturecube depth2d \
     address_space saturate clamp mix step smoothstep sign select fract dot \
     cross length distance normalize reflect refract popcount clz ctz rotate \
     mulhi madhi mad extract_bits insert_bits reverse_bits precise fast \
     is_null_buffer"

let metal_prefixes =
  [
    "simd_";
    "simdgroup_";
    "quad_";
    "atomic_";
    "as_type";
    "threadgroup_";
    "packed_";
  ]

let metal_vector =
  is_suffixed
    ~bases:
      [
        "char";
        "uchar";
        "short";
        "ushort";
        "int";
        "uint";
        "long";
        "ulong";
        "float";
        "half";
        "bool";
        "bfloat";
      ]
    ~suffixes:["2"; "3"; "4"]

let metal_matrix =
  let dims = ["2"; "3"; "4"] in
  let shapes =
    List.concat_map (fun r -> List.map (fun c -> r ^ "x" ^ c) dims) dims
  in
  is_suffixed ~bases:["half"; "float"] ~suffixes:shapes

let cuda name =
  invalid_identifier name || c_family name || cuda_specific name
  || cuda_vector name
  || String.starts_with ~prefix:"make_" name

let opencl name =
  invalid_identifier name || c_family name || opencl_specific name
  || has_prefix opencl_prefixes name
  || opencl_vector name

let metal name =
  invalid_identifier name || c_family name || metal_specific name
  || has_prefix metal_prefixes name
  || metal_vector name || metal_matrix name
