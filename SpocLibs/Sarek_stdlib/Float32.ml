(******************************************************************************
 * Sarek Float32 Standard Library
 *
 * Provides float32 math functions for use in Sarek kernels.
 * Functions are registered with Sarek_registry at module load time.
 ******************************************************************************)

open Spoc.Devices
open Sarek.Sarek_registry

(** CUDA vs OpenCL helper *)
let cuda_or_opencl dev cuda_code opencl_code =
  match dev.specific_info with
  | CudaInfo _ -> cuda_code
  | OpenCLInfo _ -> opencl_code

(* Register float32 type *)
let () = register_type "float32" ~device:(fun _ -> "float") ~size:4

(* Register float32 intrinsic functions *)
let () =
  (* Unary math functions *)
  let unary_funs =
    [
      ("sin", "sinf", "sin");
      ("cos", "cosf", "cos");
      ("tan", "tanf", "tan");
      ("asin", "asinf", "asin");
      ("acos", "acosf", "acos");
      ("atan", "atanf", "atan");
      ("sinh", "sinhf", "sinh");
      ("cosh", "coshf", "cosh");
      ("tanh", "tanhf", "tanh");
      ("exp", "expf", "exp");
      ("log", "logf", "log");
      ("log10", "log10f", "log10");
      ("sqrt", "sqrtf", "sqrt");
      ("ceil", "ceilf", "ceil");
      ("floor", "floorf", "floor");
      ("expm1", "expm1f", "expm1");
      ("log1p", "log1pf", "log1p");
      ("abs_float", "fabsf", "fabs");
      ("rsqrt", "rsqrtf", "rsqrt");
    ]
  in
  List.iter
    (fun (name, cuda, opencl) ->
      register_fun
        ~module_path:["Float32"]
        name
        ~arity:1
        ~device:(fun dev -> cuda_or_opencl dev cuda opencl)
        ~arg_types:["float32"]
        ~ret_type:"float32")
    unary_funs ;

  (* Binary math functions *)
  let binary_funs =
    [
      ("pow", "powf", "pow");
      ("atan2", "atan2f", "atan2");
      ("hypot", "hypotf", "hypot");
      ("copysign", "copysignf", "copysign");
    ]
  in
  List.iter
    (fun (name, cuda, opencl) ->
      register_fun
        ~module_path:["Float32"]
        name
        ~arity:2
        ~device:(fun dev -> cuda_or_opencl dev cuda opencl)
        ~arg_types:["float32"; "float32"]
        ~ret_type:"float32")
    binary_funs

(** OCaml implementations for host-side use *)
let sin = Stdlib.sin

let cos = Stdlib.cos

let tan = Stdlib.tan

let asin = Stdlib.asin

let acos = Stdlib.acos

let atan = Stdlib.atan

let sinh = Stdlib.sinh

let cosh = Stdlib.cosh

let tanh = Stdlib.tanh

let exp = Stdlib.exp

let log = Stdlib.log

let log10 = Stdlib.log10

let sqrt = Stdlib.sqrt

let ceil = Stdlib.ceil

let floor = Stdlib.floor

let expm1 = Stdlib.expm1

let log1p = Stdlib.log1p

let abs_float = Stdlib.abs_float

let rsqrt x = 1.0 /. Stdlib.sqrt x

let pow = Float.pow

let atan2 = Stdlib.atan2

let hypot = Stdlib.hypot

let copysign = Stdlib.copysign

(** Conversion functions *)
let of_int x = Stdlib.float_of_int x

let to_int x = Stdlib.int_of_float x

(* Register conversion functions *)
let () =
  register_fun
    "float"
    ~arity:1
    ~device:(fun _ -> "(float)")
    ~arg_types:["int32"]
    ~ret_type:"float32" ;
  register_fun
    "int_of_float"
    ~arity:1
    ~device:(fun _ -> "(int)")
    ~arg_types:["float32"]
    ~ret_type:"int32"
