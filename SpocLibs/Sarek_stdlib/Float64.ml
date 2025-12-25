(******************************************************************************
 * Sarek Float64 Standard Library
 *
 * Provides float64 (double precision) type and math functions for Sarek kernels.
 * Uses %sarek_intrinsic to define GPU intrinsics with device function pattern.
 *
 * Each intrinsic generates:
 * - func_device : device -> string  (the device code generator)
 * - func_device_ref : (device -> string) ref  (for extension chaining)
 * - func : host-side OCaml implementation
 * - Registry entry for JIT code generation
 ******************************************************************************)

open Spoc.Devices

(** CUDA vs OpenCL helper - for float64, both use the same names (no 'f' suffix)
*)
let _cuda_or_opencl dev cuda_code opencl_code =
  match dev.specific_info with
  | CudaInfo _ -> cuda_code
  | OpenCLInfo _ -> opencl_code

(******************************************************************************
 * Type registration
 ******************************************************************************)

let%sarek_intrinsic float64 =
  {device = (fun _ -> "double"); ctype = Ctypes.double}

(******************************************************************************
 * Arithmetic operators
 ******************************************************************************)

let%sarek_intrinsic (add_float64 : float -> float -> float) =
  {device = (fun _ -> "(%s + %s)"); ocaml = ( +. )}

let%sarek_intrinsic (sub_float64 : float -> float -> float) =
  {device = (fun _ -> "(%s - %s)"); ocaml = ( -. )}

let%sarek_intrinsic (mul_float64 : float -> float -> float) =
  {device = (fun _ -> "(%s * %s)"); ocaml = ( *. )}

let%sarek_intrinsic (div_float64 : float -> float -> float) =
  {device = (fun _ -> "(%s / %s)"); ocaml = ( /. )}

(******************************************************************************
 * Unary math functions
 * Note: float64 uses same function names as float32 but without 'f' suffix
 ******************************************************************************)

let%sarek_intrinsic (sin : float -> float) =
  {device = (fun _ -> "sin"); ocaml = Stdlib.sin}

let%sarek_intrinsic (cos : float -> float) =
  {device = (fun _ -> "cos"); ocaml = Stdlib.cos}

let%sarek_intrinsic (tan : float -> float) =
  {device = (fun _ -> "tan"); ocaml = Stdlib.tan}

let%sarek_intrinsic (asin : float -> float) =
  {device = (fun _ -> "asin"); ocaml = Stdlib.asin}

let%sarek_intrinsic (acos : float -> float) =
  {device = (fun _ -> "acos"); ocaml = Stdlib.acos}

let%sarek_intrinsic (atan : float -> float) =
  {device = (fun _ -> "atan"); ocaml = Stdlib.atan}

let%sarek_intrinsic (sinh : float -> float) =
  {device = (fun _ -> "sinh"); ocaml = Stdlib.sinh}

let%sarek_intrinsic (cosh : float -> float) =
  {device = (fun _ -> "cosh"); ocaml = Stdlib.cosh}

let%sarek_intrinsic (tanh : float -> float) =
  {device = (fun _ -> "tanh"); ocaml = Stdlib.tanh}

let%sarek_intrinsic (exp : float -> float) =
  {device = (fun _ -> "exp"); ocaml = Stdlib.exp}

let%sarek_intrinsic (log : float -> float) =
  {device = (fun _ -> "log"); ocaml = Stdlib.log}

let%sarek_intrinsic (log10 : float -> float) =
  {device = (fun _ -> "log10"); ocaml = Stdlib.log10}

let%sarek_intrinsic (sqrt : float -> float) =
  {device = (fun _ -> "sqrt"); ocaml = Stdlib.sqrt}

let%sarek_intrinsic (ceil : float -> float) =
  {device = (fun _ -> "ceil"); ocaml = Stdlib.ceil}

let%sarek_intrinsic (floor : float -> float) =
  {device = (fun _ -> "floor"); ocaml = Stdlib.floor}

let%sarek_intrinsic (expm1 : float -> float) =
  {device = (fun _ -> "expm1"); ocaml = Stdlib.expm1}

let%sarek_intrinsic (log1p : float -> float) =
  {device = (fun _ -> "log1p"); ocaml = Stdlib.log1p}

let%sarek_intrinsic (abs_float : float -> float) =
  {device = (fun _ -> "fabs"); ocaml = Stdlib.abs_float}

let%sarek_intrinsic (rsqrt : float -> float) =
  {device = (fun _ -> "rsqrt"); ocaml = (fun x -> 1.0 /. Stdlib.sqrt x)}

(******************************************************************************
 * Binary math functions
 ******************************************************************************)

let%sarek_intrinsic (pow : float -> float -> float) =
  {device = (fun _ -> "pow"); ocaml = Float.pow}

let%sarek_intrinsic (atan2 : float -> float -> float) =
  {device = (fun _ -> "atan2"); ocaml = Stdlib.atan2}

let%sarek_intrinsic (hypot : float -> float -> float) =
  {device = (fun _ -> "hypot"); ocaml = Stdlib.hypot}

let%sarek_intrinsic (copysign : float -> float -> float) =
  {device = (fun _ -> "copysign"); ocaml = Stdlib.copysign}

(******************************************************************************
 * Conversion functions
 ******************************************************************************)

let of_int x = Stdlib.float_of_int x

let to_int x = Stdlib.int_of_float x

let of_float32 x = x

let to_float32 x = x
