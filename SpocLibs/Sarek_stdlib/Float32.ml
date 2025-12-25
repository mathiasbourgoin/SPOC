(******************************************************************************
 * Sarek Float32 Standard Library
 *
 * Provides float32 type and math functions for use in Sarek kernels.
 * Uses %sarek_intrinsic to define GPU intrinsics with device function pattern.
 *
 * Each intrinsic generates:
 * - func_device : device -> string  (the device code generator)
 * - func_device_ref : (device -> string) ref  (for extension chaining)
 * - func : host-side OCaml implementation
 * - Registry entry for JIT code generation
 *
 * To extend for a new backend (e.g., FPGA), use %sarek_extend:
 *   let%sarek_extend Float32.sin = fun dev ->
 *     if is_fpga dev then "fpga_sin" else Float32.sin_device dev
 ******************************************************************************)

open Spoc.Devices

(** CUDA vs OpenCL helper *)
let cuda_or_opencl dev cuda_code opencl_code =
  match dev.specific_info with
  | CudaInfo _ -> cuda_code
  | OpenCLInfo _ -> opencl_code

(******************************************************************************
 * Type registration
 *
 * Register float32 as a primitive type. This must happen before any functions
 * that use float32 are registered.
 ******************************************************************************)

let%sarek_intrinsic float32 =
  {device = (fun _ -> "float"); ctype = Ctypes.float}

(******************************************************************************
 * Arithmetic operators
 *
 * These are the fundamental float32 operations used by the kernel code generator.
 ******************************************************************************)

let%sarek_intrinsic (add_float32 : float -> float -> float) =
  {device = (fun _ -> "(%s + %s)"); ocaml = ( +. )}

let%sarek_intrinsic (sub_float32 : float -> float -> float) =
  {device = (fun _ -> "(%s - %s)"); ocaml = ( -. )}

let%sarek_intrinsic (mul_float32 : float -> float -> float) =
  {device = (fun _ -> "(%s * %s)"); ocaml = ( *. )}

let%sarek_intrinsic (div_float32 : float -> float -> float) =
  {device = (fun _ -> "(%s / %s)"); ocaml = ( /. )}

(******************************************************************************
 * Unary math functions
 ******************************************************************************)

let%sarek_intrinsic (sin : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "sinf" "sin"); ocaml = Stdlib.sin}

let%sarek_intrinsic (cos : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "cosf" "cos"); ocaml = Stdlib.cos}

let%sarek_intrinsic (tan : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "tanf" "tan"); ocaml = Stdlib.tan}

let%sarek_intrinsic (asin : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "asinf" "asin"); ocaml = Stdlib.asin}

let%sarek_intrinsic (acos : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "acosf" "acos"); ocaml = Stdlib.acos}

let%sarek_intrinsic (atan : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "atanf" "atan"); ocaml = Stdlib.atan}

let%sarek_intrinsic (sinh : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "sinhf" "sinh"); ocaml = Stdlib.sinh}

let%sarek_intrinsic (cosh : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "coshf" "cosh"); ocaml = Stdlib.cosh}

let%sarek_intrinsic (tanh : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "tanhf" "tanh"); ocaml = Stdlib.tanh}

let%sarek_intrinsic (exp : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "expf" "exp"); ocaml = Stdlib.exp}

let%sarek_intrinsic (log : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "logf" "log"); ocaml = Stdlib.log}

let%sarek_intrinsic (log10 : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "log10f" "log10");
    ocaml = Stdlib.log10;
  }

let%sarek_intrinsic (sqrt : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "sqrtf" "sqrt"); ocaml = Stdlib.sqrt}

let%sarek_intrinsic (ceil : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "ceilf" "ceil"); ocaml = Stdlib.ceil}

let%sarek_intrinsic (floor : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "floorf" "floor");
    ocaml = Stdlib.floor;
  }

let%sarek_intrinsic (expm1 : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "expm1f" "expm1");
    ocaml = Stdlib.expm1;
  }

let%sarek_intrinsic (log1p : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "log1pf" "log1p");
    ocaml = Stdlib.log1p;
  }

let%sarek_intrinsic (abs_float : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "fabsf" "fabs");
    ocaml = Stdlib.abs_float;
  }

let%sarek_intrinsic (rsqrt : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "rsqrtf" "rsqrt");
    ocaml = (fun x -> 1.0 /. Stdlib.sqrt x);
  }

(******************************************************************************
 * Binary math functions
 ******************************************************************************)

let%sarek_intrinsic (pow : float -> float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "powf" "pow"); ocaml = Float.pow}

let%sarek_intrinsic (atan2 : float -> float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "atan2f" "atan2");
    ocaml = Stdlib.atan2;
  }

let%sarek_intrinsic (hypot : float -> float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "hypotf" "hypot");
    ocaml = Stdlib.hypot;
  }

let%sarek_intrinsic (copysign : float -> float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "copysignf" "copysign");
    ocaml = Stdlib.copysign;
  }

(******************************************************************************
 * Conversion functions
 ******************************************************************************)

let of_int x = Stdlib.float_of_int x

let to_int x = Stdlib.int_of_float x
