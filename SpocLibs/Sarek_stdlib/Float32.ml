(******************************************************************************
 * Sarek Float32 Standard Library
 *
 * Provides float32 math functions for use in Sarek kernels.
 * Each function is defined using %sarek_intrinsic which:
 *   1. Registers the intrinsic in Sarek_registry at library load time
 *   2. Provides an OCaml implementation for host-side use
 ******************************************************************************)

open Spoc.Devices

(** CUDA vs OpenCL helper *)
let cuda_or_opencl dev cuda_code opencl_code =
  match dev.specific_info with
  | CudaInfo _ -> cuda_code
  | OpenCLInfo _ -> opencl_code

(** Square root *)
let%sarek_intrinsic (sqrt_float32 : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "sqrtf(%s)" "sqrt(%s)");
    ocaml = sqrt;
  }

(** Reciprocal square root: 1/sqrt(x) *)
let%sarek_intrinsic (rsqrt_float32 : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "rsqrtf(%s)" "rsqrt(%s)");
    ocaml = (fun x -> 1.0 /. sqrt x);
  }

(** Sine *)
let%sarek_intrinsic (sin_float32 : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "sinf(%s)" "sin(%s)"); ocaml = sin}

(** Cosine *)
let%sarek_intrinsic (cos_float32 : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "cosf(%s)" "cos(%s)"); ocaml = cos}

(** Exponential: e^x *)
let%sarek_intrinsic (exp_float32 : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "expf(%s)" "exp(%s)"); ocaml = exp}

(** Natural logarithm *)
let%sarek_intrinsic (log_float32 : float -> float) =
  {device = (fun dev -> cuda_or_opencl dev "logf(%s)" "log(%s)"); ocaml = log}

(** Absolute value *)
let%sarek_intrinsic (abs_float32 : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "fabsf(%s)" "fabs(%s)");
    ocaml = abs_float;
  }

(** Floor *)
let%sarek_intrinsic (floor_float32 : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "floorf(%s)" "floor(%s)");
    ocaml = floor;
  }

(** Ceiling *)
let%sarek_intrinsic (ceil_float32 : float -> float) =
  {
    device = (fun dev -> cuda_or_opencl dev "ceilf(%s)" "ceil(%s)");
    ocaml = ceil;
  }

(* Expose nicer names *)
let sqrt = sqrt_float32

let rsqrt = rsqrt_float32

let sin = sin_float32

let cos = cos_float32

let exp = exp_float32

let log = log_float32

let abs = abs_float32

let floor = floor_float32

let ceil = ceil_float32
