(******************************************************************************
 * Sarek Primitive Types and Intrinsic Functions
 *
 * This module defines:
 * 1. Sarek primitive types (float32, float64, int32, etc.) as phantom types
 *    that map to OCaml types but are distinguished by the Sarek type system.
 * 2. Intrinsic functions as a GADT that carries:
 *    - Device code generation (CUDA/OpenCL)
 *    - OCaml fallback implementation
 *    - Type information via phantom types
 *
 * ARCHITECTURE NOTES:
 * - Phantom types (float32, float64, etc.) allow the PPX to infer Sarek types
 *   from OCaml type annotations while keeping runtime compatibility.
 * - The intrinsic GADT enables type-safe intrinsic definitions that can be:
 *   1. Used by the JIT for GPU code generation
 *   2. Used for pure OCaml execution (CPU fallback / testing)
 *   3. Type-checked at compile time by the PPX
 *
 * FUTURE: A Sarek_cpu runtime could use the OCaml implementations to run
 * kernels entirely on CPU for testing/debugging/fallback scenarios.
 *
 * Libraries can define new intrinsics using [%sarek_intrinsic] which:
 * 1. Registers the type with the PPX for inference
 * 2. Provides the GADT value for runtime use
 ******************************************************************************)

open Spoc.Devices

(******************************************************************************
 * Sarek Phantom Types
 *
 * These types are used in intrinsic signatures so the PPX can infer the
 * correct Sarek types. At runtime, they are just aliases to OCaml types.
 *
 * Usage in intrinsic definitions:
 *   let sin : (float32 -> float32) intrinsic = ...
 *
 * The PPX sees "float32 -> float32" and maps it to t_float32 -> t_float32.
 ******************************************************************************)

(** 32-bit floating point. Maps to OCaml [float] at runtime, but the PPX treats
    it as Sarek's float32 type. *)
type float32 = float

(** 64-bit floating point. Maps to OCaml [float] at runtime, but the PPX treats
    it as Sarek's float64 type. *)
type float64 = float

(** 32-bit integer. Maps to OCaml [int32] at runtime. *)
type int32_ = int32

(** 64-bit integer. Maps to OCaml [int64] at runtime. *)
type int64_ = int64

(** Boolean. Maps to OCaml [bool] at runtime. *)
type bool_ = bool

(** Unit type. *)
type unit_ = unit

(******************************************************************************
 * Legacy Module Signatures (kept for compatibility)
 ******************************************************************************)

module type SAREK_PRIM = sig
  type t

  val device : device -> string

  val ctype : t Ctypes.typ
end

module type SAREK_FUN1 = sig
  type arg

  type ret

  val device : device -> string

  val ocaml : arg -> ret
end

module type SAREK_FUN2 = sig
  type arg1

  type arg2

  type ret

  val device : device -> string

  val ocaml : arg1 -> arg2 -> ret
end

(******************************************************************************
 * Legacy Convenience Types (kept for compatibility)
 ******************************************************************************)

type 'a unary_intrinsic = {device : device -> string; ocaml : 'a -> 'a}

type ('a, 'b, 'c) binary_intrinsic = {
  device : device -> string;
  ocaml : 'a -> 'b -> 'c;
}

type 'a prim_type = {device : device -> string; ctype : 'a Ctypes.typ}

(******************************************************************************
 * Standard Primitive Type Modules
 ******************************************************************************)

module Float32 : SAREK_PRIM with type t = float = struct
  type t = float

  let device _ = "float"

  let ctype = Ctypes.float
end

module Float64 : SAREK_PRIM with type t = float = struct
  type t = float

  let device _ = "double"

  let ctype = Ctypes.double
end

module Int32 : SAREK_PRIM with type t = int32 = struct
  type t = int32

  let device _ = "int"

  let ctype = Ctypes.int32_t
end

module Int64 : SAREK_PRIM with type t = int64 = struct
  type t = int64

  let device _ = "long"

  let ctype = Ctypes.int64_t
end

module Bool : SAREK_PRIM with type t = bool = struct
  type t = bool

  let device _ = "int" (* bools are ints on GPU *)

  let ctype = Ctypes.bool
end

module Unit : SAREK_PRIM with type t = unit = struct
  type t = unit

  let device _ = "void"

  let ctype = Ctypes.void
end

(******************************************************************************
 * Intrinsic Function GADT
 *
 * This GADT encodes intrinsic functions with:
 * - Type information via the type parameter (using phantom types)
 * - Device-specific code generation
 * - OCaml fallback implementation
 *
 * The type parameter 'a represents the function type, e.g.:
 *   (float32 -> float32) intrinsic      -- unary float32 function
 *   (float32 -> float32 -> float32) intrinsic  -- binary float32 function
 ******************************************************************************)

(** Helper to generate device-specific code *)
let cuda_or_opencl dev cuda_code opencl_code =
  match dev.specific_info with
  | CudaInfo _ -> cuda_code
  | OpenCLInfo _ -> opencl_code

(** GADT for intrinsic functions *)
type _ intrinsic =
  | Intrinsic1 : {
      name : string;
      device : device -> string;  (** Device-specific code generator *)
      ocaml : 'a -> 'b;  (** OCaml implementation for CPU fallback *)
    }
      -> ('a -> 'b) intrinsic
  | Intrinsic2 : {
      name : string;
      device : device -> string;
      ocaml : 'a -> 'b -> 'c;
    }
      -> ('a -> 'b -> 'c) intrinsic

(** Get the OCaml function from an intrinsic. Used for: 1. Compile-time type
    checking (PPX generates references to this) 2. CPU fallback execution
    (future Sarek_cpu runtime) *)
let ocaml_of_intrinsic : type a. a intrinsic -> a = function
  | Intrinsic1 {ocaml; _} -> ocaml
  | Intrinsic2 {ocaml; _} -> ocaml

(** Get the name of an intrinsic *)
let name_of_intrinsic : type a. a intrinsic -> string = function
  | Intrinsic1 {name; _} -> name
  | Intrinsic2 {name; _} -> name

(** Get device code generator for an intrinsic. Used by the JIT to generate
    CUDA/OpenCL code. *)
let device_of_intrinsic : type a. a intrinsic -> device -> string = function
  | Intrinsic1 {device; _} -> device
  | Intrinsic2 {device; _} -> device

(******************************************************************************
 * Standard Intrinsic Functions - Float32
 *
 * These are the built-in math functions for 32-bit floats.
 * Each intrinsic provides:
 * - Device code for CUDA (typically *f suffix) and OpenCL
 * - OCaml implementation from Stdlib
 ******************************************************************************)

let sin : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "sin";
      device = (fun dev -> cuda_or_opencl dev "sinf" "sin");
      ocaml = Stdlib.sin;
    }

let cos : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "cos";
      device = (fun dev -> cuda_or_opencl dev "cosf" "cos");
      ocaml = Stdlib.cos;
    }

let tan : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "tan";
      device = (fun dev -> cuda_or_opencl dev "tanf" "tan");
      ocaml = Stdlib.tan;
    }

let asin : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "asin";
      device = (fun dev -> cuda_or_opencl dev "asinf" "asin");
      ocaml = Stdlib.asin;
    }

let acos : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "acos";
      device = (fun dev -> cuda_or_opencl dev "acosf" "acos");
      ocaml = Stdlib.acos;
    }

let atan : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "atan";
      device = (fun dev -> cuda_or_opencl dev "atanf" "atan");
      ocaml = Stdlib.atan;
    }

let sinh : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "sinh";
      device = (fun dev -> cuda_or_opencl dev "sinhf" "sinh");
      ocaml = Stdlib.sinh;
    }

let cosh : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "cosh";
      device = (fun dev -> cuda_or_opencl dev "coshf" "cosh");
      ocaml = Stdlib.cosh;
    }

let tanh : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "tanh";
      device = (fun dev -> cuda_or_opencl dev "tanhf" "tanh");
      ocaml = Stdlib.tanh;
    }

let exp : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "exp";
      device = (fun dev -> cuda_or_opencl dev "expf" "exp");
      ocaml = Stdlib.exp;
    }

let log : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "log";
      device = (fun dev -> cuda_or_opencl dev "logf" "log");
      ocaml = Stdlib.log;
    }

let log10 : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "log10";
      device = (fun dev -> cuda_or_opencl dev "log10f" "log10");
      ocaml = Stdlib.log10;
    }

let sqrt : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "sqrt";
      device = (fun dev -> cuda_or_opencl dev "sqrtf" "sqrt");
      ocaml = Stdlib.sqrt;
    }

let ceil : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "ceil";
      device = (fun dev -> cuda_or_opencl dev "ceilf" "ceil");
      ocaml = Stdlib.ceil;
    }

let floor : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "floor";
      device = (fun dev -> cuda_or_opencl dev "floorf" "floor");
      ocaml = Stdlib.floor;
    }

let abs_float : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "abs_float";
      device = (fun dev -> cuda_or_opencl dev "fabsf" "fabs");
      ocaml = Stdlib.abs_float;
    }

let expm1 : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "expm1";
      device = (fun dev -> cuda_or_opencl dev "expm1f" "expm1");
      ocaml = Stdlib.expm1;
    }

let log1p : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "log1p";
      device = (fun dev -> cuda_or_opencl dev "log1pf" "log1p");
      ocaml = Stdlib.log1p;
    }

(** Reciprocal square root: 1/sqrt(x). GPU has dedicated hardware instruction,
    OCaml computes it. *)
let rsqrt : (float32 -> float32) intrinsic =
  Intrinsic1
    {
      name = "rsqrt";
      device = (fun dev -> cuda_or_opencl dev "rsqrtf" "rsqrt");
      ocaml = (fun x -> 1.0 /. Stdlib.sqrt x);
    }

(* Binary float32 functions *)

let pow : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2
    {
      name = "pow";
      device = (fun dev -> cuda_or_opencl dev "powf" "pow");
      ocaml = Float.pow;
    }

let atan2 : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2
    {
      name = "atan2";
      device = (fun dev -> cuda_or_opencl dev "atan2f" "atan2");
      ocaml = Stdlib.atan2;
    }

let hypot : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2
    {
      name = "hypot";
      device = (fun dev -> cuda_or_opencl dev "hypotf" "hypot");
      ocaml = Stdlib.hypot;
    }

let copysign : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2
    {
      name = "copysign";
      device = (fun dev -> cuda_or_opencl dev "copysignf" "copysign");
      ocaml = Stdlib.copysign;
    }

(******************************************************************************
 * Standard Intrinsic Functions - Float64
 *
 * 64-bit float versions. CUDA/OpenCL use the same names without 'f' suffix.
 ******************************************************************************)

let sin64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "sin64"; device = (fun _ -> "sin"); ocaml = Stdlib.sin}

let cos64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "cos64"; device = (fun _ -> "cos"); ocaml = Stdlib.cos}

let tan64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "tan64"; device = (fun _ -> "tan"); ocaml = Stdlib.tan}

let asin64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "asin64"; device = (fun _ -> "asin"); ocaml = Stdlib.asin}

let acos64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "acos64"; device = (fun _ -> "acos"); ocaml = Stdlib.acos}

let atan64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "atan64"; device = (fun _ -> "atan"); ocaml = Stdlib.atan}

let sinh64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "sinh64"; device = (fun _ -> "sinh"); ocaml = Stdlib.sinh}

let cosh64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "cosh64"; device = (fun _ -> "cosh"); ocaml = Stdlib.cosh}

let tanh64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "tanh64"; device = (fun _ -> "tanh"); ocaml = Stdlib.tanh}

let exp64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "exp64"; device = (fun _ -> "exp"); ocaml = Stdlib.exp}

let log64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "log64"; device = (fun _ -> "log"); ocaml = Stdlib.log}

let log1064 : (float64 -> float64) intrinsic =
  Intrinsic1
    {name = "log1064"; device = (fun _ -> "log10"); ocaml = Stdlib.log10}

let sqrt64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "sqrt64"; device = (fun _ -> "sqrt"); ocaml = Stdlib.sqrt}

let ceil64 : (float64 -> float64) intrinsic =
  Intrinsic1 {name = "ceil64"; device = (fun _ -> "ceil"); ocaml = Stdlib.ceil}

let floor64 : (float64 -> float64) intrinsic =
  Intrinsic1
    {name = "floor64"; device = (fun _ -> "floor"); ocaml = Stdlib.floor}

let abs_float64 : (float64 -> float64) intrinsic =
  Intrinsic1
    {name = "abs_float64"; device = (fun _ -> "fabs"); ocaml = Stdlib.abs_float}

let rsqrt64 : (float64 -> float64) intrinsic =
  Intrinsic1
    {
      name = "rsqrt64";
      device = (fun _ -> "rsqrt");
      ocaml = (fun x -> 1.0 /. Stdlib.sqrt x);
    }

(* Binary float64 functions *)

let pow64 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2 {name = "pow64"; device = (fun _ -> "pow"); ocaml = Float.pow}

let atan264 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2
    {name = "atan264"; device = (fun _ -> "atan2"); ocaml = Stdlib.atan2}

let hypot64 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2
    {name = "hypot64"; device = (fun _ -> "hypot"); ocaml = Stdlib.hypot}

let copysign64 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2
    {
      name = "copysign64";
      device = (fun _ -> "copysign");
      ocaml = Stdlib.copysign;
    }

(******************************************************************************
 * Type Conversion Intrinsics
 ******************************************************************************)

let float_of_int : (int32_ -> float32) intrinsic =
  Intrinsic1
    {
      name = "float";
      device = (fun _ -> "(float)");
      ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
    }

let float64_of_int : (int32_ -> float64) intrinsic =
  Intrinsic1
    {
      name = "float64";
      device = (fun _ -> "(double)");
      ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
    }

let int_of_float : (float32 -> int32_) intrinsic =
  Intrinsic1
    {
      name = "int_of_float";
      device = (fun _ -> "(int)");
      ocaml = (fun f -> Stdlib.Int32.of_int (Stdlib.int_of_float f));
    }

let int_of_float64 : (float64 -> int32_) intrinsic =
  Intrinsic1
    {
      name = "int_of_float64";
      device = (fun _ -> "(int)");
      ocaml = (fun f -> Stdlib.Int32.of_int (Stdlib.int_of_float f));
    }

(******************************************************************************
 * GPU-specific Intrinsics (no meaningful OCaml equivalent)
 *
 * These are GPU synchronization and special operations.
 * The OCaml implementations are no-ops or raise errors.
 *
 * TODO: For future CPU fallback runtime, these would need proper
 * implementations (e.g., block_barrier could be a memory fence).
 ******************************************************************************)

let block_barrier : (unit_ -> unit_) intrinsic =
  Intrinsic1
    {
      name = "block_barrier";
      device =
        (fun dev ->
          cuda_or_opencl dev "__syncthreads()" "barrier(CLK_LOCAL_MEM_FENCE)");
      ocaml = (fun () -> ());
      (* No-op on CPU - TODO: memory fence for CPU runtime *)
    }

let return_unit : (unit_ -> unit_) intrinsic =
  Intrinsic1
    {name = "return"; device = (fun _ -> "return"); ocaml = (fun () -> ())}

(******************************************************************************
 * Integer Bitwise Operations
 *
 * These are used by GPU code for bitwise manipulation.
 ******************************************************************************)

let logical_and : (int32_ -> int32_ -> int32_) intrinsic =
  Intrinsic2
    {
      name = "logical_and";
      device = (fun _ -> "logical_and");
      ocaml = Stdlib.Int32.logand;
    }

let xor : (int32_ -> int32_ -> int32_) intrinsic =
  Intrinsic2
    {name = "xor"; device = (fun _ -> "spoc_xor"); ocaml = Stdlib.Int32.logxor}

let spoc_powint : (int32_ -> int32_ -> int32_) intrinsic =
  Intrinsic2
    {
      name = "spoc_powint";
      device = (fun _ -> "spoc_powint");
      ocaml =
        (fun base exp ->
          let rec pow b e acc =
            if e = 0l then acc
            else pow b (Stdlib.Int32.sub e 1l) (Stdlib.Int32.mul acc b)
          in
          pow base exp 1l);
    }

(******************************************************************************
 * Float32 Arithmetic Operations (spoc_* GPU helpers)
 *
 * These provide explicit arithmetic operations that map to GPU helper functions.
 * The OCaml implementations are simple arithmetic.
 ******************************************************************************)

let add_f32 : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2 {name = "add_f32"; device = (fun _ -> "spoc_fadd"); ocaml = ( +. )}

let minus_f32 : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2
    {name = "minus_f32"; device = (fun _ -> "spoc_fminus"); ocaml = ( -. )}

let mul_f32 : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2 {name = "mul_f32"; device = (fun _ -> "spoc_fmul"); ocaml = ( *. )}

let div_f32 : (float32 -> float32 -> float32) intrinsic =
  Intrinsic2 {name = "div_f32"; device = (fun _ -> "spoc_fdiv"); ocaml = ( /. )}

(******************************************************************************
 * Float64 Arithmetic Operations (spoc_* GPU helpers)
 ******************************************************************************)

let add64 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2 {name = "add64"; device = (fun _ -> "spoc_dadd"); ocaml = ( +. )}

let minus64 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2
    {name = "minus64"; device = (fun _ -> "spoc_dminus"); ocaml = ( -. )}

let mul64 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2 {name = "mul64"; device = (fun _ -> "spoc_dmul"); ocaml = ( *. )}

let div64 : (float64 -> float64 -> float64) intrinsic =
  Intrinsic2 {name = "div64"; device = (fun _ -> "spoc_ddiv"); ocaml = ( /. )}
