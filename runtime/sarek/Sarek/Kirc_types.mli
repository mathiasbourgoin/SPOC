(******************************************************************************
 * Kirc Types
 *
 * Type definitions for Sarek kernels. Separated from Kirc to allow the V2
 * path to use these types without depending on SPOC.
 ******************************************************************************)

type float64 = float

type float32 = float

type extension = ExFloat32 | ExFloat64

(** Stub type replacing Spoc.Vector.kind - V2 path doesn't use this *)
type ('a, 'b) vector_kind_stub = unit

(** Stub type replacing Spoc.Kernel.spoc_kernel - V2 path doesn't use this *)
type ('a, 'b) spoc_kernel_stub = unit

type ('a, 'b, 'c) kirc_kernel = {
  ml_kern : 'a;
  body : Kirc_Ast.k_ext;
  body_ir : Sarek_ir.kernel option;
  ret_val : Kirc_Ast.k_ext * ('b, 'c) vector_kind_stub;
  extensions : extension array;
  cpu_kern :
    (mode:Sarek_cpu_runtime.exec_mode ->
    block:int * int * int ->
    grid:int * int * int ->
    Obj.t array ->
    unit)
    option;
}

type ('a, 'b, 'c, 'd) kirc_function = {
  fun_name : string;
  ml_fun : 'a;
  funbody : Kirc_Ast.k_ext;
  fun_ret : Kirc_Ast.k_ext * ('b, 'c) vector_kind_stub;
  fastflow_acc : 'd;
  fun_extensions : extension array;
}

type ('a, 'b, 'c, 'd, 'e) sarek_kernel =
  ('a, 'b) spoc_kernel_stub * ('c, 'd, 'e) kirc_kernel

(** Constructor registry for variant types *)
val constructors : string list ref

val register_constructor_string : string -> unit
