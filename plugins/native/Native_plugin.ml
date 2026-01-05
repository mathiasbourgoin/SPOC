(******************************************************************************
 * Native Plugin Backend Implementation
 *
 * Implements the Framework_sig.BACKEND interface for CPU execution.
 * Extends Native_plugin with:
 * - Execution model discrimination (Direct)
 * - Direct execution of pre-compiled OCaml functions
 * - Intrinsic registry support
 *
 * This plugin coexists with Native_plugin during the transition period.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Reuse the existing Native implementation *)
module Native_base = struct
  include Native_plugin_base.Native
end

(** Native-specific intrinsic implementation *)
type native_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for Native-specific intrinsics *)
module Native_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = native_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard Native intrinsics - these are placeholders since
     Native execution uses pre-compiled OCaml functions *)
  let () =
    (* Thread indexing - in Native, these come from thread state *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "Sarek_cpu_runtime.thread_id_x ()";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "Sarek_cpu_runtime.thread_id_y ()";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "Sarek_cpu_runtime.thread_id_z ()";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "Sarek_cpu_runtime.block_id_x ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "Sarek_cpu_runtime.block_id_y ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "Sarek_cpu_runtime.block_id_z ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "Sarek_cpu_runtime.block_dim_x ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "Sarek_cpu_runtime.block_dim_y ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "Sarek_cpu_runtime.block_dim_z ()";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "Sarek_cpu_runtime.global_thread_id ()";
        intr_convergence = Framework_sig.Divergent;
      } ;

    (* Synchronization - Native uses threadpool barriers *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "Sarek_cpu_runtime.block_barrier ()";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_codegen = "()" (* No-op on CPU *);
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "()" (* No-op on CPU with proper memory model *);
        intr_convergence = Framework_sig.Uniform;
      }
end

(** Native Backend - implements BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  (* Include all of BACKEND from Native_base *)
  include Native_base

  (** Execution model: Native uses Direct execution *)
  let execution_model = Framework_sig.Direct

  (** Generate source - not used for Direct backend *)
  let generate_source ?block:_ (_ir : Sarek_ir_types.kernel) : string option = None

  (** Execute directly using native function from IR. Args contain vectors
      directly (not expanded buffer/length pairs). The native function uses
      Spoc_core.Vector.get/set for access. *)
  let execute_direct
      ~(native_fn :
         (block:Framework_sig.dims ->
         grid:Framework_sig.dims ->
         Obj.t array ->
         unit)
         option) ~(ir : Sarek_ir_types.kernel option)
      ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
      (args : Obj.t array) : unit =
    ignore native_fn ;
    (* We use kern_native_fn from IR *)
    match ir with
    | Some kernel -> (
        match kernel.kern_native_fn with
        | Some (Sarek_ir_types.NativeFn fn) ->
            (* Use native function - args are vectors directly *)
            let block_tuple = (block.x, block.y, block.z) in
            let grid_tuple = (grid.x, grid.y, grid.z) in
            fn ~parallel:true ~block:block_tuple ~grid:grid_tuple args
        | None ->
            (* Fall back to IR interpretation with vector support *)
            Sarek.Sarek_ir_interp.parallel_mode := true ;
            Sarek.Sarek_ir_interp.run_kernel_with_obj_args
              (Obj.magic kernel) (* TODO: Fix Sarek_ir_interp signature *)
              ~block:(block.x, block.y, block.z)
              ~grid:(grid.x, grid.y, grid.z)
              args)
    | None -> failwith "Native backend execute_direct: IR required"

  (** Native intrinsic registry *)
  module Intrinsics = Native_intrinsics

  (** {2 External Kernel Support} *)

  (** Native backend does not support external GPU sources *)
  let supported_source_langs = []

  (** External kernel execution not supported on Native backend *)
  let run_source ~source:_ ~lang ~kernel_name:_ ~block:_ ~grid:_ ~shared_mem:_
      (_args : Framework_sig.run_source_arg list) =
    let lang_str =
      match lang with
      | Framework_sig.CUDA_Source -> "CUDA source"
      | Framework_sig.OpenCL_Source -> "OpenCL source"
      | Framework_sig.PTX -> "PTX"
      | Framework_sig.SPIR_V -> "SPIR-V"
    in
    failwith
      (Printf.sprintf
         "Native backend cannot execute external %s kernels. Use CUDA or \
          OpenCL backend instead."
         lang_str)
end

(** Auto-register backend when module is loaded *)
let registered_backend =
  lazy
    (if Backend.is_available () then
       Framework_registry.register_backend
         ~priority:10
         (module Backend : Framework_sig.BACKEND))

let () = Lazy.force registered_backend

(** Force module initialization *)
let init () = Lazy.force registered_backend

(** {1 Additional Native-specific Functions} *)

(** Register a custom Native intrinsic *)
let register_intrinsic = Native_intrinsics.register

(** Look up a Native intrinsic *)
let find_intrinsic = Native_intrinsics.find

(** Execute a kernel directly with the registered function *)
let run_kernel_direct ~name
    ~(native_fn : Obj.t array -> int * int * int -> int * int * int -> unit)
    ~(args : Obj.t array) ~(grid : int * int * int) ~(block : int * int * int) :
    unit =
  (* Use the existing Native_plugin kernel registry *)
  ignore name ;
  native_fn args grid block

(** Register a kernel for direct execution (wraps
    Native_plugin_base.register_kernel) *)
let register_kernel = Native_plugin_base.register_kernel

(** Check if a kernel is registered *)
let kernel_registered = Native_plugin_base.kernel_registered

(** List all registered kernels *)
let list_kernels = Native_plugin_base.list_kernels
