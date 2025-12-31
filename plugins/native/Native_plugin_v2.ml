(******************************************************************************
 * Native Plugin V2 - Phase 4 Backend Implementation
 *
 * Implements the Framework_sig.BACKEND_V2 interface for CPU execution.
 * Extends Native_plugin with:
 * - Execution model discrimination (Direct)
 * - Direct execution of pre-compiled OCaml functions
 * - Intrinsic registry support
 *
 * This plugin coexists with Native_plugin during the transition period.
 ******************************************************************************)

open Sarek_framework

(** Reuse the existing Native implementation *)
module Native_base = struct
  include Native_plugin.Native
end

(** Intrinsic registry for Native-specific intrinsics *)
module Native_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  let table : (string, Framework_sig.intrinsic_impl) Hashtbl.t =
    Hashtbl.create 64

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
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.thread_id_x ()");
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.thread_id_y ()");
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.thread_id_z ()");
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.block_id_x ()");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.block_id_y ()");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.block_id_z ()");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.block_dim_x ()");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.block_dim_y ()");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.block_dim_z ()");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.global_thread_id ()");
        intr_convergence = Framework_sig.Divergent;
      } ;

    (* Synchronization - Native uses threadpool barriers *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_typing = (fun _ -> Sarek_ir.TUnit);
        intr_codegen = (fun _ -> "Sarek_cpu_runtime.block_barrier ()");
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_typing = (fun _ -> Sarek_ir.TUnit);
        intr_codegen = (fun _ -> "()" (* No-op on CPU *));
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_typing = (fun _ -> Sarek_ir.TUnit);
        intr_codegen =
          (fun _ -> "()" (* No-op on CPU with proper memory model *));
        intr_convergence = Framework_sig.Uniform;
      }
end

(** Native V2 Backend - implements BACKEND_V2 *)
module Native_v2 : Framework_sig.BACKEND_V2 = struct
  (* Include all of BACKEND from Native_base *)
  include Native_base

  (** Execution model: Native uses Direct execution *)
  let execution_model = Framework_sig.Direct

  (** Generate source - not used for Direct backend *)
  let generate_source (_ir : Sarek_ir.kernel) : string option = None

  (** Execute directly using pre-compiled OCaml function *)
  let execute_direct
      ~(native_fn :
         (block:Framework_sig.dims ->
         grid:Framework_sig.dims ->
         Obj.t array ->
         unit)
         option) ~(block : Framework_sig.dims) ~(grid : Framework_sig.dims)
      (args : Obj.t array) : unit =
    match native_fn with
    | Some fn -> fn ~block ~grid args
    | None -> failwith "Native_v2.execute_direct: no native function provided"

  (** Native intrinsic registry *)
  module Intrinsics = Native_intrinsics
end

(** Auto-register V2 backend when module is loaded *)
let registered_v2 =
  lazy
    (if Native_v2.is_available () then
       Framework_registry.register_backend_v2
         ~priority:10
         (module Native_v2 : Framework_sig.BACKEND_V2))

let () = Lazy.force registered_v2

(** Force module initialization *)
let init () = Lazy.force registered_v2

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

(** Register a kernel for direct execution (wraps Native_plugin.register_kernel)
*)
let register_kernel = Native_plugin.register_kernel

(** Check if a kernel is registered *)
let kernel_registered = Native_plugin.kernel_registered

(** List all registered kernels *)
let list_kernels = Native_plugin.list_kernels
