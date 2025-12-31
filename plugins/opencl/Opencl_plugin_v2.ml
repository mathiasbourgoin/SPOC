(******************************************************************************
 * OpenCL Plugin V2 - Phase 4 Backend Implementation
 *
 * Implements the Framework_sig.BACKEND_V2 interface for OpenCL devices.
 * Extends Opencl_plugin with:
 * - Execution model discrimination (JIT)
 * - IR-based source generation via Sarek_ir_opencl
 * - Intrinsic registry support
 *
 * This plugin coexists with Opencl_plugin during the transition period.
 ******************************************************************************)

open Sarek_framework

(** Reuse the existing OpenCL implementation *)
module Opencl_base = struct
  include Opencl_plugin.Opencl
end

(** Intrinsic registry for OpenCL-specific intrinsics *)
module Opencl_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  let table : (string, Framework_sig.intrinsic_impl) Hashtbl.t =
    Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard OpenCL intrinsics *)
  let () =
    (* Thread indexing - OpenCL uses get_local_id, get_group_id, etc. *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_local_id(0)");
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_local_id(1)");
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_local_id(2)");
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_group_id(0)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_group_id(1)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_group_id(2)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_local_size(0)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_local_size(1)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_local_size(2)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_x"
      {
        intr_name = "grid_dim_x";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_num_groups(0)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_y"
      {
        intr_name = "grid_dim_y";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_num_groups(1)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "grid_dim_z"
      {
        intr_name = "grid_dim_z";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_num_groups(2)");
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_global_id(0)");
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "global_size"
      {
        intr_name = "global_size";
        intr_typing = (fun _ -> Sarek_ir.TInt32);
        intr_codegen = (fun _ -> "get_global_size(0)");
        intr_convergence = Framework_sig.Uniform;
      } ;

    (* Synchronization *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_typing = (fun _ -> Sarek_ir.TUnit);
        intr_codegen = (fun _ -> "barrier(CLK_LOCAL_MEM_FENCE)");
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_typing = (fun _ -> Sarek_ir.TUnit);
        intr_codegen = (fun _ -> "sub_group_barrier(CLK_LOCAL_MEM_FENCE)");
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_typing = (fun _ -> Sarek_ir.TUnit);
        intr_codegen = (fun _ -> "mem_fence(CLK_GLOBAL_MEM_FENCE)");
        intr_convergence = Framework_sig.Uniform;
      }
end

(** OpenCL V2 Backend - implements BACKEND_V2 *)
module Opencl_v2 : Framework_sig.BACKEND_V2 = struct
  (* Include all of BACKEND from Opencl_base *)
  include Opencl_base

  (** Execution model: OpenCL uses JIT compilation *)
  let execution_model = Framework_sig.JIT

  (** Generate OpenCL source from Sarek IR *)
  let generate_source (ir : Sarek_ir.kernel) : string option =
    try Some (Sarek_ir_opencl.generate ir) with _ -> None

  (** Execute directly - not supported for JIT backend *)
  let execute_direct ~native_fn:_ ~block:_ ~grid:_ _args =
    failwith
      "Opencl_v2.execute_direct: JIT backend does not support direct execution"

  (** OpenCL intrinsic registry *)
  module Intrinsics = Opencl_intrinsics
end

(** Auto-register V2 backend when module is loaded *)
let registered_v2 =
  lazy
    (if Opencl_v2.is_available () then
       Framework_registry.register_backend_v2
         ~priority:90
         (module Opencl_v2 : Framework_sig.BACKEND_V2))

let () = Lazy.force registered_v2

(** Force module initialization *)
let init () = Lazy.force registered_v2

(** {1 Additional OpenCL-specific Functions} *)

(** Register a custom OpenCL intrinsic *)
let register_intrinsic = Opencl_intrinsics.register

(** Look up an OpenCL intrinsic *)
let find_intrinsic = Opencl_intrinsics.find

(** Generate OpenCL source with custom types *)
let generate_with_types = Sarek_ir_opencl.generate_with_types

(** Generate OpenCL source for a kernel *)
let generate_source = Sarek_ir_opencl.generate

(** Generate OpenCL source with FP64 extension if needed *)
let generate_with_fp64 = Sarek_ir_opencl.generate_with_fp64
