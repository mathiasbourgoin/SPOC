(******************************************************************************
 * CUDA Plugin V2 - Phase 4 Backend Implementation
 *
 * Implements the Framework_sig.BACKEND_V2 interface for CUDA devices.
 * Extends Cuda_plugin with:
 * - Execution model discrimination (JIT)
 * - IR-based source generation via Sarek_ir_cuda
 * - Intrinsic registry support
 *
 * This plugin coexists with Cuda_plugin during the transition period.
 ******************************************************************************)

open Sarek_framework

(** Reuse the existing CUDA implementation *)
module Cuda_base = struct
  include Cuda_plugin.Cuda
end

(** CUDA-specific intrinsic implementation *)
type cuda_intrinsic = {
  intr_name : string;
  intr_codegen : string;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for CUDA-specific intrinsics *)
module Cuda_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = cuda_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard CUDA intrinsics *)
  let () =
    (* Thread indexing *)
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_codegen = "threadIdx.x";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_y"
      {
        intr_name = "thread_id_y";
        intr_codegen = "threadIdx.y";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "thread_id_z"
      {
        intr_name = "thread_id_z";
        intr_codegen = "threadIdx.z";
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_id_x"
      {
        intr_name = "block_id_x";
        intr_codegen = "blockIdx.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_y"
      {
        intr_name = "block_id_y";
        intr_codegen = "blockIdx.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_id_z"
      {
        intr_name = "block_id_z";
        intr_codegen = "blockIdx.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_x"
      {
        intr_name = "block_dim_x";
        intr_codegen = "blockDim.x";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_y"
      {
        intr_name = "block_dim_y";
        intr_codegen = "blockDim.y";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "block_dim_z"
      {
        intr_name = "block_dim_z";
        intr_codegen = "blockDim.z";
        intr_convergence = Framework_sig.Uniform;
      } ;
    register
      "global_thread_id"
      {
        intr_name = "global_thread_id";
        intr_codegen = "(threadIdx.x + blockIdx.x * blockDim.x)";
        intr_convergence = Framework_sig.Divergent;
      } ;

    (* Synchronization *)
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_codegen = "__syncthreads()";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "warp_barrier"
      {
        intr_name = "warp_barrier";
        intr_codegen = "__syncwarp()";
        intr_convergence = Framework_sig.Sync;
      } ;
    register
      "memory_fence"
      {
        intr_name = "memory_fence";
        intr_codegen = "__threadfence()";
        intr_convergence = Framework_sig.Uniform;
      }
end

(** CUDA V2 Backend - implements BACKEND_V2 *)
module Cuda_v2 : Framework_sig.BACKEND_V2 = struct
  (* Include all of BACKEND from Cuda_base *)
  include Cuda_base

  (** Execution model: CUDA uses JIT compilation *)
  let execution_model = Framework_sig.JIT

  (** Generate CUDA source from Sarek IR (wrapped as Obj.t) *)
  let generate_source (ir_obj : Obj.t) : string option =
    try Some (Sarek.Sarek_ir_cuda.generate (Obj.obj ir_obj)) with _ -> None

  (** Execute directly - not supported for JIT backend *)
  let execute_direct ~native_fn:_ ~block:_ ~grid:_ _args =
    failwith
      "Cuda_v2.execute_direct: JIT backend does not support direct execution"

  (** CUDA intrinsic registry *)
  module Intrinsics = Cuda_intrinsics
end

(** Auto-register V2 backend when module is loaded *)
let registered_v2 =
  lazy
    (if Cuda_v2.is_available () then
       Framework_registry.register_backend_v2
         ~priority:100
         (module Cuda_v2 : Framework_sig.BACKEND_V2))

let () = Lazy.force registered_v2

(** Force module initialization *)
let init () = Lazy.force registered_v2

(** {1 Additional CUDA-specific Functions} *)

(** Register a custom CUDA intrinsic *)
let register_intrinsic = Cuda_intrinsics.register

(** Look up a CUDA intrinsic *)
let find_intrinsic = Cuda_intrinsics.find

(** Generate CUDA source with custom types *)
let generate_with_types = Sarek.Sarek_ir_cuda.generate_with_types

(** Generate CUDA source for a kernel *)
let generate_source = Sarek.Sarek_ir_cuda.generate
