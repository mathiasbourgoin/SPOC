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

open Spoc_framework
open Spoc_framework_registry

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

  (** Generate CUDA source from Sarek IR *)
  let generate_source (ir : Sarek_ir_types.kernel) : string option =
    try Some (Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir)
    with _ -> None

  (** Execute directly - not supported for JIT backend *)
  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args =
    failwith
      "Cuda_v2.execute_direct: JIT backend does not support direct execution"

  (** CUDA intrinsic registry *)
  module Intrinsics = Cuda_intrinsics

  (** {2 External Kernel Support} *)

  (** Supported source languages: CUDA only (PTX requires direct loading, not
      yet implemented) *)
  let supported_source_langs = [Framework_sig.CUDA_Source]

  (** Execute external kernel from source *)
  let run_source ~source ~lang ~kernel_name ~block ~grid ~shared_mem args =
    match lang with
    | Framework_sig.CUDA_Source ->
        (* Get current device (must be set by Execute before calling) *)
        let dev =
          match Cuda_plugin.Cuda.Device.get_current_device () with
          | Some d -> d
          | None -> failwith "run_source: no current CUDA device set"
        in

        (* Compile and get kernel *)
        let compiled = Kernel.compile_cached dev ~name:kernel_name ~source in

        (* Set up kernel arguments using typed run_source_arg list *)
        let kargs = Kernel.create_args () in
        List.iteri
          (fun i arg ->
            match arg with
            | Framework_sig.RSA_Buffer {binder; _} ->
                (* Use the binder function to properly bind the device buffer *)
                binder ~kargs:(Obj.repr kargs) ~idx:i
            | Framework_sig.RSA_Int32 n -> Kernel.set_arg_int32 kargs i n
            | Framework_sig.RSA_Int64 n -> Kernel.set_arg_int64 kargs i n
            | Framework_sig.RSA_Float32 f -> Kernel.set_arg_float32 kargs i f
            | Framework_sig.RSA_Float64 f -> Kernel.set_arg_float64 kargs i f)
          args ;

        (* Launch *)
        let stream = Stream.default dev in
        Kernel.launch
          compiled
          ~args:kargs
          ~grid
          ~block
          ~shared_mem
          ~stream:(Some stream)
    | Framework_sig.PTX ->
        failwith
          "CUDA backend does not support PTX loading yet (use CUDA source)"
    | Framework_sig.OpenCL_Source ->
        failwith "CUDA backend does not support OpenCL source"
    | Framework_sig.SPIR_V -> failwith "CUDA backend does not support SPIR-V"
end

(** Auto-register V2 backend when module is loaded *)
let registered_v2 =
  lazy
    (Spoc_core.Log.debug
       Spoc_core.Log.Device
       "Cuda_plugin_v2: checking availability" ;
     if Cuda_v2.is_available () then begin
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_plugin_v2: CUDA available, registering V2 backend" ;
       Framework_registry.register_backend_v2
         ~priority:100
         (module Cuda_v2 : Framework_sig.BACKEND_V2)
     end
     else
       Spoc_core.Log.debug
         Spoc_core.Log.Device
         "Cuda_plugin_v2: CUDA not available")

let () = Lazy.force registered_v2

(** Force module initialization *)
let init () = Lazy.force registered_v2

(** {1 Additional CUDA-specific Functions} *)

(** Register a custom CUDA intrinsic *)
let register_intrinsic = Cuda_intrinsics.register

(** Look up a CUDA intrinsic *)
let find_intrinsic = Cuda_intrinsics.find

(** Generate CUDA source with custom types *)
let generate_with_types = Sarek_ir_cuda.generate_with_types

(** Generate CUDA source for a kernel *)
let generate_source = Sarek_ir_cuda.generate
