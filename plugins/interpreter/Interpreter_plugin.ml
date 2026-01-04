(******************************************************************************
 * Interpreter Plugin Backend Implementation
 *
 * Implements the Framework_sig.BACKEND interface for CPU interpretation.
 * Extends Interpreter_plugin with:
 * - Execution model discrimination (Direct with Custom behavior)
 * - Direct execution of IR via interpretation
 * - Intrinsic registry support
 *
 * Unlike Native backend which calls pre-compiled functions, the interpreter
 * actually interprets the IR at runtime using Sarek_ir_interp.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** Reuse the existing Interpreter implementation *)
module Interpreter_base = struct
  include Interpreter_plugin_base.Interpreter
end

(** Interpreter-specific intrinsic implementation *)
type interp_intrinsic = {
  intr_name : string;
  intr_eval : Sarek.Sarek_ir_interp.value list -> Sarek.Sarek_ir_interp.value;
  intr_convergence : Framework_sig.convergence;
}

(** Intrinsic registry for Interpreter-specific intrinsics *)
module Interpreter_intrinsics : Framework_sig.INTRINSIC_REGISTRY = struct
  type intrinsic_impl = interp_intrinsic

  let table : (string, intrinsic_impl) Hashtbl.t = Hashtbl.create 64

  let register name impl = Hashtbl.replace table name impl

  let find name = Hashtbl.find_opt table name

  let list_all () =
    Hashtbl.fold (fun name _ acc -> name :: acc) table [] |> List.sort compare

  (* Register standard intrinsics *)
  let () =
    register
      "thread_id_x"
      {
        intr_name = "thread_id_x";
        intr_eval = (fun _ -> Sarek.Sarek_ir_interp.VInt32 0l);
        intr_convergence = Framework_sig.Divergent;
      } ;
    register
      "block_barrier"
      {
        intr_name = "block_barrier";
        intr_eval = (fun _ -> Sarek.Sarek_ir_interp.VUnit);
        intr_convergence = Framework_sig.Sync;
      }
end

(** Interpreter Backend - implements BACKEND *)
module Backend : Framework_sig.BACKEND = struct
  include Interpreter_base

  (** Execution model: Custom (interprets IR directly) *)
  let execution_model = Framework_sig.Custom

  (** Generate source - not used for Interpreter (returns None) *)
  let generate_source (_ir : Sarek_ir_types.kernel) : string option = None

  (** Execute directly by interpreting the IR. Interpreter always interprets,
      ignoring native_fn (use Native backend for compiled execution). Uses
      parallel or sequential mode based on current device. *)
  let execute_direct
      ~(native_fn :
         (block:Framework_sig.dims ->
         grid:Framework_sig.dims ->
         Obj.t array ->
         unit)
         option) ~(ir : Obj.t option) ~(block : Framework_sig.dims)
      ~(grid : Framework_sig.dims) (args : Obj.t array) : unit =
    ignore native_fn ;
    (* Interpreter ignores native_fn - always interprets *)
    (* Determine parallel mode based on current device *)
    let use_parallel =
      match !Device.current with
      | Some d -> Device.is_parallel d
      | None -> false
    in
    match ir with
    | Some ir_obj ->
        let kernel : Sarek.Sarek_ir.kernel = Obj.obj ir_obj in
        Sarek.Sarek_ir_interp.parallel_mode := use_parallel ;
        (* Interpreter handles vectors directly *)
        Sarek.Sarek_ir_interp.run_kernel_with_obj_args
          kernel
          ~block:(block.x, block.y, block.z)
          ~grid:(grid.x, grid.y, grid.z)
          args
    | None ->
        failwith
          "Interpreter backend execute_direct: IR required for interpretation"

  module Intrinsics = Interpreter_intrinsics

  (** {2 External Kernel Support} *)

  (** Interpreter backend does not support external GPU sources *)
  let supported_source_langs = []

  (** External kernel execution not supported on Interpreter backend *)
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
         "Interpreter backend cannot execute external %s kernels. Use CUDA or \
          OpenCL backend instead."
         lang_str)
end

(** Run IR directly without going through execute_direct. *)
let run_ir ~(ir : Sarek.Sarek_ir.kernel) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims)
    ~(args : (string * Sarek.Sarek_ir_interp.arg) list) : unit =
  Sarek.Sarek_ir_interp.run_kernel
    ir
    ~block:(block.x, block.y, block.z)
    ~grid:(grid.x, grid.y, grid.z)
    args

(** Auto-register backend when module is loaded *)
let registered_backend =
  lazy
    (if Backend.is_available () then
       Framework_registry.register_backend
         ~priority:5
         (module Backend : Framework_sig.BACKEND))

let () = Lazy.force registered_backend

let init () = Lazy.force registered_backend

(** Register a custom interpreter intrinsic *)
let register_intrinsic = Interpreter_intrinsics.register

(** Look up an interpreter intrinsic *)
let find_intrinsic = Interpreter_intrinsics.find
