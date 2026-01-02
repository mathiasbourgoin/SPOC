(******************************************************************************
 * Interpreter Plugin V2 - Phase 4 Backend Implementation
 *
 * Implements the Framework_sig.BACKEND_V2 interface for CPU interpretation.
 * Extends Interpreter_plugin with:
 * - Execution model discrimination (Direct with Custom behavior)
 * - Direct execution of IR via interpretation
 * - Intrinsic registry support
 *
 * Unlike Native_v2 which calls pre-compiled functions, Interpreter_v2
 * actually interprets the IR at runtime using Sarek_ir_interp.
 ******************************************************************************)

open Sarek_framework

(** Reuse the existing Interpreter implementation *)
module Interpreter_base = struct
  include Interpreter_plugin.Interpreter
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

(** Interpreter V2 Backend - implements BACKEND_V2 *)
module Interpreter_v2 : Framework_sig.BACKEND_V2 = struct
  include Interpreter_base

  (** Execution model: Custom (interprets IR directly) *)
  let execution_model = Framework_sig.Custom

  (** Generate source - not used for Interpreter (returns None) *)
  let generate_source (_ir_obj : Obj.t) : string option = None

  (** Execute directly by interpreting the IR. Interpreter prefers IR but can
      use native_fn if provided. Uses parallel or sequential mode based on the
      current device selection. *)
  let execute_direct
      ~(native_fn :
         (block:Framework_sig.dims ->
         grid:Framework_sig.dims ->
         Obj.t array ->
         unit)
         option) ~(ir : Obj.t option) ~(block : Framework_sig.dims)
      ~(grid : Framework_sig.dims) (args : Obj.t array) : unit =
    (* Set parallel mode based on current device *)
    let use_parallel =
      match !Device.current with
      | Some d -> Device.is_parallel d
      | None -> false
    in
    Sarek.Sarek_ir_interp.parallel_mode := use_parallel ;
    (* Interpreter prefers IR for true interpretation, falls back to native_fn *)
    match ir with
    | Some ir_obj ->
        let kernel : Sarek.Sarek_ir.kernel = Obj.obj ir_obj in
        (* Use run_kernel_with_buffers for proper buffer handling *)
        Sarek.Sarek_ir_interp.run_kernel_with_buffers
          kernel
          ~block:(block.x, block.y, block.z)
          ~grid:(grid.x, grid.y, grid.z)
          args
    | None -> (
        match native_fn with
        | Some fn -> fn ~block ~grid args
        | None ->
            failwith
              "Interpreter_v2.execute_direct: no ir or native_fn provided")

  module Intrinsics = Interpreter_intrinsics
end

(** Run IR directly without going through execute_direct. This is the preferred
    way to use the interpreter V2. *)
let run_ir ~(ir : Sarek.Sarek_ir.kernel) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims)
    ~(args : (string * Sarek.Sarek_ir_interp.arg) list) : unit =
  Sarek.Sarek_ir_interp.run_kernel
    ir
    ~block:(block.x, block.y, block.z)
    ~grid:(grid.x, grid.y, grid.z)
    args

(** Auto-register V2 backend when module is loaded *)
let registered_v2 =
  lazy
    (if Interpreter_v2.is_available () then
       Framework_registry.register_backend_v2
         ~priority:5
         (module Interpreter_v2 : Framework_sig.BACKEND_V2))

let () = Lazy.force registered_v2

let init () = Lazy.force registered_v2

(** Register a custom interpreter intrinsic *)
let register_intrinsic = Interpreter_intrinsics.register

(** Look up an interpreter intrinsic *)
let find_intrinsic = Interpreter_intrinsics.find
