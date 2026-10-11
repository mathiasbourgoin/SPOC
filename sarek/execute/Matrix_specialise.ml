(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Launch-time specialisation of matrix products (specs/matrix-pattern-dynamic.md):
   Execute asks, per launch, whether to run a rewritten kernel instead of the
   one it was given. See the interface. *)

open Sarek_ir_types
module P = Sarek_matrix_pattern
module Adv = Sarek_matrix_advice
module Map = Sarek_matrix_mapping
module Rw = Sarek_matrix_rewrite
module Direct = Sarek_matrix_direct
open Sarek_matrix_scheme
module Device = Spoc_core.Device
module Fw = Spoc_framework.Framework_sig

let ( let* ) = Result.bind

(* ---- Mode ------------------------------------------------------------------ *)

type mode = Off | Exact | Auto

let mode_name = function Off -> "off" | Exact -> "exact" | Auto -> "auto"

let mode_of_env () =
  match Sys.getenv_opt "SAREK_MATRIX_REWRITE" with
  | Some "off" -> Off
  | Some "exact" -> Exact
  | _ -> Auto

let current_mode = Atomic.make (mode_of_env ())

let set_mode m = Atomic.set current_mode m

let mode () = Atomic.get current_mode

(* ---- Outcome ------------------------------------------------------------ *)

type variant = Staged | Direct of int

type specialised = {
  ir : kernel;
  lowering : Rw.lowering;
  variant : variant;
  config : Sarek_coopmat.config;
  extra : int list;
  block : int * int * int;
  grid : int * int * int;
  via_interpreter : bool;
}

type outcome = Specialised of specialised | Original of string

let variant_name = function
  | Staged -> "staged"
  | Direct w -> Printf.sprintf "direct (%d waves/block)" w

let outcome_text = function
  | Specialised s ->
      Printf.sprintf
        "rewritten: %s, %s, %s"
        (Rw.lowering_name s.lowering)
        (variant_name s.variant)
        (Sarek_coopmat.config_name s.config)
  | Original why -> "original kernel: " ^ why

(* ---- Memos, by the IR's physical identity --------------------------------- *)

module Physical = struct
  type t = kernel

  let equal = ( == )

  let hash (k : t) = Hashtbl.hash (k.kern_name, k.kern_params)
end

module Memo = Ephemeron.K1.Make (Physical)

type plan = {gemm : P.gemm; mapping : Map.mapping; case : Adv.element_case}

type entry = {
  plan : (plan, string) result;
  mutable kernels : ((Rw.lowering * variant) * (kernel, string) result) list;
}

let memo : entry Memo.t = Memo.create 64

let memo_lock = Mutex.create ()

let only_write_is (k : kernel) arr =
  let fs acc = function
    | SAssign (LArrayElem (a, _), _) -> `Array a :: acc
    | SAssign (LArrayElemExpr _, _) | SExpr _ | SReturn _ | SCoopmat _ ->
        `Other :: acc
    | _ -> acc
  in
  let writes =
    Sarek_ir_analysis.stmt_fold
      {
        fe = (fun acc _ -> acc);
        ft = (fun acc _ -> acc);
        fs;
        fnative = (fun acc -> `Other :: acc);
        visit_lvalue = true;
      }
      []
      k.kern_body
  in
  writes = [`Array arr] && not (Sarek_ir_analysis.kernel_uses_atomics k)

let static_plan (k : kernel) =
  match P.analyze k with
  | [P.Gemm g] ->
      if not (only_write_is k g.c.array) then
        Error "the kernel does more than the matrix product"
      else
        let* mapping = Map.analyse k g in
        Ok {gemm = g; mapping; case = Adv.element_case g}
  | [] -> Error "no matrix product recognised"
  | [P.Near_miss nm] ->
      Error ("not a matrix product: " ^ P.reason_text nm.nm_reason)
  | _ -> Error "more than one reduction loop"

let entry k =
  Mutex.protect memo_lock (fun () ->
      match Memo.find_opt memo k with
      | Some e -> e
      | None ->
          let e = {plan = static_plan k; kernels = []} in
          Memo.replace memo k e ;
          e)

let build k g lowering = function
  | Staged -> Rw.kernel k g lowering
  | Direct waves -> Direct.kernel k g ~waves

let kernel_for e k g lowering variant =
  let cached () =
    Mutex.protect memo_lock (fun () ->
        List.assoc_opt (lowering, variant) e.kernels)
  in
  match cached () with
  | Some r -> r
  | None ->
      (* Built outside the lock: two domains may build the same kernel once
         each; the first stored wins, so the IR stays physically stable. *)
      let r = build k g lowering variant in
      Mutex.protect memo_lock (fun () ->
          match List.assoc_opt (lowering, variant) e.kernels with
          | Some first -> first
          | None ->
              e.kernels <- ((lowering, variant), r) :: e.kernels ;
              r)

let clear () = Mutex.protect memo_lock (fun () -> Memo.reset memo)

(* ---- Policy --------------------------------------------------------------- *)

let precision_lowering = function
  | Sarek_coopmat.Operands_f16 -> Rw.F16_operands
  | Sarek_coopmat.Operands_bf16 -> Rw.Bf16_operands
  | Sarek_coopmat.Operands_bf16x3 -> Rw.Bf16x3_operands

(* Without an explicit precision: float16, then bfloat16, whichever the
   program accepted. Both were measured faster than the scalar kernel at every
   size benchmarked; the bf16x3 split was not (it buys accuracy, not speed),
   so it is used only when asked for by [~precision]. *)
let preferred = Sarek_coopmat.[Operands_f16; Operands_bf16]

let float32_lowering precision =
  let accepted p = Sarek_coopmat.operand_rounding_reason p <> None in
  match precision with
  | Some p when accepted p -> Ok (precision_lowering p)
  | Some p ->
      Error
        (Printf.sprintf
           "operand rounding to %s was not accepted \
            (Sarek_coopmat.accept_operand_rounding)"
           (Sarek_coopmat.operand_precision_name p))
  | None -> (
      match List.find_opt accepted preferred with
      | Some p -> Ok (precision_lowering p)
      | None ->
          Error
            "a float32 product is rewritten only after \
             Sarek_coopmat.accept_operand_rounding")

let lowering_of ~mode plan precision =
  match (mode, plan.case) with
  | (Off | Exact | Auto), Adv.Int32_inputs -> Ok Rw.Int32_byte_planes
  | (Off | Exact), (Adv.F16_inputs | Adv.F32_inputs) ->
      Error
        (Printf.sprintf
           "mode %s: float products are not rewritten"
           (mode_name mode))
  | Auto, Adv.F16_inputs -> Ok Rw.F16_operands
  | Auto, Adv.F32_inputs -> float32_lowering precision
  | (Off | Exact | Auto), Adv.No_case ->
      Error "no matrix-unit configuration for these element types"

let verdict_ok = function
  | Sarek_capability.Available -> Ok ()
  | Sarek_capability.Unavailable cap -> Error cap.Sarek_capability.cap_why
  | Sarek_capability.Unknown why -> Error why

(* Laid out for 32-wide waves and executed on HIP gfx11 and the interpreter
   only. Vulkan advertises the integer configuration on the RX 7900 XTX with
   64-wide subgroups (radv), and GLSL lowers no byte-plane store. *)
let rewrite_backends = ["HIP"; "Interpreter"]

let device_admits (device : Device.t) cfg =
  let* () = verdict_ok (Sarek_coopmat.policy_verdict cfg) in
  if not (List.mem device.framework rewrite_backends) then
    Error
      (Printf.sprintf
         "the rewrite is lowered and measured on %s only, not %s"
         (String.concat " and " rewrite_backends)
         device.framework)
  else if device.framework = "Interpreter" then Ok ()
  else
    let support = device.capabilities.Fw.coopmat in
    let* () = verdict_ok (Sarek_coopmat.verdict ~support cfg) in
    match support with
    | Some s when s.Sarek_coopmat.ds_subgroup_size = 32 -> Ok ()
    | _ -> Error "the rewrite needs 32-wide subgroups"

(* ---- One launch ---------------------------------------------------------- *)

let fits_device (device : Device.t) (gx, gy, gz) =
  let mx, my, mz = device.capabilities.Fw.max_grid_dims in
  let over g m = m > 0 && g > m in
  if over gx mx || over gy my || over gz mz then
    Error
      (Printf.sprintf
         "the rewritten grid %dx%dx%d exceeds the device's %dx%dx%d"
         gx
         gy
         gz
         mx
         my
         mz)
  else Ok ()

(* The size class: the aligned float16 case loads fragments straight from
   global memory; everything else is staged. *)
let variant_for plan lowering (launch : Sarek_ir_eval_int.launch)
    (r : Map.region) =
  let g = plan.gemm in
  let k_extent =
    Sarek_ir_eval_int.eval
      launch
      (EBinop (Add, EBinop (Sub, g.k_hi, g.k_lo), EConst (CInt32 1l)))
  in
  match k_extent with
  | Some k
    when lowering = Rw.F16_operands && Direct.eligible g
         && Direct.fits ~me:r.me ~ne:r.ne ~k_extent:(Int32.to_int k) ~d:r.d ->
      let waves = Direct.waves_for ~me:r.me ~ne:r.ne in
      (Direct waves, Direct.launch_dims ~me:r.me ~ne:r.ne ~ze:r.ze ~waves)
  | _ -> (Staged, Rw.launch_dims ~me:r.me ~ne:r.ne ~ze:r.ze)

(* Before any analysis: a device the rewrite can never use costs nothing.
   The interpreter is rewritten only on an explicit request: by default it
   stays the independent oracle (and the rewrite is slower there). *)
let device_possible (device : Device.t) ~explicit ~stream =
  match device.framework with
  | "HIP" -> Ok ()
  | "Interpreter" when explicit && not stream -> Ok ()
  | "Interpreter" when explicit -> Error "a stream launch on the interpreter"
  | "Interpreter" ->
      Error "the interpreter is rewritten only on request (Sarek_matrix.run)"
  | fw -> Error (Printf.sprintf "no matrix-unit rewrite on %s" fw)

let decide ?precision ~explicit ~stream ~(device : Device.t) ~ir ~launch () =
  let mode = mode () in
  let* () = if mode = Off then Error "matrix rewrite switched off" else Ok () in
  let* () = device_possible device ~explicit ~stream in
  let e = entry ir in
  let* plan = e.plan in
  let* () =
    if uses_launch_dims plan.gemm then
      Error
        "a bound, the initial value or the epilogue reads block_dim/grid_dim"
    else Ok ()
  in
  let* lowering = lowering_of ~mode plan precision in
  let cfg = Rw.config_of lowering ~operand_elt:plan.gemm.a.elt in
  let* () = device_admits device cfg in
  let* launch =
    Option.to_result ~none:"the arguments do not match the kernel" (launch ())
  in
  let* r = Map.region plan.mapping launch in
  let variant, (block, grid) = variant_for plan lowering launch r in
  let* () = fits_device device grid in
  let* rk = kernel_for e ir plan.gemm lowering variant in
  Ok
    {
      ir = rk;
      lowering;
      variant;
      config = cfg;
      extra = [r.me; r.ne; r.ze; r.d; r.t];
      block;
      grid;
      via_interpreter = device.framework = "Interpreter";
    }

(* ---- Report ---------------------------------------------------------------- *)

type report = {kernel : string; device : string; outcome : outcome}

let reporter : (report -> unit) option Atomic.t = Atomic.make None

let set_reporter r = Atomic.set reporter r

let env_report = Sys.getenv_opt "SAREK_MATRIX_REPORT" = Some "1"

let reported : (string * string * string, unit) Hashtbl.t = Hashtbl.create 16

let report ~kernel ~device outcome =
  Option.iter (fun f -> f {kernel; device; outcome}) (Atomic.get reporter) ;
  if env_report then
    let text = outcome_text outcome in
    Mutex.protect memo_lock (fun () ->
        if not (Hashtbl.mem reported (kernel, device, text)) then (
          Hashtbl.replace reported (kernel, device, text) () ;
          Printf.eprintf "[sarek matrix] %s on %s: %s\n%!" kernel device text))

(* A kernel with no matrix product, a device that can never rewrite, and mode
   Off are the common cases: not reported. *)
let reportable (device : Device.t) ~explicit ir =
  mode () <> Off
  && Result.is_ok (device_possible device ~explicit ~stream:false)
  && Result.is_ok (entry ir).plan

let choose ?precision ?(explicit = false) ?(stream = false) ~(device : Device.t)
    ~ir ~launch () =
  let outcome =
    match decide ?precision ~explicit ~stream ~device ~ir ~launch () with
    | Ok s -> Specialised s
    | Error why -> Original why
    | exception e ->
        (* The analysis runs on the default launch path: a failure in it must
           never fail a launch that works without it. *)
        Original ("matrix analysis failed: " ^ Printexc.to_string e)
  in
  (match outcome with
  | Original _ when not (reportable device ~explicit ir) -> ()
  | _ -> report ~kernel:ir.kern_name ~device:device.name outcome) ;
  outcome

(* A cleared backend cache (an intrinsic or type replaced) must not keep
   rewritten kernels built against the old one. *)
let () = Spoc_framework.Cache_hooks.on_clear_all clear
