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
module Adm = Matrix_admission

let ( let* ) = Result.bind

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
  evidence : Sarek_policy.evidence;
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

type plan = {gemm : P.gemm; mapping : Map.mapping; case : Adv.element_case}

type entry = {
  generation : int;  (** the {!generation} it belongs to *)
  plan : (plan, string) result;
  mutable kernels :
    ((Rw.lowering * variant * int * int) * (kernel, string) result) list;
      (** keyed by lowering, variant, subgroup width and fragment edge *)
}

(* A slot of each IR's Ir_memo record; an entry of an older generation is
   stale ({!clear}). Guarded by [memo_lock]. *)
let memo_key : entry Ir_memo.key = Ir_memo.new_key ()

let generation = ref 0

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

(* Test hook: make the analysis raise, as a defect in it would. *)
let fail_analysis = Atomic.make false

let fail_analysis_for_tests b = Atomic.set fail_analysis b

(* The analysis runs on the default launch path: a failure in it is cached as
   the plan's error and never fails a launch. Out_of_memory and Sys.Break
   are not analysis failures and propagate. *)
let guarded_plan k =
  match
    if Atomic.get fail_analysis then failwith "injected analysis failure"
    else static_plan k
  with
  | p -> p
  | exception ((Out_of_memory | Sys.Break) as e) -> raise e
  | exception e -> Error ("matrix analysis failed: " ^ Printexc.to_string e)

(* [Sarek_ir_analysis.kernel_uses_warp_ops] is a whole-IR scan and a pure
   function of the immutable IR: recorded once per IR in its own slot, outside
   the generation-scoped [entry] (a cache clear must not repeat it) and ahead of
   the matrix analysis (a warp kernel is refused without being analysed). *)
let warp_ops_key : bool Ir_memo.key = Ir_memo.new_key ()

let uses_warp_ops (ir : kernel) : bool =
  let facts = Ir_memo.facts ir in
  match Ir_memo.find facts warp_ops_key with
  | Some verdict -> verdict
  | None ->
      let verdict = Sarek_ir_analysis.kernel_uses_warp_ops ir in
      Ir_memo.set facts warp_ops_key verdict ;
      verdict

let entry k =
  let facts = Ir_memo.facts k in
  Mutex.protect memo_lock (fun () ->
      match Ir_memo.find facts memo_key with
      | Some e when e.generation = !generation -> e
      | Some _ | None ->
          let e =
            {generation = !generation; plan = guarded_plan k; kernels = []}
          in
          Ir_memo.set facts memo_key e ;
          e)

(* The rewritten kernel is renamed (<name>_sarek_mm): it has the budget of the
   kernel it came from, for the capability gate and the interpreter. *)
let build k g lowering ~subgroup ~frag variant =
  let built =
    match variant with
    | Staged -> Rw.kernel ~subgroup ~frag k g lowering
    | Direct waves -> Direct.kernel k g ~waves
  in
  Result.iter
    (fun (rk : kernel) ->
      Sarek_policy.register_derived ~name:rk.kern_name ~from:k.kern_name)
    built ;
  built

let kernel_for e k g lowering variant ~subgroup ~frag =
  let key = (lowering, variant, subgroup, frag) in
  let cached () =
    Mutex.protect memo_lock (fun () -> List.assoc_opt key e.kernels)
  in
  match cached () with
  | Some r -> r
  | None ->
      (* Built outside the lock: two domains may build the same kernel once
         each; the first stored wins, so the IR stays physically stable. *)
      let r = build k g lowering ~subgroup ~frag variant in
      Mutex.protect memo_lock (fun () ->
          match List.assoc_opt key e.kernels with
          | Some first -> first
          | None ->
              e.kernels <- (key, r) :: e.kernels ;
              r)

let clear () = Mutex.protect memo_lock (fun () -> incr generation)

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

let float32_lowering ~budget precision =
  let accepted p =
    Sarek_policy.Budget.permits
      budget
      (Sarek_policy.Deviation (Sarek_policy.Operand_rounding p))
  in
  match precision with
  | Some p when accepted p -> Ok (precision_lowering p)
  | Some p
    when Sarek_policy.revoked
           (Sarek_policy.Deviation (Sarek_policy.Operand_rounding p)) ->
      Error
        (Printf.sprintf
           "operand rounding to %s is revoked by SAREK_OPT_DISABLE"
           (Sarek_coopmat.operand_precision_name p))
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

(* The automatic path (not [explicit]) rewrites a program's own float kernels
   only after its own opt-in; the accuracy opt-ins are checked with the
   configuration ([device_admits]) and by [float32_lowering]. *)
let automatic_float_optin ~budget ~explicit =
  let consent = Sarek_policy.Consent Sarek_policy.Automatic_float_rewrite in
  if explicit || Sarek_policy.Budget.permits budget consent then Ok ()
  else if Sarek_policy.revoked consent then
    Error
      "float products are not rewritten automatically: automatic-float-rewrite \
       is revoked (SAREK_OPT_DISABLE); Sarek_matrix.run does not need it"
  else
    Error
      "float products are rewritten automatically only after \
       Sarek_coopmat.accept_automatic_float_rewrite (plus the accuracy \
       opt-ins); Sarek_matrix.run does not need it"

let lowering_for ~budget ~explicit case precision =
  match case with
  | Adv.Int32_inputs -> Ok Rw.Int32_byte_planes
  | Adv.F16_inputs ->
      let* () = automatic_float_optin ~budget ~explicit in
      Ok Rw.F16_operands
  | Adv.F32_inputs ->
      let* () = automatic_float_optin ~budget ~explicit in
      float32_lowering ~budget precision
  | Adv.No_case -> Error "no matrix-unit configuration for these element types"

(* What justifies a rewrite, for the explain report: int32 is bit-identical;
   a float one rests on the grants it was admitted under (the consent only for
   the automatic path), with their reasons. *)
let evidence_for ~budget ~explicit ~case lowering variant cfg =
  let description =
    Printf.sprintf
      "%s, %s, %s"
      (Rw.lowering_name lowering)
      (variant_name variant)
      (Sarek_coopmat.config_name cfg)
  in
  match lowering with
  | Rw.Int32_byte_planes ->
      Sarek_policy.Proof (description ^ ": bit-identical modulo 2^32")
  | Rw.F16_operands | Rw.Bf16_operands | Rw.Bf16x3_operands ->
      let rounding p =
        Sarek_policy.Deviation (Sarek_policy.Operand_rounding p)
      in
      let grants =
        (if explicit then []
         else [Sarek_policy.Consent Sarek_policy.Automatic_float_rewrite])
        @ [Sarek_policy.Deviation Sarek_policy.Accumulation_order]
        @
        match (case, lowering) with
        | _, Rw.Bf16_operands -> [rounding Sarek_policy.Operands_bf16]
        | _, Rw.Bf16x3_operands -> [rounding Sarek_policy.Operands_bf16x3]
        | Adv.F32_inputs, _ -> [rounding Sarek_policy.Operands_f16]
        | _ -> []
      in
      let reasons =
        List.filter_map (Sarek_policy.Budget.reason budget) grants
      in
      Sarek_policy.Granted
        {
          grants;
          reason = Sarek_policy.join_reasons reasons;
          lowering = Some description;
        }

let subgroup_override_for_tests = Adm.subgroup_override_for_tests

let fragment_override_for_tests = Adm.fragment_override_for_tests

(* ---- One launch ---------------------------------------------------------- *)

(* The rewritten launch against the device's limits: grid dimensions, threads
   per block and shared-memory bytes (a limit of 0 is "unknown"). *)
let fits_device (device : Device.t) ~shared_bytes (bx, by, bz) (gx, gy, gz) =
  let caps = device.capabilities in
  let threads = bx * by * bz in
  let mx, my, mz = caps.Fw.max_grid_dims in
  let over g m = m > 0 && g > m in
  if
    caps.Fw.max_threads_per_block > 0 && threads > caps.Fw.max_threads_per_block
  then
    Error
      (Printf.sprintf
         "the rewritten block of %d threads exceeds the device's %d"
         threads
         caps.Fw.max_threads_per_block)
  else if
    caps.Fw.shared_mem_per_block > 0
    && shared_bytes > caps.Fw.shared_mem_per_block
  then
    Error
      (Printf.sprintf
         "the rewritten kernel needs %d bytes of shared memory, the device has \
          %d"
         shared_bytes
         caps.Fw.shared_mem_per_block)
  else if over gx mx || over gy my || over gz mz then
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
   global memory (laid out for 32-wide waves of 16x16x16 fragments only, so any
   other width or fragment stages); everything else is staged. *)
let variant_for plan lowering ~subgroup ~frag
    (launch : Sarek_ir_eval_int.launch) (r : Map.region) =
  let g = plan.gemm in
  let k_extent =
    Sarek_ir_eval_int.eval
      launch
      (EBinop (Add, EBinop (Sub, g.k_hi, g.k_lo), EConst (CInt32 1l)))
  in
  match k_extent with
  | Some k
    when lowering = Rw.F16_operands && subgroup = 32 && frag = 16
         && Direct.eligible g
         && Direct.fits ~me:r.me ~ne:r.ne ~k_extent:(Int32.to_int k) ~d:r.d ->
      let waves = Direct.waves_for ~me:r.me ~ne:r.ne in
      (Direct waves, Direct.launch_dims ~me:r.me ~ne:r.ne ~ze:r.ze ~waves)
  | _ -> (Staged, Rw.launch_dims ~subgroup ~me:r.me ~ne:r.ne ~ze:r.ze ())

(* Before any analysis: a device the rewrite can never use costs nothing.
   The interpreter is rewritten only on an explicit request: by default it
   stays the independent oracle (and the rewrite is slower there). *)
let device_possible (device : Device.t) ~explicit ~stream =
  match device.framework with
  | "HIP" | "Vulkan" | "Metal" -> Ok ()
  | "CUDA/C" when explicit -> Ok ()
  | "CUDA/C" ->
      Error
        "CUDA/C is rewritten only on request (Sarek_matrix.run): the lowering \
         is compiler-output evidence only, never executed"
  | "Interpreter" when explicit && not stream -> Ok ()
  | "Interpreter" when explicit -> Error "a stream launch on the interpreter"
  | "Interpreter" ->
      Error "the interpreter is rewritten only on request (Sarek_matrix.run)"
  | fw -> Error (Printf.sprintf "no matrix-unit rewrite on %s" fw)

let decide ?precision ~explicit ~stream ~(device : Device.t) ~ir ~launch () =
  let* () =
    if Sarek_policy.restricted Sarek_policy.Matrix_rewrite then
      Error Sarek_policy.disabled_reason
    else Ok ()
  in
  let* () = device_possible device ~explicit ~stream in
  let* () =
    (* The rewrite chooses its own geometry; a warp shuffle is defined over the
       warp the kernel was written for. *)
    if uses_warp_ops ir then
      Error "the kernel uses a warp operation (shuffle, warp_size or lane_id)"
    else Ok ()
  in
  let e = entry ir in
  let* plan = e.plan in
  let* () =
    if uses_launch_dims plan.gemm then
      Error
        "a bound, the initial value or the epilogue reads block_dim/grid_dim"
    else Ok ()
  in
  (* The grants of this kernel: a Kernel-scoped opt-in is read for it only. *)
  let budget = Sarek_policy.budget ~kernels:[ir.kern_name] in
  let* lowering = lowering_for ~budget ~explicit plan.case precision in
  let operand_elt = plan.gemm.a.elt in
  let frag = Adm.fragment_of device lowering ~operand_elt in
  let cfg = Rw.config_of ~frag lowering ~operand_elt in
  let* () = Adm.device_admits ~budget device ~explicit lowering cfg in
  let subgroup = Adm.subgroup_of device in
  let* launch =
    Option.to_result ~none:"the arguments do not match the kernel" (launch ())
  in
  let* r = Map.region plan.mapping launch in
  let variant, (block, grid) =
    variant_for plan lowering ~subgroup ~frag launch r
  in
  let shared_bytes =
    match variant with
    | Direct _ -> 0
    | Staged ->
        let s = scheme ~operand_elt:plan.gemm.a.elt lowering in
        shared_bytes s ~bk:(choose_bk s)
  in
  let* () = fits_device device ~shared_bytes block grid in
  let* rk = kernel_for e ir plan.gemm lowering variant ~subgroup ~frag in
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
      evidence =
        evidence_for ~budget ~explicit ~case:plan.case lowering variant cfg;
    }

(* ---- Report ---------------------------------------------------------------- *)

(* The outcome as a policy decision, for the explain report. *)
let policy_decision = function
  | Specialised s -> Sarek_policy.Applied s.evidence
  | Original why -> Sarek_policy.Refused why

let report ~kernel ~target outcome =
  Sarek_policy.report
    {
      kernel;
      target;
      optimisation = Sarek_policy.Matrix_rewrite;
      decision = policy_decision outcome;
    }

(* A kernel with no matrix product, and a device that can never rewrite, are
   the common cases: not reported. *)
let reportable (device : Device.t) ~explicit ir =
  Sarek_policy.reporting ()
  && Result.is_ok (device_possible device ~explicit ~stream:false)
  && Result.is_ok (entry ir).plan

let choose ?precision ?(explicit = false) ?(stream = false) ~(device : Device.t)
    ~ir ~launch () =
  let outcome =
    match decide ?precision ~explicit ~stream ~device ~ir ~launch () with
    | Ok s -> Specialised s
    | Error why -> Original why
    | exception ((Out_of_memory | Sys.Break) as e) -> raise e
    | exception e ->
        (* The rewrite builders and the region evaluation run on the default
           launch path too: a failure there must not fail the launch. *)
        Original ("matrix rewrite failed: " ^ Printexc.to_string e)
  in
  (match outcome with
  | Original _ when not (reportable device ~explicit ir) -> ()
  | _ ->
      report ~kernel:ir.kern_name ~target:(Launch_policy.target device) outcome) ;
  outcome

(* A cleared backend cache (an intrinsic or type replaced) must not keep
   rewritten kernels built against the old one. *)
let () = Spoc_framework.Cache_hooks.on_clear_all clear
