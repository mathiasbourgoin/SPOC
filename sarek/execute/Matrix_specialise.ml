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

let accepted_modes = "off/0/false/no, exact, auto/on/1/true/yes"

let parse_mode = function
  | None -> (Auto, None)
  | Some raw -> (
      match String.lowercase_ascii (String.trim raw) with
      | "off" | "0" | "false" | "no" -> (Off, None)
      | "exact" -> (Exact, None)
      | "auto" | "on" | "1" | "true" | "yes" | "" -> (Auto, None)
      | _ ->
          ( Auto,
            Some
              (Printf.sprintf
                 "[sarek matrix] SAREK_MATRIX_REWRITE=%S is not recognised \
                  (accepted: %s); using auto"
                 raw
                 accepted_modes) ))

let mode_of_env () =
  let mode, warning = parse_mode (Sys.getenv_opt "SAREK_MATRIX_REWRITE") in
  Option.iter (Printf.eprintf "%s\n%!") warning ;
  mode

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

let build k g lowering ~subgroup ~frag = function
  | Staged -> Rw.kernel ~subgroup ~frag k g lowering
  | Direct waves -> Direct.kernel k g ~waves

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

(* The automatic path (not [explicit]) rewrites a program's own float kernels
   only after its own opt-in; the accuracy opt-ins are checked with the
   configuration ([device_admits]) and by [float32_lowering]. *)
let automatic_float_optin ~explicit =
  if explicit || Sarek_coopmat.automatic_float_rewrite_reason () <> None then
    Ok ()
  else
    Error
      "float products are rewritten automatically only after \
       Sarek_coopmat.accept_automatic_float_rewrite (plus the accuracy \
       opt-ins); Sarek_matrix.run does not need it"

let lowering_for ~mode ~explicit case precision =
  match (mode, case) with
  | (Off | Exact | Auto), Adv.Int32_inputs -> Ok Rw.Int32_byte_planes
  | (Off | Exact), (Adv.F16_inputs | Adv.F32_inputs) ->
      Error
        (Printf.sprintf
           "mode %s: float products are not rewritten"
           (mode_name mode))
  | Auto, Adv.F16_inputs ->
      let* () = automatic_float_optin ~explicit in
      Ok Rw.F16_operands
  | Auto, Adv.F32_inputs ->
      let* () = automatic_float_optin ~explicit in
      float32_lowering precision
  | (Off | Exact | Auto), Adv.No_case ->
      Error "no matrix-unit configuration for these element types"

(* A policy refusal (an accuracy opt-in not given) carries its remedy, which
   names the call to make; a device refusal has none worth quoting. *)
let verdict_ok = function
  | Sarek_capability.Available -> Ok ()
  | Sarek_capability.Unavailable
      {cap_kind = Sarek_capability.Policy; cap_why; cap_remedy = Some remedy; _}
    ->
      Error (cap_why ^ ". " ^ remedy)
  | Sarek_capability.Unavailable cap -> Error cap.Sarek_capability.cap_why
  | Sarek_capability.Unknown why -> Error why

(* The frameworks the rewrite is lowered on, and what each one lowers: HIP gfx11
   (32-wide waves, 16x16x16 fragments) every lowering; Vulkan (RADV, 64-wide
   subgroups) the integer byte planes only, because GLSL lowers no float
   cooperative-matrix component; CUDA/C (nvcuda::wmma, 32-wide warps) the
   integer byte planes only, because its bf16-from-float32 arm is an #error and
   its f16 path is unmeasured, and only on an explicit request, because the
   lowering has compiler-output evidence and has never been executed; Metal
   (simdgroup_matrix, 32-wide SIMD-groups, 8x8x8 fragments) the float16
   operands only, because MSL has no integer simdgroup_matrix and its bfloat
   fragment is unmeasured; the interpreter every lowering (also on request
   only). *)
let rewrite_backends = ["HIP"; "Vulkan"; "CUDA/C"; "Metal"; "Interpreter"]

(* Why [lowering] cannot be lowered on [device], or [None]. *)
let lowering_refusal (device : Device.t) lowering =
  match (device.framework, lowering) with
  | ("Vulkan" | "CUDA/C"), Rw.Int32_byte_planes -> None
  | ("Vulkan" | "CUDA/C"), _ ->
      Some
        (Printf.sprintf
           "%s lowers only the integer byte planes, not %s"
           device.framework
           (Rw.lowering_name lowering))
  | "Metal", Rw.F16_operands -> None
  | "Metal", Rw.Int32_byte_planes ->
      Some
        "Metal has no integer simdgroup_matrix (MSL offers half, float and \
         bfloat elements only), so the integer byte-plane rewrite cannot be \
         lowered there"
  | "Metal", _ ->
      Some
        (Printf.sprintf
           "Metal lowers only the f16 operands (f16*f16+f32->f32), not %s: its \
            bfloat fragment is unmeasured"
           (Rw.lowering_name lowering))
  | _ -> None

let subgroup_override : int option Atomic.t = Atomic.make None

let subgroup_override_for_tests w = Atomic.set subgroup_override w

(* Lanes per subgroup the templates are laid out for: the device's own width
   (32 on HIP, 64 on RADV), 32 on the interpreter, whose cooperative-matrix
   subgroup is 32 lanes. *)
let subgroup_of (device : Device.t) =
  match (Atomic.get subgroup_override, device.framework) with
  | Some w, "Interpreter" -> w
  | _ -> (
      match (device.framework, device.capabilities.Fw.coopmat) with
      | "Interpreter", _ | _, None -> 32
      | _, Some s -> s.Sarek_coopmat.ds_subgroup_size)

let fragment_override : int option Atomic.t = Atomic.make None

let fragment_override_for_tests e = Atomic.set fragment_override e

(* The fragment edge the templates are laid out for: the largest of
   {!Sarek_matrix_scheme.fragment_edges} whose configuration the device
   advertises (16 on HIP, Vulkan and CUDA, 8 on Metal), or 16 when none is, so
   that the capability verdict that follows names the 16x16x16 configuration
   the device lacks. The interpreter holds whole matrices: 16 unless a test
   overrides it. *)
let fragment_of (device : Device.t) lowering ~operand_elt =
  let advertised support edge =
    Sarek_capability.permits
      (Sarek_coopmat.verdict
         ~support
         (Rw.config_of ~frag:edge lowering ~operand_elt))
  in
  match (Atomic.get fragment_override, device.framework) with
  | Some e, "Interpreter" -> e
  | _, "Interpreter" -> 16
  | _ -> (
      let support = device.capabilities.Fw.coopmat in
      match List.find_opt (advertised support) fragment_edges with
      | Some edge -> edge
      | None -> 16)

(* Vulkan is rewritten automatically only at the width that was measured: 64
   (RADV). The probe falls back to 32 when the driver reports nothing, so 64 is
   always a probed value; a 32-wide Vulkan device is rewritten on request
   only, until measured. *)
let vulkan_measured_width = 64

let vulkan_width_ok (device : Device.t) ~explicit =
  if
    device.framework <> "Vulkan"
    || explicit
    || subgroup_of device = vulkan_measured_width
  then Ok ()
  else
    Error
      (Printf.sprintf
         "Vulkan is rewritten automatically on %d-wide subgroups only (the \
          measured width), not %d; Sarek_matrix.run rewrites it on request"
         vulkan_measured_width
         (subgroup_of device))

(* The device's own facts: the configuration is advertised, the vulkan width
   was measured, and the templates cover the subgroup width. *)
let device_offers (device : Device.t) ~explicit cfg =
  let support = device.capabilities.Fw.coopmat in
  let* () = verdict_ok (Sarek_coopmat.verdict ~support cfg) in
  let* () = vulkan_width_ok device ~explicit in
  if supported_subgroup (subgroup_of device) then Ok ()
  else
    Error
      (Printf.sprintf
         "the rewrite has no template for %d-wide subgroups"
         (subgroup_of device))

let device_admits (device : Device.t) ~explicit lowering cfg =
  let* () = verdict_ok (Sarek_coopmat.policy_verdict cfg) in
  if not (List.mem device.framework rewrite_backends) then
    Error
      (Printf.sprintf
         "the rewrite is lowered and measured on %s only, not %s"
         (String.concat ", " rewrite_backends)
         device.framework)
  else
    match (lowering_refusal device lowering, device.framework) with
    | Some why, _ -> Error why
    | None, "Interpreter" -> Ok ()
    | None, _ -> device_offers device ~explicit cfg

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
  let* lowering = lowering_for ~mode ~explicit plan.case precision in
  let operand_elt = plan.gemm.a.elt in
  let frag = fragment_of device lowering ~operand_elt in
  let cfg = Rw.config_of ~frag lowering ~operand_elt in
  let* () = device_admits device ~explicit lowering cfg in
  let subgroup = subgroup_of device in
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
    | exception ((Out_of_memory | Sys.Break) as e) -> raise e
    | exception e ->
        (* The rewrite builders and the region evaluation run on the default
           launch path too: a failure there must not fail the launch. *)
        Original ("matrix rewrite failed: " ^ Printexc.to_string e)
  in
  (match outcome with
  | Original _ when not (reportable device ~explicit ir) -> ()
  | _ -> report ~kernel:ir.kern_name ~device:device.name outcome) ;
  outcome

(* A cleared backend cache (an intrinsic or type replaced) must not keep
   rewritten kernels built against the old one. *)
let () = Spoc_framework.Cache_hooks.on_clear_all clear
