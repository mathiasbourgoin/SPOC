(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Launch a kernel, running a recognised GEMM nest on matrix units when that is
   allowed and possible. specs/matrix-pattern-rewrite.md. *)

open Sarek_ir_types
module P = Sarek_matrix_pattern
module Adv = Sarek_matrix_advice
module Map = Sarek_matrix_mapping
module Rw = Sarek_matrix_rewrite
module Execute = Sarek.Execute

(* The rewritten kernels call Int32.bits_of_float / float_of_bits (the bf16
   split) and the Gpu intrinsics: make sure their registrations are linked. *)
let () = Sarek_stdlib.force_init ()

type outcome =
  | Rewritten of {lowering : Rw.lowering; config : Sarek_coopmat.config}
  | Original of string

let ( let* ) = Result.bind

(* ---- Static plan, once per kernel ----------------------------------------- *)

type plan = {gemm : P.gemm; mapping : Map.mapping; case : Adv.element_case}

let only_write_is (k : kernel) arr =
  let writes =
    Sarek_ir_analysis.stmt_fold
      {
        fe = (fun acc _ -> acc);
        ft = (fun acc _ -> acc);
        fs =
          (fun acc s ->
            match s with
            | SAssign (LArrayElem (a, _), _) -> `Array a :: acc
            | SAssign (LArrayElemExpr _, _) | SExpr _ | SReturn _ | SCoopmat _
              ->
                `Other :: acc
            | _ -> acc);
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

(* Physical-identity memo, as Execute's launch memo: a PPX kernel hands the same
   IR record to every launch. *)
let memo_lock = Mutex.create ()

let plans : (kernel * (plan, string) result) list ref = ref []

let rewritten : (kernel * Rw.lowering * (kernel, string) result) list ref =
  ref []

let memo_plan k =
  Mutex.protect memo_lock (fun () ->
      match List.assq_opt k !plans with
      | Some p -> p
      | None ->
          let p = static_plan k in
          plans := (k, p) :: !plans ;
          p)

let memo_rewrite k g lowering =
  Mutex.protect memo_lock (fun () ->
      match
        List.find_opt (fun (k', l, _) -> k' == k && l = lowering) !rewritten
      with
      | Some (_, _, r) -> r
      | None ->
          let r = Rw.kernel k g lowering in
          rewritten := (k, lowering, r) :: !rewritten ;
          r)

(* ---- Policy and device ------------------------------------------------------ *)

let precision_lowering = function
  | Sarek_coopmat.Operands_f16 -> Rw.F16_operands
  | Sarek_coopmat.Operands_bf16 -> Rw.Bf16_operands
  | Sarek_coopmat.Operands_bf16x3 -> Rw.Bf16x3_operands

let lowering_of plan precision =
  match (plan.case, precision) with
  | Adv.Int32_inputs, _ -> Ok Rw.Int32_byte_planes
  | Adv.F16_inputs, _ -> Ok Rw.F16_operands
  | Adv.F32_inputs, None ->
      Error
        "a float32 product is rewritten only with ~precision (operand rounding)"
  | Adv.F32_inputs, Some p -> (
      match Sarek_coopmat.operand_rounding_reason p with
      | Some _ -> Ok (precision_lowering p)
      | None ->
          Error
            (Printf.sprintf
               "operand rounding to %s was not accepted \
                (Sarek_coopmat.accept_operand_rounding)"
               (Sarek_coopmat.operand_precision_name p)))
  | Adv.No_case, _ ->
      Error "no matrix-unit configuration for these element types"

let verdict_ok = function
  | Sarek_capability.Available -> Ok ()
  | Sarek_capability.Unavailable cap -> Error cap.Sarek_capability.cap_why
  | Sarek_capability.Unknown why -> Error why

(* The rewritten kernel is laid out for 32-wide waves (4 per block) and its
   lowering has been executed on HIP gfx11 and the interpreter only. Vulkan
   advertises the integer configuration on the RX 7900 XTX, but with 64-wide
   subgroups under radv, and GLSL lowers no byte-plane store: declined there
   rather than run unmeasured. *)
let rewrite_backends = ["HIP"; "Interpreter"]

let device_admits (device : Spoc_core.Device.t) cfg =
  let* () = verdict_ok (Sarek_coopmat.policy_verdict cfg) in
  if not (List.mem device.framework rewrite_backends) then
    Error
      (Printf.sprintf
         "the rewrite is lowered and measured on %s only, not %s"
         (String.concat " and " rewrite_backends)
         device.framework)
  else if device.framework = "Interpreter" then Ok ()
  else
    let support = device.capabilities.Spoc_framework.Framework_sig.coopmat in
    let* () = verdict_ok (Sarek_coopmat.verdict ~support cfg) in
    match support with
    | Some s when s.Sarek_coopmat.ds_subgroup_size = 32 -> Ok ()
    | _ -> Error "the rewrite needs 32-wide subgroups"

(* The interpreter device's [run] refuses cooperative-matrix kernels (it
   probes no configuration); the interpreter itself evaluates them, so the
   rewritten kernel goes through its own entry point. *)
let launch (device : Spoc_core.Device.t) ~ir ~args ~block ~grid =
  if device.framework = "Interpreter" then
    let parallel =
      let n = device.name in
      let p = "Parallel" in
      let rec has i =
        i + String.length p <= String.length n
        && (String.sub n i (String.length p) = p || has (i + 1))
      in
      has 0
    in
    Execute.run_interpreter_vectors ~ir ~args ~block ~grid ~parallel
  else Execute.run_vectors ~device ~ir ~args ~block ~grid ()

(* ---- One launch ----------------------------------------------------------- *)

let scalar_of_arg = function
  | Execute.Int i -> Some (Int32.of_int i)
  | Execute.Int32 i -> Some i
  | Execute.Int64 i -> Some (Int64.to_int32 i)
  | Execute.Vec _ | Execute.Float32 _ | Execute.Float64 _ -> None

let launch_facts (k : kernel) args ~(block : Spoc_framework.Framework_sig.dims)
    ~(grid : Spoc_framework.Framework_sig.dims) =
  if List.length args <> List.length k.kern_params then
    Error "the arguments do not match the kernel's parameters"
  else
    let values =
      List.filter_map
        (fun (d, a) ->
          match d with
          | DParam (v, None) ->
              Option.map (fun x -> (v.var_id, x)) (scalar_of_arg a)
          | _ -> None)
        (List.combine k.kern_params args)
    in
    Ok
      {
        Sarek_ir_eval_int.grid = (grid.x, grid.y, grid.z);
        block = (block.x, block.y, block.z);
        scalar = (fun v -> List.assoc_opt v.var_id values);
      }

let decide ?precision ~device ~ir ~args ~block ~grid () =
  let* plan = memo_plan ir in
  let* lowering = lowering_of plan precision in
  let cfg = Rw.config_of lowering ~operand_elt:plan.gemm.a.elt in
  let* () = device_admits device cfg in
  let* facts = launch_facts ir args ~block ~grid in
  let* region = Map.region plan.mapping facts in
  let* rk = memo_rewrite ir plan.gemm lowering in
  Ok (lowering, cfg, region, rk)

let run ?precision ~device ~ir ~args ~block ~grid () =
  match decide ?precision ~device ~ir ~args ~block ~grid () with
  | Error why ->
      Execute.run_vectors ~device ~ir ~args ~block ~grid () ;
      Original why
  | Ok (lowering, config, (r : Map.region), rk) ->
      let (bx, by, bz), (gx, gy, gz) =
        Rw.launch_dims ~me:r.me ~ne:r.ne ~ze:r.ze
      in
      let extra =
        List.map
          (fun x -> Execute.Int32 (Int32.of_int x))
          [r.me; r.ne; r.ze; r.d; r.t]
      in
      launch
        device
        ~ir:rk
        ~args:(args @ extra)
        ~block:(Execute.dims3d bx by bz)
        ~grid:(Execute.dims3d gx gy gz) ;
      Rewritten {lowering; config}

let outcome_text = function
  | Rewritten {lowering; config} ->
      Printf.sprintf
        "rewritten: %s, %s"
        (Rw.lowering_name lowering)
        (Sarek_coopmat.config_name config)
  | Original why -> "original kernel: " ^ why
