(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Which devices the launch-time matrix rewrite is admitted on, and with which
   subgroup width and fragment edge (specs/matrix-pattern-rewrite.md): the
   device-dependent half of Matrix_specialise, split out of it. *)

module Rw = Sarek_matrix_rewrite
module Device = Spoc_core.Device
module Fw = Spoc_framework.Framework_sig
open Sarek_matrix_scheme

let ( let* ) = Result.bind

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

(* Fragments other than 16x16x16 have been executed on Metal's f16 operands
   only (8x8x8, an Apple M4); a device elsewhere that advertises only another
   edge is not rewritten. The interpreter holds whole matrices of any edge. *)
let fragment_refusal (device : Device.t) lowering ~frag =
  match (device.framework, lowering, frag) with
  | _, _, 16 | "Interpreter", _, _ | "Metal", Rw.F16_operands, 8 -> None
  | fw, _, _ ->
      Some
        (Printf.sprintf
           "%dx%dx%d fragments have been executed on Metal's f16 operands \
            only, not on %s with %s"
           frag
           frag
           frag
           fw
           (Rw.lowering_name lowering))

(* What [device.framework] lowers, independently of the fragment edge. *)
let framework_refusal (device : Device.t) lowering =
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

(* Why [lowering] on [frag]-edge fragments cannot be lowered on [device], or
   [None]. *)
let lowering_refusal (device : Device.t) lowering ~frag =
  match framework_refusal device lowering with
  | Some _ as refusal -> refusal
  | None -> fragment_refusal device lowering ~frag

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
    match
      ( lowering_refusal device lowering ~frag:cfg.Sarek_coopmat.cfg_shape.m,
        device.framework )
    with
    | Some why, _ -> Error why
    | None, "Interpreter" -> Ok ()
    | None, _ -> device_offers device ~explicit cfg
