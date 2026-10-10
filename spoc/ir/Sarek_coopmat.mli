(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix: the DEVICE-facing half.

    Every type and value of {!Sarek_coopmat_types} is re-exported here by
    [include module type of], with its type equalities intact, so
    [Sarek_coopmat.fragment], [Sarek_coopmat.Uint8] and the rest mean exactly
    what they meant before slice 3 split the module. Read
    {!Sarek_coopmat_types}'s interface for the vocabulary and the design
    rationale behind it.

    What is defined HERE is the part that consults {!Sarek_capability}, and it
    is separated for a mechanical reason as much as a conceptual one: the IR
    names a fragment, {!Sarek_capability} depends on the IR, and a vocabulary
    that also depended on {!Sarek_capability} would close a module cycle. *)

(* [struct include ... end] and not the bare module path: the bare form
   re-declares each type ABSTRACTLY, so [Sarek_coopmat.config] would be a
   distinct type from [Sarek_coopmat_types.config] and every value crossing
   between the IR and the capability layer would need a conversion that does
   nothing. This form preserves the type equalities, which is the whole point of
   the re-export. *)
include module type of struct
  include Sarek_coopmat_types
end

(** [verdict ~support cfg] judges a requested configuration against a device.

    [support = None] yields {!Sarek_capability.Unknown}, which
    {!Sarek_capability.permits} refuses. That is the reason this takes an option
    and the reason {!Framework_sig.capabilities} stores an option: an empty
    configuration list and an unprobed device are different facts, and only one
    of them is evidence.

    {b The gate must be keyed on the feature, not on the configuration list},
    and this is measured rather than argued. On this workstation
    [vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR] returns [VK_SUCCESS] and
    FOURTEEN configurations for the Raphael iGPU — a device that does not
    advertise [VK_KHR_cooperative_matrix] and whose [cooperativeMatrix] feature
    reads false. A probe that populated [ds_configs] from that call without
    first checking the extension would report the iGPU as fully capable, and
    this verdict would say [Available] for a device that cannot run the
    instruction. The Vulkan probe therefore leaves [ds_configs] empty unless the
    extension is advertised AND the feature is true; see
    [sarek-vulkan/Vulkan_api_device.ml]. *)
val verdict :
  support:device_support option -> config -> Sarek_capability.verdict

(** The {!Sarek_capability.Device_optional} record for a device that provides no
    cooperative-matrix support at all, or none matching the request. [evidence]
    is the probe's {!device_support.ds_evidence}; {!verdict} passes it, so the
    refusal says which backend rule refused this device. *)
val device_lacks_config : ?evidence:string -> config -> Sarek_capability.t

(** [fragment_verdict ~support f]: does the device offer any configuration one
    of whose four fragments is [f]? A kernel that declares, loads or stores a
    fragment without ever multiplying still needs the device's per-lane layout
    for it, so the launch gate asks this for every fragment as well as
    {!verdict} for every multiply-add. [None] support is [Unknown]. *)
val fragment_verdict :
  support:device_support option -> fragment -> Sarek_capability.verdict

(** {1 The accuracy policy (docs/design/f16-relaxed-accuracy.md §6.1)}

    A float configuration is {!Relaxed_bounded}: the hardware chooses the order
    of the additions, so its result is within a derived bound of the exact value
    but not bit-identical to the interpreter. The owner's decision of 2026-07-27
    is that such a deviation needs an EXPLICIT opt-in and that the launch fails
    without it; these three values are that opt-in. *)

(** [accept_relaxed_accumulation ~reason] admits the float configurations that
    have a derived bound (today only [f16 * f16 + f32 -> f32]) for the rest of
    the process. One-way: there is no withdrawal, so a capability check that has
    passed stays valid. The first non-empty [reason] is kept.

    @raise Invalid_argument if [reason] is empty or blank. *)
val accept_relaxed_accumulation : reason:string -> unit

(** The reason recorded by {!accept_relaxed_accumulation}, if it was called. *)
val relaxed_accumulation_reason : unit -> string option

(** [policy_verdict cfg] judges the accuracy regime, independently of any
    device: configurations whose four component types are all integer are
    [Available] (strict: exact accumulation); [f16 * f16 + f32 -> f32] is
    [Available] after {!accept_relaxed_accumulation} and otherwise [Unavailable]
    with a {!Sarek_capability.Policy} record naming the call; every other
    configuration with a float component is [Unavailable] (no derived bound,
    §5.4). The launch gate and the interpreter apply it after the device
    verdict. *)
val policy_verdict : config -> Sarek_capability.verdict
