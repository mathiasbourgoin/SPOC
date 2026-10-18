(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Capability: can a device copy straight from a file-backed host mapping that
    has been registered with the driver?

    {!Transfer.register_host} takes any Bigarray and may succeed where the
    registration cannot serve a copy (measured on HIP, RX 7900 XTX, ROCm 7.2.4:
    [hipHostRegister] returns success for a [PROT_NONE] mapping, a tmpfs file, a
    range registered twice and overlapping ranges, and [hipHostUnregister] then
    fails with error 713). So the answer to "is it usable here" is a verdict
    reached in two steps, and neither is the return code of the driver:

    + {!static_verdict}: decided from the device and the backend alone. It
      registers and copies nothing; for HIP and CUDA it reads one device
      attribute (integrated or not) through
      {!Spoc_framework.Shared_memory_query}. Vulkan is [Refused] here, whatever
      the opt-in environment says, so registration is never attempted on it
      (kernel oops, Sarek #503/#504).
    + {!probe}: a self-test on a region of a real file that copies the
      registered pages to the device and back and compares the bytes with an
      independent read of the file, then times the registered copy against the
      staged one it would replace.

    Conditions under which a [Usable] verdict holds, and that the caller must
    keep:

    - the mapping is read-only and shared ([PROT_READ], [MAP_SHARED]) of a
      regular file on a filesystem that supports it (the probe is run on the
      file the pages will come from);
    - registered windows are disjoint, whole pages, and each is released with
      {!Transfer.unregister_host} before the mapping is unmapped;
    - the verdict is a property of the device, driver, kernel, filesystem,
      locked-memory limit and SPOC revision (the static refusals change with it)
      at the time of the probe, and [Usable] covers the two sampled windows
      only, read as they were when probed (pages already in the page cache: a
      cold first copy is not measured). The caller keys any cache on those;
      {!cacheable} says which verdicts may be cached at all.

    This module has no cache and no mapping code: the caller maps the file (SPOC
    has no read-only shared mapping helper) and owns the cache, because both
    depend on facts SPOC does not hold (the model file, the process-lifetime
    policy). *)

(** The bytes of a mapping or a buffer. *)
type host =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(** Why a device is refused. *)
type refusal =
  | Integrated_gpu
      (** The device shares system memory: registration gains nothing and the
          probe would load the memory fabric of the machine. *)
  | Device_kind_unknown
      (** The backend registers host memory but did not say whether the device
          is integrated ({!Shared_memory_query}): refused rather than guessed.
      *)
  | Kernel_hazard of string
      (** Registering is known to harm the machine on this backend (Vulkan on
          AMD: kernel oops, Sarek #503/#504). Never attempted. *)
  | Not_measured of string
      (** No measurement shows the capability on this backend. *)
  | Wrong_bytes
      (** The driver accepted the registration and the copy returned bytes that
          differ from the file, while the mapping and the reference agree: a
          false success of the device. *)

type verdict =
  | Usable
  | Not_needed of string
      (** The device reads host memory directly (Metal unified memory, CPU
          backends): copy from the mapping, register nothing. *)
  | Refused of refusal
  | Too_slow of string
      (** Bytes right, but the registered copy was slower than the staged one by
          more than the margin: a property of the load at that moment. *)
  | Inconclusive of string
      (** An operation raised, or the region was too small: no verdict. *)
  | Still_registered of string
      (** The probe could not release a window it registered (the driver's
          unregister failed after a retry). The window stays in {!Transfer}'s
          table (so [Transfer.is_pinned] stays true and the mapping stays
          rooted) and may still be registered with the driver: the whole [map]
          given to {!probe} must not be unmapped, registered again or probed
          again in this process (either window may be the one left). Probe a
          scratch mapping, not the one the weights are read from. Not cacheable.
      *)

(** Result of {!static_verdict}. *)
type static = Decided of verdict | Self_test_required

val refusal_to_string : refusal -> string

val verdict_to_string : verdict -> string

(** [true] only for what the self-test established about the machine: [Usable]
    and [Refused Wrong_bytes]. Everything else is [false]: the static verdicts
    cost nothing to recompute and some rest on a query that can fail for a
    moment ([Device_kind_unknown], [Integrated_gpu] when HIP could not read the
    attribute), and [Too_slow], [Inconclusive] and [Still_registered] depend on
    load or on a transient failure. Match [verdict] with a wildcard: variants
    may be added. *)
val cacheable : verdict -> bool

(** Decided from the device and its backend alone. Never registers or copies
    anything; Vulkan is answered before any query is made. For HIP and CUDA it
    asks the backend's hook, which reads one device attribute (and may create
    the backend's device object on a first call). A HIP device whose attribute
    could not be read is reported as [Refused Integrated_gpu] (the backend
    counts a failed query as integrated); it is not cacheable. Framework names
    that are neither listed in the source nor family variants of a listed one,
    and GPU OpenCL, are [Refused (Not_measured _)]; a CPU device is
    [Not_needed]. *)
val static_verdict : Device.t -> static

(** The ways to talk to the device that the self-test needs. {!probe} builds
    them from a device ({!with_device_ops}); the record is public so that a
    backend or a test can drive {!probe_with} against a double. *)
type ops = {
  register : host -> unit;  (** may raise on a refusal *)
  unregister : host -> unit;
  alloc_host : int -> host;  (** page-locked when the backend can *)
  free_host : host -> unit;
  round_trip : host -> into:host -> unit;
      (** copy a host range to the device, then back into [into] *)
  upload : host -> unit;
      (** the host-to-device half alone: what the speed test times *)
}

(** [with_device_ops dev ~bytes f] runs [f] on operations backed by {!Transfer}
    (a device buffer of [bytes] bytes, at least half the mapping; released when
    [f] returns or raises; a release that raises is ignored, so that it replaces
    neither the result of [f] nor its exception). [release] is how the device
    buffer is freed ({!Transfer.free_all_buffers} by default); it exists for
    tests. Raises [Invalid_argument] for a device whose static verdict is
    [Refused _] (Vulkan above all): registration is never reachable through this
    module on a refused device. *)
val with_device_ops :
  ?release:((char, Bigarray.int8_unsigned_elt) Vector.t -> unit) ->
  Device.t ->
  bytes:int ->
  (ops -> 'a) ->
  'a

(** The self-test, without the static check: two disjoint halves of [map] are
    registered, copied to the device and back, compared with the same halves of
    [reference] (the bytes of the same file region read independently, by
    [pread]; [map] starts on a page boundary, as a mapping does), checked equal
    to [map] first (a reference that differs is the caller's fault:
    [Inconclusive], nothing registered; likewise when the two windows hold the
    same bytes, which could not tell a stale copy from a good one: pick another
    region), timed against a staged copy (memcpy into page-locked memory, then
    the same copy), and released. [map] and [reference] have the same length, at
    least two units of 64 KiB (the largest page size in use, so that windows
    never share a page). Each window is half the length rounded down to a
    multiple of 64 KiB: bytes beyond the two windows are neither registered nor
    compared. Everything registered or allocated is released, or its release is
    tried twice and reported, whatever happens; [Out_of_memory],
    [Stack_overflow] and [Sys.Break] are re-raised, every other exception is
    [Inconclusive].

    [min_relative_speed] (default 0.8) is the fraction of the staged copy's
    speed that the registered copy must reach, best of 7 against best of 7. It
    is a margin for noise, not a measurement. *)
val probe_with :
  ?log:(string -> unit) ->
  ?min_relative_speed:float ->
  ops ->
  map:host ->
  reference:host ->
  verdict

(** The capability: {!static_verdict}, then, if it asks for it, the self-test on
    [dev] ({!probe_with} over {!with_device_ops}). A device decided statically
    is answered without reading [map] or [reference] and without registering or
    copying anything. [map] must not overlap a registration that is live in the
    process (the driver cannot tell, and HIP reports a false success). The
    release of the windows can fail (HIP: error 713); the verdict is then
    [Still_registered] (or [Inconclusive] if a second try succeeds), the release
    error is passed to [log], and the failed window stays in {!Transfer}'s table
    (its mapping rooted) until the process ends: never unmap that range.

    [map] must start on a page boundary of the host (checked at 4 KiB only) and
    so must every offset at which a caller slices it: a sub-slice that starts
    inside a page of a larger-page host makes the two windows share a page. A
    mapping returned by [mmap] satisfies this. *)
val probe :
  ?log:(string -> unit) ->
  ?min_relative_speed:float ->
  Device.t ->
  map:host ->
  reference:host ->
  verdict
