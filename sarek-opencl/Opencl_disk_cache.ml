(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL persistent compile cache - key and entry format.
 *
 * OpenCL builds from source in every process (clBuildProgram). The program
 * binary the driver hands back (clGetProgramInfo CL_PROGRAM_BINARIES) is
 * stored through {!Spoc_framework_registry.Framework_cache}, the same on-disk
 * store the Vulkan backend uses for SPIR-V, and reloaded with
 * clCreateProgramWithBinary.
 *
 * This module is pure (no OpenCL entry point): the identity of the build target
 * is passed in, so the key and the entry format are testable without a device.
 * [Opencl_api.Program.build] does the driver calls.
 ******************************************************************************)

module Cache = Spoc_framework_registry.Framework_cache

(** Everything that changes the binary a driver produces from a source.

    A driver update changes [driver_version] (and usually [device_version]); a
    different compiler front end changes [platform_*]; the screened and
    capability-gated option string is [build_options]. Any difference must miss,
    so each is a key component. *)
type target = {
  platform_name : string;
  platform_version : string;
  device_name : string;
  device_version : string;
  driver_version : string;
  build_options : string;
}

(** The cache key of [source] built for [target].

    The program binary holds every kernel of the source, so the key is per
    source, not per kernel name: keying by name would store one identical binary
    per kernel of a multi-kernel source. The fixed name below only fills the
    shared key shape. *)
let key (target : target) ~source =
  let dev_name =
    String.concat
      "\000"
      [
        "OpenCL";
        target.platform_name;
        target.platform_version;
        target.device_name;
        target.device_version;
      ]
  in
  Cache.compute_key_with_options
    ~options:[("build_options", target.build_options)]
    ~dev_name
    ~driver_version:target.driver_version
    ~name:"opencl-program"
    ~source

(** An entry is the binary prefixed by its own MD5, so a truncated or bit-rotted
    file is rejected here instead of being handed to the driver. A driver is
    free to accept garbage; MD5 is for corruption, not for an adversary. *)
let seal binary = Digest.string binary ^ binary

let digest_length = 16

let unseal entry =
  let length = String.length entry in
  if length <= digest_length then None
  else
    let binary = String.sub entry digest_length (length - digest_length) in
    if String.equal (String.sub entry 0 digest_length) (Digest.string binary)
    then Some binary
    else None

(** The stored binary for [key], or [None] on a miss, a disabled cache, an
    unusable cache directory, or a corrupt entry. Never raises: the cache is an
    optimisation. *)
let load ~key =
  match Cache.get ~key with
  | exception Spoc_framework_registry.Framework_error.Framework_error _ -> None
  | None -> None
  | Some entry -> unseal entry

(** Store [binary] under [key]. A driver that returned no binary is skipped.
    Never raises. *)
let store ~key binary =
  if String.length binary > 0 then
    try Cache.put ~key ~data:(seal binary)
    with Spoc_framework_registry.Framework_error.Framework_error _ -> ()
