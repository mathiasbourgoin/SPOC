(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Whether a Vulkan device may import host memory
    ([VK_EXT_external_memory_host]), i.e. whether [alloc_pinned] and
    [register_host] work.

    Importing pageable anonymous memory makes the driver register a userptr
    object with the kernel's MMU notifiers. On amdgpu (RADV, Linux 7.1.2) that
    path was followed by a kernel NULL dereference in
    [amdgpu_hmm_invalidate_gfx], hit by [kcompactd] migrating the pages, after
    which processes stayed in uninterruptible sleep until reboot (Sarek #503).

    So the import is an explicit opt-in on every driver, and refused on AMD
    drivers even then, until a design that keeps the imported pages from being
    migrated is known to be safe. Pure: the caller reads the environment. *)

type decision = Allowed | Not_opted_in | Amdgpu_refused

let opt_in_variable = "SPOC_VULKAN_HOST_IMPORT"

let amdgpu_override_variable =
  "SPOC_VULKAN_HOST_IMPORT_AMDGPU_I_ACCEPT_KERNEL_OOPS"

let amd_vendor_id = 0x1002

(* VK_DRIVER_ID_AMD_PROPRIETARY, AMD_OPEN_SOURCE, MESA_RADV. *)
let amd_driver_ids = [1; 2; 3]

let is_amd ~vendor_id ~driver_id =
  vendor_id = amd_vendor_id || List.mem driver_id amd_driver_ids

let decide ~opted_in ~amdgpu_override ~vendor_id ~driver_id =
  if not opted_in then Not_opted_in
  else if is_amd ~vendor_id ~driver_id && not amdgpu_override then
    Amdgpu_refused
  else Allowed

let env_enabled name =
  match Sys.getenv_opt name with Some "1" -> true | _ -> false

let of_environment ~vendor_id ~driver_id =
  decide
    ~opted_in:(env_enabled opt_in_variable)
    ~amdgpu_override:(env_enabled amdgpu_override_variable)
    ~vendor_id
    ~driver_id

let refusal_reason = function
  | Allowed -> None
  | Not_opted_in ->
      Some
        (Printf.sprintf
           "pinned and registered host memory are off by default on Vulkan \
            (set %s=1 to opt in)"
           opt_in_variable)
  | Amdgpu_refused ->
      Some
        "pinned and registered host memory are refused on AMD Vulkan drivers: \
         importing host pages was followed by a kernel oops in \
         amdgpu_hmm_invalidate_gfx (Sarek #503)"
