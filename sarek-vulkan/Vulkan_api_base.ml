(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Ctypes
open Vulkan_types

(** memcpy from libc for memory transfers *)
let memcpy dst src size =
  let memcpy_c =
    Foreign.foreign
      "memcpy"
      (ptr void @-> ptr void @-> size_t @-> returning (ptr void))
  in
  memcpy_c dst src size

(** Helper to convert OCaml int to Unsigned.UInt32.t *)
let u32 = Unsigned.UInt32.of_int

(** {1 Exceptions} *)

(** Deprecated: no longer raised. [check] below now raises the canonical
    {!Vulkan_error.Vulkan_error} (a [Backend_error] alias) via the shared
    {!Sarek_backend_error.Backend_error.Make.check} funnel. Kept only so
    out-of-tree code matching on [Vulkan_api_base.Vk_result_error (code, ctx)]
    (or the re-exported [Vulkan_api.Vk_result_error]) still compiles (this
    library is opam-published). *)
exception
  Vk_result_error of vk_result * string
      [@ocaml.deprecated
        "no longer raised; Vulkan_api_base.check now raises \
         Vulkan_error.Vulkan_error (Backend_error) - catch that instead"]

(** Check Vulkan result and raise a canonical {!Backend_error} on failure. *)
let check (ctx : string) (result : vk_result) : unit =
  (match result with
  | VK_SUCCESS -> ()
  | err ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Device
        "[Vulkan] %s failed with %s"
        ctx
        (string_of_vk_result err)) ;
  Vulkan_error.check
    ~is_success:(fun r -> r = VK_SUCCESS)
    ~to_string:string_of_vk_result
    ctx
    result

(** {1 Fences} *)

(** Interval between warnings while a fence wait goes on, in nanoseconds. *)
let fence_warn_interval_ns = 60_000_000_000L

(** Optional bound on a fence wait, from [SPOC_VULKAN_FENCE_TIMEOUT_S] (seconds,
    positive), read once when the library is loaded. Unset means unbounded: a
    wait on submitted work ends only when the work completes or the device
    reports an error, because a caller that gave up early could free memory the
    GPU still uses. The bound exists for tests that use it as a hang detector. A
    malformed value is kept as [Error] and refused by {!check_fence_wait_bound},
    called when a device is created, so it never surfaces after a submission. *)
let fence_wait_bound_parsed : (int64 option, string) result =
  match Sys.getenv_opt "SPOC_VULKAN_FENCE_TIMEOUT_S" with
  | None | Some "" -> Ok None
  | Some text -> (
      match float_of_string_opt text with
      | Some seconds when seconds > 0.0 ->
          Ok (Some (Int64.of_float (seconds *. 1e9)))
      | _ ->
          Error
            (Printf.sprintf
               "expected a positive number of seconds, got %S"
               text))

let check_fence_wait_bound () =
  match fence_wait_bound_parsed with
  | Ok _ -> ()
  | Error reason ->
      Vulkan_error.raise_error
        (Vulkan_error.context_error "SPOC_VULKAN_FENCE_TIMEOUT_S" reason)

(** Test hook: while [Some r], every fence wait fails with [r] before waiting
    (the caller should have made the queue idle first and clears the hook). *)
let inject_wait_failure : vk_result option ref = ref None

(** Test instrumentation, not an API: number of [vkWaitForFences] sequences
    started and of fence-carrying [vkQueueSubmit]s that succeeded, since the
    library was loaded. Instrumentation for tests that assert how many host
    waits a run of launches costs. *)
let fence_wait_count = Atomic.make 0

let submission_count = Atomic.make 0

(** Wait for [fence] on [device] until it is signalled. Logs a warning naming
    [ctx] every {!fence_warn_interval_ns}. Raises the canonical error of
    {!check} on a device error, and [Synchronization_failed] only when
    {!fence_wait_bound_parsed} sets a bound and it is reached. *)
let wait_fence (ctx : string) device fence =
  Atomic.incr fence_wait_count ;
  (match !inject_wait_failure with
  | Some result -> check ctx result
  | None -> ()) ;
  let bound = Result.value fence_wait_bound_parsed ~default:None in
  let fence_ptr = allocate vk_fence fence in
  let rec loop waited =
    let slice =
      match bound with
      | None -> fence_warn_interval_ns
      | Some b -> Int64.min fence_warn_interval_ns (Int64.sub b waited)
    in
    match
      Vulkan_bindings.vkWaitForFences
        device
        (u32 1)
        fence_ptr
        vk_true
        (Unsigned.UInt64.of_int64 slice)
    with
    | VK_TIMEOUT -> (
        let waited = Int64.add waited slice in
        let waited_ms = Int64.div waited 1_000_000L in
        match bound with
        | Some b when waited >= b ->
            Vulkan_error.raise_error
              (Vulkan_error.synchronization_failed
                 (Printf.sprintf
                    "%s: fence not signalled after %Ld ms \
                     (SPOC_VULKAN_FENCE_TIMEOUT_S)"
                    ctx
                    waited_ms))
        | _ ->
            Spoc_core.Log.warnf
              Spoc_core.Log.Device
              "[Vulkan] %s: fence still not signalled after %Ld ms"
              ctx
              waited_ms ;
            loop waited)
    | result -> check ctx result
  in
  loop 0L ;
  ignore (Sys.opaque_identity fence_ptr)

(** Test hook: while [Some r], every fence status query returns [r]. *)
let inject_status_failure : vk_result option ref = ref None

(** Whether [fence] has signalled, without blocking. Raises the canonical error
    of {!check} on a device error. *)
let fence_signalled (ctx : string) device fence =
  match
    match !inject_status_failure with
    | Some result -> result
    | None -> Vulkan_bindings.vkGetFenceStatus device fence
  with
  | VK_SUCCESS -> true
  | VK_NOT_READY -> false
  | result ->
      check ctx result ;
      false

(** Reset [fence] on [device]. The fence must not be pending in a queue. *)
let reset_fence (ctx : string) device fence =
  let fence_ptr = allocate vk_fence fence in
  check ctx (Vulkan_bindings.vkResetFences device (u32 1) fence_ptr) ;
  ignore (Sys.opaque_identity fence_ptr)

(** A fence with its submission state. Invariant: [pending = true] implies the
    fence was handed to a successful [vkQueueSubmit] and has not been waited for
    since. Waits happen only when [pending] is set, so a fence that was never
    submitted (or whose submission failed) is never waited on. *)
type tracked_fence = {handle : vk_fence; mutable pending : bool}

let create_tracked_fence device =
  let fence_info = make vk_fence_create_info in
  setf fence_info fence_create_sType (u32 vk_structure_type_fence_create_info) ;
  setf fence_info fence_create_pNext null ;
  setf fence_info fence_create_flags (u32 0) ;
  let fence = allocate vk_fence vk_null_handle in
  check
    "vkCreateFence"
    (Vulkan_bindings.vkCreateFence device (addr fence_info) null fence) ;
  {handle = !@fence; pending = false}

(** Wait for [fence] if it is pending, then mark it not pending. *)
let await_fence (ctx : string) device fence =
  if fence.pending then begin
    wait_fence ctx device fence.handle ;
    fence.pending <- false
  end

(** Reset [fence] and pass it to [submit]; mark it pending only when [submit]
    returns [VK_SUCCESS]. The fence must not be pending. *)
let submit_with_fence (ctx : string) device fence submit =
  if fence.pending then
    Vulkan_error.raise_error
      (Vulkan_error.context_error ctx "fence is still pending") ;
  reset_fence ctx device fence.handle ;
  check ctx (submit fence.handle) ;
  Atomic.incr submission_count ;
  fence.pending <- true

(** Wait out a pending submission, then destroy [fence]
    (VUID-vkDestroyFence-fence-01120). *)
let destroy_tracked_fence (ctx : string) device fence =
  await_fence ctx device fence ;
  Vulkan_bindings.vkDestroyFence device fence.handle null

(** {1 SPIR-V Compilation} *)

(** A shader that uses subgroup operations ([GL_KHR_shader_subgroup_*], emitted
    by the warp shuffles) needs SPIR-V 1.3, i.e. the Vulkan 1.1 target
    environment. It is requested for those shaders ONLY: Vulkan 1.1 changes the
    storage class of buffer blocks, which the note below keeps away from every
    other shader. The device side is already at 1.2 (Vulkan_api_device). *)
let needs_vulkan_1_1 (glsl_source : string) : bool =
  let needle = "GL_KHR_shader_subgroup" in
  let n = String.length needle and m = String.length glsl_source in
  let rec at i =
    i + n <= m && (String.sub glsl_source i n = needle || at (i + 1))
  in
  at 0

(** Compile GLSL to SPIR-V using glslangValidator. Requires glslangValidator in
    PATH. *)
let compile_glsl_to_spirv_cli ~(entry_point : string) (glsl_source : string) :
    string =
  (* Write GLSL to temp file *)
  let glsl_file = Filename.temp_file "sarek_" ".comp" in
  let spirv_file = Filename.temp_file "sarek_" ".spv" in
  let oc = open_out glsl_file in
  output_string oc glsl_source ;
  close_out oc ;

  (* Compile with glslangValidator *)
  (* NOTE: Don't use --target-env vulkan1.1 - it changes storage classes from
     Uniform to StorageBuffer which may cause issues with some drivers *)
  let cmd =
    Printf.sprintf
      "glslangValidator -V %s-S comp -e %s -o %s %s 2>&1"
      (if needs_vulkan_1_1 glsl_source then "--target-env vulkan1.1 " else "")
      entry_point
      spirv_file
      glsl_file
  in
  Spoc_core.Log.debugf
    Spoc_core.Log.Device
    "[Vulkan] Compiling GLSL to SPIR-V: %s"
    cmd ;
  let ic = Unix.open_process_in cmd in
  let output = Buffer.create 256 in
  (try
     while true do
       Buffer.add_string output (input_line ic ^ "\n")
     done
   with End_of_file -> ()) ;
  let status = Unix.close_process_in ic in

  (* Check result *)
  (match status with
  | Unix.WEXITED 0 -> (
      Spoc_core.Log.debugf
        Spoc_core.Log.Device
        "[Vulkan] SPIR-V compilation succeeded, file size: %d bytes"
        Unix.((stat spirv_file).st_size) ;
      (* Save both SPIR-V and GLSL for debugging if logging is enabled *)
      if Spoc_core.Log.is_enabled Spoc_core.Log.Device then
        let debug_spirv = "/tmp/sarek_debug.spv" in
        let debug_glsl = "/tmp/sarek_debug.comp" in
        try
          let cmd =
            Printf.sprintf
              "cp %s %s && cp %s %s"
              spirv_file
              debug_spirv
              glsl_file
              debug_glsl
          in
          ignore (Sys.command cmd) ;
          Spoc_core.Log.debugf
            Spoc_core.Log.Device
            "[Vulkan] Saved SPIR-V to %s and GLSL to %s for debugging"
            debug_spirv
            debug_glsl
        with _ -> ())
  | _ ->
      (try Unix.unlink spirv_file with _ -> ()) ;
      Vulkan_error.raise_error
        (Vulkan_error.compilation_failed
           ""
           (Printf.sprintf
              "glslangValidator failed:\n%s"
              (Buffer.contents output)))) ;

  (* Clean up GLSL file *)
  (try Unix.unlink glsl_file with _ -> ()) ;

  (* Read SPIR-V binary *)
  let ic = open_in_bin spirv_file in
  let size = in_channel_length ic in
  let spirv = really_input_string ic size in
  close_in ic ;

  (* Clean up SPIR-V file *)
  (try Unix.unlink spirv_file with _ -> ()) ;

  spirv

(** Compile GLSL to SPIR-V using Shaderc if available, otherwise fallback to CLI
*)
let compile_glsl_to_spirv ~(entry_point : string) (glsl_source : string) :
    string =
  if Shaderc.is_available () then begin
    Spoc_core.Log.debug
      Spoc_core.Log.Device
      "[Vulkan] Compiling with libshaderc" ;
    try
      Shaderc.compile_glsl_to_spirv
        ~vulkan_1_1:(needs_vulkan_1_1 glsl_source)
        ~entry_point
        glsl_source
    with e ->
      Spoc_core.Log.errorf
        Spoc_core.Log.Device
        "[Vulkan] libshaderc failed: %s"
        (Printexc.to_string e) ;
      compile_glsl_to_spirv_cli ~entry_point glsl_source
  end
  else begin
    Spoc_core.Log.debug
      Spoc_core.Log.Device
      "[Vulkan] libshaderc not found, using glslangValidator" ;
    compile_glsl_to_spirv_cli ~entry_point glsl_source
  end

(** Check if glslangValidator is available *)
let glslang_available () : bool =
  try
    let ic = Unix.open_process_in "glslangValidator --version 2>&1" in
    let _ = input_line ic in
    let status = Unix.close_process_in ic in
    match status with Unix.WEXITED 0 -> true | _ -> false
  with _ -> false
