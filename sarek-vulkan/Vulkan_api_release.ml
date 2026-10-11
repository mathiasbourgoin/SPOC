(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Deferred release of device buffers, and the allocation retry that depends on
    it. [free] runs from [Gc.finalise] finalisers
    (Gpu_memory.register_finalizer), which OCaml 5 runs at any allocation point
    of any domain, even inside [batch_lock] or [queue_lock]. A free therefore
    never submits, records, waits, blocks on a lock or consumes an error: it
    queues its release and asks the stream layer to run the queue if the device
    is idle. A queue that stays non-empty is run at the next drain, idle
    [Stream.synchronize], allocation failure, device destroy, or once it is over
    its bounds (see {!Vulkan_api_stream.settle}). *)

module Device = Vulkan_api_device
module BE = Sarek_backend_error.Backend_error

(** Device buffers allocated and not yet released (test instrumentation: a
    buffer whose release failed silently stays counted). *)
let live_allocations = Atomic.make 0

(** Test hook: the next [n] allocation attempts fail as out-of-device-memory
    errors. *)
let inject_alloc_failure = ref 0

let out_of_device_memory = "VK_ERROR_OUT_OF_DEVICE_MEMORY"

let fail_if_injected () =
  if !inject_alloc_failure > 0 then begin
    decr inject_alloc_failure ;
    Vulkan_error.raise_error
      (Vulkan_error.context_error "vkAllocateMemory" out_of_device_memory)
  end

let is_out_of_memory = function
  | Vulkan_error.Vulkan_error
      (BE.Runtime {error = BE.Context_error {reason; _}; _}) ->
      String.equal reason out_of_device_memory
      || String.equal reason "VK_ERROR_OUT_OF_HOST_MEMORY"
  | _ -> false

(** Run [attempt]; if it fails for lack of memory while releases are queued,
    complete the launches, run the releases (keeping other streams' errors) and
    try once more. A failing retry re-raises the original error. Allocation
    never runs in a finaliser, so taking the batch lock here is safe. *)
let with_release_retry (device : Device.t) attempt =
  try attempt ()
  with e when is_out_of_memory e && Atomic.get device.deferred_count > 0 -> (
    try
      !Device.settle_pending device ;
      attempt ()
    with _ -> raise e)

(** Queue [release] ([bytes] of device memory) until no launch recorded so far
    can use the resource, and run the queue now if the device is idle. A handle
    of a destroyed device only queues: the device took its resources with it. *)
let free (device : Device.t) ~bytes release =
  Device.defer_free device ~bytes release ;
  if not (Atomic.get device.destroyed) then
    try !Device.try_release_idle device with _ -> ()
