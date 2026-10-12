(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Recording, submitting and completing launch batches
    (specs/vulkan-async-launches.md). Every function that takes a stream runs
    with [stream.device.batch_lock] held, except {!try_release_idle}, which
    takes it with [try_lock]. *)

open Ctypes
open Vulkan_types
open Vulkan_bindings
open Vulkan_api_base
open Vulkan_api_slot
module Device = Vulkan_api_device
module BE = Sarek_backend_error.Backend_error

(** Dispatches one command buffer collects before it is submitted, from
    [SPOC_VULKAN_BATCH] (a positive integer; anything else gives the default).
    [1] submits and waits at every launch: the behaviour before batching.

    Batching is opt-in ([SPOC_VULKAN_BATCH=32] restores the batched behaviour
    measured in CHANGES.md): [test_vulkan_async] produced GPU write page faults
    and gfx ring resets on RADV (RX 7900 XTX, Linux 7.1.2) and the cause is not
    found (Sarek #504). Until it is, a launch submits and waits. *)
let default_batch = 1

let batch_limit =
  ref
    (match
       Option.bind (Sys.getenv_opt "SPOC_VULKAN_BATCH") int_of_string_opt
     with
    | Some n when n >= 1 -> n
    | Some _ | None -> default_batch)

(** Queued buffer releases (count, bytes) past which the next launch first waits
    for all launches and runs them. A bound acted on at the next launch, not a
    hard cap: a program that queues releases and never launches again keeps them
    until a drain point or an allocation failure
    ({!Vulkan_api_release.with_release_retry}). *)
let max_deferred_frees = 64

let max_deferred_bytes = ref (256 * 1024 * 1024)

(** The device reported that it is lost: the work of every submission is over.
*)
let is_device_lost = function
  | BE.Runtime {error = BE.Context_error {reason; _}; _} ->
      String.equal reason "VK_ERROR_DEVICE_LOST"
  | _ -> false

(** Kernels and reason of a batch failure, from the error it raised. *)
let failure_of_error = function
  | BE.Runtime {error = BE.Kernel_launch_failed {kernel_name; reason}; _} ->
      (kernel_name, reason)
  | e -> ("", BE.to_string e)

(** The slot's fields back to idle; allocation-free, so no asynchronous
    exception can interrupt it. Pool counters may be stale afterwards. *)
let force_idle slot =
  slot.fence.pending <- false ;
  slot.kernels <- [] ;
  slot.dispatches <- 0 ;
  slot.recording <- false ;
  slot.armed <- false ;
  slot.ended <- false

(** {!retire}, retried when an exception interrupts it (asynchronous ones can
    land at any allocation); after [attempts] failures only the fields are
    reset. *)
let rec retire_robust attempts stream slot =
  match retire stream slot with
  | () -> ()
  | exception _ ->
      if attempts > 0 then retire_robust (attempts - 1) stream slot
      else force_idle slot

(** Abandon a slot whose work is over or can never be observed. *)
let abandon stream slot =
  slot.fence.pending <- false ;
  retire_robust 100 stream slot

(** Wait for the slot's submission, if any, and retire it. A failing wait raises
    the batch error naming its kernels. Only a lost device ends the slot's work:
    the slot is then abandoned and later waits do not raise the error again. Any
    other failure (a timeout under [SPOC_VULKAN_FENCE_TIMEOUT_S], out of memory)
    does not say the work is over, so the slot stays in flight and the next wait
    tries again. *)
let await_slot stream slot =
  if slot.fence.pending then begin
    match
      await_fence
        "vkWaitForFences (launch batch)"
        stream.device.Device.device
        slot.fence
    with
    | () -> retire stream slot
    | exception Vulkan_error.Vulkan_error e ->
        let kernels = slot.kernels in
        if is_device_lost e then abandon stream slot ;
        Vulkan_error.raise_error
          (Vulkan_error.kernel_launch_failed
             (names_of_kernels kernels)
             (BE.to_string e))
  end

let raise_deferred stream =
  match stream.deferred with
  | None -> ()
  | Some (kernels, reason) ->
      stream.deferred <- None ;
      Vulkan_error.raise_error
        (Vulkan_error.kernel_launch_failed kernels reason)

(** Like {!await_slot}, but a failure is kept as the stream's deferred error
    (the first one wins) instead of being raised. *)
let keep_wait_error stream slot =
  try await_slot stream slot
  with Vulkan_error.Vulkan_error e ->
    if stream.deferred = None then stream.deferred <- Some (failure_of_error e)

let submit_info_of slot =
  let cmd_buf_ptr = allocate vk_command_buffer_ptr slot.command_buffer in
  let submit_info = make vk_submit_info in
  setf submit_info submit_sType (u32 vk_structure_type_submit_info) ;
  setf submit_info submit_pNext null ;
  setf submit_info submit_waitSemaphoreCount (u32 0) ;
  setf submit_info submit_pWaitSemaphores (from_voidp vk_semaphore null) ;
  setf submit_info submit_pWaitDstStageMask (from_voidp vk_flags null) ;
  setf submit_info submit_commandBufferCount (u32 1) ;
  setf submit_info submit_pCommandBuffers cmd_buf_ptr ;
  setf submit_info submit_signalSemaphoreCount (u32 0) ;
  setf submit_info submit_pSignalSemaphores (from_voidp vk_semaphore null) ;
  (submit_info, cmd_buf_ptr)

(** End the slot's command buffer and submit it with its fence. The fence is
    reset first and [armed] set right after: an exception that escapes while
    armed may have come after the queue accepted the batch. *)
let submit_slot stream slot =
  let device = stream.device in
  Vulkan_api_barrier.end_of_batch slot.command_buffer ;
  slot.ended <- true ;
  check "vkEndCommandBuffer" (vkEndCommandBuffer slot.command_buffer) ;
  slot.cb_open <- false ;
  reset_fence
    "vkResetFences (launch batch)"
    device.Device.device
    slot.fence.handle ;
  slot.armed <- true ;
  let submit_info, cmd_buf_ptr = submit_info_of slot in
  check
    "vkQueueSubmit (launch batch)"
    (Device.queue_submit device (u32 1) (addr submit_info) slot.fence.handle) ;
  Atomic.incr submission_count ;
  slot.fence.pending <- true ;
  slot.recording <- false ;
  ignore (Sys.opaque_identity cmd_buf_ptr) ;
  ignore (Sys.opaque_identity submit_info)

(** Put the slot's command buffer back in the initial state, whatever state an
    interrupted end-of-batch step left it in. A device error is ignored: the
    buffer is then unusable anyway. *)
let reset_command_buffer slot =
  match
    check
      "vkResetCommandBuffer"
      (vkResetCommandBuffer slot.command_buffer (u32 0))
  with
  | () -> slot.cb_open <- false
  | exception Vulkan_error.Vulkan_error _ -> ()

let wait_queue_idle device =
  Device.with_queue_lock device (fun () ->
      ignore (vkQueueWaitIdle device.Device.compute_queue))

(** An exception that is not a Vulkan failure (an asynchronous one: [Sys.Break],
    [Out_of_memory], a raising finaliser or memprof callback) escaped
    {!submit_slot}. Before the end-of-batch step the batch is intact and stays
    open. After it began, the command buffer may be recording, executable or
    submitted, and the batch may or may not run: if the fence was armed, wait
    for the queue to be idle, reset the command buffer so that the next begin is
    on an initial-state buffer, then abandon the slot (its launches may or may
    not have run). Another exception can land in this recovery itself, so every
    step is retried, nothing is allocated outside the protected region, and
    [attempts] bounds the retries. The caller re-raises the original exception
    as it is. *)
let rec recover_from_exception attempts stream slot =
  match
    if slot.ended then begin
      if slot.armed then wait_queue_idle stream.device ;
      reset_command_buffer slot ;
      abandon stream slot
    end
  with
  | () -> ()
  | exception _ ->
      if attempts > 0 then recover_from_exception (attempts - 1) stream slot
      else force_idle slot

(** Submit the open batch of [stream], without waiting for it. A Vulkan failure
    retires the batch; it is raised now when [defer] is false, and otherwise
    kept for the next wait of the stream (the first failure wins). Any other
    exception propagates unchanged after {!recover_from_exception}. *)
let submit_open ~defer stream =
  let slot = stream.slots.(stream.active) in
  if slot.recording && slot.dispatches > 0 then begin
    match submit_slot stream slot with
    | () -> stream.active <- (stream.active + 1) mod Array.length stream.slots
    | exception Vulkan_error.Vulkan_error e ->
        let kernels = slot.kernels in
        abandon stream slot ;
        let kernels = names_of_kernels kernels in
        if not defer then
          Vulkan_error.raise_error
            (Vulkan_error.kernel_launch_failed kernels (BE.to_string e))
        else if stream.deferred = None then
          stream.deferred <- Some (kernels, BE.to_string e)
    | exception e ->
        recover_from_exception 1000 stream slot ;
        raise e
  end

(** Submit and wait for everything launched on [stream]; raises the first
    failure. *)
let drain_stream stream =
  submit_open ~defer:false stream ;
  let count = Array.length stream.slots in
  for i = 1 to count do
    await_slot stream stream.slots.((stream.active + i) mod count)
  done ;
  raise_deferred stream

(** Run [f] on every stream of [device], then re-raise the first exception: one
    failing stream must not leave the others undrained. *)
let on_all_streams device f =
  let first_error = ref None in
  List.iter
    (fun stream ->
      try f stream with e -> if !first_error = None then first_error := Some e)
    (streams_of device) ;
  Option.iter raise !first_error

(** Run the queued buffer releases. Caller holds [batch_lock] and every launch
    recorded so far has completed. *)
let release_deferred device =
  List.iter (fun release -> release ()) (Device.take_deferred device)

(** Forget the awaited events of [stream] whose work has completed (their fence
    is no longer pending or reads signalled), so a stream that waits on many
    events does not keep them all. An event whose lock another domain holds is
    kept: its fence is being recorded or waited and must not be queried. Caller
    holds [batch_lock]. *)
let prune_awaited stream =
  let still_awaited ev =
    if not (Mutex.try_lock ev.event_lock) then true
    else
      Fun.protect
        ~finally:(fun () -> Mutex.unlock ev.event_lock)
        (fun () ->
          ev.event_fence.pending
          && not
               (fence_signalled
                  "vkGetFenceStatus (awaited event)"
                  stream.device.Device.device
                  ev.event_fence.handle))
  in
  stream.awaited <- List.filter still_awaited stream.awaited

let drain_all device =
  on_all_streams device drain_stream ;
  on_all_streams device prune_awaited ;
  release_deferred device

let drain_device device =
  Mutex.protect device.Device.batch_lock (fun () -> drain_all device)

let () = Device.drain_pending := drain_device

(** Wait for every launch of [device], then run the queued releases, without
    consuming any stream's deferred error: it belongs to the wait of that
    stream, not to the launch that happens to trigger this. A failing wait is
    kept as that stream's deferred error; the releases run only if every slot
    ended up idle. *)
let settle device =
  on_all_streams device (fun stream ->
      submit_open ~defer:true stream ;
      Array.iter (keep_wait_error stream) stream.slots) ;
  if all_idle device then release_deferred device

let () =
  Device.settle_pending :=
    fun device ->
      Mutex.protect device.Device.batch_lock (fun () -> settle device)

(** Retire the launch of [slot] if its fence has already signalled, without
    waiting. A lost device ends the work: the error is kept once as the stream's
    deferred error and the slot is abandoned. Other status errors are left to
    the next wait, which reports them. *)
let reap stream slot =
  if slot.fence.pending then
    match
      fence_signalled
        "vkGetFenceStatus (launch batch)"
        stream.device.Device.device
        slot.fence.handle
    with
    | true ->
        slot.fence.pending <- false ;
        retire stream slot
    | false -> ()
    | exception Vulkan_error.Vulkan_error e ->
        if is_device_lost e then begin
          let kernels = slot.kernels in
          abandon stream slot ;
          if stream.deferred = None then
            stream.deferred <- Some (names_of_kernels kernels, BE.to_string e)
        end

let reap_all device =
  List.iter
    (fun stream -> Array.iter (reap stream) stream.slots)
    (streams_of device)

(** The free path of {!Vulkan_api_release}: never submits, records, waits or
    consumes an error, and never blocks on the batch lock (a lock held, even by
    this domain, leaves the release queued; so does re-entry on the registry
    lock, which raises and is caught by the caller). Queued releases run only if
    no stream has an open batch or a launch in flight, so a free is prompt when
    the device is idle and otherwise waits for the next drain point. The
    destroyed flag is re-checked under the lock, which [Device.destroy] holds
    from its last release to [vkDestroyDevice]. *)
let try_release_idle (device : Device.t) =
  if
    (not (Atomic.get device.destroyed))
    && Mutex.try_lock device.Device.batch_lock
  then
    Fun.protect
      ~finally:(fun () -> Mutex.unlock device.Device.batch_lock)
      (fun () ->
        if not (Atomic.get device.destroyed) then begin
          reap_all device ;
          if all_idle device then release_deferred device
        end)

let () = Device.try_release_idle := try_release_idle

(** Submit the open batch of every stream of [device] without waiting. *)
let flush_device device =
  Mutex.protect device.Device.batch_lock (fun () ->
      on_all_streams device (submit_open ~defer:true))

(** Test instrumentation: begins attempted on a command buffer still recording.
*)
let begin_on_open_buffer = Atomic.make 0

(** Make the active slot of [stream] ready to record: wait for its previous
    submission if it is still in flight, then begin the command buffer. *)
let open_active stream =
  let slot = stream.slots.(stream.active) in
  if not slot.recording then begin
    await_slot stream slot ;
    if slot.cb_open then begin
      Atomic.incr begin_on_open_buffer ;
      reset_command_buffer slot
    end ;
    let begin_info = make vk_command_buffer_begin_info in
    setf
      begin_info
      cmd_buf_begin_sType
      (u32 vk_structure_type_command_buffer_begin_info) ;
    setf begin_info cmd_buf_begin_pNext null ;
    setf
      begin_info
      cmd_buf_begin_flags
      (u32 vk_command_buffer_usage_one_time_submit_bit) ;
    setf begin_info cmd_buf_begin_pInheritanceInfo null ;
    (* Marked open before the call: an exception right after it must not leave a
       recording buffer unaccounted for. *)
    slot.cb_open <- true ;
    check
      "vkBeginCommandBuffer"
      (vkBeginCommandBuffer slot.command_buffer (addr begin_info)) ;
    ignore (Sys.opaque_identity begin_info) ;
    slot.recording <- true
  end ;
  slot

(** Record one dispatch of [kernel] on [stream]. [record command_buffer set]
    issues the bind, push-constant and dispatch commands (after the barrier
    recorded here) with [set], a descriptor set of [layout] that it must fill.
    Nothing is recorded when the set cannot be allocated. *)
let record_dispatch stream ~kernel ~layout ~descriptors ~record =
  let device = stream.device in
  Mutex.protect device.Device.batch_lock (fun () ->
      (* Bound the memory held by freed buffers. *)
      if
        Atomic.get device.Device.deferred_count >= max_deferred_frees
        || Atomic.get device.Device.deferred_bytes >= !max_deferred_bytes
      then settle device ;
      on_all_streams device (fun other ->
          if other != stream then submit_open ~defer:true other) ;
      let slot = open_active stream in
      let set =
        Vulkan_api_pools.allocate device slot.pools layout ~descriptors
      in
      Vulkan_api_barrier.before_dispatch slot.command_buffer ;
      record slot.command_buffer set ;
      slot.kernels <- kernel :: slot.kernels ;
      slot.dispatches <- slot.dispatches + 1 ;
      if !batch_limit <= 1 then drain_stream stream
      else if slot.dispatches >= !batch_limit then
        submit_open ~defer:true stream)

(** Record that [stream] waits for [event]. The event is not added twice, and
    events whose work has completed are dropped first, so the list stays bounded
    by the events actually outstanding. {!synchronize} waits on the event's
    fence as it is then: an event re-recorded after the wait was issued is
    waited for too, which over-waits and is safe. *)
let add_awaited stream event =
  Mutex.protect stream.device.Device.batch_lock (fun () ->
      prune_awaited stream ;
      if
        not
          (List.exists
             (fun a -> a.event_fence == event.event_fence)
             stream.awaited)
      then stream.awaited <- event :: stream.awaited)

(** Wait for the events [stream] awaits, then forget them. Every wait is
    attempted; the first failure is re-raised after the others. Takes the
    events' locks, so it runs without [batch_lock] ([Vulkan_api_event.record]
    takes an event's lock after releasing it, never the reverse). *)
let await_events stream events =
  let first_error = ref None in
  let failed = ref [] in
  List.iter
    (fun ev ->
      try
        Mutex.protect ev.event_lock (fun () ->
            await_fence
              "vkWaitForFences (stream wait on event)"
              stream.device.Device.device
              ev.event_fence)
      with e ->
        failed := ev :: !failed ;
        if !first_error = None then first_error := Some e)
    events ;
  (* An event whose wait failed is still owed: a retried synchronize waits again. *)
  List.iter (fun ev -> add_awaited stream ev) (List.rev !failed) ;
  Option.iter raise !first_error

(** Complete everything launched on [stream] and the events it awaits; also reap
    the other streams, so a device that is now idle runs its queued releases. *)
let synchronize stream =
  let device = stream.device in
  let events =
    Mutex.protect device.Device.batch_lock (fun () ->
        drain_stream stream ;
        reap_all device ;
        if all_idle device then release_deferred device ;
        let events = stream.awaited in
        stream.awaited <- [] ;
        events)
  in
  await_events stream events

(** Launches recorded on [device] that have not completed: open batches and
    batches in flight. Zero after any synchronisation point. *)
let pending_launches device =
  Mutex.protect device.Device.batch_lock (fun () ->
      List.fold_left
        (fun total stream ->
          Array.fold_left
            (fun total slot ->
              if slot.recording || slot.fence.pending then
                total + slot.dispatches
              else total)
            total
            stream.slots)
        0
        (streams_of device))
