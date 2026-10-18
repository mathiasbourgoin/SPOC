(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

type host =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type refusal =
  | Integrated_gpu
  | Device_kind_unknown
  | Kernel_hazard of string
  | Not_measured of string
  | Wrong_bytes

type verdict =
  | Usable
  | Not_needed of string
  | Refused of refusal
  | Too_slow of string
  | Inconclusive of string
  | Still_registered of string

type static = Decided of verdict | Self_test_required

let refusal_to_string = function
  | Integrated_gpu -> "integrated GPU (shared host memory)"
  | Device_kind_unknown ->
      "the backend did not say whether the device is integrated"
  | Kernel_hazard why -> why
  | Not_measured why -> why
  | Wrong_bytes -> "copied bytes differ from the file (false success)"

let verdict_to_string = function
  | Usable -> "usable"
  | Not_needed why -> "not needed: " ^ why
  | Refused r -> "refused: " ^ refusal_to_string r
  | Too_slow why -> "too slow: " ^ why
  | Inconclusive why -> "inconclusive: " ^ why
  | Still_registered why -> "still registered: " ^ why

(* Only what the self-test established. The static verdicts are recomputed at
   no cost, and some rest on a query that can fail for a moment. *)
let cacheable = function
  | Usable | Refused Wrong_bytes -> true
  | Not_needed _ | Refused _ | Too_slow _ | Inconclusive _ | Still_registered _
    ->
      false

(* ---- the decision that needs no driver ---- *)

let vulkan_hazard =
  Kernel_hazard
    "Vulkan: registering host pages is never attempted (importing host pages \
     on amdgpu was followed by a kernel oops, Sarek #503/#504)"

(* HIP and CUDA register host memory with the driver; only a discrete device
   gains from it. *)
let registering_framework (dev : Device.t) =
  dev.framework = "HIP" || Device.is_cuda_framework dev.framework

let static_of_registering_device (dev : Device.t) =
  match
    Spoc_framework.Shared_memory_query.query
      ~framework:dev.framework
      ~backend_id:dev.backend_id
  with
  | Some false -> Self_test_required
  | Some true -> Decided (Refused Integrated_gpu)
  | None -> Decided (Refused Device_kind_unknown)

let static_verdict (dev : Device.t) : static =
  let framework = dev.framework in
  if String.starts_with ~prefix:"Vulkan" framework then
    Decided (Refused vulkan_hazard)
  else if Device.is_cpu dev || framework = "Interpreter" then
    Decided (Not_needed (framework ^ ": host memory, nothing to register"))
  else if framework = "Metal" then
    Decided (Not_needed "Metal: unified memory, copy from the mapping")
  else if registering_framework dev then static_of_registering_device dev
  else
    Decided
      (Refused
         (Not_measured (framework ^ ": no registration of file pages measured")))

(* ---- the self-test ---- *)

type ops = {
  register : host -> unit;
  unregister : host -> unit;
  alloc_host : int -> host;
  free_host : host -> unit;
  round_trip : host -> into:host -> unit;
  upload : host -> unit;
}

(* The unit of a window: the largest page size in use (64 KiB on aarch64 Linux;
   x86-64 uses 4 KiB, Apple silicon 16 KiB), so that two windows never share a
   page whatever the host. *)
let page = 65536

let timing_trials = 7

let default_min_relative_speed = 0.8

let seconds f =
  let t0 = Unix.gettimeofday () in
  f () ;
  (* The clock is not monotonic: a null or negative duration is a clock step,
     not a measurement, and must not pass the speed test ([run] reports it). *)
  let d = Unix.gettimeofday () -. t0 in
  if d > 0. then d else nan

(* Best of [n]; [nan] when no trial gave a valid duration. *)
let best_of n f =
  List.fold_left
    (fun best d -> if Float.is_nan d then best else Float.min best d)
    infinity
    (List.init n (fun _ -> seconds f))
  |> fun b -> if b = infinity then nan else b

let equal_bytes (a : host) (b : host) =
  Bigarray.Array1.dim a = Bigarray.Array1.dim b
  &&
  let n = Bigarray.Array1.dim a in
  let rec go i =
    i >= n
    || Bigarray.Array1.unsafe_get a i = Bigarray.Array1.unsafe_get b i
       && go (i + 1)
  in
  go 0

(* The exceptions that must reach the caller: an interrupt, the end of memory. *)
let is_fatal = function
  | Out_of_memory | Stack_overflow | Sys.Break -> true
  | _ -> false

let attempt what f =
  match f () with
  | () -> Ok ()
  | exception e when is_fatal e -> raise e
  | exception e -> Error (Printf.sprintf "%s: %s" what (Printexc.to_string e))

(* What the probe holds on the device: released whatever happens. *)
type held = {mutable windows : host list; mutable buffers : host list}

let alloc ops held n =
  let b = ops.alloc_host n in
  held.buffers <- b :: held.buffers ;
  b

(* Best effort, after the verdict. What is held is taken out of [held] before
   it is released, so that nothing is released twice; everything is tried and
   the first fatal exception is raised at the end. A window is unregistered
   with a second try when the first fails. Returns the first non-fatal failure
   of each kind: an unregister that failed twice means pages may still be
   registered, a buffer that could not be freed does not. *)
type release_outcome = {
  unregister_failure : string option;
  free_failure : string option;
}

let release ops held =
  let fatal = ref None
  and unregister_failure = ref None
  and free_failure = ref None in
  let each failure release_one x =
    try release_one x
    with e ->
      if is_fatal e then (if Option.is_none !fatal then fatal := Some e)
      else if Option.is_none !failure then
        failure := Some (Printexc.to_string e)
  in
  let unregister_twice w =
    try ops.unregister w with e when not (is_fatal e) -> ops.unregister w
  in
  let windows = held.windows and buffers = held.buffers in
  held.windows <- [] ;
  held.buffers <- [] ;
  List.iter (each unregister_failure unregister_twice) windows ;
  List.iter (each free_failure ops.free_host) buffers ;
  Option.iter raise !fatal ;
  {unregister_failure = !unregister_failure; free_failure = !free_failure}

let ( let* ) = Result.bind

let register_all ops held windows =
  List.fold_left
    (fun acc w ->
      let* () = acc in
      match attempt "register" (fun () -> ops.register w) with
      | Ok () ->
          held.windows <- w :: held.windows ;
          Ok ()
      | Error e -> Error (Inconclusive e))
    (Ok ())
    windows

(* Copy [window] to the device and back, and compare with the file's bytes. *)
let check_bytes ops ~into (window, expected) =
  (* Every byte starts as the complement of the expected one, so that a copy
     that writes nothing can match no content whatever (an earlier fixed
     sentinel passed on a file holding that byte). *)
  for i = 0 to Bigarray.Array1.dim into - 1 do
    Bigarray.Array1.unsafe_set
      into
      i
      (Char.unsafe_chr (255 - Char.code (Bigarray.Array1.get expected i)))
  done ;
  match attempt "copy" (fun () -> ops.round_trip window ~into) with
  | Error e -> Error (Inconclusive e)
  | Ok () ->
      if equal_bytes into expected then Ok () else Error (Refused Wrong_bytes)

(* Every registration is released and the first failure is the verdict. A
   release that fails is [Still_registered] unless the second try in [release]
   succeeds (then [Inconclusive]): it can be transient, and a cached refusal
   would switch the capability off for good. The window that could not be
   released stays in [held] so that [release] tries it once more; a window is
   out of [held] while it is being released, so an interrupt during the call
   leaves none that would be unregistered twice. *)
let unregister_checked ops held =
  let first_error = ref None in
  let rec go failed = function
    | [] -> held.windows <- failed
    | w :: pending -> (
        held.windows <- List.rev_append failed pending ;
        match attempt "unregister" (fun () -> ops.unregister w) with
        | Ok () -> go failed pending
        | Error e ->
            if Option.is_none !first_error then first_error := Some e ;
            go (w :: failed) pending)
  in
  go [] held.windows ;
  match !first_error with None -> Ok () | Some e -> Error (Still_registered e)

let timing_line ~min_relative_speed ~t_registered ~t_staged =
  Printf.sprintf
    "probe timing: registered %.2f ms, staged %.2f ms (best of %d, registered \
     = %.2f of staged speed, accepted from %.2f)"
    (t_registered *. 1e3)
    (t_staged *. 1e3)
    timing_trials
    (t_staged /. t_registered)
    min_relative_speed

(* A reference that differs from the mapping is the caller's fault (the file
   changed between the mapping and the read, a wrong region): no verdict on
   the device, and nothing registered yet. *)
let check_inputs ~window ~ref_window =
  if
    equal_bytes (window 0) (ref_window 0)
    && equal_bytes (window 1) (ref_window 1)
  then Ok ()
  else Error (Inconclusive "reference differs from the mapping (file changed?)")

(* Two windows with the same bytes cannot tell a copy of the right window from
   a copy of the other one (a device that always reads the first would pass):
   the caller picks another region (not an all-zero or repeated one). *)
let check_windows_differ ~ref_window =
  if equal_bytes (ref_window 0) (ref_window 1) then
    Error (Inconclusive "the two probe windows hold the same bytes")
  else Ok ()

let run ops held ~log ~min_relative_speed ~window ~ref_window ~half =
  let* () = check_inputs ~window ~ref_window in
  let* () = check_windows_differ ~ref_window in
  let into = alloc ops held half in
  let* () = register_all ops held [window 0; window 1] in
  (* A mismatch is the device's only if the file still holds what the caller
     read: another process may have written it meanwhile. *)
  let check_window k =
    match check_bytes ops ~into (window k, ref_window k) with
    | Error (Refused Wrong_bytes) ->
        let* () = check_inputs ~window ~ref_window in
        Error (Refused Wrong_bytes)
    | r -> r
  in
  let* () = check_window 0 in
  let* () = check_window 1 in
  let staging = alloc ops held half in
  let staged () =
    Bigarray.Array1.blit (window 0) staging ;
    ops.upload staging
  in
  let t_registered = best_of timing_trials (fun () -> ops.upload (window 0)) in
  let t_staged = best_of timing_trials staged in
  log (timing_line ~min_relative_speed ~t_registered ~t_staged) ;
  let* () = unregister_checked ops held in
  if Float.is_nan t_registered || Float.is_nan t_staged then
    Ok (Inconclusive "the clock gave no valid duration (stepped?)")
  else if t_registered *. min_relative_speed <= t_staged then Ok Usable
  else
    Ok
      (Too_slow
         (Printf.sprintf
            "registered copy slower than the staged one (%.2f ms against %.2f \
             ms, best of %d; timing only, probe again later)"
            (t_registered *. 1e3)
            (t_staged *. 1e3)
            timing_trials))

(* Whatever the machine raises, the probe has no verdict. *)
let guard f =
  match f () with
  | v -> v
  | exception e when is_fatal e -> raise e
  | exception e -> Inconclusive ("probe raised " ^ Printexc.to_string e)

let half_of len = len / 2 / page * page

let probe_with ?(log = ignore)
    ?(min_relative_speed = default_min_relative_speed) ops ~(map : host)
    ~(reference : host) =
  let len = Bigarray.Array1.dim map in
  let half = half_of len in
  if half < page || Bigarray.Array1.dim reference <> len then
    Inconclusive "probe region too small"
  else if Nativeint.rem (Transfer.bigarray_addr map) 4096n <> 0n then
    Inconclusive "the mapping does not start on a page boundary"
  else begin
    let sub a b = Bigarray.Array1.sub a (if b = 0 then 0 else half) half in
    let held = {windows = []; buffers = []} in
    let outcome =
      try
        Ok
          (guard (fun () ->
               match
                 run
                   ops
                   held
                   ~log
                   ~min_relative_speed
                   ~window:(sub map)
                   ~ref_window:(sub reference)
                   ~half
               with
               | Ok v | Error v -> v))
      with e -> Error e
    in
    (* Cleanup first: a logger that raises must not skip it. *)
    let released = release ops held in
    let log line = try log line with e when is_fatal e -> raise e | _ -> () in
    (match outcome with
    | Ok (Still_registered first) -> log ("release failed: " ^ first)
    | Ok _ | Error _ -> ()) ;
    Option.iter
      (fun why -> log ("release failed: " ^ why))
      (match released.unregister_failure with
      | Some _ as f -> f
      | None -> released.free_failure) ;
    match (outcome, released) with
    | Error e, _ -> raise e
    | Ok _, {unregister_failure = Some why; _} -> (
        match outcome with
        | Ok (Still_registered _ as v) -> v
        | _ -> Still_registered ("release after the probe failed: " ^ why))
    | Ok (Still_registered first), {free_failure = None; _} ->
        (* The retry released it: transient, nothing stays registered. *)
        Inconclusive ("a release failed once, then succeeded: " ^ first)
    | Ok _, {free_failure = Some why; _} ->
        Inconclusive ("a buffer could not be freed: " ^ why)
    | Ok v, _ -> v
  end

(* ---- operations over a real device ---- *)

let with_device_ops (dev : Device.t) ~bytes f =
  (match static_verdict dev with
  | Decided (Refused r) ->
      invalid_arg ("Mapped_registration.with_device_ops: " ^ refusal_to_string r)
  | Decided _ | Self_test_required -> ()) ;
  let device_buffer = Transfer.create_device_only dev Vector.char bytes in
  Fun.protect
    ~finally:(fun () -> Transfer.free_all_buffers device_buffer)
    (fun () ->
      f
        {
          register = (fun w -> Transfer.register_host dev w);
          unregister = (fun w -> Transfer.unregister_host w);
          alloc_host = (fun n -> Transfer.alloc_pinned dev Bigarray.char n);
          free_host = Transfer.free_pinned;
          upload =
            (fun src ->
              Transfer.copy_bytes_to_device
                src
                ~dst:device_buffer
                ~dst_byte_offset:0);
          round_trip =
            (fun src ~into ->
              Transfer.copy_bytes_to_device
                src
                ~dst:device_buffer
                ~dst_byte_offset:0 ;
              Transfer.copy_bytes_to_host
                ~src:device_buffer
                ~src_byte_offset:0
                into);
        })

let probe ?log ?min_relative_speed (dev : Device.t) ~(map : host)
    ~(reference : host) =
  match static_verdict dev with
  | Decided v -> v
  | Self_test_required ->
      let bytes = max page (half_of (Bigarray.Array1.dim map)) in
      guard (fun () ->
          with_device_ops dev ~bytes (fun ops ->
              probe_with ?log ?min_relative_speed ops ~map ~reference))
