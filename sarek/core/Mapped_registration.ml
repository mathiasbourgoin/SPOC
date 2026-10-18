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

let cacheable = function
  | Usable | Not_needed _ | Refused _ -> true
  | Too_slow _ | Inconclusive _ -> false

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
}

let page = 4096

let timing_trials = 7

let default_min_relative_speed = 0.8

let seconds f =
  let t0 = Unix.gettimeofday () in
  f () ;
  Unix.gettimeofday () -. t0

let best_of n f = List.fold_left min infinity (List.init n (fun _ -> seconds f))

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
   the first fatal exception is raised at the end. *)
let release ops held =
  let fatal = ref None in
  let each release_one x =
    try release_one x
    with e -> if is_fatal e && Option.is_none !fatal then fatal := Some e
  in
  let windows = held.windows and buffers = held.buffers in
  held.windows <- [] ;
  held.buffers <- [] ;
  List.iter (each ops.unregister) windows ;
  List.iter (each ops.free_host) buffers ;
  Option.iter raise !fatal

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

let sentinel = '\xA5'

(* Copy [window] to the device and back, and compare with the file's bytes. *)
let check_bytes ops ~into (window, expected) =
  Bigarray.Array1.fill into sentinel ;
  match attempt "copy" (fun () -> ops.round_trip window ~into) with
  | Error e -> Error (Inconclusive e)
  | Ok () ->
      if equal_bytes into expected then Ok () else Error (Refused Wrong_bytes)

(* Every registration is released and the first failure is the verdict. A
   release that fails is [Inconclusive]: it can be transient, and a cached
   refusal would switch the capability off for good. The window that could not
   be released stays in [held] so that [release] tries it once more; a window
   is out of [held] while it is being released, so an interrupt during the call
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
  match !first_error with None -> Ok () | Some e -> Error (Inconclusive e)

let timing_line ~min_relative_speed ~t_registered ~t_staged =
  Printf.sprintf
    "probe timing: registered %.2f ms, staged %.2f ms (best of %d, registered \
     = %.2f of staged speed, accepted from %.2f)"
    (t_registered *. 1e3)
    (t_staged *. 1e3)
    timing_trials
    (t_staged /. t_registered)
    min_relative_speed

let run ops held ~log ~min_relative_speed ~window ~ref_window ~half =
  let into = alloc ops held half in
  let* () = register_all ops held [window 0; window 1] in
  let* () = check_bytes ops ~into (window 0, ref_window 0) in
  let* () = check_bytes ops ~into (window 1, ref_window 1) in
  let staging = alloc ops held half in
  let staged () =
    Bigarray.Array1.blit (ref_window 0) staging ;
    ops.round_trip staging ~into
  in
  let t_registered =
    best_of timing_trials (fun () -> ops.round_trip (window 0) ~into)
  in
  let t_staged = best_of timing_trials staged in
  log (timing_line ~min_relative_speed ~t_registered ~t_staged) ;
  let* () = unregister_checked ops held in
  if t_registered *. min_relative_speed <= t_staged then Ok Usable
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
    release ops held ;
    match outcome with Ok v -> v | Error e -> raise e
  end

(* ---- operations over a real device ---- *)

let with_device_ops (dev : Device.t) ~bytes f =
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
