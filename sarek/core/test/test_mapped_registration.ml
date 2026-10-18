(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Mapped_registration: the capability to copy from registered file-backed
 * pages. No GPU: the static decision is checked against fake devices (the
 * driver is never reachable from them, so a verdict instead of
 * [Inconclusive] proves it was not called), the self-test against doubles that
 * fail the way hipHostRegister does (a success that copies nothing, a release
 * that fails), and the real Transfer plumbing on the Native device (where
 * registration is a no-op and the mapping is read-write, Unix.map_file: it
 * checks the plumbing, not file-backed registration, which no test here runs).
 ******************************************************************************)

open Spoc_core
module M = Mapped_registration
module Q = Spoc_framework.Shared_memory_query

let () = ignore (Lazy.force Sarek_native.Native_plugin.registered_backend)

let failures = ref 0

let check name cond =
  if cond then Printf.printf "ok   %s\n%!" name
  else begin
    incr failures ;
    Printf.printf "FAIL %s\n%!" name
  end

let caps ~is_cpu : Spoc_framework.Framework_sig.capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = [];
    coopmat = None;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu;
  }

(* A device no backend serves: any driver call on it raises. *)
let fake ?(is_cpu = false) ?(backend_id = 0) framework : Device.t =
  {
    id = 0;
    backend_id;
    name = "fake " ^ framework;
    framework;
    capabilities = caps ~is_cpu;
  }

let page = 65536

let host n : M.host = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let host_of_len n : M.host =
  Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let pattern n : M.host =
  let a = host n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set
      a
      i
      (Char.chr
         (((i * 7) + ((i lsr 8) * 13) + (i lsr 12) + ((i lsr 20) * 5)) land 0xff))
  done ;
  a

let static_is dev expected = M.static_verdict dev = expected

let refused r = M.Decided (M.Refused r)

let is_refused_with p = function M.Decided (M.Refused r) -> p r | _ -> false

(* ---- static decision ---- *)

let test_static () =
  check
    "Vulkan is refused as a kernel hazard"
    (is_refused_with
       (function M.Kernel_hazard _ -> true | _ -> false)
       (M.static_verdict (fake "Vulkan"))) ;
  check
    "Vulkan on a CPU device is still refused"
    (is_refused_with
       (function M.Kernel_hazard _ -> true | _ -> false)
       (M.static_verdict (fake ~is_cpu:true "Vulkan"))) ;
  check
    "Metal needs no registration"
    (match M.static_verdict (fake "Metal") with
    | M.Decided (M.Not_needed _) -> true
    | _ -> false) ;
  List.iter
    (fun fw ->
      check
        (fw ^ " needs no registration")
        (match M.static_verdict (fake ~is_cpu:(fw = "Native") fw) with
        | M.Decided (M.Not_needed _) -> true
        | _ -> false))
    ["Native"; "Interpreter"] ;
  check
    "OpenCL GPU is not measured"
    (is_refused_with
       (function M.Not_measured _ -> true | _ -> false)
       (M.static_verdict (fake "OpenCL"))) ;
  check
    "OpenCL CPU device reads host memory"
    (match M.static_verdict (fake ~is_cpu:true "OpenCL") with
    | M.Decided (M.Not_needed _) -> true
    | _ -> false) ;
  check
    "unknown framework is not measured"
    (is_refused_with
       (function M.Not_measured _ -> true | _ -> false)
       (M.static_verdict (fake "Future")))

let test_static_registering () =
  (* No query installed yet for these frameworks: refused, not guessed. *)
  List.iter
    (fun fw ->
      check
        (fw ^ " without a query is Device_kind_unknown")
        (static_is (fake fw) (refused M.Device_kind_unknown)))
    ["HIP"; "CUDA/PTX"; "CUDA/C"] ;
  Q.set ~framework:"HIP" (function
    | 0 -> false
    | 1 -> true
    | _ -> failwith "no such device") ;
  check
    "discrete HIP asks for the self-test"
    (static_is (fake "HIP") M.Self_test_required) ;
  check
    "integrated HIP is refused"
    (static_is (fake ~backend_id:1 "HIP") (refused M.Integrated_gpu)) ;
  check
    "a query that raises is Device_kind_unknown"
    (static_is (fake ~backend_id:2 "HIP") (refused M.Device_kind_unknown)) ;
  Q.set ~framework:"CUDA/PTX" (fun _ -> false) ;
  check
    "discrete CUDA/PTX asks for the self-test"
    (static_is (fake "CUDA/PTX") M.Self_test_required) ;
  check
    "CUDA/C is independent of CUDA/PTX"
    (static_is (fake "CUDA/C") (refused M.Device_kind_unknown)) ;
  Q.set ~framework:"CUDA/C" (fun _ -> raise Sys.Break) ;
  check
    "an interrupt in a query reaches the caller"
    (match M.static_verdict (fake "CUDA/C") with
    | _ -> false
    | exception Sys.Break -> true) ;
  Q.set ~framework:"CUDA/PTX" (fun _ -> true) ;
  check
    "a later query replaces the earlier one"
    (static_is (fake "CUDA/PTX") (refused M.Integrated_gpu))

(* The probe of a device decided statically never reaches the driver nor reads
   the arrays: a fake device has no backend, so a driver call would raise and
   the verdict would be Inconclusive. *)
let test_probe_static_never_touches () =
  Q.set ~framework:"HIP" (function 1 -> true | _ -> false) ;
  let map = pattern (4 * page) and reference = pattern (4 * page) in
  let v = M.probe (fake "Vulkan") ~map ~reference in
  check
    "probe on Vulkan is Refused (Kernel_hazard)"
    (match v with M.Refused (M.Kernel_hazard _) -> true | _ -> false) ;
  List.iter
    (fun (var, value) -> Unix.putenv var value)
    [
      ("SPOC_VULKAN_HOST_IMPORT", "1");
      ("SPOC_VULKAN_HOST_IMPORT_AMDGPU_I_ACCEPT_KERNEL_OOPS", "1");
    ] ;
  check
    "the Vulkan opt-in and the amdgpu override do not change the answer"
    (match M.probe (fake "Vulkan") ~map ~reference with
    | M.Refused (M.Kernel_hazard _) -> true
    | _ -> false) ;
  Unix.putenv "SPOC_VULKAN_HOST_IMPORT" "0" ;
  Unix.putenv "SPOC_VULKAN_HOST_IMPORT_AMDGPU_I_ACCEPT_KERNEL_OOPS" "0" ;
  check
    "with_device_ops refuses a Vulkan device before any registration"
    (match M.with_device_ops (fake "Vulkan") ~bytes:page (fun _ -> ()) with
    | () -> false
    | exception Invalid_argument _ -> true) ;
  check
    "probe on an integrated HIP device is Refused before any driver call"
    (match M.probe (fake ~backend_id:1 "HIP") ~map ~reference with
    | M.Refused M.Integrated_gpu -> true
    | _ -> false) ;
  check
    "probe on Metal is Not_needed whatever the arrays hold"
    (match M.probe (fake "Metal") ~map ~reference:(host 1) with
    | M.Not_needed _ -> true
    | _ -> false)

let test_cacheable () =
  check "Usable is cacheable" (M.cacheable M.Usable) ;
  check
    "static verdicts are not cacheable"
    ((not (M.cacheable (M.Not_needed "x")))
    && (not (M.cacheable (M.Refused M.Integrated_gpu)))
    && not (M.cacheable (M.Refused M.Device_kind_unknown))) ;
  check "Wrong_bytes is cacheable" (M.cacheable (M.Refused M.Wrong_bytes)) ;
  check
    "Refused Device_kind_unknown is not cacheable"
    (not (M.cacheable (M.Refused M.Device_kind_unknown))) ;
  check "Too_slow is not cacheable" (not (M.cacheable (M.Too_slow "x"))) ;
  check "Inconclusive is not cacheable" (not (M.cacheable (M.Inconclusive "x")))

(* ---- the self-test against doubles ---- *)

type double = {
  ops : M.ops;
  registered : int ref;
  unregistered : int ref;
  allocated : M.host list ref;
  freed : int ref;
  mutable failing_unregister : bool;
}

(* A device that copies faithfully; the registered path is [registered_delay]
   slower than the staged one. *)
let double ?(register = fun _ -> ()) ?(round_trip = None)
    ?(registered_delay = 0.) () =
  let registered = ref 0 and unregistered = ref 0 and freed = ref 0 in
  let allocated = ref [] in
  let d = ref None in
  let faithful src ~into = Bigarray.Array1.blit src into in
  let ops =
    {
      M.register =
        (fun w ->
          register w ;
          incr registered);
      unregister =
        (fun _ ->
          match !d with
          | Some x when x.failing_unregister -> failwith "unregister: 713"
          | _ -> incr unregistered);
      alloc_host =
        (fun n ->
          let b = host n in
          allocated := b :: !allocated ;
          b);
      free_host = (fun _ -> incr freed);
      round_trip = (match round_trip with Some f -> f | None -> faithful);
      upload =
        (fun src ->
          if not (List.memq src !allocated) then Unix.sleepf registered_delay);
    }
  in
  let x =
    {
      ops;
      registered;
      unregistered;
      allocated;
      freed;
      failing_unregister = false;
    }
  in
  d := Some x ;
  x

let balanced d = !(d.registered) = !(d.unregistered)

let nothing_left d = balanced d && !(d.freed) = List.length !(d.allocated)

let any_speed = 0.

let is_inconclusive = function M.Inconclusive _ -> true | _ -> false

let halves_differ (a : M.host) =
  let half = Bigarray.Array1.dim a / 2 in
  let lo = Bigarray.Array1.sub a 0 half
  and hi = Bigarray.Array1.sub a half half in
  let rec go i =
    i < half
    && (Bigarray.Array1.get lo i <> Bigarray.Array1.get hi i || go (i + 1))
  in
  go 0

let test_engine () =
  let map = pattern (8 * page) and reference = pattern (8 * page) in
  let d = double () in
  check
    "faithful device is Usable"
    (M.probe_with ~min_relative_speed:any_speed d.ops ~map ~reference = M.Usable) ;
  check "two disjoint windows registered, released" (!(d.registered) = 2) ;
  check "everything released" (nothing_left d) ;
  (* A file that holds the byte a fixed sentinel would have been. *)
  let a5 = host (8 * page) in
  Bigarray.Array1.fill a5 '\xA5' ;
  let a5_ref = host (8 * page) in
  Bigarray.Array1.fill a5_ref '\xA5' ;
  (* the windows differ in their last byte only *)
  Bigarray.Array1.set a5 ((8 * page) - 1) '\x01' ;
  Bigarray.Array1.set a5_ref ((8 * page) - 1) '\x01' ;
  check
    "identical windows give no verdict (Inconclusive)"
    (is_inconclusive
       (M.probe_with
          ~min_relative_speed:any_speed
          (double ()).ops
          ~map:
            ( host (8 * page) |> fun h ->
              Bigarray.Array1.fill h 'x' ;
              h )
          ~reference:
            ( host (8 * page) |> fun h ->
              Bigarray.Array1.fill h 'x' ;
              h ))) ;
  check
    "a copy that writes nothing is Wrong_bytes even on a file of 0xA5"
    (M.probe_with
       ~min_relative_speed:any_speed
       (double ~round_trip:(Some (fun _ ~into:_ -> ())) ()).ops
       ~map:a5
       ~reference:a5_ref
    = M.Refused M.Wrong_bytes) ;
  let wrong = double ~round_trip:(Some (fun _ ~into:_ -> ())) () in
  check
    "a copy that returns the sentinel is Wrong_bytes (false success)"
    (M.probe_with ~min_relative_speed:any_speed wrong.ops ~map ~reference
    = M.Refused M.Wrong_bytes) ;
  check "released after Wrong_bytes" (nothing_left wrong) ;
  let zeros =
    double
      ~round_trip:(Some (fun _ ~into -> Bigarray.Array1.fill into '\000'))
      ()
  in
  check
    "a copy of zeros is Wrong_bytes"
    (M.probe_with ~min_relative_speed:any_speed zeros.ops ~map ~reference
    = M.Refused M.Wrong_bytes) ;
  let other = pattern (8 * page) in
  Bigarray.Array1.set other (page + 5) '\255' ;
  let d2 = double () in
  check
    "a reference that differs from the mapping is the caller's fault: \
     Inconclusive, nothing registered"
    (is_inconclusive
       (M.probe_with ~min_relative_speed:any_speed d2.ops ~map ~reference:other)
    && !(d2.registered) = 0) ;
  (* A device that reads the first window whatever it is given. *)
  let first_window = Bigarray.Array1.sub map 0 (4 * page) in
  let stale =
    double
      ~round_trip:(Some (fun _ ~into -> Bigarray.Array1.blit first_window into))
      ()
  in
  check "the windows of the pattern differ" (halves_differ map) ;
  check
    "a copy that always reads the first window is Wrong_bytes"
    (M.probe_with ~min_relative_speed:any_speed stale.ops ~map ~reference
    = M.Refused M.Wrong_bytes) ;
  (* A release that fails after a Wrong_bytes is not a cacheable refusal. *)
  let leaky = double ~round_trip:(Some (fun _ ~into:_ -> ())) () in
  leaky.failing_unregister <- true ;
  check
    "Wrong_bytes with a failing release is Inconclusive"
    (is_inconclusive
       (M.probe_with ~min_relative_speed:any_speed leaky.ops ~map ~reference))

let test_failures () =
  let map = pattern (8 * page) and reference = pattern (8 * page) in
  let r = ref 0 in
  let d =
    double
      ~register:(fun _ ->
        incr r ;
        if !r = 2 then failwith "hipErrorOutOfMemory")
      ()
  in
  check
    "second registration refused: Inconclusive"
    (is_inconclusive
       (M.probe_with ~min_relative_speed:any_speed d.ops ~map ~reference)) ;
  check "the first window is released" (!(d.unregistered) = 1) ;
  let d = double () in
  d.failing_unregister <- true ;
  check
    "a release that fails (713) is Inconclusive, not a cached refusal"
    (is_inconclusive
       (M.probe_with ~min_relative_speed:any_speed d.ops ~map ~reference)) ;
  check
    "Inconclusive is never cacheable"
    (not
       (M.cacheable
          (M.probe_with ~min_relative_speed:any_speed d.ops ~map ~reference))) ;
  let d =
    double ~round_trip:(Some (fun _ ~into:_ -> failwith "copy failed")) ()
  in
  check
    "a copy that raises is Inconclusive"
    (is_inconclusive
       (M.probe_with ~min_relative_speed:any_speed d.ops ~map ~reference)) ;
  check "released after a failing copy" (nothing_left d) ;
  let d = double ~round_trip:(Some (fun _ ~into:_ -> raise Out_of_memory)) () in
  check
    "Out_of_memory reaches the caller"
    (match M.probe_with ~min_relative_speed:any_speed d.ops ~map ~reference with
    | _ -> false
    | exception Out_of_memory -> true) ;
  check "released after Out_of_memory" (nothing_left d) ;
  check
    "region too small is Inconclusive"
    (is_inconclusive
       (M.probe_with (double ()).ops ~map:(host page) ~reference:(host page))) ;
  check
    "reference of another length is Inconclusive"
    (is_inconclusive (M.probe_with (double ()).ops ~map ~reference:(host page)))

let test_timing () =
  let map = pattern (8 * page) and reference = pattern (8 * page) in
  let slow = double ~registered_delay:0.01 () in
  check
    "a registered copy 10 ms slower than the staged one is Too_slow"
    (match M.probe_with slow.ops ~map ~reference with
    | M.Too_slow _ -> true
    | _ -> false) ;
  check "Too_slow leaves nothing registered" (nothing_left slow) ;
  let logged = ref [] in
  let ok =
    M.probe_with
      ~log:(fun l -> logged := l :: !logged)
      ~min_relative_speed:any_speed
      (double ()).ops
      ~map
      ~reference
  in
  check "the timing is logged" (ok = M.Usable && List.length !logged = 1) ;
  let strict = double ~registered_delay:0.001 () in
  check
    "the margin is a parameter: demanding 1000x the staged speed refuses"
    (match
       M.probe_with ~min_relative_speed:1000. strict.ops ~map ~reference
     with
    | M.Too_slow _ -> true
    | _ -> false)

(* ---- the real plumbing, on the Native device ---- *)

let native () =
  Array.to_list (Device.init ~frameworks:["Native"] ())
  |> List.find (fun d -> d.Device.framework = "Native")

let with_file_region ~len f =
  let path = Filename.temp_file "mapped_registration" ".bin" in
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () ->
      let fd = Unix.openfile path [Unix.O_RDWR] 0o600 in
      let content = pattern len in
      let oc = open_out_bin path in
      for i = 0 to len - 1 do
        output_char oc (Bigarray.Array1.get content i)
      done ;
      close_out oc ;
      (* The reference is read back from the file, not the generated bytes. *)
      let ic = open_in_bin path in
      let s = really_input_string ic len in
      close_in ic ;
      let reference = host_of_len len in
      String.iteri (fun i c -> Bigarray.Array1.set reference i c) s ;
      let map =
        Bigarray.array1_of_genarray
          (Unix.map_file fd Bigarray.char Bigarray.c_layout true [|len|])
      in
      Unix.close fd ;
      f ~map ~reference)

let test_native_plumbing () =
  let dev = native () in
  let len = 16 * page in
  with_file_region ~len (fun ~map ~reference ->
      check
        "probe on Native is Not_needed"
        (match M.probe dev ~map ~reference with
        | M.Not_needed _ -> true
        | _ -> false) ;
      let verdict =
        M.with_device_ops dev ~bytes:(len / 2) (fun ops ->
            M.probe_with ~min_relative_speed:any_speed ops ~map ~reference)
      in
      check
        "the real Transfer round trip returns the file's bytes"
        (verdict = M.Usable) ;
      check
        "the windows are no longer registered afterwards"
        (not (Transfer.is_pinned (Bigarray.Array1.sub map 0 (len / 2)))) ;
      let changed = pattern len in
      Bigarray.Array1.set changed ((len / 2) + 9) '\255' ;
      check
        "a reference that differs is Inconclusive through the real plumbing"
        (is_inconclusive
           (M.with_device_ops dev ~bytes:(len / 2) (fun ops ->
                M.probe_with
                  ~min_relative_speed:any_speed
                  ops
                  ~map
                  ~reference:changed))) ;
      check
        "no window stays registered afterwards"
        ((not (Transfer.is_pinned (Bigarray.Array1.sub map 0 (len / 2))))
        && not
             (Transfer.is_pinned (Bigarray.Array1.sub map (len / 2) (len / 2)))
        ))

let () =
  test_static () ;
  test_static_registering () ;
  test_probe_static_never_touches () ;
  test_cacheable () ;
  test_engine () ;
  test_failures () ;
  test_timing () ;
  test_native_plumbing () ;
  if !failures > 0 then begin
    Printf.printf "%d FAILED\n%!" !failures ;
    exit 1
  end
  else print_endline "all passed"
