(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Device-only vectors, ranged copies, pinned memory, stream launches and
 * events on Metal hardware (specs/metal-ranged-copies.md): the same suite
 * test_device_buffers.exe runs on the CPU backends and HIP runs from
 * sarek-hip/test, plus the Metal-only checks (maxBufferLength refusal, page
 * alignment of pinned memory, copy -> kernel -> copy ordering on one buffer).
 * Self-skips without a Metal device. Small sizes: the GPU is shared.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Suite = Device_buffers_suite
module Metal = Sarek_metal.Metal_plugin_base.Metal

let () = Sarek_metal.Metal_plugin.init ()

(* FR-002: one byte over maxBufferLength is refused, naming the limit, before
   Metal is asked. Checked on the backend allocation every Metal buffer goes
   through, and through Transfer.create_device_only when the element count
   fits an int32 (on large-memory Macs limit/8 exceeds Int32.max_int, which
   core refuses first, with Invalid_argument). *)
let test_max_buffer_length (dev : Device.t) =
  let api_dev = Sarek_metal.Metal_api.Device.get dev.Device.backend_id in
  let limit = Sarek_metal.Metal_api.Device.max_buffer_length api_dev in
  Printf.printf
    "    info maxBufferLength = %d bytes, recommendedMaxWorkingSetSize = %d \
     bytes, total_global_mem = %Ld bytes\n\
     %!"
    limit
    (Sarek_metal.Metal_api.Device.recommended_working_set api_dev)
    dev.Device.capabilities.total_global_mem ;
  Suite.check "maxBufferLength is positive" (limit > 0) ;
  Suite.check
    "total_global_mem is recommendedMaxWorkingSetSize"
    (Int64.to_int dev.Device.capabilities.total_global_mem
    = Sarek_metal.Metal_api.Device.recommended_working_set api_dev) ;
  let mdev = Metal.Device.get dev.Device.backend_id in
  let names_limit f =
    match f () with
    | _ -> false
    | exception (Sarek_backend_error.Backend_error.Backend_error _ as e) ->
        let msg = Printexc.to_string e in
        let has sub =
          let n = String.length sub and m = String.length msg in
          let rec go i =
            i + n <= m && (String.sub msg i n = sub || go (i + 1))
          in
          go 0
        in
        has "maxBufferLength" && has (string_of_int limit)
    | exception e ->
        Printf.printf "      (unexpected %s)\n" (Printexc.to_string e) ;
        false
  in
  Suite.check
    "backend allocation above maxBufferLength refused, naming the limit"
    (names_limit (fun () ->
         Metal.Memory.alloc_custom mdev ~size:(limit + 1) ~elem_size:1)) ;
  let elems = (limit / 8) + 1 in
  if elems <= Int32.to_int Int32.max_int then
    Suite.check
      "create_device_only above maxBufferLength refused, naming the limit"
      (names_limit (fun () ->
           Transfer.create_device_only dev Vector.int64 elems))
  else
    Printf.printf
      "    info create_device_only cannot reach maxBufferLength (%d int64 > \
       Int32.max_int): backend check only\n"
      elems

(* FR-009: page-aligned pinned memory; unknown addresses refused. *)
let test_pinned_alignment (dev : Device.t) =
  let page = Sarek_metal.Metal_api.page_size () in
  Printf.printf "    info page size = %d bytes\n%!" page ;
  let ba = Transfer.alloc_pinned dev Bigarray.char 100 in
  let addr =
    Ctypes.raw_address_of_ptr
      (Ctypes.to_voidp (Ctypes.bigarray_start Ctypes.array1 ba))
  in
  Suite.check
    (Printf.sprintf "pinned address is %d-byte aligned" page)
    (Nativeint.rem addr (Nativeint.of_int page) = 0n) ;
  Bigarray.Array1.fill ba 'z' ;
  Transfer.free_pinned ba ;
  let mdev = Metal.Device.get dev.Device.backend_id in
  let ordinary = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 64 in
  Suite.check
    "backend free_pinned of a non-pinned address refused"
    (Suite.refuses_invalid (fun () ->
         Metal.Async.free_pinned
           mdev
           (Ctypes.raw_address_of_ptr
              (Ctypes.to_voidp (Ctypes.bigarray_start Ctypes.array1 ordinary))))) ;
  ignore (Sys.opaque_identity ordinary) ;
  Suite.check
    "register_host of zero bytes refused by the backend"
    (Suite.refuses_invalid (fun () ->
         Metal.Async.register_host mdev 0n ~byte_size:0))

(* EC-1: a zero-byte copy at offset = size is accepted and changes nothing. *)
let test_zero_bytes (dev : Device.t) =
  let v = Transfer.create_device_only dev Vector.int32 4 in
  Transfer.copy_to_device
    (Bigarray.Array1.of_array
       Bigarray.int32
       Bigarray.c_layout
       [|1l; 2l; 3l; 4l|])
    ~dst:v
    ~dst_offset:0 ;
  Transfer.copy_bytes_to_device
    (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 0)
    ~dst:v
    ~dst_byte_offset:16 ;
  let out = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 4 in
  Transfer.copy_to_host ~src:v ~src_offset:0 out ;
  Suite.check
    "zero-byte copy at offset = size accepted, contents unchanged"
    (Bigarray.Array1.get out 0 = 1l && Bigarray.Array1.get out 3 = 4l) ;
  Transfer.free_all_buffers v

(* Asynchronous launches (Metal_api.Pending): many dependent launches with no
   synchronisation in between, then one host read. Each check would see a
   stale or torn value if a launch did not wait for the work it depends on. *)
let all_equal (v : (int32, Bigarray.int32_elt) Vector.t) n want =
  let out = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  Transfer.copy_to_host ~src:v ~src_offset:0 out ;
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if Bigarray.Array1.get out i <> want then incr bad
  done ;
  !bad

let test_async_launches (dev : Device.t) =
  let n = 1 lsl 20 in
  let x = Transfer.create_device_only dev Vector.int32 n in
  let y = Transfer.create_device_only dev Vector.int32 n in
  let zeros = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  Bigarray.Array1.fill zeros 0l ;
  (* Same queue, read-after-write chain: 400 launches, one read. *)
  Transfer.copy_to_device zeros ~dst:x ~dst_offset:0 ;
  for _ = 1 to 200 do
    Suite.launch dev x y n ;
    Suite.launch dev y x n
  done ;
  let bad = all_equal x n 400l in
  Suite.check
    (Printf.sprintf
       "400 dependent launches without sync, then one read (%d mismatches)"
       bad)
    (bad = 0) ;
  (* Write-after-read: a ranged copy into x right after a launch that reads x
     must wait for it; y keeps the old x + 1. *)
  let fresh = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  Bigarray.Array1.fill fresh 1000l ;
  Suite.launch dev x y n ;
  Transfer.copy_to_device fresh ~dst:x ~dst_offset:0 ;
  let bad_y = all_equal y n 401l and bad_x = all_equal x n 1000l in
  Suite.check
    (Printf.sprintf
       "copy into a buffer a pending kernel reads waits for it (%d/%d \
        mismatches)"
       bad_y
       bad_x)
    (bad_y = 0 && bad_x = 0) ;
  (* Two streams (two command queues) alternating on the same buffers: the
     cross-queue wait orders them. *)
  let a = Transfer.create_stream dev and b = Transfer.create_stream dev in
  Transfer.copy_to_device zeros ~dst:x ~dst_offset:0 ;
  for _ = 1 to 100 do
    Suite.launch ~stream:a dev x y n ;
    Suite.launch ~stream:b dev y x n
  done ;
  let bad = all_equal x n 200l in
  Suite.check
    (Printf.sprintf
       "200 launches alternating over two streams, no sync (%d mismatches)"
       bad)
    (bad = 0) ;
  (* Events: recorded after pending work, complete once synchronised. *)
  let e = Transfer.create_event dev in
  Suite.launch ~stream:a dev x y n ;
  Transfer.record_event e a ;
  Transfer.synchronize_event e ;
  Suite.check "event complete after synchronize_event" (Transfer.event_query e) ;
  Transfer.destroy_event e ;
  Transfer.synchronize_stream a ;
  Transfer.synchronize_stream b ;
  Transfer.destroy_stream a ;
  Transfer.destroy_stream b ;
  (* Ordinary vectors through the auto-transfer path: host element reads
     after unsynchronised launches. *)
  let u = Vector.create Vector.int32 4096
  and w = Vector.create Vector.int32 4096 in
  Vector.fill u 7l ;
  for _ = 1 to 50 do
    Suite.launch dev u w 4096 ;
    Suite.launch dev w u 4096
  done ;
  Suite.check
    "auto-transfer read-back after 100 unsynchronised launches"
    (Vector.get u 0 = 107l && Vector.get u 4095 = 107l) ;
  Transfer.free_all_buffers x ;
  Transfer.free_all_buffers y

(* FR-015 / FR-006, deterministically: right after a launch its command
   buffer is still in the pending table (a synchronous launch would have
   waited for and dropped it), and a launch on a stream lands on that
   stream's queue, not the default one. *)
let test_async_structure (dev : Device.t) =
  let n = 4096 in
  let x = Transfer.create_device_only dev Vector.int32 n in
  let y = Transfer.create_device_only dev Vector.int32 n in
  let zeros = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  Bigarray.Array1.fill zeros 0l ;
  Transfer.copy_to_device zeros ~dst:x ~dst_offset:0 ;
  Transfer.flush dev ;
  let pending () = Sarek_metal.Metal_api.Pending.queues_with_pending () in
  Suite.check "nothing pending after flush" (pending () = 0) ;
  Suite.launch dev x y n ;
  Suite.check "a launch returns before it is waited for" (pending () = 1) ;
  let s = Transfer.create_stream dev in
  let u = Transfer.create_device_only dev Vector.int32 n in
  let w = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device zeros ~dst:u ~dst_offset:0 ;
  Suite.launch ~stream:s dev u w n ;
  Suite.check
    "a launch on a stream goes to the stream's own queue"
    (pending () = 2) ;
  Transfer.synchronize_stream s ;
  Suite.check "synchronize_stream waits for its queue only" (pending () = 1) ;
  Transfer.flush dev ;
  Suite.check "flush waits for everything" (pending () = 0) ;
  (* Batching: 200 launches over 1 Mi elements keep the GPU busier than the
     host, so dispatches share command buffers. A per-launch commit (each
     batch ended after one dispatch) gives exactly 200. *)
  let big = 1 lsl 20 in
  let bx = Transfer.create_device_only dev Vector.int32 big in
  let by = Transfer.create_device_only dev Vector.int32 big in
  let c0 = Atomic.get Sarek_metal.Metal_api.Pending.commits in
  for _ = 1 to 100 do
    Suite.launch dev bx by big ;
    Suite.launch dev by bx big
  done ;
  let commits = Atomic.get Sarek_metal.Metal_api.Pending.commits - c0 in
  Transfer.flush dev ;
  Printf.printf "    info 200 launches -> %d command buffers\n%!" commits ;
  Suite.check
    "dispatches are batched into shared command buffers"
    (commits < 100) ;
  Transfer.free_all_buffers bx ;
  Transfer.free_all_buffers by ;
  Transfer.destroy_stream s ;
  (* Event timing (backend Event.elapsed): GPU end times of the recorded
     command buffers, a sane positive span (it used to be host record times,
     and for a fresh start event about the epoch). *)
  let mdev = Metal.Device.get dev.Device.backend_id in
  let q = Metal.Stream.default mdev in
  let e1 = Metal.Event.create () and e2 = Metal.Event.create () in
  Suite.launch dev u w n ;
  Metal.Event.record e1 q ;
  for _ = 1 to 50 do
    Suite.launch dev u w n ;
    Suite.launch dev w u n
  done ;
  Metal.Event.record e2 q ;
  let ms = Metal.Event.elapsed ~start:e1 ~stop:e2 in
  Printf.printf "    info 100 launches between two events: %.3f ms\n%!" ms ;
  Suite.check "event elapsed is a sane span" (ms > 0.0 && ms < 10_000.0) ;
  (* An event recorded after the GPU went idle must not be dated back to the
     end of the earlier command buffer: host time before the record is not
     part of the span. *)
  Suite.launch dev u w n ;
  Unix.sleepf 0.05 ;
  Metal.Event.record e1 q ;
  Suite.launch dev u w n ;
  Metal.Event.record e2 q ;
  let ms2 = Metal.Event.elapsed ~start:e1 ~stop:e2 in
  Printf.printf "    info one launch after a 50 ms host pause: %.3f ms\n%!" ms2 ;
  Suite.check "an event's time is not earlier than its record" (ms2 < 40.0) ;
  Metal.Event.destroy e1 ;
  Metal.Event.destroy e2 ;
  List.iter Transfer.free_all_buffers [x; y; u; w]

(* Two domains at once: one launches dependent kernels on its buffers while
   the other copies into and reads back its own buffers. Both results must be
   exact and nothing may raise (Pending's table is shared). *)
let test_two_domains (dev : Device.t) =
  let n = 1 lsl 16 in
  let mk () = Transfer.create_device_only dev Vector.int32 n in
  let x = mk () and y = mk () and c = mk () in
  let zeros = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  Bigarray.Array1.fill zeros 0l ;
  Transfer.copy_to_device zeros ~dst:x ~dst_offset:0 ;
  let copier =
    Domain.spawn (fun () ->
        let src = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
        let out = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
        let bad = ref 0 in
        for r = 1 to 200 do
          Bigarray.Array1.fill src (Int32.of_int r) ;
          Transfer.copy_to_device src ~dst:c ~dst_offset:0 ;
          Transfer.copy_to_host ~src:c ~src_offset:0 out ;
          if Bigarray.Array1.get out (n - 1) <> Int32.of_int r then incr bad
        done ;
        !bad)
  in
  for _ = 1 to 200 do
    Suite.launch dev x y n ;
    Suite.launch dev y x n
  done ;
  let copy_bad = Domain.join copier in
  let bad = all_equal x n 400l in
  Suite.check
    (Printf.sprintf
       "two domains: 400 launches (%d mismatches) alongside 200 copies (%d)"
       bad
       copy_bad)
    (bad = 0 && copy_bad = 0) ;
  List.iter Transfer.free_all_buffers [x; y; c]

(* Copy -> kernel -> copy -> kernel on the SAME buffers, no flush in between:
   each launch must see the copy issued just before it, and each copy must
   not land while the previous kernel still reads. Repeated so a race would
   show. *)
let test_ordering (dev : Device.t) =
  let n = 1 lsl 16 in
  let x = Transfer.create_device_only dev Vector.int32 n in
  let y = Transfer.create_device_only dev Vector.int32 n in
  let src = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  let out = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  let bad = ref 0 in
  for round = 0 to 49 do
    Bigarray.Array1.fill src (Int32.of_int (round * 1000)) ;
    Transfer.copy_to_device src ~dst:x ~dst_offset:0 ;
    Suite.launch dev x y n ;
    Transfer.copy_to_host ~src:y ~src_offset:0 out ;
    let want = Int32.of_int ((round * 1000) + 1) in
    for i = 0 to n - 1 do
      if Bigarray.Array1.get out i <> want then incr bad
    done
  done ;
  Suite.check
    (Printf.sprintf
       "copy -> kernel -> copy ordering over 50 rounds (%d mismatches)"
       !bad)
    (!bad = 0) ;
  Transfer.free_all_buffers x ;
  Transfer.free_all_buffers y

let () =
  let devices = Device.init () in
  let metal =
    Array.to_list devices |> List.filter (fun d -> d.Device.framework = "Metal")
  in
  match metal with
  | [] ->
      print_endline "test_metal_device_buffers: [SKIP] no Metal device present" ;
      exit 0
  | devs ->
      List.iter
        (fun d ->
          Suite.run_on d ;
          test_max_buffer_length d ;
          test_pinned_alignment d ;
          test_zero_bytes d ;
          test_ordering d ;
          test_async_launches d ;
          test_async_structure d ;
          test_two_domains d)
        devs ;
      exit (Suite.exit_code ())
