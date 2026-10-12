(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Ranged copies and pinned / registered host memory on Vulkan
 * (specs/vulkan-async-api.md), on a real device.
 *
 * - Byte-exact round trips at odd offsets, for every pairing of device buffer
 *   (host-mapped / forced unmapped) and host range (pageable / pinned /
 *   registered); bytes outside the range stay untouched. A positive control
 *   shows the comparison goes red on a single flipped byte.
 * - A pinned or registered range takes the staging-free path: the staging
 *   buffer count does not move (a silent fallback to staging would pass the
 *   byte comparison and fail here).
 * - A copy is ordered against launches that were not synchronised: the host
 *   read after a launch sees its result, a launch after a copy sees the copy.
 * - Misuse is refused: out-of-range copies, a free of an unknown address, a
 *   double registration, unregistering an allocation.
 *
 * Skips when no Vulkan device is present.
 ******************************************************************************)

open Sarek_vulkan
open Vulkan_async_support
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Stream = Vulkan_api_stream
module Transfer = Vulkan_api_transfer
module Host = Vulkan_api_host

type host = {
  label : string;
  bytes :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  addr : nativeint;
  release : unit -> unit;
}

let char_array n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let address_of ba =
  Ctypes.raw_address_of_ptr (Spoc_core.Memory.borrow_bigarray_ptr ba)

let pageable n =
  let bytes = char_array n in
  {label = "pageable"; bytes; addr = address_of bytes; release = ignore}

let registered device n =
  let backing = char_array (n + 4096) in
  (* An odd start: the import must round outwards to whole pages. *)
  let bytes = Bigarray.Array1.sub backing 123 n in
  let addr = address_of bytes in
  Host.register device addr ~byte_size:n ;
  {
    label = "registered";
    bytes;
    addr;
    release =
      (fun () ->
        Host.unregister device addr ;
        ignore (Sys.opaque_identity backing));
  }

let pinned device n =
  let addr = Host.alloc_pinned device ~byte_size:n in
  let bytes =
    Bigarray.array1_of_genarray
      ( Ctypes.bigarray_of_ptr
          Ctypes.array1
          n
          Bigarray.char
          (Ctypes.coerce
             Ctypes.(ptr void)
             Ctypes.(ptr char)
             (Ctypes.ptr_of_raw_address addr))
      |> fun a -> Bigarray.genarray_of_array1 a )
  in
  {
    label = "pinned";
    bytes;
    addr;
    release = (fun () -> Host.free_pinned device addr);
  }

let fill_random ba =
  for i = 0 to Bigarray.Array1.dim ba - 1 do
    Bigarray.Array1.unsafe_set ba i (Char.chr (Random.int 256))
  done

(** Number of bytes of [a] and [b] that differ over [len] bytes. *)
let mismatches a ~a_off b ~b_off len =
  let bad = ref 0 in
  for i = 0 to len - 1 do
    if Bigarray.Array1.get a (a_off + i) <> Bigarray.Array1.get b (b_off + i)
    then incr bad
  done ;
  !bad

let with_unmapped unmapped f =
  Atomic.set Memory.force_unmapped unmapped ;
  Fun.protect ~finally:(fun () -> Atomic.set Memory.force_unmapped false) f

let size = 1 lsl 20

let range_off = 4097

let range_len = 300_001

let host_kinds device = [pageable; registered device; pinned device]

(* One pairing: zero the device buffer, write a random range at an odd offset,
   read the whole buffer back. *)
let roundtrip device ~unmapped make_host =
  with_unmapped unmapped (fun () ->
      let buf = Memory.alloc device size Bigarray.char in
      Alcotest.(check bool)
        "forced unmapped buffer has no host mapping"
        true
        ((not unmapped) || buf.Memory.mapped_ptr = None) ;
      let blank = pageable size in
      Bigarray.Array1.fill blank.bytes '\000' ;
      Transfer.host_to_device
        ~src_ptr:blank.addr
        ~dst:buf
        ~dst_offset:0
        ~byte_size:size ;
      let src = make_host size in
      fill_random src.bytes ;
      let staging0 = Atomic.get Memory.staging_buffers_created in
      Transfer.host_to_device
        ~src_ptr:(Nativeint.add src.addr (Nativeint.of_int 17))
        ~dst:buf
        ~dst_offset:range_off
        ~byte_size:range_len ;
      let out = make_host size in
      Bigarray.Array1.fill out.bytes '\255' ;
      Transfer.device_to_host
        ~src:buf
        ~src_offset:0
        ~dst_ptr:out.addr
        ~byte_size:size ;
      let staged = Atomic.get Memory.staging_buffers_created - staging0 in
      let what = Printf.sprintf "%s, unmapped=%b" src.label unmapped in
      Alcotest.(check int)
        (what ^ ": range bytes")
        0
        (mismatches src.bytes ~a_off:17 out.bytes ~b_off:range_off range_len) ;
      Alcotest.(check int)
        (what ^ ": bytes before the range stay zero")
        0
        (mismatches blank.bytes ~a_off:0 out.bytes ~b_off:0 range_off) ;
      let after = range_off + range_len in
      Alcotest.(check int)
        (what ^ ": bytes after the range stay zero")
        0
        (mismatches
           blank.bytes
           ~a_off:after
           out.bytes
           ~b_off:after
           (size - after)) ;
      (* Positive control: the comparison must see one flipped byte. *)
      Bigarray.Array1.set
        out.bytes
        (range_off + 5)
        (Char.chr
           ((Char.code (Bigarray.Array1.get out.bytes (range_off + 5)) + 1)
           land 255)) ;
      Alcotest.(check int)
        (what ^ ": control (one flipped byte is detected)")
        1
        (mismatches src.bytes ~a_off:17 out.bytes ~b_off:range_off range_len) ;
      if src.label <> "pageable" && unmapped then
        Alcotest.(check int)
          (what ^ ": staging buffers used by imported copies")
          0
          staged ;
      if src.label = "pageable" && unmapped then
        Alcotest.(check bool)
          (what ^ ": pageable copies stage")
          true
          (staged >= 2) ;
      src.release () ;
      out.release () ;
      blank.release () ;
      Memory.free buf)

let test_roundtrips device =
  List.iter
    (fun unmapped ->
      List.iter
        (fun make -> roundtrip device ~unmapped (fun n -> make n))
        (host_kinds device))
    [false; true]

(* The mapped, cached or uncached, read path and the DMA path agree. *)
let test_large_dma_write_into_mapped device =
  let n = Transfer.dma_min_bytes + 4099 in
  let buf = Memory.alloc device n Bigarray.char in
  let src = pinned device n in
  fill_random src.bytes ;
  Transfer.host_to_device ~src_ptr:src.addr ~dst:buf ~dst_offset:0 ~byte_size:n ;
  let out = pageable n in
  Transfer.device_to_host ~src:buf ~src_offset:0 ~dst_ptr:out.addr ~byte_size:n ;
  Alcotest.(check int)
    "pinned write of dma_min_bytes+ into a buffer, read back"
    0
    (mismatches src.bytes ~a_off:0 out.bytes ~b_off:0 n) ;
  src.release () ;
  Memory.free buf

let test_ordering_with_launches device =
  let kernel = step_kernel device in
  let bytes = n * 4 in
  let a = Memory.alloc device n Bigarray.float32 in
  let b = Memory.alloc device n Bigarray.float32 in
  let host = pinned device bytes in
  let floats =
    Bigarray.array1_of_genarray
      ( Ctypes.bigarray_of_ptr
          Ctypes.array1
          n
          Bigarray.float32
          (Ctypes.coerce
             Ctypes.(ptr void)
             Ctypes.(ptr float)
             (Ctypes.ptr_of_raw_address host.addr))
      |> fun x -> Bigarray.genarray_of_array1 x )
  in
  let stream = Stream.create device in
  let other = Stream.create device in
  for round = 0 to 19 do
    Bigarray.Array1.fill floats (float_of_int (round * 10)) ;
    (* copy -> launch (unsynchronised, on a stream) -> copy back at once *)
    Transfer.host_to_device
      ~src_ptr:host.addr
      ~dst:a
      ~dst_offset:0
      ~byte_size:bytes ;
    launch_step ~stream kernel [a; b] ;
    launch_step ~stream:other kernel [b; a] ;
    Transfer.device_to_host
      ~src:a
      ~src_offset:0
      ~dst_ptr:host.addr
      ~byte_size:bytes ;
    let want = float_of_int ((round * 10) + 2) in
    let wrong = ref 0 in
    for i = 0 to n - 1 do
      if Bigarray.Array1.get floats i <> want then incr wrong
    done ;
    Alcotest.(check int)
      (Printf.sprintf "round %d: copy -> launch -> launch -> copy back" round)
      0
      !wrong
  done ;
  (* A copy over the input of a launch that has not run yet must not change what
     the launch reads. *)
  for round = 0 to 19 do
    Bigarray.Array1.fill floats (float_of_int round) ;
    Transfer.host_to_device
      ~src_ptr:host.addr
      ~dst:a
      ~dst_offset:0
      ~byte_size:bytes ;
    launch_step ~stream kernel [a; b] ;
    Bigarray.Array1.fill floats (-1000.0) ;
    Transfer.host_to_device
      ~src_ptr:host.addr
      ~dst:a
      ~dst_offset:0
      ~byte_size:bytes ;
    Transfer.device_to_host
      ~src:b
      ~src_offset:0
      ~dst_ptr:host.addr
      ~byte_size:bytes ;
    let want = float_of_int (round + 1) in
    let wrong = ref 0 in
    for i = 0 to n - 1 do
      if Bigarray.Array1.get floats i <> want then incr wrong
    done ;
    Alcotest.(check int)
      (Printf.sprintf "round %d: copy over a pending launch's input" round)
      0
      !wrong
  done ;
  Stream.destroy stream ;
  Stream.destroy other ;
  host.release () ;
  Memory.free a ;
  Memory.free b

let expect_invalid what f =
  match f () with
  | () -> Alcotest.failf "%s: no Invalid_argument" what
  | exception Invalid_argument _ -> ()

let test_refusals device =
  let buf = Memory.alloc device 1024 Bigarray.char in
  let host = pageable 4096 in
  expect_invalid "range past the end" (fun () ->
      Transfer.host_to_device
        ~src_ptr:host.addr
        ~dst:buf
        ~dst_offset:1000
        ~byte_size:25) ;
  expect_invalid "negative offset" (fun () ->
      Transfer.device_to_host
        ~src:buf
        ~src_offset:(-1)
        ~dst_ptr:host.addr
        ~byte_size:4) ;
  Transfer.host_to_device
    ~src_ptr:host.addr
    ~dst:buf
    ~dst_offset:1024
    ~byte_size:0 ;
  expect_invalid "free_pinned of an unknown address" (fun () ->
      Host.free_pinned device host.addr) ;
  expect_invalid "unregister of an unknown address" (fun () ->
      Host.unregister device host.addr) ;
  let reg = registered device 5000 in
  expect_invalid "double registration" (fun () ->
      Host.register device reg.addr ~byte_size:5000) ;
  expect_invalid "free_pinned of a registered range" (fun () ->
      Host.free_pinned device reg.addr) ;
  reg.release () ;
  let alloc = Host.alloc_pinned device ~byte_size:5000 in
  expect_invalid "unregister of an allocation" (fun () ->
      Host.unregister device alloc) ;
  Host.free_pinned device alloc ;
  expect_invalid "double free_pinned" (fun () -> Host.free_pinned device alloc) ;
  expect_invalid "alloc_pinned of 0 bytes" (fun () ->
      ignore (Host.alloc_pinned device ~byte_size:0)) ;
  Memory.free buf

let test_event_query device =
  let module Event = Vulkan_api_event in
  let kernel = step_kernel device in
  let e = Event.create_with_device device in
  Alcotest.(check bool) "never recorded: complete" true (Event.query e) ;
  let a = upload device zeros in
  let b = upload device zeros in
  let stream = Stream.create device in
  for _ = 1 to 20 do
    launch_step ~stream kernel [a; b]
  done ;
  Event.record e stream ;
  Event.synchronize e ;
  Alcotest.(check bool) "complete after synchronize" true (Event.query e) ;
  Event.record e stream ;
  let deadline = Unix.gettimeofday () +. 20.0 in
  while (not (Event.query e)) && Unix.gettimeofday () < deadline do
    Unix.sleepf 0.001
  done ;
  Alcotest.(check bool) "complete by polling alone" true (Event.query e) ;
  Event.destroy e ;
  Stream.destroy stream ;
  Memory.free a ;
  Memory.free b

let tests =
  [
    Alcotest.test_case
      "byte-exact round trips, every path"
      `Quick
      (with_device test_roundtrips);
    Alcotest.test_case
      "pinned write above the DMA threshold"
      `Quick
      (with_device test_large_dma_write_into_mapped);
    Alcotest.test_case
      "ordering against unsynchronised launches"
      `Quick
      (with_device test_ordering_with_launches);
    Alcotest.test_case "misuse is refused" `Quick (with_device test_refusals);
    Alcotest.test_case "event_query" `Quick (with_device test_event_query);
  ]

let () = Alcotest.run "vulkan_ranged_copies" [("ranged copies", tests)]
