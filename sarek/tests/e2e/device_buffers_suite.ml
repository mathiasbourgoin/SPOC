(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Device-only vectors, ranged/stream-ordered copies, pinned host memory,
 * stream launches and events (spec: specs/device-buffers-async-copy.md).
 *
 * One body, run per device: by test_device_buffers.exe on every CPU device
 * (Native, Interpreter), CUDA and Metal device present, by
 * sarek-hip/test/test_hip_device_buffers.exe on HIP devices and by
 * sarek-metal/test/test_metal_device_buffers.exe on Metal devices. Sizes are small
 * on purpose: the GPU these tests run on is shared.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let add_one =
  [%kernel
    fun (x : int32 vector) (y : int32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then y.(tid) <- x.(tid) + 1l]

let add_one_ir =
  lazy
    (let _, kirc = add_one in
     match kirc.Sarek.Kirc_types.body_ir with
     | Some ir -> ir
     | None -> failwith "add_one: no IR")

let accumulate =
  [%kernel
    fun (acc : int32 vector) (x : int32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then acc.(tid) <- acc.(tid) + x.(tid)]

let accumulate_ir =
  lazy
    (let _, kirc = accumulate in
     match kirc.Sarek.Kirc_types.body_ir with
     | Some ir -> ir
     | None -> failwith "accumulate: no IR")

(* One thread loops [iters] times: keeps the default stream busy so that what
   is enqueued behind it is observably still pending. *)
let spin =
  [%kernel
    fun (out : int32 vector) (iters : int32) ->
      let open Std in
      let tid = global_thread_id in
      let acc = mut 0l in
      let i = mut 0l in
      while i < iters do
        acc := (acc * 3l) + i ;
        i := i + 1l
      done ;
      if tid = 0l then out.(0) <- acc]

let spin_ir =
  lazy
    (let _, kirc = spin in
     match kirc.Sarek.Kirc_types.body_ir with
     | Some ir -> ir
     | None -> failwith "spin: no IR")

let failures = ref 0

let check name cond =
  if cond then Printf.printf "    ok   %s\n%!" name
  else begin
    incr failures ;
    Printf.printf "    FAIL %s\n%!" name
  end

(** [true] iff [f ()] raises [Invalid_argument]. *)
let refuses_invalid f =
  match f () with
  | _ -> false
  | exception Invalid_argument _ -> true
  | exception e ->
      Printf.printf "      (unexpected %s)\n" (Printexc.to_string e) ;
      false

(** [true] iff [f ()] raises the [Feature_not_supported] capability refusal. *)
let refuses_capability f =
  match f () with
  | _ -> false
  | exception
      Sarek_backend_error.Backend_error.Backend_error
        (Sarek_backend_error.Backend_error.Plugin
           {
             error = Sarek_backend_error.Backend_error.Feature_not_supported _;
             _;
           }) ->
      true
  | exception e ->
      Printf.printf "      (unexpected %s)\n" (Printexc.to_string e) ;
      false

let i32_ba n f =
  let ba = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set ba i (f i)
  done ;
  ba

let read_all (v : (int32, Bigarray.int32_elt) Vector.t) =
  let out = i32_ba (Vector.length v) (fun _ -> -1l) in
  Transfer.copy_to_host ~src:v ~src_offset:0 out ;
  out

let launch ?stream dev x y n =
  let block = 64 in
  Execute.run_vectors
    ?stream
    ~device:dev
    ~ir:(Lazy.force add_one_ir)
    ~args:[Execute.Vec x; Execute.Vec y; Execute.Int n]
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d ((n + block - 1) / block))
    ()

(* ---------------------------------------------------------------- US-1 *)

let test_device_only dev =
  let n = 1000 in
  let v = Transfer.create_device_only dev Vector.int32 n in
  let on_gpu = not dev.Device.capabilities.is_cpu in
  check
    "device-only vector is device-resident"
    (match Vector.location v with Vector.GPU d -> d.id = dev.id | _ -> false) ;
  check
    "is_device_only (true off-CPU; CPU devices keep host storage)"
    (Vector.is_device_only v = on_gpu) ;
  if on_gpu then begin
    check "Vector.get refused" (refuses_invalid (fun () -> Vector.get v 0)) ;
    check "Vector.set refused" (refuses_invalid (fun () -> Vector.set v 0 1l)) ;
    check "Vector.fill refused" (refuses_invalid (fun () -> Vector.fill v 0l)) ;
    check "Vector.copy refused" (refuses_invalid (fun () -> Vector.copy v)) ;
    check
      "Vector.sub_vector refused"
      (refuses_invalid (fun () -> Vector.sub_vector v ~start:0 ~len:1 ())) ;
    check
      "Vector.to_bigarray refused"
      (refuses_invalid (fun () -> Vector.to_bigarray v)) ;
    check
      "Transfer.to_cpu refused"
      (refuses_invalid (fun () -> Transfer.to_cpu ~force:true v))
  end ;
  check
    "length 0 refused"
    (refuses_invalid (fun () -> Transfer.create_device_only dev Vector.int32 0)) ;
  check
    "length > Int32.max_int refused"
    (refuses_invalid (fun () ->
         Transfer.create_device_only
           dev
           Vector.int32
           (Int32.to_int Int32.max_int + 1))) ;
  (* Kernel argument: fill x by ranged copy, run y = x + 1 into device-only y. *)
  let x = Transfer.create_device_only dev Vector.int32 n in
  let y = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device
    (i32_ba n (fun i -> Int32.of_int (3 * i)))
    ~dst:x
    ~dst_offset:0 ;
  launch dev x y n ;
  Transfer.flush dev ;
  let out = read_all y in
  let ok = ref true in
  for i = 0 to n - 1 do
    if Bigarray.Array1.get out i <> Int32.of_int ((3 * i) + 1) then ok := false
  done ;
  check "device-only vectors as kernel arguments (y = x + 1)" !ok ;
  (* Freeing discards; the vector is unusable afterwards. *)
  Transfer.free_all_buffers y ;
  check
    "ranged copy into a freed device-only vector refused"
    ((not on_gpu)
    || refuses_invalid (fun () ->
        Transfer.copy_to_device (i32_ba 1 (fun _ -> 0l)) ~dst:y ~dst_offset:0))

(* ---------------------------------------------------------------- US-2 *)

let test_ranged dev =
  let n = 1000 in
  let v = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device (i32_ba n (fun _ -> 0l)) ~dst:v ~dst_offset:0 ;
  let ramp base len = i32_ba len (fun i -> Int32.of_int (base + i)) in
  Transfer.copy_to_device (ramp 10_000 10) ~dst:v ~dst_offset:0 ;
  Transfer.copy_to_device (ramp 20_000 10) ~dst:v ~dst_offset:500 ;
  Transfer.copy_to_device (ramp 30_000 10) ~dst:v ~dst_offset:(n - 10) ;
  check
    "offset past the end refused"
    (refuses_invalid (fun () ->
         Transfer.copy_to_device (ramp 0 10) ~dst:v ~dst_offset:(n - 5))) ;
  check
    "negative offset refused"
    (refuses_invalid (fun () ->
         Transfer.copy_to_device (ramp 0 10) ~dst:v ~dst_offset:(-1))) ;
  (* 4 int32 values as raw bytes at byte offset 8 = elements 2..5. *)
  let bytes = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  let vals = [|7l; -8l; 0x01020304l; Int32.max_int|] in
  Array.iteri
    (fun k x ->
      for b = 0 to 3 do
        Bigarray.Array1.set
          bytes
          ((4 * k) + b)
          (Char.chr
             (Int32.to_int
                (Int32.logand (Int32.shift_right_logical x (8 * b)) 0xffl)))
      done)
    vals ;
  Transfer.copy_bytes_to_device bytes ~dst:v ~dst_byte_offset:(8 + (4 * 100)) ;
  (* Raw-pointer variant: caller keeps [src] alive (it is live here). *)
  let src = ramp 40_000 3 in
  Transfer.copy_ptr_to_device
    ~src_ptr:
      (Ctypes.raw_address_of_ptr
         (Ctypes.to_voidp (Ctypes.bigarray_start Ctypes.array1 src)))
    ~byte_size:12
    ~dst:v
    ~dst_byte_offset:(4 * 700)
    () ;
  ignore (Sys.opaque_identity src) ;
  let out = read_all v in
  let expect i =
    if i < 10 then 10_000 + i
    else if i >= 500 && i < 510 then 20_000 + (i - 500)
    else if i >= n - 10 then 30_000 + (i - (n - 10))
    else if i >= 102 && i < 106 then Int32.to_int vals.(i - 102)
    else if i >= 700 && i < 703 then 40_000 + (i - 700)
    else 0
  in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if Bigarray.Array1.get out i <> Int32.of_int (expect i) then incr bad
  done ;
  check
    (Printf.sprintf "ranged copies land exactly in range (%d mismatches)" !bad)
    (!bad = 0) ;
  (* Partial read-back at an offset. *)
  let part = i32_ba 4 (fun _ -> -1l) in
  Transfer.copy_to_host ~src:v ~src_offset:(n - 4) part ;
  check
    "copy_to_host at an offset"
    (Bigarray.Array1.get part 3 = Int32.of_int (30_000 + 9)) ;
  (* Location rules on an ordinary vector. *)
  let o = Vector.create Vector.int32 16 in
  Vector.fill o 5l ;
  check
    "ranged write onto a host-authoritative vector refused"
    (refuses_invalid (fun () ->
         Transfer.copy_to_device (ramp 0 1) ~dst:o ~dst_offset:0)) ;
  Transfer.to_device o dev ;
  Transfer.copy_to_device (ramp 77 1) ~dst:o ~dst_offset:3 ;
  check
    "Both becomes Stale_CPU off-CPU (device copy is newer)"
    (dev.Device.capabilities.is_cpu
    || match Vector.location o with Vector.Stale_CPU _ -> true | _ -> false) ;
  check
    "auto-sync then reads the ranged write"
    (Vector.get o 3 = 77l && Vector.get o 2 = 5l)

(* ---------------------------------------------------------- US-3, US-4 *)

let test_streams_pinned dev =
  let n = 4096 in
  let pinned = Transfer.alloc_pinned dev Bigarray.int32 n in
  check "alloc_pinned buffer is_pinned" (Transfer.is_pinned pinned) ;
  for i = 0 to n - 1 do
    Bigarray.Array1.set pinned i (Int32.of_int (i * 7))
  done ;
  let x = Transfer.create_device_only dev Vector.int32 n in
  let y = Transfer.create_device_only dev Vector.int32 n in
  let a = Transfer.create_stream dev in
  let b = Transfer.create_stream dev in
  let e = Transfer.create_event dev in
  (* M3 pattern: copy on a, event, b waits, kernel on b. *)
  Transfer.copy_to_device ~stream:a pinned ~dst:x ~dst_offset:0 ;
  Transfer.record_event e a ;
  Transfer.stream_wait_event b e ;
  launch ~stream:b dev x y n ;
  check
    "free_pinned refused while a stream-ordered copy references it"
    (refuses_invalid (fun () -> Transfer.free_pinned pinned)) ;
  check
    "freeing a vector in flight refused"
    (refuses_invalid (fun () -> Transfer.free_all_buffers x)) ;
  Transfer.synchronize_stream b ;
  check
    "sub-array free refused"
    (refuses_invalid (fun () ->
         Transfer.free_pinned (Bigarray.Array1.sub pinned 0 16))) ;
  (* Neither stream a nor the event has been synchronised or queried: only
     the TRANSITIVE release through b's wait on e (FR-020) can have let go of
     the copy that read [pinned]. Checked before anything touches e. *)
  check
    "syncing the waiting stream releases the copy stream (transitive)"
    (match Transfer.free_pinned pinned with
    | () -> true
    | exception Invalid_argument _ -> false) ;
  check "event complete after the waiting stream" (Transfer.event_query e) ;
  let out = read_all y in
  let ok = ref true in
  for i = 0 to n - 1 do
    if Bigarray.Array1.get out i <> Int32.of_int ((i * 7) + 1) then ok := false
  done ;
  check "copy stream -> event -> compute stream gives x + 1" !ok ;
  check "is_pinned false after free" (not (Transfer.is_pinned pinned)) ;
  check
    "double free refused"
    (refuses_invalid (fun () -> Transfer.free_pinned pinned)) ;
  (* Device->host on a stream + synchronize_event. *)
  let back = i32_ba n (fun _ -> 0l) in
  Transfer.copy_to_host ~stream:a ~src:y ~src_offset:0 back ;
  Transfer.record_event e a ;
  Transfer.synchronize_event e ;
  check
    "async copy_to_host after synchronize_event"
    (Bigarray.Array1.get back (n - 1) = Int32.of_int (((n - 1) * 7) + 1)) ;
  (* Register an ordinary Bigarray. ZLUDA (CUDA on AMD) does not implement
     cuMemHostRegister (observed 2026-10-10: CUDA_ERROR_OTHER), so on a ZLUDA
     device this part is reported as a known driver limitation, not skipped
     silently; every other device must pass it. *)
  let reg = i32_ba 1024 (fun i -> Int32.of_int i) in
  match Transfer.register_host dev reg with
  | exception (Sarek_backend_error.Backend_error.Backend_error _ as exn)
    when String.ends_with ~suffix:"[ZLUDA]" dev.Device.name ->
      Printf.printf
        "    KNOWN register_host unsupported by ZLUDA (%s)\n"
        (Printexc.to_string exn) ;
      Transfer.destroy_event e ;
      Transfer.destroy_stream a ;
      Transfer.destroy_stream b ;
      Transfer.free_all_buffers x ;
      Transfer.free_all_buffers y
  | () ->
      check "registered buffer is_pinned" (Transfer.is_pinned reg) ;
      check
        "free_pinned on a registered buffer refused"
        (refuses_invalid (fun () -> Transfer.free_pinned reg)) ;
      Transfer.copy_to_device ~stream:a reg ~dst:x ~dst_offset:0 ;
      Transfer.synchronize_stream a ;
      Transfer.unregister_host reg ;
      check "unregistered" (not (Transfer.is_pinned reg)) ;
      Transfer.destroy_event e ;
      Transfer.destroy_stream a ;
      Transfer.destroy_stream b ;
      Transfer.free_all_buffers x ;
      Transfer.free_all_buffers y

(* ------------------------------------------------ keep-alive across GC *)

(* Out of line, so that no register or stack slot of the caller holds the
   source: only the stream's ledger can keep it alive. *)
let[@inline never] copy_unrooted stream ~dst collected =
  let src = i32_ba (Vector.length dst) (fun i -> Int32.of_int i) in
  Gc.finalise (fun _ -> collected := true) src ;
  Transfer.copy_to_device ~stream src ~dst ~dst_offset:0

let[@inline never] retain_unrooted stream collected =
  let cell = Bytes.create 64 in
  Gc.finalise (fun _ -> collected := true) cell ;
  Transfer.retain_on_stream stream ~keep:(Transfer.Keep cell) ~vec_ids:[]

(* A stream copy's source and a [retain_on_stream] value survive
   [Gc.full_major] until the stream is synchronised, and are released after. *)
let test_keep_alive dev =
  let dst = Transfer.create_device_only dev Vector.int32 256 in
  let s = Transfer.create_stream dev in
  let copied = ref false and retained = ref false in
  copy_unrooted s ~dst copied ;
  retain_unrooted s retained ;
  Gc.full_major () ;
  check "stream copy source alive across Gc.full_major" (not !copied) ;
  check "retain_on_stream value alive across Gc.full_major" (not !retained) ;
  Transfer.synchronize_stream s ;
  Gc.full_major () ;
  check "stream copy source released after the sync" !copied ;
  check "retain_on_stream value released after the sync" !retained ;
  Transfer.destroy_stream s ;
  Transfer.free_all_buffers dst

(* ------------------------------------- byte-level read-back (bytes-d2h) *)

(* Byte [b] of int32 [x] as stored in host memory (native byte order): the
   copy moves raw bytes, so the expected bytes are the host's own encoding,
   whatever its endianness. The int32 values reach the device through
   [copy_to_device], which moves the same raw bytes. *)
let ne_byte x b =
  let buf = Bytes.create 4 in
  Bytes.set_int32_ne buf 0 x ;
  Bytes.get buf b

let char_ba n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

(* [true] iff [bytes] holds the native-order encoding of [value (first + k)]
   for k = 0 .. len/4 - 1. *)
let holds_ne bytes ~first value =
  let ok = ref true in
  for k = 0 to (Bigarray.Array1.dim bytes / 4) - 1 do
    for b = 0 to 3 do
      if
        Bigarray.Array1.get bytes ((4 * k) + b) <> ne_byte (value (first + k)) b
      then ok := false
    done
  done ;
  !ok

let bytes_value i = Int32.of_int (i * 2654435 lxor 0x5a5a5a5a)

(* spec: specs/copy-bytes-to-host.md, scenarios 1 and 2. [v] holds
   [bytes_value] over [n] elements. *)
let test_bytes_to_host_data dev v n =
  let value = bytes_value in
  (* Byte offsets: 0, an odd multiple of 4, 4096 (one page), the last
     element. *)
  let read ~off ~len =
    let dst = char_ba len in
    Bigarray.Array1.fill dst '\xee' ;
    Transfer.copy_bytes_to_host ~src:v ~src_byte_offset:off dst ;
    dst
  in
  List.iter
    (fun (off, len) ->
      check
        (Printf.sprintf "int32 -> char bytes [%d, %d)" off (off + len))
        (holds_ne (read ~off ~len) ~first:(off / 4) value))
    [(0, 64); (4, 8); (4 * 333, 40); (4096, 4096); ((4 * n) - 4, 4)] ;
  (* Unaligned within an element: bytes 1..6. *)
  let mid = read ~off:5 ~len:6 in
  check
    "byte offset inside an element"
    (List.for_all
       (fun k ->
         Bigarray.Array1.get mid k
         = ne_byte (value ((5 + k) / 4)) ((5 + k) mod 4))
       [0; 1; 2; 3; 4; 5]) ;
  (* Another destination kind: float32 view of the int32 bits. *)
  let f = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 4 in
  Transfer.copy_bytes_to_host ~src:v ~src_byte_offset:(4 * 10) f ;
  check
    "int32 -> float32 destination keeps the bits"
    (List.for_all
       (fun k -> Int32.bits_of_float (Bigarray.Array1.get f k) = value (10 + k))
       [0; 1; 2; 3]) ;
  (* Round trip through copy_bytes_to_device: bytes [4096, 8192) of v land at
     element 7 of w. *)
  let w = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device (i32_ba n (fun _ -> 0l)) ~dst:w ~dst_offset:0 ;
  Transfer.copy_bytes_to_device
    (read ~off:4096 ~len:4096)
    ~dst:w
    ~dst_byte_offset:(4 * 7) ;
  let back = read_all w in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let expect = if i >= 7 && i < 7 + 1024 then value (1024 + i - 7) else 0l in
    if Bigarray.Array1.get back i <> expect then incr bad
  done ;
  check
    (Printf.sprintf
       "char round trip through copy_bytes_to_device (%d mismatches)"
       !bad)
    (!bad = 0) ;
  Transfer.free_all_buffers w

(* spec: specs/copy-bytes-to-host.md, scenarios 3, 4, 6 and 8. *)
let test_bytes_to_host_edges dev v n =
  Transfer.copy_bytes_to_host ~src:v ~src_byte_offset:(4 * n) (char_ba 0) ;
  check
    "zero-length read at the end accepted; one byte past it refused"
    (refuses_invalid (fun () ->
         Transfer.copy_bytes_to_host
           ~src:v
           ~src_byte_offset:((4 * n) + 1)
           (char_ba 0))) ;
  let untouched = char_ba 8 in
  Bigarray.Array1.fill untouched '\x11' ;
  check
    "range past the end refused, naming the public call (before the backend)"
    (match
       Transfer.copy_bytes_to_host
         ~src:v
         ~src_byte_offset:((4 * n) - 7)
         untouched
     with
    | () -> false
    | exception Invalid_argument msg ->
        String.starts_with ~prefix:"Transfer.copy_bytes_to_host: byte range" msg) ;
  (* An offset near max_int: [off + size] wraps negative, the check must not. *)
  check
    "offset near max_int refused (no overflow in the range check)"
    (refuses_invalid (fun () ->
         Transfer.copy_bytes_to_host
           ~src:v
           ~src_byte_offset:(max_int - 2)
           untouched)
    && refuses_invalid (fun () ->
        Transfer.copy_bytes_to_device
          untouched
          ~dst:v
          ~dst_byte_offset:(max_int - 2))
    && refuses_invalid (fun () ->
        Transfer.copy_to_host
          ~src:v
          ~src_offset:(max_int - 1)
          (i32_ba 4 (fun _ -> 0l)))) ;
  check
    "negative byte offset refused"
    (refuses_invalid (fun () ->
         Transfer.copy_bytes_to_host ~src:v ~src_byte_offset:(-1) untouched)) ;
  check
    "refused copies leave the destination untouched"
    (List.for_all
       (fun k -> Bigarray.Array1.get untouched k = '\x11')
       [0; 1; 2; 3; 4; 5; 6; 7]) ;
  check
    "copy_to_host keeps its element-range refusal"
    (match
       Transfer.copy_to_host ~src:v ~src_offset:(n - 1) (i32_ba 2 (fun _ -> 0l))
     with
    | () -> false
    | exception Invalid_argument msg ->
        String.starts_with ~prefix:"Transfer.copy_to_host: elements" msg) ;
  let o = Vector.create Vector.int32 16 in
  Vector.fill o 5l ;
  check
    "read-back of a host-authoritative vector refused"
    (refuses_invalid (fun () ->
         Transfer.copy_bytes_to_host ~src:o ~src_byte_offset:0 (char_ba 4))) ;
  let s = Transfer.create_stream dev in
  Transfer.destroy_stream s ;
  check
    "byte read-back on a destroyed stream refused"
    (refuses_invalid (fun () ->
         Transfer.copy_bytes_to_host
           ~stream:s
           ~src:v
           ~src_byte_offset:0
           (char_ba 4))) ;
  check
    "src location unchanged by a read-back"
    (match Vector.location v with Vector.GPU d -> d.id = dev.id | _ -> false) ;
  (* An ordinary vector in [Both] stays [Both]: a read-back must not mark the
     host copy stale as a ranged write does. *)
  let b = Vector.create Vector.int32 16 in
  Vector.fill b 9l ;
  Transfer.to_device b dev ;
  let is_both () =
    match Vector.location b with Vector.Both d -> d.id = dev.id | _ -> false
  in
  let was_both = is_both () in
  let nine = char_ba 8 in
  Transfer.copy_bytes_to_host ~src:b ~src_byte_offset:8 nine ;
  check
    "a Both vector keeps its location across a read-back"
    (was_both && is_both ()) ;
  check "read-back of a Both vector" (holds_ne nine ~first:0 (fun _ -> 9l))

let test_bytes_to_host dev =
  let n = 2048 in
  let v = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device (i32_ba n bytes_value) ~dst:v ~dst_offset:0 ;
  test_bytes_to_host_data dev v n ;
  test_bytes_to_host_edges dev v n ;
  Transfer.free_all_buffers v

(* spec: specs/copy-bytes-to-host.md, scenario 5. The copy is ordered after a
   launch that writes [src] on the same stream; its pinned destination is in
   flight until the stream is synchronised. *)
let test_bytes_to_host_stream dev =
  let n = 4096 in
  let x = Transfer.create_device_only dev Vector.int32 n in
  let y = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device
    (i32_ba n (fun i -> Int32.of_int (5 * i)))
    ~dst:x
    ~dst_offset:0 ;
  Transfer.copy_to_device (i32_ba n (fun _ -> 0l)) ~dst:y ~dst_offset:0 ;
  let s = Transfer.create_stream dev in
  let pinned = Transfer.alloc_pinned dev Bigarray.char (4 * n) in
  launch ~stream:s dev x y n ;
  Transfer.copy_bytes_to_host ~stream:s ~src:y ~src_byte_offset:0 pinned ;
  check
    "free_pinned refused while a byte read-back targets it"
    (refuses_invalid (fun () -> Transfer.free_pinned pinned)) ;
  Transfer.synchronize_stream s ;
  check
    "stream byte read-back ordered after the launch (y = x + 1)"
    (holds_ne pinned ~first:0 (fun i -> Int32.of_int ((5 * i) + 1))) ;
  (* Pageable destination, at an offset, released through an event. *)
  let e = Transfer.create_event dev in
  let page = char_ba 4096 in
  Transfer.copy_bytes_to_host ~stream:s ~src:y ~src_byte_offset:8192 page ;
  (* Only the read-back references [y] now (the launch was synchronised). *)
  check
    "freeing the source of a pending read-back refused"
    (refuses_invalid (fun () -> Transfer.free_all_buffers y)) ;
  Transfer.record_event e s ;
  Transfer.synchronize_event e ;
  check
    "stream byte read-back into pageable memory after synchronize_event"
    (holds_ne page ~first:2048 (fun i -> Int32.of_int ((5 * i) + 1))) ;
  Transfer.free_pinned pinned ;
  Transfer.destroy_event e ;
  Transfer.destroy_stream s ;
  Transfer.free_all_buffers x ;
  Transfer.free_all_buffers y

(* Destroyed handles are refused: the driver may recycle them. *)
let test_destroyed dev =
  let v = Transfer.create_device_only dev Vector.int32 8 in
  let s = Transfer.create_stream dev in
  let e = Transfer.create_event dev in
  Transfer.destroy_stream s ;
  Transfer.destroy_event e ;
  check
    "copy on a destroyed stream refused"
    (refuses_invalid (fun () ->
         Transfer.copy_to_device
           ~stream:s
           (i32_ba 1 (fun _ -> 0l))
           ~dst:v
           ~dst_offset:0)) ;
  check
    "destroying a stream twice refused"
    (refuses_invalid (fun () -> Transfer.destroy_stream s)) ;
  check
    "query of a destroyed event refused"
    (refuses_invalid (fun () -> Transfer.event_query e)) ;
  Transfer.free_all_buffers v

(* ------------------------------------------- staged small writes (US-1) *)

let test_staged dev =
  let n = 1000 in
  let v = Transfer.create_device_only dev Vector.int32 n in
  (* A non-zero background: fresh staging memory is zero, so an over-long copy
     would show. *)
  Transfer.copy_to_device (i32_ba n (fun _ -> -1l)) ~dst:v ~dst_offset:0 ;
  (* One source reused for every write, overwritten right after each call:
     the device must see the values at call time (FR-002). *)
  let src = i32_ba 10 (fun _ -> 0l) in
  let stage base off =
    for i = 0 to 9 do
      Bigarray.Array1.set src i (Int32.of_int (base + i))
    done ;
    Transfer.stage_to_device src ~dst:v ~dst_offset:off ;
    Bigarray.Array1.fill src (-7l)
  in
  stage 10_000 0 ;
  stage 20_000 500 ;
  stage 30_000 (n - 10) ;
  Transfer.flush dev ;
  let out = read_all v in
  let expect i =
    if i < 10 then 10_000 + i
    else if i >= 500 && i < 510 then 20_000 + (i - 500)
    else if i >= n - 10 then 30_000 + (i - (n - 10))
    else -1
  in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if Bigarray.Array1.get out i <> Int32.of_int (expect i) then incr bad
  done ;
  check
    (Printf.sprintf
       "staged writes land exactly in range, source reusable at once (%d \
        mismatches)"
       !bad)
    (!bad = 0) ;
  check
    "staged write past the end refused"
    (refuses_invalid (fun () ->
         Transfer.stage_to_device src ~dst:v ~dst_offset:(n - 5))) ;
  check
    "staged write above max_staged_bytes refused"
    (refuses_invalid (fun () ->
         Transfer.stage_to_device
           (i32_ba ((Transfer.max_staged_bytes / 4) + 1) (fun _ -> 0l))
           ~dst:(Transfer.create_device_only dev Vector.int32 (1 lsl 15))
           ~dst_offset:0)) ;
  (* Zero length: accepted at offset = length, changes nothing; refused past
     it. *)
  Transfer.stage_to_device (i32_ba 0 (fun _ -> 0l)) ~dst:v ~dst_offset:n ;
  check
    "zero-length staged write past the end refused"
    (refuses_invalid (fun () ->
         Transfer.stage_to_device
           (i32_ba 0 (fun _ -> 0l))
           ~dst:v
           ~dst_offset:(n + 1))) ;
  (* Exactly max_staged_bytes is accepted, and lands. *)
  let maxn = Transfer.max_staged_bytes / 4 in
  let big = Transfer.create_device_only dev Vector.int32 maxn in
  Transfer.stage_to_device
    (i32_ba maxn (fun i -> Int32.of_int (i * 3)))
    ~dst:big
    ~dst_offset:0 ;
  Transfer.flush dev ;
  let back = read_all big in
  check
    "staged write of exactly max_staged_bytes lands"
    (Bigarray.Array1.get back 0 = 0l
    && Bigarray.Array1.get back (maxn - 1) = Int32.of_int ((maxn - 1) * 3)) ;
  Transfer.free_all_buffers big ;
  (* Another element size, at a non-zero offset. *)
  let f = Transfer.create_device_only dev Vector.float64 16 in
  let fsrc = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 16 in
  Bigarray.Array1.fill fsrc 0.0 ;
  Transfer.copy_to_device fsrc ~dst:f ~dst_offset:0 ;
  let three = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 3 in
  Bigarray.Array1.set three 0 1.5 ;
  Bigarray.Array1.set three 1 (-2.25) ;
  Bigarray.Array1.set three 2 1e300 ;
  Transfer.stage_to_device three ~dst:f ~dst_offset:5 ;
  Transfer.flush dev ;
  let fback = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 16 in
  Transfer.copy_to_host ~src:f ~src_offset:0 fback ;
  check
    "staged float64 write at an offset"
    (Bigarray.Array1.get fback 4 = 0.0
    && Bigarray.Array1.get fback 5 = 1.5
    && Bigarray.Array1.get fback 6 = -2.25
    && Bigarray.Array1.get fback 7 = 1e300
    && Bigarray.Array1.get fback 8 = 0.0) ;
  Transfer.free_all_buffers f ;
  (* Stale_GPU (host newer) is refused on a GPU: a partial device write would
     leave neither copy authoritative. *)
  let sg = Vector.create Vector.int32 16 in
  Vector.fill sg 1l ;
  Transfer.to_device sg dev ;
  Vector.set sg 0 2l ;
  check
    "staged write onto a Stale_GPU vector refused off-CPU"
    (dev.Device.capabilities.is_cpu
    || refuses_invalid (fun () ->
        Transfer.stage_to_device (i32_ba 1 (fun _ -> 9l)) ~dst:sg ~dst_offset:3)
    ) ;
  let o = Vector.create Vector.int32 16 in
  Vector.fill o 5l ;
  check
    "staged write onto a host-authoritative vector refused"
    (refuses_invalid (fun () ->
         Transfer.stage_to_device (i32_ba 1 (fun _ -> 9l)) ~dst:o ~dst_offset:0)) ;
  Transfer.to_device o dev ;
  Transfer.stage_to_device (i32_ba 1 (fun _ -> 77l)) ~dst:o ~dst_offset:3 ;
  check
    "staged: Both becomes Stale_CPU off-CPU, auto-sync reads it"
    ((dev.Device.capabilities.is_cpu
     || match Vector.location o with Vector.Stale_CPU _ -> true | _ -> false)
    && Vector.get o 3 = 77l
    && Vector.get o 2 = 5l) ;
  Transfer.free_all_buffers v

(* Ordering and snapshot across ring wrap-around: each iteration stages x_i
   (16 KiB) and launches acc += x on the same stream, without synchronising.
   300 x 16 KiB is several times the 1 MiB staging ring. *)
let is_async (dev : Device.t) =
  let (module B : Spoc_framework.Framework_sig.BACKEND) =
    Transfer.backend_of dev
  in
  B.Async.support.async_copy

(* Queue a long single-thread kernel on the default stream and return an event
   recorded after it. *)
let busy dev =
  let out = Transfer.create_device_only dev Vector.int32 1 in
  Execute.run_vectors
    ~device:dev
    ~ir:(Lazy.force spin_ir)
    ~args:[Execute.Vec out; Execute.Int 100_000_000]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    () ;
  let e = Transfer.create_event dev in
  Transfer.record_event e (Transfer.default_stream dev) ;
  e

(* HIP/CUDA only: staging does not wait for the GPU (FR-003), snapshots a
   source the DMA would otherwise read late (FR-002, pinned source), and keeps
   the destination referenced while the copy is pending. *)
let test_staged_async dev =
  let n = 64 in
  let v = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device (i32_ba n (fun _ -> -1l)) ~dst:v ~dst_offset:0 ;
  let e = busy dev in
  let p = Transfer.alloc_pinned dev Bigarray.int32 8 in
  for i = 0 to 7 do
    Bigarray.Array1.set p i (Int32.of_int (100 + i))
  done ;
  Transfer.stage_to_device p ~dst:v ~dst_offset:0 ;
  Bigarray.Array1.fill p (-7l) ;
  Transfer.stage_to_device
    (i32_ba 8 (fun i -> Int32.of_int (200 + i)))
    ~dst:v
    ~dst_offset:8 ;
  check
    "stage_to_device returns while the GPU is still busy"
    (not (Transfer.event_query e)) ;
  check
    "freeing the destination of a pending staged copy refused"
    (refuses_invalid (fun () -> Transfer.free_all_buffers v)) ;
  Transfer.flush dev ;
  let out = read_all v in
  check
    "pinned source overwritten after the call: device has call-time values"
    (Bigarray.Array1.get out 0 = 100l
    && Bigarray.Array1.get out 7 = 107l
    && Bigarray.Array1.get out 8 = 200l
    && Bigarray.Array1.get out 16 = -1l) ;
  Transfer.free_pinned p ;
  Transfer.destroy_event e ;
  Transfer.free_all_buffers v

let test_staged_ordering ?stream ?(blocked = false) dev =
  let n = 4096 and iters = 300 in
  (* Run the finalisers of earlier tests' dead vectors now. Left to fire
     during the loop (hipFree from Gpu_memory's finaliser), they made the 7950X
     iGPU under HIP lose launches: 4 of 4 suite runs failed without this, 0 of
     4 with it (issue #427). This test is about staging, not that bug. *)
  Gc.full_major () ;
  let acc = Transfer.create_device_only dev Vector.int32 n in
  let x = Transfer.create_device_only dev Vector.int32 n in
  Transfer.copy_to_device (i32_ba n (fun _ -> 0l)) ~dst:acc ~dst_offset:0 ;
  Transfer.copy_to_device (i32_ba n (fun _ -> 0l)) ~dst:x ~dst_offset:0 ;
  let src = i32_ba n (fun _ -> 0l) in
  let value it i = Int32.of_int (((it * 31) + i) mod 1009) in
  (* Blocked: every staged copy is enqueued behind a long kernel, so the ring
     fills with copies still pending and must wait for them. *)
  let e = if blocked then Some (busy dev) else None in
  for it = 0 to iters - 1 do
    for i = 0 to n - 1 do
      Bigarray.Array1.set src i (value it i)
    done ;
    Transfer.stage_to_device ?stream src ~dst:x ~dst_offset:0 ;
    Execute.run_vectors
      ?stream
      ~device:dev
      ~ir:(Lazy.force accumulate_ir)
      ~args:[Execute.Vec acc; Execute.Vec x; Execute.Int n]
      ~block:(Execute.dims1d 64)
      ~grid:(Execute.dims1d (n / 64))
      ()
  done ;
  (match stream with
  | Some s -> Transfer.synchronize_stream s
  | None -> Transfer.flush dev) ;
  let out = read_all acc in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    let sum = ref 0 in
    for it = 0 to iters - 1 do
      sum := !sum + Int32.to_int (value it i)
    done ;
    if Bigarray.Array1.get out i <> Int32.of_int !sum then incr bad
  done ;
  check
    (Printf.sprintf
       "staged writes ordered with launches across ring wrap-around (%s, %d \
        mismatches)"
       (match (stream, blocked) with
       | None, true -> "default stream, behind a long kernel"
       | None, false -> "default stream"
       | Some _, _ -> "created stream")
       !bad)
    (!bad = 0) ;
  Option.iter Transfer.destroy_event e ;
  Transfer.free_all_buffers acc ;
  Transfer.free_all_buffers x

(* Run last: on the 7950X iGPU (HIP) destroying a stream made the later
   ordering cases lose launches far more often (5 of 5 suite runs against 1 of
   5 without it; issue #427). *)
let test_staged_destroyed_stream dev =
  let v = Transfer.create_device_only dev Vector.int32 8 in
  let s = Transfer.create_stream dev in
  Transfer.destroy_stream s ;
  check
    "staged write on a destroyed stream refused"
    (refuses_invalid (fun () ->
         Transfer.stage_to_device
           ~stream:s
           (i32_ba 1 (fun _ -> 0l))
           ~dst:v
           ~dst_offset:0)) ;
  Transfer.free_all_buffers v

let run_on (dev : Device.t) =
  Printf.printf "  [%s] %s\n%!" dev.Device.framework dev.Device.name ;
  test_device_only dev ;
  test_ranged dev ;
  test_streams_pinned dev ;
  test_keep_alive dev ;
  test_bytes_to_host dev ;
  test_bytes_to_host_stream dev ;
  test_destroyed dev ;
  test_staged dev ;
  test_staged_ordering dev ;
  if is_async dev then begin
    test_staged_async dev ;
    test_staged_ordering ~blocked:true dev
  end ;
  let s = Transfer.create_stream dev in
  test_staged_ordering ~stream:s dev ;
  Transfer.destroy_stream s ;
  test_staged_destroyed_stream dev

(** On a backend without ranged copies every entry point refuses with the
    capability error, never a fallback. *)
let check_refused (dev : Device.t) =
  Printf.printf
    "  [%s] %s (refusing backend)\n%!"
    dev.Device.framework
    dev.Device.name ;
  check
    "create_device_only refused (Feature_not_supported)"
    (refuses_capability (fun () ->
         Transfer.create_device_only dev Vector.int32 16)) ;
  check
    "alloc_pinned refused (Feature_not_supported)"
    (refuses_capability (fun () -> Transfer.alloc_pinned dev Bigarray.char 16)) ;
  let v = Vector.create Vector.int32 16 in
  Vector.fill v 0l ;
  Transfer.to_device v dev ;
  check
    "ranged copy refused (Feature_not_supported)"
    (refuses_capability (fun () ->
         Transfer.copy_to_device (i32_ba 1 (fun _ -> 0l)) ~dst:v ~dst_offset:0)) ;
  check
    "copy_bytes_to_host refused (Feature_not_supported)"
    (refuses_capability (fun () ->
         Transfer.copy_bytes_to_host
           ~src:v
           ~src_byte_offset:0
           (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4))) ;
  check
    "stage_to_device refused (Feature_not_supported)"
    (refuses_capability (fun () ->
         Transfer.stage_to_device (i32_ba 1 (fun _ -> 0l)) ~dst:v ~dst_offset:0)) ;
  let s = Transfer.create_stream dev in
  check
    "launch on a stream refused before running (Feature_not_supported)"
    (refuses_capability (fun () -> launch ~stream:s dev v v 16)) ;
  Transfer.destroy_stream s

let exit_code () = if !failures = 0 then 0 else 1
