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

let run_on (dev : Device.t) =
  Printf.printf "  [%s] %s\n%!" dev.Device.framework dev.Device.name ;
  test_device_only dev ;
  test_ranged dev ;
  test_streams_pinned dev ;
  test_destroyed dev

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
  let s = Transfer.create_stream dev in
  check
    "launch on a stream refused before running (Feature_not_supported)"
    (refuses_capability (fun () -> launch ~stream:s dev v v 16)) ;
  Transfer.destroy_stream s

let exit_code () = if !failures = 0 then 0 else 1
