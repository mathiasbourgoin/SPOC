(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Ranged copies issued from several domains at once on Vulkan
 * (specs/vulkan-async-api.md). A regression guard, not proof of a property:
 * each domain checks its own bytes exactly, and it caught a use-after-collect
 * of the submission structures (see H7 keep-alive runs). That every copy has a
 * command pool of its own holds by construction; this test does not
 * discriminate a shared-pool mutant. Skips when no Vulkan device is present.
 ******************************************************************************)

open Sarek_vulkan
open Vulkan_async_support
module Memory = Vulkan_api_memory
module Transfer = Vulkan_api_transfer

let domains = 4

let rounds = 60

let size = 64 * 1024

let char_array n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let address_of ba =
  Ctypes.raw_address_of_ptr (Spoc_core.Memory.borrow_bigarray_ptr ba)

(** Bytes that differ after writing a domain-specific pattern and reading it
    back [rounds] times through an unmapped (staged) buffer. *)
let worker device id =
  let buf = Memory.alloc device size Bigarray.char in
  let src = char_array size and out = char_array size in
  let bad = ref 0 in
  for round = 1 to rounds do
    Bigarray.Array1.fill src (Char.chr (((id * 31) + round) land 255)) ;
    Transfer.host_to_device
      ~src_ptr:(address_of src)
      ~dst:buf
      ~dst_offset:0
      ~byte_size:size ;
    Bigarray.Array1.fill out '\000' ;
    Transfer.device_to_host
      ~src:buf
      ~src_offset:0
      ~dst_ptr:(address_of out)
      ~byte_size:size ;
    for i = 0 to size - 1 do
      if Bigarray.Array1.get out i <> Bigarray.Array1.get src i then incr bad
    done
  done ;
  ignore (Sys.opaque_identity (src, out)) ;
  Memory.free buf ;
  !bad

let test_concurrent_copies device =
  Atomic.set Memory.force_unmapped true ;
  (* The FFI bindings are lazy and forcing one from two domains at once raises
     [Lazy.Undefined]: a pre-existing property of the backend, so warm up. *)
  ignore (worker device 99) ;
  let results =
    Fun.protect
      ~finally:(fun () -> Atomic.set Memory.force_unmapped false)
      (fun () ->
        List.init domains (fun id -> Domain.spawn (fun () -> worker device id))
        |> List.map Domain.join)
  in
  Alcotest.(check (list int))
    "bytes differing per domain"
    (List.init domains (fun _ -> 0))
    results

let () =
  Alcotest.run
    "vulkan_copy_domains"
    [
      ( "domains",
        [
          Alcotest.test_case
            "concurrent ranged copies from several domains"
            `Quick
            (with_device test_concurrent_copies);
        ] );
    ]
