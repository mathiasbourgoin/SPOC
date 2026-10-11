(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host-side push-constant block of the Vulkan backend (no device): the block
 * is laid out by Sarek_push_constants.place, exactly at the 128-byte limit it
 * is accepted, one scalar more it is refused with a backend error, and a
 * mixed 32/64-bit block has the padding the launch window's byte count
 * assumes.
 ******************************************************************************)

open Sarek_vulkan
module K = Vulkan_api_kernel
module BE = Sarek_backend_error.Backend_error

let int32_args n =
  let args = K.create_args () in
  for i = 0 to n - 1 do
    K.set_arg_int32 args i (Int32.of_int i)
  done ;
  args

let size_of = function None -> 0 | Some pc -> Bytes.length pc

let test_exactly_at_the_limit () =
  Alcotest.(check int)
    "32 int32 scalars fill the 128-byte block"
    Sarek_push_constants.vulkan_block_limit
    (size_of (K.build_push_constants (int32_args 32)))

let test_one_past_the_limit_is_refused () =
  match K.build_push_constants (int32_args 33) with
  | _ -> Alcotest.fail "a 132-byte block was accepted"
  | exception BE.Backend_error _ -> ()

let test_padding_matches_the_shared_layout () =
  let args = K.create_args () in
  K.set_arg_int32 args 0 7l ;
  K.set_arg_int64 args 1 9L ;
  K.set_arg_int32 args 2 5l ;
  let pc =
    match K.build_push_constants args with
    | Some pc -> pc
    | None -> Alcotest.fail "no block for three scalars"
  in
  let offsets, total =
    Sarek_push_constants.place [Sarek_push_constants.W4; W8; W4]
  in
  Alcotest.(check (list int)) "offsets" [0; 8; 16] offsets ;
  Alcotest.(check int) "size" total (Bytes.length pc) ;
  Alcotest.(check int32) "int32 at 0" 7l (Bytes.get_int32_le pc 0) ;
  Alcotest.(check int64) "int64 at 8" 9L (Bytes.get_int64_le pc 8) ;
  Alcotest.(check int32) "int32 at 16" 5l (Bytes.get_int32_le pc 16)

let () =
  Alcotest.run
    "Vulkan_push_constant_budget"
    [
      ( "block",
        [
          Alcotest.test_case
            "exactly at the limit"
            `Quick
            test_exactly_at_the_limit;
          Alcotest.test_case
            "one past the limit is refused"
            `Quick
            test_one_past_the_limit_is_refused;
          Alcotest.test_case
            "padding matches the shared layout"
            `Quick
            test_padding_matches_the_shared_layout;
        ] );
    ]
