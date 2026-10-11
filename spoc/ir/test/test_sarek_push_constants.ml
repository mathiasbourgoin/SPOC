(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_push_constants: std430 placement of the push-constant block, and the
 * size of a fused kernel's block (vector lengths first, scalars after, the
 * (offset, size) pair of every member last).
 ******************************************************************************)

open Sarek_ir_types
module P = Sarek_push_constants

let param ?(id = 0) name ty =
  DParam
    ({var_name = name; var_id = id; var_type = ty; var_mutable = false}, None)

let vec name = param name (TVec TFloat32)

let int_list = Alcotest.(list int)

let test_place_aligns_to_own_size () =
  Alcotest.check
    (Alcotest.pair int_list Alcotest.int)
    "int32, int64: 4 bytes of padding"
    ([0; 8], 16)
    (P.place [P.W4; P.W8]) ;
  Alcotest.check
    (Alcotest.pair int_list Alcotest.int)
    "int32, int32, int64: aligned already"
    ([0; 4; 8], 16)
    (P.place [P.W4; P.W4; P.W8]) ;
  Alcotest.check
    (Alcotest.pair int_list Alcotest.int)
    "no fields"
    ([], 0)
    (P.place [])

let test_block_bytes () =
  Alcotest.(check int)
    "two vectors and an int32 and a float64"
    (* lengths 0,4; int32 at 8; float64 at 16 *)
    24
    (P.block_bytes [vec "a"; param "n" TInt32; vec "b"; param "d" TFloat64])

let test_fused_block_bytes () =
  let member =
    [vec "x"; param "a" TInt32; param "b" TInt32; param "c" TInt32]
  in
  (* 16 bytes of its own + (offset, size) = 24 per member. *)
  Alcotest.(check int)
    "five members fill 120"
    120
    (P.fused_block_bytes (List.init 5 (fun _ -> member))) ;
  Alcotest.(check int)
    "six members need 144"
    144
    (P.fused_block_bytes (List.init 6 (fun _ -> member)))

let test_fused_lengths_come_first () =
  (* Two members of (vec, int64): the lengths (8 bytes) precede both int64
     fields, so the block is 8 + 8 + 8 + 8 (positions) = 32; the members' own
     blocks (16 each) plus positions (16) would give 48. *)
  let member = [vec "x"; param "wide" TInt64] in
  Alcotest.(check int)
    "fused block is not the sum of the members' blocks"
    32
    (P.fused_block_bytes [member; member])

let test_locals_do_not_count () =
  Alcotest.(check int)
    "only parameters occupy the block"
    4
    (P.block_bytes
       [
         vec "x";
         DShared ("tile", TFloat32, None);
         DLocal
           ( {var_name = "t"; var_id = 1; var_type = TInt32; var_mutable = true},
             None );
       ])

let () =
  Alcotest.run
    "Sarek_push_constants"
    [
      ( "layout",
        [
          Alcotest.test_case
            "fields align to their own size"
            `Quick
            test_place_aligns_to_own_size;
          Alcotest.test_case "block bytes" `Quick test_block_bytes;
          Alcotest.test_case "fused block bytes" `Quick test_fused_block_bytes;
          Alcotest.test_case
            "fused lengths come first"
            `Quick
            test_fused_lengths_come_first;
          Alcotest.test_case
            "locals do not count"
            `Quick
            test_locals_do_not_count;
        ] );
    ]
