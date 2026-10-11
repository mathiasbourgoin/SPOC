(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The epilogue coverage check (Launch_epilogue.covers,
 * specs/epilogue-fusion.md): P stores each element of the intermediate at most
 * once, at an index the device can address, and every element C writes is one
 * P stores. "Covers" is only ever answered with proof; anything else refuses
 * the fusion.
 ******************************************************************************)

open Alcotest
open Coverage_fixtures

let identity i _ = i

let shifted i off = i +! off

let verdict ?(p_total = 1000) ?(c_total = 1000) ?(block = 256) ?(off = 0)
    ?(cons = consumer ()) ~p_n ~c_n prod =
  covers
    ~p:(launch_1d prod ~total:p_total ~block)
    ~p_scalars:[("n", p_n); ("off", off)]
    ~c:(launch_1d cons ~total:c_total ~block)
    ~c_scalars:[("n", c_n)]

let test_covered () =
  check
    bool
    "same domain"
    true
    (verdict ~p_n:1000 ~c_n:1000 (producer identity)) ;
  check bool "C smaller" true (verdict ~p_n:1000 ~c_n:500 (producer identity)) ;
  check bool "C empty" true (verdict ~p_n:10 ~c_n:0 (producer identity))

let test_not_covered () =
  check bool "P smaller" false (verdict ~p_n:500 ~c_n:1000 (producer identity)) ;
  check
    bool
    "unguarded C writes past P"
    false
    (verdict
       ~p_n:1000
       ~c_n:1000
       ~c_total:1024
       ~cons:(consumer ~guarded:false ())
       (producer identity))

let test_partial () =
  check
    bool
    "P shifted up misses the bottom"
    false
    (verdict ~off:4 ~p_n:1000 ~c_n:1000 (producer shifted)) ;
  check
    bool
    "P shifted up, C empty"
    true
    (verdict ~off:4 ~p_n:1000 ~c_n:0 (producer shifted))

let test_out_of_bounds () =
  (* P stores [-4, 1000): it covers C's [0, 1000) but writes below the array;
     the fused C would then run on negative indices too. *)
  check
    bool
    "negative store index"
    false
    (verdict ~off:(-4) ~p_total:1004 ~p_n:1004 ~c_n:1000 (producer shifted))

let test_duplicates () =
  check
    bool
    "dst.(i/2)"
    false
    (verdict ~p_n:1000 ~c_n:400 (producer (fun i _ -> i /! int 2))) ;
  check
    bool
    "two stores of dst.(i)"
    false
    (verdict
       ~p_n:1000
       ~c_n:1000
       (producer_with (fun i -> SSeq [store i; store i])))

let test_non_affine () =
  let ident i _ = (i /! int 2 *! int 2) +! (i %! int 2) in
  check
    bool
    "identity through / and mod"
    true
    (verdict ~p_n:1000 ~c_n:1000 (producer ident)) ;
  check
    bool
    "loop stores [0, 2n)"
    true
    (verdict ~p_n:500 ~c_n:1000 loop_producer) ;
  check
    bool
    "loop stores [0, 2n), C beyond"
    false
    (verdict ~p_n:500 ~c_n:1001 ~c_total:1001 loop_producer)

let tile ~guard_col ~cols ~bx =
  let m = 64 in
  covers
    ~p:
      (launch
         (tile_producer ~guard_col)
         ~grid:(Dims.dims_2d bx (m / 16))
         ~block:(Dims.dims_2d 16 16))
    ~p_scalars:[("n", cols); ("off", 0); ("m", m)]
    ~c:(launch_1d (consumer ()) ~total:(m * cols) ~block:256)
    ~c_scalars:[("n", m * cols)]

let test_tile () =
  check bool "16-aligned columns" true (tile ~guard_col:true ~cols:96 ~bx:6) ;
  check
    bool
    "ragged columns, guarded"
    true
    (tile ~guard_col:true ~cols:100 ~bx:7) ;
  check
    bool
    "ragged columns, unguarded"
    false
    (tile ~guard_col:false ~cols:100 ~bx:7)

(* Beyond what host enumeration can afford: an affine pair is still proven. *)
let test_large_affine_proven () =
  let total = 1 lsl 25 in
  check
    bool
    "2^25 threads, identity"
    true
    (verdict
       ~p_total:total
       ~c_total:total
       ~p_n:total
       ~c_n:total
       (producer identity))

(* A pair that needs enumeration and exceeds its budget is not proven. *)
let test_over_budget_not_proven () =
  let total = 1 lsl 22 in
  let ident i _ = (i /! int 2 *! int 2) +! (i %! int 2) in
  check
    bool
    "2^22 threads, non-affine"
    false
    (verdict
       ~p_total:total
       ~c_total:total
       ~p_n:total
       ~c_n:total
       (producer ident))

let () =
  run
    "epilogue_coverage"
    [
      ( "covers",
        [
          test_case "covered" `Quick test_covered;
          test_case "not covered" `Quick test_not_covered;
          test_case "partial" `Quick test_partial;
          test_case "out of bounds" `Quick test_out_of_bounds;
          test_case "duplicates" `Quick test_duplicates;
          test_case "non-affine" `Quick test_non_affine;
          test_case "2-D tile" `Quick test_tile;
        ] );
      ( "cost",
        [
          test_case "large affine pair proven" `Quick test_large_affine_proven;
          test_case "over budget not proven" `Quick test_over_budget_not_proven;
        ] );
    ]
