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
  let ident i _ = (i /! int 3 *! int 3) +! (i %! int 3) in
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
  let ident i _ = (i /! int 3 *! int 3) +! (i %! int 3) in
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

module LE = Sarek_execute.Launch_epilogue

(* Which path decided: the counters move by exactly one per check. *)
let decided_by f =
  let before = LE.proof_counts () in
  ignore (f ()) ;
  let after = LE.proof_counts () in
  match
    ( after.affine - before.affine,
      after.enumerated - before.enumerated,
      after.over_budget - before.over_budget )
  with
  | 1, 0, 0 -> "affine"
  | 0, 1, 0 -> "enumerated"
  | 0, 0, 1 -> "over budget"
  | 0, 0, 0 -> "unknown"
  | _ -> "several"

(* Warp-per-row matrix-vector producer: [rows] rows per batch, [batches]
   batches, [gx] work-groups of [block] threads per batch. *)
let warp_pair ?(block = 256) ?(gx = 128) ?(batches = 4) ?(rows = 1024) ?c_total
    prod =
  let c_total = Option.value ~default:(rows * batches) c_total in
  ( launch prod ~grid:(Dims.dims_2d gx batches) ~block:(Dims.dims_1d block),
    [("n", rows); ("off", 0)],
    launch_1d (consumer ()) ~total:c_total ~block:256,
    [("n", c_total)] )

let warp ?block ?gx ?batches ?rows ?c_total prod =
  let p, p_scalars, c, c_scalars =
    warp_pair ?block ?gx ?batches ?rows ?c_total prod
  in
  both_paths ~p ~p_scalars ~c ~c_scalars

let test_paths () =
  let total = 128 * 3584 in
  let path f = decided_by f in
  check
    string
    "prefill-shaped 1-D pair"
    "affine"
    (path (fun () ->
         verdict
           ~p_total:total
           ~c_total:total
           ~p_n:total
           ~c_n:total
           (producer identity))) ;
  check
    string
    "2-D tile"
    "affine"
    (path (fun () -> tile ~guard_col:true ~cols:100 ~bx:7)) ;
  check
    string
    "warp-per-row matvec"
    "affine"
    (path (fun () ->
         let p, p_scalars, c, c_scalars = warp_pair (warp_producer ()) in
         covers ~p ~p_scalars ~c ~c_scalars)) ;
  check
    string
    "non-affine index"
    "enumerated"
    (path (fun () ->
         verdict
           ~p_n:1000
           ~c_n:1000
           (producer (fun i _ -> (i /! int 3 *! int 3) +! (i %! int 3))))) ;
  check
    string
    "loop"
    "enumerated"
    (path (fun () -> verdict ~p_n:500 ~c_n:1000 loop_producer))

(* The affine proof agrees with enumeration wherever both answer. *)
let agree what (affine, enumerated) =
  match affine with
  | None -> failf "%s: expected the affine form" what
  | Some a -> check bool what enumerated a

let one_d_indices =
  [
    ("i + off", (fun i off -> i +! off), `Affine);
    ("2i + off", (fun i off -> (int 2 *! i) +! off), `Maybe);
    ("off - i", (fun i off -> off +! EBinop (Sub, int 0, i)), `Maybe);
    ("0 * i + off", (fun i off -> (int 0 *! i) +! off), `Affine);
  ]

(* (off, P's n, C's n, C's threads) *)
let one_d_shapes =
  [
    (0, 300, 300, 300);
    (0, 250, 250, 256);
    (0, 300, 301, 320);
    (3, 300, 300, 300);
    (-3, 300, 200, 224);
    (0, 0, 0, 32);
    (0, 300, 0, 32);
    (5, 100, 105, 128);
    (-1, 300, 299, 299);
  ]

let one_d_case (name, index, form) (off, p_n, c_n, c_total) guarded =
  let what =
    Printf.sprintf
      "%s off=%d p_n=%d c_n=%d c_total=%d guarded=%b"
      name
      off
      p_n
      c_n
      c_total
      guarded
  in
  match
    both_paths
      ~p:(launch_1d (producer index) ~total:300 ~block:64)
      ~p_scalars:[("n", p_n); ("off", off)]
      ~c:(launch_1d (consumer ~guarded ()) ~total:c_total ~block:32)
      ~c_scalars:[("n", c_n)]
  with
  | None, _ when form = `Maybe -> ()
  | r -> agree what r

let test_affine_matches_enumeration_1d () =
  List.iter
    (fun index ->
      List.iter
        (fun shape -> List.iter (one_d_case index shape) [true; false])
        one_d_shapes)
    one_d_indices

let tile_case (guard_col, cols, bx) =
  let r =
    both_paths
      ~p:
        (launch
           (tile_producer ~guard_col)
           ~grid:(Dims.dims_2d bx 2)
           ~block:(Dims.dims_2d 16 16))
      ~p_scalars:[("n", cols); ("off", 0); ("m", 32)]
      ~c:(launch_1d (consumer ()) ~total:(32 * cols) ~block:64)
      ~c_scalars:[("n", 32 * cols)]
  in
  let what = Printf.sprintf "tile cols=%d bx=%d" cols bx in
  match r with
  | None, enumerated ->
      (* Rows further apart than a tile row is wide leave gaps. *)
      check bool (what ^ ": gaps") true (cols > 16 * bx) ;
      check bool (what ^ ": enumeration refuses") false enumerated
  | r -> agree what r

let test_affine_matches_enumeration_tile () =
  List.iter
    tile_case
    [
      (true, 32, 2); (true, 30, 2); (false, 30, 2); (false, 32, 2); (true, 40, 2);
    ]

(* The verdict reports the scalars it read, so the per-valuation cache cannot
   reuse it for other values. *)
let test_used_scalars () =
  let _, used =
    LE.covers
      ~p:(launch_1d (producer shifted) ~total:256 ~block:64)
      ~p_array:"dst"
      ~p_scalars:(scalars [("n", 256); ("off", 0)])
      ~c:(launch_1d (consumer ()) ~total:256 ~block:64)
      ~shape:(shape_of (launch_1d (consumer ()) ~total:256 ~block:64))
      ~c_scalars:(scalars [("n", 256)])
  in
  check
    (list (pair string string))
    "p n, p off, c n"
    [("c", "n"); ("p", "n"); ("p", "off")]
    (List.sort compare used)

(* A refusal found by enumeration depends on the scalar values: it must not be
   reused for another valuation (an early exit used to report no scalar read,
   so the cache served it for every value). *)
let test_refusal_per_valuation () =
  LE.clear () ;
  let ident i off = (i /! int 3 *! int 3) +! (i %! int 3) +! off in
  let p = launch_1d ~ir_id:101 (producer ident) ~total:1024 ~block:256 in
  let c = launch_1d ~ir_id:102 (consumer ()) ~total:1000 ~block:250 in
  let shape = shape_of c in
  let cached off =
    LE.cached_covers
      ~p
      ~p_array:"dst"
      ~p_scalars:(scalars [("n", 1024); ("off", off)])
      ~c
      ~shape
      ~c_scalars:(scalars [("n", 1000)])
  in
  check bool "off = -1: negative index, refused" false (cached (-1)) ;
  check bool "off = 0 afterwards: covered" true (cached 0) ;
  let dup i _ = (i /! int 3) +! (i %! int 3 *! int 0) in
  let p2 = launch_1d ~ir_id:103 (producer dup) ~total:1024 ~block:256 in
  let cached_dup n =
    LE.cached_covers
      ~p:p2
      ~p_array:"dst"
      ~p_scalars:(scalars [("n", n); ("off", 0)])
      ~c
      ~shape
      ~c_scalars:(scalars [("n", 0)])
  in
  check bool "n = 1024: stores twice, refused" false (cached_dup 1024) ;
  check bool "n = 1 afterwards: one store, C empty, covered" true (cached_dup 1)

let test_warp_matvec () =
  let affine what expected r =
    match r with
    | Some a, e ->
        check bool (what ^ " (affine)") expected a ;
        check bool (what ^ " (enumeration agrees)") e a
    | None, _ -> failf "%s: expected the affine form" what
  in
  affine "land 31, lane == 0" true (warp (warp_producer ())) ;
  affine "mod 32" true (warp (warp_producer ~use_mod:true ())) ;
  affine
    "fewer rows than threads cover"
    true
    (warp ~rows:1000 (warp_producer ())) ;
  affine
    "C past the last row"
    false
    (warp ~rows:1000 ~c_total:((1000 * 4) + 1) (warp_producer ())) ;
  affine "two lanes store a row" false (warp (warp_producer ~lane_bound:2 ())) ;
  (match warp (warp_producer ~lanes_per_row:16 ()) with
  | None, e -> check bool "rows overlap between work-groups: enumerated" false e
  | Some a, e -> check bool "rows overlap between work-groups" e a) ;
  (* Rows of 1024 apart, 800 stored per batch: gaps, so enumerated. *)
  (match warp ~gx:100 (warp_producer ()) with
  | None, e -> check bool "too few work-groups: enumerated" false e
  | Some a, e -> check bool "too few work-groups" e a) ;
  match warp ~block:250 (warp_producer ()) with
  | None, e -> check bool "block of 250: enumerated, covered" true e
  | Some a, e -> check bool "block of 250 agrees" e a

(* A launch with a zero extent runs no thread: P stores nothing, so it
   covers only a C that writes nothing (the fused kernel runs on P's grid). *)
let test_zero_threads () =
  let c1 = launch_1d (consumer ()) ~total:1 ~block:1 in
  let case what ?(c = c1) ?(c_n = 1) ~grid ~block index =
    let p =
      launch
        (producer index)
        ~grid:(Dims.dims_1d grid)
        ~block:(Dims.dims_1d block)
    in
    let p_scalars = [("n", 100); ("off", 0)] and c_scalars = [("n", c_n)] in
    let affine, enumerated = both_paths ~p ~p_scalars ~c ~c_scalars in
    let expected = c_n = 0 in
    check bool (what ^ " (enumeration)") expected enumerated ;
    Option.iter (check bool (what ^ " (affine)") expected) affine ;
    check
      bool
      (what ^ " (covers)")
      expected
      (covers ~p ~p_scalars ~c ~c_scalars)
  in
  case "grid 0, identity" ~grid:0 ~block:64 identity ;
  case "grid 0, constant index" ~grid:0 ~block:64 (fun _ off -> off) ;
  case "block 0, constant index" ~grid:1 ~block:0 (fun _ off -> off) ;
  case "block 0, identity" ~grid:4 ~block:0 identity ;
  case "grid 0, C writes nothing" ~c_n:0 ~grid:0 ~block:64 identity

(* Over the enumeration bound, a refusal reached because the affine proof
   failed for these scalar values (here a possibly negative dividend) must not
   be reused for values the proof accepts. *)
let test_over_budget_refusal_per_valuation () =
  LE.clear () ;
  let total = 1 lsl 20 in
  let index i off = (i +! off) /! int 1 in
  let p = launch_1d ~ir_id:201 (producer index) ~total ~block:256 in
  let c = launch_1d ~ir_id:202 (consumer ()) ~total ~block:256 in
  let shape = shape_of c in
  let cached off =
    LE.cached_covers
      ~p
      ~p_array:"dst"
      ~p_scalars:(scalars [("n", total); ("off", off)])
      ~c
      ~shape
      ~c_scalars:(scalars [("n", total)])
  in
  check
    bool
    "off = -1: affine unknown, over budget, refused"
    false
    (cached (-1)) ;
  check bool "off = 0 afterwards: proven" true (cached 0)

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
          test_case "zero-thread launches" `Quick test_zero_threads;
          test_case "warp-per-row matvec" `Quick test_warp_matvec;
          test_case
            "affine = enumeration, 1-D"
            `Quick
            test_affine_matches_enumeration_1d;
          test_case
            "affine = enumeration, tile"
            `Quick
            test_affine_matches_enumeration_tile;
          test_case "scalars read" `Quick test_used_scalars;
        ] );
      ( "cost",
        [
          test_case "large affine pair proven" `Quick test_large_affine_proven;
          test_case "over budget not proven" `Quick test_over_budget_not_proven;
          test_case "decided by" `Quick test_paths;
          test_case "refusal per valuation" `Quick test_refusal_per_valuation;
          test_case
            "over-budget refusal per valuation"
            `Quick
            test_over_budget_refusal_per_valuation;
        ] );
    ]
