(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The warp shuffle contract of docs/plans/warp-shuffle-intake-2026-10-17.md
   section 4.3, as Sarek_warp.source_lane implements it, over the boundary
   operands of the plan and every warp width the backends meet. *)

open Alcotest

let widths = [8; 16; 32; 64]

let boundary w = [0; 1; w - 1; w; w + 1; -1; (2 * w) + 1; -w; min_int / 2]

let test_idx_masks_the_operand () =
  List.iter
    (fun w ->
      List.iter
        (fun x ->
          for lane = 0 to w - 1 do
            check
              int
              (Printf.sprintf "idx W=%d x=%d" w x)
              (x land (w - 1))
              (Sarek_warp.source_lane Sarek_warp.Idx ~width:w ~lane x)
          done)
        (boundary w))
    widths

let test_relative_out_of_range_is_own () =
  List.iter
    (fun w ->
      for lane = 0 to w - 1 do
        for x = -3 to (2 * w) + 3 do
          let down = Sarek_warp.source_lane Sarek_warp.Down ~width:w ~lane x in
          let up = Sarek_warp.source_lane Sarek_warp.Up ~width:w ~lane x in
          let xr = Sarek_warp.source_lane Sarek_warp.Xor ~width:w ~lane x in
          check
            int
            "down"
            (if x < 0 || lane + x >= w then lane else lane + x)
            down ;
          check int "up" (if x < 0 || lane - x < 0 then lane else lane - x) up ;
          check
            int
            "xor"
            (if x < 0 || lane lxor x >= w then lane else lane lxor x)
            xr ;
          List.iter
            (fun r -> check bool "source in [0, W)" true (r >= 0 && r < w))
            [down; up; xr]
        done
      done)
    widths

(* Spot values worth reading as numbers: W=32. *)
let test_known_values () =
  let s op lane x = Sarek_warp.source_lane op ~width:32 ~lane x in
  check int "idx 33 -> 1" 1 (s Sarek_warp.Idx 5 33) ;
  check int "idx -1 -> 31" 31 (s Sarek_warp.Idx 5 (-1)) ;
  check int "down 1 at lane 31 -> own" 31 (s Sarek_warp.Down 31 1) ;
  check int "down 32 -> own" 5 (s Sarek_warp.Down 5 32) ;
  check int "up 1 at lane 0 -> own" 0 (s Sarek_warp.Up 0 1) ;
  check int "up 33 -> own" 5 (s Sarek_warp.Up 5 33) ;
  check int "xor 4 at lane 3 -> 7" 7 (s Sarek_warp.Xor 3 4) ;
  check int "xor 32 -> own" 3 (s Sarek_warp.Xor 3 32) ;
  check int "xor -1 -> own" 3 (s Sarek_warp.Xor 3 (-1))

let test_names () =
  check int "twelve typed shuffles" 12 (List.length Sarek_warp.typed_names) ;
  List.iter
    (fun (name, call) ->
      check
        bool
        (name ^ " classifies")
        true
        (Sarek_warp.classify name = Some call) ;
      check bool (name ^ " is a shuffle") true (Sarek_warp.is_shuffle_name name) ;
      check bool (name ^ " is a warp name") true (Sarek_warp.is_warp_name name))
    Sarek_warp.typed_names ;
  check
    bool
    "warp_size"
    true
    (Sarek_warp.classify "warp_size" = Some Sarek_warp.Warp_size) ;
  check
    bool
    "lane_id"
    true
    (Sarek_warp.classify "lane_id" = Some Sarek_warp.Lane_id) ;
  check
    bool
    "lane_id is not a shuffle"
    false
    (Sarek_warp.is_shuffle_name "lane_id") ;
  check
    bool
    "the user-facing names are not IR names"
    true
    (Sarek_warp.classify "warp_shuffle" = None
    && Sarek_warp.is_user_shuffle_name "warp_shuffle") ;
  check
    bool
    "an unrelated name"
    true
    (Sarek_warp.classify "block_barrier" = None)

let test_simulated_width () =
  check int "default" 32 (Sarek_warp.simulated_width ()) ;
  Sarek_warp.with_width 16 (fun () ->
      check int "inside" 16 (Sarek_warp.simulated_width ())) ;
  check int "restored" 32 (Sarek_warp.simulated_width ()) ;
  (match Sarek_warp.with_width 8 (fun () -> failwith "boom") with
  | () -> fail "expected the exception"
  | exception Failure _ ->
      check int "restored after an exception" 32 (Sarek_warp.simulated_width ())) ;
  check
    int
    "lane of thread 70 at W=32"
    6
    (Sarek_warp.lane_of_thread ~width:32 70) ;
  check
    int
    "linear id"
    (3 + (8 * (2 + (4 * 1))))
    (Sarek_warp.linear_thread_id ~block:(8, 4, 2) (3, 2, 1)) ;
  check
    bool
    "whole warps"
    true
    (Sarek_warp.block_is_whole_warps ~block:(8, 4, 1) ~width:32) ;
  check
    bool
    "partial warp"
    false
    (Sarek_warp.block_is_whole_warps ~block:(8, 5, 1) ~width:32)

let () =
  run
    "Sarek_warp"
    [
      ( "contract",
        [
          test_case "idx masks the operand" `Quick test_idx_masks_the_operand;
          test_case
            "relative modes: out of range is own"
            `Quick
            test_relative_out_of_range_is_own;
          test_case "known values" `Quick test_known_values;
        ] );
      ( "names and simulation",
        [
          test_case "names" `Quick test_names;
          test_case "simulated width" `Quick test_simulated_width;
        ] );
    ]
