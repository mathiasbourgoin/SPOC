(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host cost of one epilogue coverage check (Launch_epilogue.covers), the
 * uncached path a fresh process pays on its first prompt chunk. Shapes follow
 * a 128-token prefill chunk of a ~0.8B model (hidden 1024, FFN 3584): 1-D
 * element-wise producers, a 2-D output tile, a warp-per-row matvec (4M
 * threads), and a producer whose index is not affine (enumerated). No device.
 *
 *   SPOC_BENCH_REPS=7 dune exec sarek/execute/test/bench_epilogue_coverage.exe
 ******************************************************************************)

open Coverage_fixtures

let reps =
  match Sys.getenv_opt "SPOC_BENCH_REPS" with
  | Some s -> int_of_string s
  | None -> 7

let time f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  ((Unix.gettimeofday () -. t0) *. 1e6, r)

let one_d prod total () =
  covers
    ~p:(launch_1d prod ~total ~block:256)
    ~p_scalars:[("n", total); ("off", 0)]
    ~c:(launch_1d (consumer ()) ~total ~block:256)
    ~c_scalars:[("n", total)]

let tile ~rows ~cols () =
  covers
    ~p:
      (launch
         (tile_producer ~guard_col:true)
         ~grid:(Dims.dims_2d ((cols + 15) / 16) ((rows + 15) / 16))
         ~block:(Dims.dims_2d 16 16))
    ~p_scalars:[("n", cols); ("off", 0); ("m", rows)]
    ~c:(launch_1d (consumer ()) ~total:(rows * cols) ~block:256)
    ~c_scalars:[("n", rows * cols)]

(* Warp-per-row matvec over a 128-token batch: 1024 rows, 8 per work-group. *)
let warp_matvec () =
  covers
    ~p:
      (launch
         (warp_producer ())
         ~grid:(Dims.dims_2d 128 128)
         ~block:(Dims.dims_1d 256))
    ~p_scalars:[("n", 1024); ("off", 0)]
    ~c:(launch_1d (consumer ()) ~total:(128 * 1024) ~block:256)
    ~c_scalars:[("n", 128 * 1024)]

let non_affine = producer (fun i _ -> (i /! int 3 *! int 3) +! (i %! int 3))

let cases =
  [
    ("1-D 128x1024", one_d (producer (fun i _ -> i)) (128 * 1024));
    ("1-D 128x3584", one_d (producer (fun i _ -> i)) (128 * 3584));
    ("2-D tile 128x3584", tile ~rows:128 ~cols:3584);
    ("warp matvec 128x1024", warp_matvec);
    ("non-affine 128x1024", one_d non_affine (128 * 1024));
  ]

let summary xs =
  let xs = List.sort compare xs in
  let n = List.length xs in
  (List.nth xs (n / 2), List.hd xs, List.nth xs (n - 1))

let () =
  let load =
    try In_channel.with_open_text "/proc/loadavg" input_line
    with Sys_error _ -> "n/a"
  in
  Printf.printf "# reps %d, load %s\n" reps load ;
  List.iter
    (fun (name, f) ->
      let runs = List.init reps (fun _ -> time f) in
      let med, lo, hi = summary (List.map fst runs) in
      let verdicts = List.sort_uniq compare (List.map snd runs) in
      Printf.printf
        "%-22s us/check median %10.1f  min %10.1f  max %10.1f  covers %s\n%!"
        name
        med
        lo
        hi
        (String.concat "/" (List.map string_of_bool verdicts)))
    cases
