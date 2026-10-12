(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Matrix-product rewrite, int32 differential (specs/matrix-pattern-rewrite.md).
 *
 * For every kernel and shape: the ORIGINAL kernel and Sarek_matrix.run (the
 * rewrite) are launched with the same arguments and geometry on the same
 * device, from the same initial C (a sentinel, so an output the original
 * leaves alone must stay alone), and compared element by element; both are
 * also compared with a host reference. Bit-identical on the interpreter and
 * on every gfx11 HIP device; every other device must decline (Original).
 ******************************************************************************)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
open Matrix_rewrite_support
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let run_case dev ~expect_rewrite case (m, n, k) =
  let nb = if case.batched then 3 else 1 in
  let alpha = 7l in
  let a = Array.init (nb * m * k) (fun _ -> random_int32 ()) in
  let b = Array.init (nb * k * n) (fun _ -> random_int32 ()) in
  let c0 =
    Array.init
      (nb * m * n)
      (fun i ->
        if case.batched then random_int32 ()
        else Int32.add sentinel (Int32.of_int i))
  in
  let block, grid = case.geometry ~m ~n ~nb in
  let go ~rewrite =
    let cv = vec c0 in
    let i x = Execute.Int32 (Int32.of_int x) in
    let args =
      [Execute.Vec (vec a); Execute.Vec (vec b); Execute.Vec cv; i m; i n; i k]
      @ if case.batched then [i nb; Execute.Int32 alpha] else []
    in
    let outcome = launch ~rewrite dev case.kernel ~args ~block ~grid in
    Transfer.flush dev ;
    (Vector.to_array cv, outcome)
  in
  let original, _ = go ~rewrite:false in
  let rewritten, outcome = go ~rewrite:true in
  let what =
    Printf.sprintf "%s %dx%dx%d on %s" case.name m n k dev.Device.name
  in
  let rewrote =
    match outcome with
    | Sarek_matrix.Rewritten _ -> true
    | Sarek_matrix.Original _ -> false
  in
  if rewrote <> expect_rewrite then
    fail
      "%s: expected %s, got %s"
      what
      (if expect_rewrite then "a rewrite" else "the original")
      (Sarek_matrix.outcome_text outcome) ;
  let same =
    compare_arrays (what ^ " rewrite vs original") original rewritten
  in
  let host_ok =
    (not case.full)
    ||
    let init x = if case.batched then x else 0l in
    let epi x = if case.batched then Int32.add (Int32.mul alpha x) 3l else x in
    compare_arrays
      (what ^ " original vs host")
      (reference ~layout:case.layout ~m ~n ~k ~nb ~init ~epi a b c0)
      original
  in
  if same && host_ok && rewrote = expect_rewrite then
    ok "%s: %s, bit-identical" what (Sarek_matrix.outcome_text outcome)

let shapes = [(1, 1, 1); (17, 33, 5); (64, 64, 64); (100, 37, 129)]

(* A kernel that does more than the product is never rewritten: the rewrite
   would drop the other write. *)
let k_extra_write =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (d : int32 vector)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let sum = mut 0l in
      for i = 0 to k - 1l do
        sum := sum + (a.((row * k) + i) * b.((i * n) + col))
      done ;
      c.((row * n) + col) <- sum ;
      d.(col) <- 1l]

let check_extra_write dev =
  let v len = Execute.Vec (vec (Array.make len 1l)) in
  let i x = Execute.Int32 (Int32.of_int x) in
  match
    Sarek_matrix.run
      ~device:dev
      ~ir:(ir_of k_extra_write)
      ~args:[v 256; v 256; v 256; v 16; i 16; i 16]
      ~block:(Execute.dims3d 16 16 1)
      ~grid:(Execute.dims3d 1 1 1)
      ()
  with
  | Sarek_matrix.Original why ->
      ok "extra write on %s: original (%s)" dev.Device.name why
  | Sarek_matrix.Rewritten _ ->
      fail
        "extra write on %s: rewritten, the other write would be lost"
        dev.Device.name

(* ---- Products the rewrite must decline (review of PR 2) ------------------- *)

let k_else_branch =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row >= m || col >= n then ()
      else begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

let k_repeated =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (n : int32)
        (k : int32)
        (reps : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      for _r = 0 to reps - 1l do
        let o = (row * n) + col in
        let sum = mut c.(o) in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.(o) <- sum
      done]

let k_while =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (n : int32)
        (k : int32)
        (go : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      let cnt = mut 0l in
      while cnt < go do
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum ;
        cnt := cnt + 1l
      done]

let k_mod_batch =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32)
        (nb : int32) ->
      let z = global_idx_x mod nb in
      let row = global_idx_y in
      if row < m then
        for col = 0 to n - 1l do
          let sum = mut 0l in
          for i = 0 to k - 1l do
            sum :=
              sum
              + a.((z * m * k) + (row * k) + i)
                * b.((z * k * n) + (i * n) + col)
          done ;
          c.((z * m * n) + (row * n) + col) <- sum
        done]

(* Rows on x: the original grid is legal, the rewritten one (rows on y) is
   not. *)
let k_rows_on_x =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_x in
      let col = global_idx_y in
      if row < m && col < n then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

(* The original and Sarek_matrix.run from the same inputs: the rewrite must
   decline (Original) and the outputs agree. *)
let check_declined dev what ir ~sizes:(la, lb, lc) ~scalars ~block ~grid =
  let a = Array.init la (fun i -> Int32.of_int ((i * 7) - 3)) in
  let b = Array.init lb (fun i -> Int32.of_int ((i * 5) + 1)) in
  let c0 = Array.init lc (fun i -> Int32.add sentinel (Int32.of_int i)) in
  let go rewrite =
    let cv = vec c0 in
    let args =
      [Execute.Vec (vec a); Execute.Vec (vec b); Execute.Vec cv]
      @ List.map (fun x -> Execute.Int32 (Int32.of_int x)) scalars
    in
    let outcome = launch ~rewrite dev ir ~args ~block ~grid in
    Transfer.flush dev ;
    (Vector.to_array cv, outcome)
  in
  let original, _ = go false in
  let rewritten, outcome = go true in
  let label = Printf.sprintf "%s on %s" what dev.Device.name in
  (match outcome with
  | Sarek_matrix.Original why -> ok "%s: declined (%s)" label why
  | Sarek_matrix.Rewritten _ -> fail "%s: rewritten" label) ;
  ignore (compare_arrays (label ^ ": results") original rewritten)

let check_declines dev =
  let d2 = Execute.dims3d 16 16 1 and d8 = Execute.dims3d 8 8 1 in
  check_declined
    dev
    "guard in an else branch"
    (ir_of k_else_branch)
    ~sizes:(64, 64, 64)
    ~scalars:[8; 8; 8]
    ~block:d2
    ~grid:(Execute.dims3d 2 2 1) ;
  List.iter
    (fun reps ->
      check_declined
        dev
        (Printf.sprintf "non-index loop, %d repetitions" reps)
        (ir_of k_repeated)
        ~sizes:(64, 64, 64)
        ~scalars:[8; 8; reps]
        ~block:d8
        ~grid:(Execute.dims3d 1 1 1))
    [0; 2] ;
  check_declined
    dev
    "product inside a while"
    (ir_of k_while)
    ~sizes:(64, 64, 64)
    ~scalars:[8; 8; 0]
    ~block:d8
    ~grid:(Execute.dims3d 1 1 1) ;
  check_declined
    dev
    "batch = thread id mod nb"
    (ir_of k_mod_batch)
    ~sizes:(3 * 8 * 8, 3 * 8 * 8, 3 * 8 * 8)
    ~scalars:[8; 8; 8; 3]
    ~block:(Execute.dims3d 1 8 1)
    ~grid:(Execute.dims3d 1 1 1)

let check_grid_limit dev =
  let m = 4_200_000 in
  check_declined
    dev
    "rows on x beyond the y grid limit"
    (ir_of k_rows_on_x)
    ~sizes:(m, 1, m)
    ~scalars:[m; 1; 1]
    ~block:(Execute.dims1d 256)
    ~grid:(Execute.dims1d ((m + 255) / 256))

let () =
  Random.init 20261011 ;
  let devices = Array.to_list (Device.init ()) in
  List.iter
    (fun dev ->
      let expect_rewrite = offers_int dev in
      check_extra_write dev ;
      check_declines dev ;
      if dev.Device.framework = "HIP" then check_grid_limit dev ;
      Printf.printf
        "device: %s [%s], rewrite expected: %b\n%!"
        dev.Device.name
        dev.Device.framework
        expect_rewrite ;
      let shapes =
        if dev.Device.framework = "Interpreter" then
          [(1, 1, 1); (17, 33, 5); (40, 24, 20)]
        else shapes
      in
      List.iter
        (fun case -> List.iter (run_case dev ~expect_rewrite case) shapes)
        cases)
    devices ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
