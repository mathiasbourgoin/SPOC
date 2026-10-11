(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test of the launch window and horizontal fusion (specs/launch-window.md)
 * on every enumerated device.
 *
 * Each scenario runs twice on a device: with the window enabled and with it
 * disabled (every launch issued immediately, the pre-change behaviour). The
 * results must be BIT-identical. On a device whose backend defers launches,
 * the window's statistics must also show the expected number of backend
 * launches:
 *   - independent: two 8-rows-per-group matvecs (same kernel), a one-row-per-
 *     group matvec with a 256-thread shared-memory tree, and a kernel that
 *     writes its own block_idx_x / grid_dim_x / thread_idx_x / global ids:
 *     4 launches -> 1 fused launch;
 *   - dependent: matvec, residual add of its output, matvec of the sum:
 *     3 launches stay 3, in order;
 *   - host write between two readers: the first launch must see the old value
 *     (Vector.set is a synchronisation point);
 *   - profiling timers around a launch flush it.
 * Separately, the fused kernel itself is run on the Interpreter and compared
 * bit for bit with the members run one by one.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Profiling = Spoc_core.Profiling

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

(* 8 rows per work-group of 256 threads, one 32-lane slice per row: the shape
   of spoc_llm's wave matvecs. y[row] = sum_b x[row*nb + b] * w[b]. *)
let mv_wave =
  [%kernel
    fun (x : float32 vector)
        (w : float32 vector)
        (y : float32 vector)
        (rows : int32)
        (nb : int32) ->
      let%shared (red : float32) = 256l in
      let tid = thread_idx_x in
      let lane = tid land 31l in
      let row = (block_idx_x * 8l) + (tid / 32l) in
      let acc = mut 0.0 in
      let%superstep partial =
        if row < rows then begin
          let b = mut lane in
          while b < nb do
            acc := acc +. (x.((row * nb) + b) *. w.(b)) ;
            b := b + 32l
          done
        end ;
        red.(tid) <- acc
      in
      let%superstep r16 =
        if lane < 16l then red.(tid) <- red.(tid) +. red.(tid + 16l)
      in
      let%superstep r8 =
        if lane < 8l then red.(tid) <- red.(tid) +. red.(tid + 8l)
      in
      let%superstep r4 =
        if lane < 4l then red.(tid) <- red.(tid) +. red.(tid + 4l)
      in
      let%superstep r2 =
        if lane < 2l then red.(tid) <- red.(tid) +. red.(tid + 2l)
      in
      if lane = 0l && row < rows then y.(row) <- red.(tid) +. red.(tid + 1l)]

(* One row per work-group, 256-thread tree reduction. *)
let mv_row =
  [%kernel
    fun (x : float32 vector)
        (w : float32 vector)
        (y : float32 vector)
        (n : int32) ->
      let%shared (red : float32) = 256l in
      let tid = thread_idx_x in
      let base = block_idx_x * n in
      let acc = mut 0.0 in
      let%superstep partial =
        let i = mut tid in
        while i < n do
          acc := acc +. (x.(base + i) *. w.(i)) ;
          i := i + 256l
        done ;
        red.(tid) <- acc
      in
      let%superstep r128 =
        if tid < 128l then red.(tid) <- red.(tid) +. red.(tid + 128l)
      in
      let%superstep r64 =
        if tid < 64l then red.(tid) <- red.(tid) +. red.(tid + 64l)
      in
      let%superstep r32 =
        if tid < 32l then red.(tid) <- red.(tid) +. red.(tid + 32l)
      in
      let%superstep r16 =
        if tid < 16l then red.(tid) <- red.(tid) +. red.(tid + 16l)
      in
      let%superstep r8 =
        if tid < 8l then red.(tid) <- red.(tid) +. red.(tid + 8l)
      in
      let%superstep r4 =
        if tid < 4l then red.(tid) <- red.(tid) +. red.(tid + 4l)
      in
      let%superstep r2 =
        if tid < 2l then red.(tid) <- red.(tid) +. red.(tid + 2l)
      in
      if tid = 0l then y.(block_idx_x) <- red.(0l) +. red.(1l)]

(* Writes what the work-group position intrinsics return, so a wrong
   rewrite in the fused kernel shows up as a wrong value. *)
let positions =
  [%kernel
    fun (p : int32 vector) (q : int32 vector) (n : int32) ->
      (* Locals named like the fused kernel's position parameters: renaming
         must not let them capture those (review finding B1). *)
      let off = n / 2l in
      let gdx = n / 4l in
      let i = global_thread_id in
      if i < n then begin
        p.(i) <- (block_idx_x * 1000000l) + (off * 1000l) + thread_idx_x - gdx ;
        q.(i) <- global_idx_x - i
      end]

(* Reads grid_dim_x, which horizontal fusion refuses (an unsigned value on the
   C-family backends): never fused, still correct. *)
let grid_size =
  [%kernel
    fun (p : int32 vector) (n : int32) ->
      let i = global_thread_id in
      if i < n then p.(i) <- grid_dim_x]

let add_inplace =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (n : int32) ->
      let i = global_thread_id in
      if i < n then x.(i) <- x.(i) +. y.(i)]

let ir_of kern =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* Deterministic inputs with many distinct bits. *)
let fill v seed =
  for i = 0 to Vector.length v - 1 do
    Vector.set v i ((Float.of_int (((i * 7919) + seed) mod 1013) /. 97.0) -. 5.0)
  done

let fvec n seed =
  let v = Vector.create Vector.float32 n in
  fill v seed ;
  v

let bits_f v =
  Array.init (Vector.length v) (fun i -> Int32.bits_of_float (Vector.get v i))

let bits_i v = Array.init (Vector.length v) (fun i -> Vector.get v i)

let d1 = Execute.dims1d

let launch dev ir args ~block ~grid =
  Execute.run_vectors ~device:dev ~ir ~args ~block ~grid ()

(* Problem sizes: rows not a multiple of 8, n not a multiple of 256. *)
let rows = 203

let nb = 77

let n_row = 300

let row_groups = 5

let n_pos = 1000

(* Scenario 1: four independent launches. Returns every output's bits. *)
let independent dev =
  let x1 = fvec (rows * nb) 1 and x2 = fvec (rows * nb) 2 in
  let w = fvec nb 3
  and xr = fvec (n_row * row_groups) 4
  and wr = fvec n_row 5 in
  let y1 = fvec rows 0 and y2 = fvec rows 0 and y3 = fvec row_groups 0 in
  let p = Vector.create Vector.int32 n_pos
  and q = Vector.create Vector.int32 n_pos in
  let g8 = d1 ((rows + 7) / 8) in
  let open Execute in
  launch
    dev
    (ir_of mv_wave)
    [Vec x1; Vec w; Vec y1; Int rows; Int nb]
    ~block:(d1 256)
    ~grid:g8 ;
  launch
    dev
    (ir_of mv_wave)
    [Vec x2; Vec w; Vec y2; Int rows; Int nb]
    ~block:(d1 256)
    ~grid:g8 ;
  launch
    dev
    (ir_of mv_row)
    [Vec xr; Vec wr; Vec y3; Int n_row]
    ~block:(d1 256)
    ~grid:(d1 row_groups) ;
  launch
    dev
    (ir_of positions)
    [Vec p; Vec q; Int n_pos]
    ~block:(d1 256)
    ~grid:(d1 ((n_pos + 255) / 256)) ;
  (bits_f y1, bits_f y2, bits_f y3, bits_i p, bits_i q)

(* Scenario 1b: a matvec and a kernel reading grid_dim_x: not fused. *)
let not_fusable dev =
  let x = fvec (rows * nb) 15 and w = fvec nb 16 and y = fvec rows 0 in
  let p = Vector.create Vector.int32 n_pos in
  let open Execute in
  launch
    dev
    (ir_of mv_wave)
    [Vec x; Vec w; Vec y; Int rows; Int nb]
    ~block:(d1 256)
    ~grid:(d1 ((rows + 7) / 8)) ;
  launch
    dev
    (ir_of grid_size)
    [Vec p; Int n_pos]
    ~block:(d1 256)
    ~grid:(d1 ((n_pos + 255) / 256)) ;
  (bits_f y, bits_i p)

(* Scenario 2: t = W x; acc += t; z = W' acc. *)
let dependent dev =
  let x = fvec (rows * nb) 6 and w = fvec nb 7 in
  let t = fvec rows 0 and acc = fvec rows 8 and z = fvec 1 0 in
  let open Execute in
  launch
    dev
    (ir_of mv_wave)
    [Vec x; Vec w; Vec t; Int rows; Int nb]
    ~block:(d1 256)
    ~grid:(d1 ((rows + 7) / 8)) ;
  launch
    dev
    (ir_of add_inplace)
    [Vec acc; Vec t; Int rows]
    ~block:(d1 256)
    ~grid:(d1 1) ;
  launch
    dev
    (ir_of mv_row)
    [Vec acc; Vec acc; Vec z; Int rows]
    ~block:(d1 256)
    ~grid:(d1 1) ;
  (bits_f t, bits_f acc, bits_f z)

(* Scenario 3: reader, host write of its input, second reader. *)
let host_write dev =
  let x = fvec n_row 9 and w = fvec n_row 10 in
  let y_old = fvec 1 0 and y_new = fvec 1 0 in
  let open Execute in
  launch
    dev
    (ir_of mv_row)
    [Vec x; Vec w; Vec y_old; Int n_row]
    ~block:(d1 256)
    ~grid:(d1 1) ;
  Vector.set w 0 1000.0 ;
  launch
    dev
    (ir_of mv_row)
    [Vec x; Vec w; Vec y_new; Int n_row]
    ~block:(d1 256)
    ~grid:(d1 1) ;
  (bits_f y_old, bits_f y_new)

let with_window enabled f =
  let was = Execute.Window.enabled () in
  Execute.Window.set_enabled enabled ;
  Fun.protect ~finally:(fun () -> Execute.Window.set_enabled was) f

let issued_by f =
  Execute.Window.flush () ;
  Execute.Window.reset_stats () ;
  let r = f () in
  Execute.Window.flush () ;
  (r, (Execute.Window.stats ()).issued)

let failures = ref []

let check (dev : Device.t) label ok =
  Printf.printf
    "  %-11s %-34s %-24s %s\n%!"
    dev.Device.framework
    dev.Device.name
    label
    (if ok then "PASS" else "FAIL") ;
  if not ok then failures := (dev.Device.framework ^ " " ^ label) :: !failures

let run_device (dev : Device.t) =
  let defers = Execute.Window.defers_device dev in
  let scenario label f ~fused_launches =
    let off, _ = with_window false (fun () -> issued_by (fun () -> f dev)) in
    let on, issued = issued_by (fun () -> f dev) in
    check dev (label ^ " bit-identical") (on = off) ;
    if defers then
      check
        dev
        (Printf.sprintf "%s %d launch(es)" label fused_launches)
        (issued = fused_launches)
  in
  scenario "independent" independent ~fused_launches:1 ;
  scenario "dependent" dependent ~fused_launches:3 ;
  scenario "grid_dim_x member" not_fusable ~fused_launches:2 ;
  scenario "host write" host_write ~fused_launches:2 ;
  (* A profiling timer is a synchronisation point: the launch it brackets is
     issued before the start event and before the stop event is recorded.
     Not on Vulkan: there [timer_elapsed_ms] does not return, with or without
     the window (measured on origin/main 86451180, RADV NAVI31). *)
  if defers && dev.Device.framework <> "Vulkan" then begin
    let (), issued =
      issued_by (fun () ->
          let t = Profiling.create_timer dev in
          Profiling.timer_start t ;
          ignore (independent dev) ;
          Profiling.timer_stop t ;
          ignore (Profiling.timer_elapsed_ms t))
    in
    check dev "timer flushes" (issued = 1)
  end

(* The fused kernel on the Interpreter, against its members one by one. *)
let interpreter_fused () =
  let members = [ir_of mv_wave; ir_of mv_row; ir_of positions] in
  let fused = Sarek_execute.Fusion_horizontal.fuse ~name:"fused_test" members in
  let mk () =
    let x = fvec (rows * nb) 11 and w = fvec nb 12 in
    let xr = fvec (n_row * row_groups) 13 and wr = fvec n_row 14 in
    let y = fvec rows 0 and yr = fvec row_groups 0 in
    let p = Vector.create Vector.int32 n_pos
    and q = Vector.create Vector.int32 n_pos in
    (x, w, xr, wr, y, yr, p, q)
  in
  let gx = [(rows + 7) / 8; row_groups; (n_pos + 255) / 256] in
  let open Execute in
  let interp ir args g =
    run_interpreter_vectors
      ~ir
      ~args
      ~block:(d1 256)
      ~grid:(d1 g)
      ~parallel:false
  in
  let x, w, xr, wr, y, yr, p, q = mk () in
  interp (ir_of mv_wave) [Vec x; Vec w; Vec y; Int rows; Int nb] (List.nth gx 0) ;
  interp (ir_of mv_row) [Vec xr; Vec wr; Vec yr; Int n_row] (List.nth gx 1) ;
  interp (ir_of positions) [Vec p; Vec q; Int n_pos] (List.nth gx 2) ;
  let x', w', xr', wr', y', yr', p', q' = mk () in
  let positions_args =
    let off = ref 0 in
    List.concat_map
      (fun g ->
        let a = [Int !off; Int g] in
        off := !off + g ;
        a)
      gx
  in
  interp
    fused
    ([
       Vec x';
       Vec w';
       Vec y';
       Int rows;
       Int nb;
       Vec xr';
       Vec wr';
       Vec yr';
       Int n_row;
       Vec p';
       Vec q';
       Int n_pos;
     ]
    @ positions_args)
    (List.fold_left ( + ) 0 gx) ;
  ignore (x, w, xr, wr, x', w', xr', wr') ;
  bits_f y = bits_f y'
  && bits_f yr = bits_f yr'
  && bits_i p = bits_i p'
  && bits_i q = bits_i q'

let () =
  print_endline "=== launch window and horizontal fusion, E2E ===" ;
  let ok = interpreter_fused () in
  Printf.printf
    "  %-46s %-24s %s\n%!"
    "Interpreter"
    "fused IR = members"
    (if ok then "PASS" else "FAIL") ;
  if not ok then failures := "interpreter fused IR" :: !failures ;
  Array.iter
    (fun (dev : Device.t) ->
      match run_device dev with
      | () -> ()
      | exception e ->
          (* A GPU backend that cannot launch here (missing driver feature) is
             reported; the host backends must run. *)
          let host =
            dev.framework = "Native" || dev.framework = "Interpreter"
          in
          Printf.printf
            "  %-11s %s: %s\n%!"
            dev.framework
            (if host then "FAIL" else "SKIP")
            (Printexc.to_string e) ;
          if host then failures := (dev.framework ^ " raised") :: !failures)
    (Device.init ()) ;
  if !failures <> [] then begin
    Printf.printf
      "test_launch_window: FAILED (%s)\n"
      (String.concat ", " !failures) ;
    exit 1
  end ;
  print_endline "test_launch_window: PASSED"
