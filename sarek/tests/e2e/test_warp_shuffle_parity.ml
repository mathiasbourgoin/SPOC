(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E parity test for warp_shuffle, warp_shuffle_down, warp_shuffle_up and
 * warp_shuffle_xor (docs/plans/warp-shuffle-intake-2026-10-17.md, AC-2).
 *
 * For every primitive, on int32, float32 and (where the device has it)
 * float64, every enumerated device runs the kernel over the boundary operands
 * {0, 1, W-1, W, W+1, -1, 2W+1} (mixed per lane) with one block of 4*W threads
 * and a 2-D block (8, 4*W/8), and the output is compared BIT FOR BIT with
 * Sarek_warp.source_lane, the executable form of the section-4.3 table. Native
 * and the Interpreter run at W in {8, 16, 32, 64} through the Sarek_warp test
 * hook; a hardware device runs at the width it reports.
 *
 * Native and the Interpreter must run; a GPU backend that is not there is
 * skipped; a device that refuses a shuffle kernel through the capability model
 * is reported as REFUSED (that refusal is the specified behaviour); a wrong
 * result or a codegen/compile failure is a failure.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

(* Thread t of the whole launch, whatever the block shape. *)
let shuffle_idx_i32 =
  [%kernel
    fun (src : int32 vector) (arg : int32 vector) (dst : int32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle src.(t) arg.(t)]

let shuffle_down_i32 =
  [%kernel
    fun (src : int32 vector) (arg : int32 vector) (dst : int32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_down src.(t) arg.(t)]

let shuffle_up_i32 =
  [%kernel
    fun (src : int32 vector) (arg : int32 vector) (dst : int32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_up src.(t) arg.(t)]

let shuffle_xor_i32 =
  [%kernel
    fun (src : int32 vector) (arg : int32 vector) (dst : int32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_xor src.(t) arg.(t)]

let shuffle_idx_f32 =
  [%kernel
    fun (src : float32 vector) (arg : int32 vector) (dst : float32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle src.(t) arg.(t)]

let shuffle_down_f32 =
  [%kernel
    fun (src : float32 vector) (arg : int32 vector) (dst : float32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_down src.(t) arg.(t)]

let shuffle_up_f32 =
  [%kernel
    fun (src : float32 vector) (arg : int32 vector) (dst : float32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_up src.(t) arg.(t)]

let shuffle_xor_f32 =
  [%kernel
    fun (src : float32 vector) (arg : int32 vector) (dst : float32 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_xor src.(t) arg.(t)]

let shuffle_idx_f64 =
  [%kernel
    fun (src : float64 vector) (arg : int32 vector) (dst : float64 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle src.(t) arg.(t)]

let shuffle_down_f64 =
  [%kernel
    fun (src : float64 vector) (arg : int32 vector) (dst : float64 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_down src.(t) arg.(t)]

let shuffle_up_f64 =
  [%kernel
    fun (src : float64 vector) (arg : int32 vector) (dst : float64 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_up src.(t) arg.(t)]

let shuffle_xor_f64 =
  [%kernel
    fun (src : float64 vector) (arg : int32 vector) (dst : float64 vector) ->
      let open Std in
      let t =
        (block_idx_x * block_dim_x * block_dim_y)
        + thread_idx_x
        + (block_dim_x * thread_idx_y)
      in
      dst.(t) <- warp_shuffle_xor src.(t) arg.(t)]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* (label, op, value type, IR). *)
let cases =
  let open Sarek_warp in
  [
    ("idx/i32", Idx, I32, ir_of shuffle_idx_i32);
    ("down/i32", Down, I32, ir_of shuffle_down_i32);
    ("up/i32", Up, I32, ir_of shuffle_up_i32);
    ("xor/i32", Xor, I32, ir_of shuffle_xor_i32);
    ("idx/f32", Idx, F32, ir_of shuffle_idx_f32);
    ("down/f32", Down, F32, ir_of shuffle_down_f32);
    ("up/f32", Up, F32, ir_of shuffle_up_f32);
    ("xor/f32", Xor, F32, ir_of shuffle_xor_f32);
    ("idx/f64", Idx, F64, ir_of shuffle_idx_f64);
    ("down/f64", Down, F64, ir_of shuffle_down_f64);
    ("up/f64", Up, F64, ir_of shuffle_up_f64);
    ("xor/f64", Xor, F64, ir_of shuffle_xor_f64);
  ]

let is_host (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

(* Boundary operands of section 4.3, mixed per thread so neighbouring lanes
   disagree. *)
let operand ~w t =
  let boundary = [|0; 1; w - 1; w; w + 1; -1; (2 * w) + 1; 2; w / 2; 3|] in
  boundary.(((t * 3) + (t / w)) mod Array.length boundary)

(* Run one case at one block shape; returns the number of wrong lanes. *)
let run_case (type a b) (kind : (a, b) Vector.kind) ~(of_int : int -> a)
    ~(bits : a -> int64) dev (op, ir) ~w ~(block : int * int) =
  let bx, by = block in
  let per_block = bx * by in
  let blocks = 2 in
  let n = blocks * per_block in
  let src = Vector.create kind n in
  let dst = Vector.create kind n in
  let arg = Vector.create Vector.int32 n in
  for t = 0 to n - 1 do
    Vector.set src t (of_int ((t * 7) + 3)) ;
    Vector.set dst t (of_int (-1)) ;
    Vector.set arg t (Int32.of_int (operand ~w t))
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec src; Execute.Vec arg; Execute.Vec dst]
    ~block:(Execute.dims2d bx by)
    ~grid:(Execute.dims1d blocks)
    () ;
  Transfer.flush dev ;
  let bad = ref 0 in
  for t = 0 to n - 1 do
    let local = t mod per_block in
    let lane = local mod w in
    let source = Sarek_warp.source_lane op ~width:w ~lane (operand ~w t) in
    let expected = Vector.get src (t - lane + source) in
    if bits (Vector.get dst t) <> bits expected then begin
      if !bad < 3 then
        Printf.printf
          "      thread %d lane %d arg %d: got %Ld expected %Ld\n%!"
          t
          lane
          (operand ~w t)
          (bits (Vector.get dst t))
          (bits expected) ;
      incr bad
    end
  done ;
  !bad

let run_typed dev (op, ty, ir) ~w ~block =
  match ty with
  | Sarek_warp.I32 ->
      run_case
        Vector.int32
        ~of_int:Int32.of_int
        ~bits:Int64.of_int32
        dev
        (op, ir)
        ~w
        ~block
  | Sarek_warp.F32 ->
      run_case
        Vector.float32
        ~of_int:(fun i -> float_of_int i *. 0.5)
        ~bits:(fun x -> Int64.of_int32 (Int32.bits_of_float x))
        dev
        (op, ir)
        ~w
        ~block
  | Sarek_warp.F64 ->
      run_case
        Vector.float64
        ~of_int:(fun i -> float_of_int i *. 0.5)
        ~bits:Int64.bits_of_float
        dev
        (op, ir)
        ~w
        ~block

(* Both shapes of AC-2: one 1-D block of 4W and a 2-D block (8, 4W/8). *)
let shapes ~w = (4 * w, 1) :: (if 4 * w mod 8 = 0 then [(8, 4 * w / 8)] else [])

let widths dev =
  if is_host dev then [8; 16; 32; 64]
  else [dev.Device.capabilities.Spoc_framework.Framework_sig.warp_size]

let max_block (dev : Device.t) =
  dev.Device.capabilities.Spoc_framework.Framework_sig.max_threads_per_block

(* The capability model's refusal names the feature: "<device>: warp-shuffle
   unavailable - ...". *)
let is_capability_refusal message =
  let needle = "warp-shuffle unavailable" in
  let n = String.length needle and m = String.length message in
  let rec at i =
    i + n <= m && (String.sub message i n = needle || at (i + 1))
  in
  at 0

let failed = ref false

let host_ran = ref false

let report dev label ~w (bx, by) outcome =
  Printf.printf
    "  %-11s %-30s %-9s W=%-2d block %dx%-3d %s\n%!"
    dev.Device.framework
    dev.Device.name
    label
    w
    bx
    by
    outcome

let run_one dev (label, op, ty, ir) ~w ~block =
  let go () = run_typed dev (op, ty, ir) ~w ~block in
  let outcome =
    match if is_host dev then Sarek_warp.with_width w go else go () with
    | 0 ->
        if is_host dev then host_ran := true ;
        "PASS"
    | bad ->
        failed := true ;
        Printf.sprintf "FAIL (%d wrong lanes)" bad
    | exception Sarek.Execute_error.Execution_error (Backend_error {message; _})
      when (not (is_host dev)) && is_capability_refusal message ->
        "REFUSED (capability model)"
    | exception e ->
        let regression =
          is_host dev || Test_helpers.Backend_failure.verdict e = `Fail
        in
        if regression then failed := true ;
        Printf.sprintf
          "%s (%s)"
          (if regression then "FAIL" else "SKIP")
          (Printexc.to_string e)
  in
  report dev label ~w block outcome

(* A device that does not list the feature must REFUSE a shuffle kernel through
   the capability model; one launch of one case shows it. *)
let run_refusal_check dev =
  let w = max 1 (List.hd (widths dev)) in
  match cases with
  | case :: _ -> run_one dev case ~w ~block:(4 * w, 1)
  | [] -> ()

let run_supported dev =
  List.iter
    (fun w ->
      List.iter
        (fun ((_, _, ty, _) as case) ->
          let skip_f64 = ty = Sarek_warp.F64 && not (Device.allows_fp64 dev) in
          if not skip_f64 then
            List.iter
              (fun block ->
                if fst block * snd block <= max_block dev then
                  run_one dev case ~w ~block)
              (shapes ~w))
        cases)
    (widths dev)

let run_device (dev : Device.t) =
  if is_host dev || Device.provides dev Sarek_ir_analysis.Warp_shuffle then
    run_supported dev
  else run_refusal_check dev

let () =
  Test_helpers.Backend_failure.self_check ~test_name:"test_warp_shuffle_parity" ;
  print_endline "=== warp shuffle parity vs Sarek_warp.source_lane ===" ;
  Array.iter run_device (Device.init ()) ;
  if not !host_ran then begin
    print_endline
      "test_warp_shuffle_parity: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_warp_shuffle_parity: FAILED" ;
    exit 1
  end ;
  print_endline "test_warp_shuffle_parity: PASSED"
