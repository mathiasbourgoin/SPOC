(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Native-backend matvec benchmark.

    A block-quantised (Q8_0-like) matrix-vector product at an LLM decode shape:
    [m] = 1024 rows, [k] = 3072 columns, 8 selections (grid y), one 32-lane wave
    per row, 8 rows per 256-thread work-group, partial sums reduced in shared
    memory over 4 supersteps. This is the geometry of the expert matvec kernels
    of a downstream LLM runtime, where Native measured ~300x slower than an
    OpenCL CPU runtime on the same kernel.

    Three kernels isolate the costs:
    - [wave]: the full kernel (barriers, shared memory, int32 + float32 reads);
    - [row]: one thread per output row, no barrier, same arithmetic;
    - [copy]: y.(i) <- x.(i), one element per thread (per-access cost).

    Usage: bench_native_matvec.exe <device> <reps> [kernel] device = native |
    opencl-cpu ; kernel = wave | row | copy | all (default). Prints one CSV line
    per sample: kernel,device,us_per_launch,checksum. *)

[@@@warning "-33"] (* the kernel PPX emits an open that is unused here *)

module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Execute = Sarek.Execute

let rows = 1024

let k = 3072

let n_sel = 8

let nb = k / 32

let rec_words = rows * nb * 9

let o_d = 0

let o_qs = rows * nb

let wave =
  [%kernel
    fun (er : int32 vector)
        (rec_words : int32)
        (o_d : int32)
        (o_qs : int32)
        (x : float32 vector)
        (y : float32 vector)
        (rows : int32)
        (nb : int32) ->
      let%shared (red : float32) = 256l in
      let tid = thread_idx_x in
      let lane = tid land 31l in
      let row = (block_idx_x * 8l) + (tid / 32l) in
      let base = block_idx_y * rec_words in
      let yo = block_idx_y * rows in
      let acc = mut 0.0 in
      let%superstep partial =
        if row < rows then begin
          let b = mut lane in
          while b < nb do
            let blk = (row * nb) + b in
            let wb = base + o_qs + (blk * 8l) in
            let xb = b * 32l in
            let s = mut 0.0 in
            for wi = 0 to 7l do
              let word = er.(wb + wi) in
              let xi = xb + (wi * 4l) in
              let q0 = float ((word lsl 24l) asr 24l) in
              let q1 = float ((word lsl 16l) asr 24l) in
              let q2 = float ((word lsl 8l) asr 24l) in
              let q3 = float (word asr 24l) in
              s :=
                s
                +. (q0 *. x.(xi))
                +. (q1 *. x.(xi + 1l))
                +. (q2 *. x.(xi + 2l))
                +. (q3 *. x.(xi + 3l))
            done ;
            acc := acc +. (Int32.float_of_bits er.(base + o_d + blk) *. s) ;
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
      if lane = 0l && row < rows then
        y.(yo + row) <- red.(tid) +. red.(tid + 1l)]

let row =
  [%kernel
    fun (er : int32 vector)
        (rec_words : int32)
        (o_d : int32)
        (o_qs : int32)
        (x : float32 vector)
        (y : float32 vector)
        (rows : int32)
        (nb : int32) ->
      let row = (block_idx_x * block_dim_x) + thread_idx_x in
      let base = block_idx_y * rec_words in
      if row < rows then begin
        let acc = mut 0.0 in
        let b = mut 0l in
        while b < nb do
          let blk = (row * nb) + b in
          let wb = base + o_qs + (blk * 8l) in
          let xb = b * 32l in
          let s = mut 0.0 in
          for wi = 0 to 7l do
            let word = er.(wb + wi) in
            let xi = xb + (wi * 4l) in
            let q0 = float ((word lsl 24l) asr 24l) in
            let q1 = float ((word lsl 16l) asr 24l) in
            let q2 = float ((word lsl 8l) asr 24l) in
            let q3 = float (word asr 24l) in
            s :=
              s
              +. (q0 *. x.(xi))
              +. (q1 *. x.(xi + 1l))
              +. (q2 *. x.(xi + 2l))
              +. (q3 *. x.(xi + 3l))
          done ;
          acc := acc +. (Int32.float_of_bits er.(base + o_d + blk) *. s) ;
          b := b + 1l
        done ;
        y.((block_idx_y * rows) + row) <- acc
      end]

let copy =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (n : int32) ->
      let i = (block_idx_x * block_dim_x) + thread_idx_x in
      if i < n then y.(i) <- x.(i)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> invalid_arg "bench_native_matvec: kernel has no IR"

let make_records () =
  let ba = Bigarray.(Array1.create int32 c_layout (rec_words * n_sel)) in
  let s = ref 0x2545F491l in
  for i = 0 to Bigarray.Array1.dim ba - 1 do
    (s := Int32.(add (mul !s 1664525l) 1013904223l)) ;
    Bigarray.Array1.unsafe_set ba i !s
  done ;
  let scale = Int32.bits_of_float 0.01 in
  for r = 0 to n_sel - 1 do
    for i = 0 to (rows * nb) - 1 do
      Bigarray.Array1.unsafe_set ba ((r * rec_words) + o_d + i) scale
    done
  done ;
  Vector.of_bigarray Vector.Int32 ba

let pick_device name =
  let devs = Array.to_list (Device.init ~frameworks:["OpenCL"; "Native"] ()) in
  let wanted (d : Device.t) =
    match name with
    | "native" -> d.framework = "Native"
    | "opencl-cpu" -> Device.is_opencl d && Device.is_cpu d
    | _ -> false
  in
  match List.find_opt wanted devs with
  | Some d -> d
  | None -> invalid_arg ("bench_native_matvec: no device " ^ name)

let checksum y =
  let acc = ref 0.0 in
  for i = 0 to Vector.length y - 1 do
    acc := !acc +. Vector.get y i
  done ;
  !acc

let time_launch dev launch =
  let t0 = Unix.gettimeofday () in
  launch () ;
  Device.synchronize dev ;
  (Unix.gettimeofday () -. t0) *. 1e6

let matvec_launch dev kern ~grid ~block er x y () =
  let i32 = Int32.of_int in
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of kern)
    ~args:
      Execute.
        [
          Vec er;
          Int32 (i32 rec_words);
          Int32 (i32 o_d);
          Int32 (i32 o_qs);
          Vec x;
          Vec y;
          Int32 (i32 rows);
          Int32 (i32 nb);
        ]
    ~block
    ~grid
    ()

let run_case dev dev_name reps name launch y =
  launch () ;
  Device.synchronize dev ;
  for _ = 1 to reps do
    let us = time_launch dev launch in
    Printf.printf "%s,%s,%.1f,%.6g\n%!" name dev_name us (checksum y)
  done

let run_matvec_cases dev dev_name reps wanted =
  let er = make_records () in
  let x = Vector.create Vector.float32 k in
  for i = 0 to k - 1 do
    Vector.set x i ((float (i * 37 mod 101) /. 50.0) -. 1.0)
  done ;
  let y = Vector.create Vector.float32 (rows * n_sel) in
  let case name kern ~groups_x =
    if wanted name then
      run_case
        dev
        dev_name
        reps
        name
        (matvec_launch
           dev
           kern
           ~grid:(Execute.dims2d groups_x n_sel)
           ~block:(Execute.dims1d 256)
           er
           x
           y)
        y
  in
  case "wave" wave ~groups_x:(rows / 8) ;
  case "row" row ~groups_x:(rows / 256)

let run_copy_case dev dev_name reps =
  let n = 1 lsl 20 in
  let xs = Vector.create Vector.float32 n
  and ys = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set xs i (float i)
  done ;
  run_case
    dev
    dev_name
    reps
    "copy"
    (fun () ->
      Execute.run_vectors
        ~device:dev
        ~ir:(ir_of copy)
        ~args:Execute.[Vec xs; Vec ys; Int32 (Int32.of_int n)]
        ~block:(Execute.dims1d 256)
        ~grid:(Execute.dims1d (n / 256))
        ())
    ys

let usage () =
  prerr_endline
    "usage: bench_native_matvec.exe native|opencl-cpu <reps> \
     [wave|row|copy|all]" ;
  exit 2

let parse_args () =
  let argc = Array.length Sys.argv in
  if argc < 3 then usage () ;
  match int_of_string_opt Sys.argv.(2) with
  | Some reps when reps > 0 ->
      let which = if argc > 3 then Sys.argv.(3) else "all" in
      (Sys.argv.(1), reps, which)
  | _ -> usage ()

let () =
  let dev_name, reps, which = parse_args () in
  Benchmark_backends.Backend_loader.init () ;
  let dev = pick_device dev_name in
  Printf.printf "# device %s (%s)\n" dev.Device.name dev.Device.framework ;
  let wanted s = which = "all" || which = s in
  run_matvec_cases dev dev_name reps wanted ;
  if wanted "copy" then run_copy_case dev dev_name reps
