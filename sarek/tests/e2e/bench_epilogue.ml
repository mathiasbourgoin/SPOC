(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Epilogue fusion benchmark (specs/epilogue-fusion.md), every enumerated
 * device whose launches are deferred.
 *
 * A launch-bound decode chain with residual-add epilogues: per layer
 *   t = W x ; x' = x + t
 * (a matvec, then the residual add of its output; x and x' alternate, since
 * an epilogue consumer may not write what its producer reads). A "token" is
 * [layers] layers followed by a flush.
 *
 * For epilogue off and on, alternating (off, on, off, on, ...) over [reps]
 * pairs, it prints per device: backend launches per token, wall microseconds
 * per token (host + device, synchronised per token) as min / median / max,
 * and whether the final residual stream is bit-identical between the modes.
 *
 *   SPOC_BENCH_TOKENS=50 SPOC_BENCH_REPS=7 SPOC_BENCH_LAYERS=24 \
 *     dune exec sarek/tests/e2e/bench_epilogue.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let env name default =
  match Sys.getenv_opt name with Some s -> int_of_string s | None -> default

let tokens = env "SPOC_BENCH_TOKENS" 50

let reps = env "SPOC_BENCH_REPS" 7

let layers = env "SPOC_BENCH_LAYERS" 24

let size = env "SPOC_BENCH_SIZE" 512

(* y[row] = sum_b w[row*nb + b] * x[b], 8 rows per 256-thread work-group. *)
let matvec =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
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
            acc := acc +. (w.((row * nb) + b) *. x.(b)) ;
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

let residual =
  [%kernel
    fun (o : float32 vector)
        (x : float32 vector)
        (t : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then o.(i) <- x.(i) +. t.(i)]

let ir_of kern =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let fvec n seed scale =
  let v = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set
      v
      i
      (scale *. ((Float.of_int (((i * 7919) + seed) mod 1013) /. 506.0) -. 1.0))
  done ;
  v

let d1 = Execute.dims1d

let bits v =
  Array.init (Vector.length v) (fun i -> Int32.bits_of_float (Vector.get v i))

(* One token: [layers] x (t = W x ; x' = x + t), then a flush. [layers] is
   even, so the residual stream ends in [x]. *)
let token dev ~w ~x ~x' ~t =
  let open Execute in
  for layer = 1 to layers do
    let src, dst = if layer mod 2 = 1 then (x, x') else (x', x) in
    run_vectors
      ~device:dev
      ~ir:(ir_of matvec)
      ~args:[Vec w; Vec src; Vec t; Int size; Int size]
      ~block:(d1 256)
      ~grid:(d1 ((size + 7) / 8))
      () ;
    run_vectors
      ~device:dev
      ~ir:(ir_of residual)
      ~args:[Vec dst; Vec src; Vec t; Int size]
      ~block:(d1 256)
      ~grid:(d1 ((size + 255) / 256))
      ()
  done ;
  Transfer.flush dev

let set_epilogue (dev : Device.t) on =
  let backend = dev.Device.framework in
  (* Fusion off HIP and Metal needs the contraction grant (no-op there). *)
  if on then
    ignore
      (Sarek_policy.grant
         Sarek_policy.Process
         (Sarek_policy.Deviation Sarek_policy.Cross_kernel_contraction)
         ~reason:"bench_epilogue: measuring fused against unfused") ;
  Execute.Window.set_backend_enabled ~backend true ;
  Execute.Window.set_epilogue_backend ~backend on

(* [tokens] tokens with the epilogue [on]/off: (us per token, launches per
   token, final x). *)
let measure dev ~on =
  set_epilogue dev on ;
  let w = fvec (size * size) 1 0.0002
  and x = fvec size 2 1.0
  and x' = fvec size 0 0.0
  and t = fvec size 0 0.0 in
  token dev ~w ~x ~x' ~t ;
  Execute.Window.reset_stats () ;
  let t0 = Unix.gettimeofday () in
  for _ = 1 to tokens do
    token dev ~w ~x ~x' ~t
  done ;
  let dt = Unix.gettimeofday () -. t0 in
  let issued = (Execute.Window.stats ()).issued in
  ( dt *. 1e6 /. Float.of_int tokens,
    Float.of_int issued /. Float.of_int tokens,
    bits x )

let stats xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  (a.(0), a.(Array.length a / 2), a.(Array.length a - 1))

let bench (dev : Device.t) =
  let off = ref [] and on = ref [] and equal = ref true in
  let launches_off = ref 0.0 and launches_on = ref 0.0 in
  for _ = 1 to reps do
    let us, l, bits_off = measure dev ~on:false in
    off := us :: !off ;
    launches_off := l ;
    let us, l, bits_on = measure dev ~on:true in
    on := us :: !on ;
    launches_on := l ;
    if bits_on <> bits_off then equal := false
  done ;
  let line label (lo, med, hi) launches =
    Printf.printf
      "  %-8s %-26s %-4s %5.1f launches/token  us/token min %8.1f  median \
       %8.1f  max %8.1f\n\
       %!"
      dev.Device.framework
      (String.sub dev.Device.name 0 (min 26 (String.length dev.Device.name)))
      label
      launches
      lo
      med
      hi
  in
  line "off" (stats !off) !launches_off ;
  line "on" (stats !on) !launches_on ;
  Printf.printf
    "  %-8s bit-identical between the modes: %b; median speed-up %.2fx\n%!"
    dev.Device.framework
    !equal
    (let _, a, _ = stats !off and _, b, _ = stats !on in
     a /. b)

let () =
  Printf.printf
    "=== epilogue fusion bench: %d layers, size %d, %d tokens x %d pairs ===\n\
     %!"
    layers
    size
    tokens
    reps ;
  Array.iter
    (fun (dev : Device.t) ->
      if not (Device.is_cpu dev || dev.framework = "Interpreter") then
        match bench dev with
        | () -> ()
        | exception e ->
            Printf.printf
              "  %-8s SKIP: %s\n%!"
              dev.framework
              (Printexc.to_string e))
    (Device.init ())
