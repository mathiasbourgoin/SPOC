(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The matrix-product rewrite (specs/matrix-pattern-rewrite.md) on the RX 7900
   XTX: a naive DSL GEMM launched as written (scalar) and through
   Sarek_matrix.run (rewritten onto WMMA), per element case and precision,
   against the hand-written SCoopmat GEMM (Sarek_gemm.Coopmat, f16) and the
   scalar tiled Sarek_gemm (f32).

   Method as bench_hip_coopmat_gemm: GPU events around batches of [batch]
   back-to-back launches, [samples] batches per round, [rounds] rounds;
   median and min-max per launch. The rewritten launch includes Sarek_matrix's
   host decision (memoised plan, region evaluation). The GPU is shared.

   Usage: bench_matrix_rewrite.exe [rounds] [samples] [batch] *)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Profiling = Spoc_core.Profiling
module Gemm = Sarek_gemm.Coopmat

let () = Sarek_hip.Hip_plugin.init ()

let arg i default =
  if Array.length Sys.argv > i then int_of_string Sys.argv.(i) else default

let rounds = arg 1 3

let samples = arg 2 10

let batch = arg 3 10

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let measure dev launch =
  launch () ;
  Transfer.flush dev ;
  let timer = Profiling.create_timer dev in
  let all =
    List.concat
      (List.init rounds (fun _ ->
           List.init samples (fun _ ->
               Profiling.timer_start timer ;
               for _ = 1 to batch do
                 launch ()
               done ;
               Profiling.timer_stop timer ;
               Profiling.timer_elapsed_ms timer *. 1000.0 /. float_of_int batch)))
  in
  Profiling.timer_destroy timer ;
  all

let report ~name ~m ~n ~k all =
  let tflops us =
    2.0 *. float m *. float n *. float k /. (us *. 1e-6) /. 1e12
  in
  let lo = List.fold_left min infinity all
  and hi = List.fold_left max 0.0 all in
  let med = median all in
  Printf.printf
    "%-34s M=%-4d N=%-5d K=%-4d median %9.1f us (%6.2f T/s)  min %9.1f max \
     %9.1f us\n\
     %!"
    name
    m
    n
    k
    med
    (tflops med)
    lo
    hi

let k_int =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

let k_f16 =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            fma
              (float32_of_float16 a.((row * k) + i))
              (float32_of_float16 b.((i * n) + col))
              sum
        done ;
        c.((row * n) + col) <- sum
      end]

let k_f32 =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with Some ir -> ir | None -> failwith "no IR"

let fill kind len f =
  let v = Vector.create kind len in
  for i = 0 to len - 1 do
    Vector.set v i (f i)
  done ;
  v

let fval i = float_of_int ((i * 7919 mod 17) - 8) /. 16.0

let geometry ~m ~n =
  (Execute.dims2d 16 16, Execute.dims2d ((n + 15) / 16) ((m + 15) / 16))

let bench_naive dev ~m ~n ~k ~label ~ir ~args ?precision () =
  let block, grid = geometry ~m ~n in
  let original () = Execute.run_vectors ~device:dev ~ir ~args ~block ~grid () in
  let rewritten () =
    match Sarek_matrix.run ?precision ~device:dev ~ir ~args ~block ~grid () with
    | Sarek_matrix.Rewritten _ -> ()
    | Sarek_matrix.Original why -> failwith ("not rewritten: " ^ why)
  in
  if precision = None || precision = Some Sarek_coopmat.Operands_f16 then
    report ~name:(label ^ " naive (scalar)") ~m ~n ~k (measure dev original) ;
  report ~name:(label ^ " rewritten") ~m ~n ~k (measure dev rewritten)

let bench_shape dev (m, n, k) =
  let i x = Execute.Int32 (Int32.of_int x) in
  let dims = [i m; i n; i k] in
  let ints len =
    fill Vector.int32 len (fun x -> Int32.of_int (x * 2654435761))
  in
  bench_naive
    dev
    ~m
    ~n
    ~k
    ~label:"int32"
    ~ir:(ir_of k_int)
    ~args:
      ([
         Execute.Vec (ints (m * k));
         Execute.Vec (ints (k * n));
         Execute.Vec (ints (m * n));
       ]
      @ dims)
    () ;
  let h len = fill Vector.float16 len fval
  and f len = fill Vector.float32 len fval in
  bench_naive
    dev
    ~m
    ~n
    ~k
    ~label:"f16"
    ~ir:(ir_of k_f16)
    ~args:
      ([
         Execute.Vec (h (m * k));
         Execute.Vec (h (k * n));
         Execute.Vec (f (m * n));
       ]
      @ dims)
    () ;
  let f32_args =
    [Execute.Vec (f (m * k)); Execute.Vec (f (k * n)); Execute.Vec (f (m * n))]
    @ dims
  in
  List.iter
    (fun p ->
      bench_naive
        dev
        ~m
        ~n
        ~k
        ~label:("f32/" ^ Sarek_coopmat.operand_precision_name p)
        ~ir:(ir_of k_f32)
        ~args:f32_args
        ~precision:p
        ())
    [
      Sarek_coopmat.Operands_f16;
      Sarek_coopmat.Operands_bf16;
      Sarek_coopmat.Operands_bf16x3;
    ] ;
  (if m mod 32 = 0 && n mod 32 = 0 && k mod 16 = 0 then
     let a = h (m * k) and b = h (k * n) and c = f (m * n) and d = f (m * n) in
     let tiling = Gemm.default_tiling in
     report
       ~name:"Sarek_gemm.Coopmat f16 (2x2/w2)"
       ~m
       ~n
       ~k
       (measure dev (fun () ->
            Gemm.run ~device:dev ~tiling ~a ~b ~c ~d ~m ~n ~k ()))) ;
  let scalar_ir = ir_of Sarek_gemm.sgemm_tiled_kernel in
  let a = f (m * k) and b = f (k * n) and c = f (m * n) in
  report
    ~name:"Sarek_gemm tiled f32 (scalar)"
    ~m
    ~n
    ~k
    (measure dev (fun () ->
         Execute.run_vectors
           ~device:dev
           ~ir:scalar_ir
           ~args:
             Execute.
               [Vec a; Vec b; Vec c; i m; i n; i k; Float32 1.0; Float32 0.0]
           ~block:(Sarek_gemm.Host.block ())
           ~grid:(Sarek_gemm.Host.grid ~m ~n)
           ()))

let () =
  match
    List.filter
      (fun d ->
        d.Device.framework = "HIP"
        && Sarek_capability.permits
             (Sarek_coopmat.verdict
                ~support:
                  d.Device.capabilities.Spoc_framework.Framework_sig.coopmat
                Gemm.config))
      (Array.to_list (Device.init ()))
  with
  | [] -> print_endline "[SKIP] no gfx11 HIP device"
  | dev :: _ ->
      Sarek_coopmat.accept_relaxed_accumulation ~reason:"benchmark" ;
      List.iter
        (fun p ->
          Sarek_coopmat.accept_operand_rounding ~precision:p ~reason:"benchmark")
        [
          Sarek_coopmat.Operands_f16;
          Sarek_coopmat.Operands_bf16;
          Sarek_coopmat.Operands_bf16x3;
        ] ;
      Printf.printf
        "device: %s; rounds=%d samples=%d batch=%d\n%!"
        dev.Device.name
        rounds
        samples
        batch ;
      List.iter
        (bench_shape dev)
        [(256, 1024, 3072); (256, 3072, 3072); (1000, 1000, 1000)]
