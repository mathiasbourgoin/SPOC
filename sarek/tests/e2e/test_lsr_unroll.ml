(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for [lsr] (Lshr32/Lshr64, one native unsigned shift per backend)
 * and for the automatic unroll hint on small constant-trip loops, on every
 * enumerated device. specs/codegen-lsr-unroll.md.
 *
 * Kernels:
 *   - lsr32:  out = a lsr s over 24 int32 values x every amount 0..31
 *   - expr:   out = (a + 1) lsr s, a non-trivial operand (refused before)
 *   - lsr64:  the same over int64 values x every amount 0..63
 *   - loops:  a q5_K-like byte extraction in 8 x 4 constant loops, a downto
 *             loop, a parameter-bound loop, a trip-32 loop, an explicit
 *             [pragma ["unroll 1"]] loop and a 4 x 4 x 8 nest
 *
 * Every expected value is computed with OCaml's Int32/Int64 on the host.
 * Amounts outside 0..width-1 are unspecified (as in OCaml) and not tested.
 *
 * Run with the CUDA/PTX device surfaced (ZLUDA):
 *   LD_LIBRARY_PATH=$HOME/opt/zluda dune exec sarek/tests/e2e/test_lsr_unroll.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let lsr32_kernel =
  [%kernel
    fun (a : int32 vector)
        (s : int32 vector)
        (out : int32 vector)
        (n : int32) ->
      let tid = global_thread_id in
      if tid < n then out.(tid) <- a.(tid) lsr s.(tid)]

let expr_kernel =
  [%kernel
    fun (a : int32 vector)
        (s : int32 vector)
        (out : int32 vector)
        (n : int32) ->
      let tid = global_thread_id in
      if tid < n then out.(tid) <- (a.(tid) + 1l) lsr s.(tid)]

let lsr64_kernel =
  [%kernel
    fun (a : int64 vector)
        (s : int64 vector)
        (out : int64 vector)
        (n : int32) ->
      let tid = global_thread_id in
      if tid < n then out.(tid) <- a.(tid) lsr s.(tid)]

let loops_kernel =
  [%kernel
    fun (w : int32 vector) (out : int32 vector) (n : int32) (m : int32) ->
      let tid = global_thread_id in
      if tid < n then begin
        let acc = mut 0l in
        for wi = 0 to 7 do
          let word = w.((tid * 8) + wi) in
          for k = 0 to 3 do
            acc := acc + ((word lsr (8 * k)) land 255l * (wi + 1))
          done
        done ;
        for j = 3 downto 0 do
          acc := (acc * 3l) + j
        done ;
        for j = 0 to m - 1 do
          acc := acc + j
        done ;
        for j = 0 to 31 do
          acc := acc lxor (j lsl 3)
        done ;
        pragma
          ["unroll 1"]
          (for j = 0 to 3 do
             acc := acc + (j * 5l)
           done) ;
        for a = 0 to 3 do
          for b = 0 to 3 do
            for c = 0 to 7 do
              acc := acc + (a * b * c)
            done
          done
        done ;
        out.(tid) <- acc
      end]

(* The host reference of [loops_kernel], in OCaml's int32 arithmetic. *)
let loops_reference (w : int32 array) ~m tid =
  let open Int32 in
  let acc = ref 0l in
  for wi = 0 to 7 do
    let word = w.((tid * 8) + wi) in
    for k = 0 to 3 do
      let byte = logand (shift_right_logical word (8 * k)) 255l in
      acc := add !acc (mul byte (of_int (wi + 1)))
    done
  done ;
  for j = 3 downto 0 do
    acc := add (mul !acc 3l) (of_int j)
  done ;
  for j = 0 to m - 1 do
    acc := add !acc (of_int j)
  done ;
  for j = 0 to 31 do
    acc := logxor !acc (of_int (j lsl 3))
  done ;
  for j = 0 to 3 do
    acc := add !acc (of_int (j * 5))
  done ;
  for a = 0 to 3 do
    for b = 0 to 3 do
      for c = 0 to 7 do
        acc := add !acc (of_int (a * b * c))
      done
    done
  done ;
  !acc

let values32 =
  [|
    0l;
    1l;
    -1l;
    -2l;
    -16l;
    Int32.min_int;
    Int32.max_int;
    0x80000001l;
    0x40000000l;
    0x12345678l;
    0xdeadbeefl;
    0x7fffffffl;
    0xfffffffel;
    0x0000ffffl;
    0xffff0000l;
    0x55555555l;
    0xaaaaaaaal;
    3l;
    255l;
    256l;
    -256l;
    0x00800000l;
    0xff800000l;
    0x7f800001l;
  |]

let values64 =
  [|
    0L;
    1L;
    -1L;
    -16L;
    Int64.min_int;
    Int64.max_int;
    0x8000000000000001L;
    0x123456789abcdef0L;
    0xdeadbeefcafebabeL;
    0x00000000ffffffffL;
    0xffffffff00000000L;
    0x5555555555555555L;
    -4294967296L;
    4294967296L;
  |]

(* Rows (value, amount), every amount for every value. *)
let rows32 =
  Array.concat
    (Array.to_list
       (Array.map
          (fun v -> Array.init 32 (fun s -> (v, Int32.of_int s)))
          values32))

let rows64 =
  Array.concat
    (Array.to_list
       (Array.map
          (fun v -> Array.init 64 (fun s -> (v, Int64.of_int s)))
          values64))

let launch dev ir ~n args =
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d ((n + 63) / 64))
    () ;
  Transfer.flush dev

(* Returns (wrong rows, total), printing the first few. *)
let compare_rows label n ok show =
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if not (ok i) then begin
      if !bad < 4 then Printf.printf "    %s row %d: %s\n%!" label i (show i) ;
      incr bad
    end
  done ;
  (!bad, n)

let run_shift32 ~operand dev ir =
  let n = Array.length rows32 in
  let a = Vector.create Vector.int32 n in
  let s = Vector.create Vector.int32 n in
  let out = Vector.create Vector.int32 n in
  Array.iteri
    (fun i (v, amt) ->
      Vector.set a i v ;
      Vector.set s i amt ;
      Vector.set out i 0x5a5a5a5al)
    rows32 ;
  launch
    dev
    ir
    ~n
    [Execute.Vec a; Execute.Vec s; Execute.Vec out; Execute.Int n] ;
  let expected i =
    let v, amt = rows32.(i) in
    Int32.shift_right_logical (operand v) (Int32.to_int amt)
  in
  compare_rows
    "lsr32"
    n
    (fun i -> Vector.get out i = expected i)
    (fun i ->
      let v, amt = rows32.(i) in
      Printf.sprintf
        "0x%08lx lsr %ld: got 0x%08lx, expected 0x%08lx"
        v
        amt
        (Vector.get out i)
        (expected i))

let run_lsr64 dev ir =
  let n = Array.length rows64 in
  let a = Vector.create Vector.int64 n in
  let s = Vector.create Vector.int64 n in
  let out = Vector.create Vector.int64 n in
  Array.iteri
    (fun i (v, amt) ->
      Vector.set a i v ;
      Vector.set s i amt ;
      Vector.set out i 0x5a5a5a5aL)
    rows64 ;
  launch
    dev
    ir
    ~n
    [Execute.Vec a; Execute.Vec s; Execute.Vec out; Execute.Int n] ;
  let expected i =
    let v, amt = rows64.(i) in
    Int64.shift_right_logical v (Int64.to_int amt)
  in
  compare_rows
    "lsr64"
    n
    (fun i -> Vector.get out i = expected i)
    (fun i ->
      let v, amt = rows64.(i) in
      Printf.sprintf
        "0x%016Lx lsr %Ld: got 0x%016Lx, expected 0x%016Lx"
        v
        amt
        (Vector.get out i)
        (expected i))

let loop_threads = 96

let loop_m = 5

let words =
  Array.init (loop_threads * 8) (fun i ->
      Int32.(logxor (mul (of_int i) 0x9e3779b1l) (shift_left (of_int i) 13)))

let run_loops dev ir =
  let n = loop_threads in
  let w = Vector.create Vector.int32 (n * 8) in
  let out = Vector.create Vector.int32 n in
  Array.iteri (fun i x -> Vector.set w i x) words ;
  for i = 0 to n - 1 do
    Vector.set out i 0x5a5a5a5al
  done ;
  launch
    dev
    ir
    ~n
    [Execute.Vec w; Execute.Vec out; Execute.Int n; Execute.Int loop_m] ;
  let expected i = loops_reference words ~m:loop_m i in
  compare_rows
    "loops"
    n
    (fun i -> Vector.get out i = expected i)
    (fun i ->
      Printf.sprintf "got %ld, expected %ld" (Vector.get out i) (expected i))

let generator (dev : Device.t) =
  let open Sarek_codegen in
  match dev.Device.framework with
  | "CUDA" | "CUDA/C" | "HIP" -> Some Sarek_ir_cuda.generate
  | "OpenCL" -> Some Sarek_ir_opencl.generate
  | "Metal" -> Some Sarek_ir_metal.generate
  | "Vulkan" -> Some (fun k -> Sarek_ir_glsl.generate k)
  | "WebGPU" -> Some (fun k -> Sarek_ir_wgsl.generate k)
  | "CUDA/PTX" -> Some (fun k -> Sarek_ir_ptx.generate k)
  | _ -> None

(* A GPU failure that must FAIL this test: codegen raised, the device compiler
   or loader rejected the source, or the launch failed. A backend that is not
   there stays a skip. *)
let rejected_source = function
  | Sarek_backend_error.Backend_error.Backend_error
      ( Codegen _
      | Runtime
          {
            error = Compilation_failed _ | Module_load_failed _ | Context_error _;
            _;
          } ) ->
      true
  | _ -> false

let is_host (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

(* int64 needs a device with 64-bit integers: WebGPU has none, and refuses
   the kernel at code generation. *)
let supports_int64 (dev : Device.t) = dev.Device.framework <> "WebGPU"

let ir_of kern name =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith (name ^ " kernel has no IR")

let () =
  let cases =
    [
      ( "lsr32",
        run_shift32 ~operand:Fun.id,
        ir_of lsr32_kernel "lsr32",
        Fun.const true );
      ( "expr",
        run_shift32 ~operand:(fun v -> Int32.add v 1l),
        ir_of expr_kernel "expr",
        Fun.const true );
      ("lsr64", run_lsr64, ir_of lsr64_kernel "lsr64", supports_int64);
      ("loops", run_loops, ir_of loops_kernel "loops", Fun.const true);
    ]
  in
  let devs = Device.init () in
  print_endline "=== lsr and automatic unroll, E2E ===" ;
  let host_ran = ref false in
  let failed = ref false in
  let report (dev : Device.t) label status detail =
    Printf.printf
      "  %-11s %-34s %-6s %s %s\n%!"
      dev.Device.framework
      dev.Device.name
      label
      status
      detail
  in
  Array.iter
    (fun (dev : Device.t) ->
      let host = is_host dev in
      List.iter
        (fun (label, runner, ir, applies) ->
          if not (applies dev) then report dev label "SKIP" "(no int64)"
          else begin
            (match generator dev with
            | None -> ()
            | Some gen -> (
                match gen ir with
                | _ -> ()
                | exception e ->
                    report dev label "FAIL" ("codegen: " ^ Printexc.to_string e) ;
                    failed := true)) ;
            match runner dev ir with
            | bad, total ->
                report
                  dev
                  label
                  (if bad = 0 then "PASS" else "FAIL")
                  (Printf.sprintf "(%d/%d ok)" (total - bad) total) ;
                if bad <> 0 then failed := true ;
                if host then host_ran := true
            | exception e ->
                let regression = host || rejected_source e in
                report
                  dev
                  label
                  (if regression then "FAIL" else "SKIP")
                  (Printexc.to_string e) ;
                if regression then failed := true
          end)
        cases)
    devs ;
  if not !host_ran then begin
    print_endline "test_lsr_unroll: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_lsr_unroll: FAILED" ;
    exit 1
  end ;
  print_endline "test_lsr_unroll: PASSED"
