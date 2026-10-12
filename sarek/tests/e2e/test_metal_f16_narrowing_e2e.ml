(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Scalar f16 on Metal, executed from SAREK-GENERATED source
 * (specs/coopmat-metal-simdgroup.md, docs/fp-contraction-policy.md §10.14).
 *
 * The narrowing [float16_of_float32] is emitted on Metal as a plain [(half)]
 * cast, with no opacity barrier, because the Metal compiler was measured not to
 * fuse the producing f32 operation into it
 * (tools/probes/metal_f16_narrowing_probe.m, hand-written MSL). This test runs
 * the kernel Sarek GENERATES, over every finite binary16 value (63488), for the
 * two shapes where AMDGPU/rusticl fuse:
 *
 *   mul     f16(x * 1.1)
 *   chain   f16(f16(x * 1.1) + 1000)
 *
 * and compares bit for bit with (a) the Interpreter and (b) an independent
 * OCaml model: round to binary32 after every operation, narrow with the
 * interpreter's binary16 rounding.
 *
 * Positive controls (they go red on the same data, so the comparison can see a
 * wrongly rounded or mis-constant result):
 *   demoted the arithmetic done in binary16 with 1.1 narrowed to binary16: what
 *           a compiler that fuses the narrowing into the operation can produce;
 *   const   the constant one binary32 ulp above 1.1.
 * Each control is required to differ from the Interpreter (host only, any
 * machine) and from the device output.
 *
 * The device part self-skips (exit 0) without a Metal device; the host checks
 * (interpreter = model, controls red) always run.
 *
 * Run on a Mac: dune build @sarek/tests/e2e/e2e-metal --force
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Gpu = Sarek_stdlib.Gpu

let () = Test_helpers.Benchmarks.init_backends ()

let k_mul =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1)]

let k_chain =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (float32_of_float16
               (float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1))
            +. 1000.0)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* ---- the domain and the independent model ---- *)

let round16 = Sarek_interp.Sarek_float16.to_float16

(* Plain IEEE round-to-nearest-even binary32 (not the interpreter's
   subnormal-flushing [Sarek_float32.to_float32]). *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let c11 = f32 1.1

(* Control constants: 1.1 narrowed to binary16 (a compiler demoting the
   arithmetic), and the next binary32 above [c11]. *)
let c11_16 = round16 1.1

let c11_up = Int32.float_of_bits (Int32.succ (Int32.bits_of_float c11))

let finite_f16 =
  let value bits =
    let sign = if bits land 0x8000 <> 0 then -1.0 else 1.0 in
    let exp = (bits lsr 10) land 0x1f and man = bits land 0x3ff in
    if exp = 31 then None
    else if exp = 0 then Some (sign *. float_of_int man *. ldexp 1.0 (-24))
    else Some (sign *. float_of_int (1024 + man) *. ldexp 1.0 (exp - 25))
  in
  List.init 0x10000 Fun.id |> List.filter_map value |> Array.of_list

let count = Array.length finite_f16

type shape = {
  name : string;
  ir : Sarek_ir_types.kernel;
  model : float -> float;  (** f32 after every operation, then narrow *)
  demoted : float -> float;  (** control: arithmetic done in binary16 *)
  const : float -> float;  (** control: constant one binary32 ulp above 1.1 *)
}

let shapes =
  [
    {
      name = "mul    f16(x*1.1)";
      ir = ir_of k_mul;
      model = (fun x -> round16 (f32 (x *. c11)));
      demoted = (fun x -> round16 (x *. c11_16));
      const = (fun x -> round16 (f32 (x *. c11_up)));
    };
    {
      name = "chain  f16(f16(x*1.1)+1000)";
      ir = ir_of k_chain;
      model = (fun x -> round16 (f32 (round16 (f32 (x *. c11)) +. 1000.0)));
      demoted = (fun x -> round16 (round16 (x *. c11_16) +. 1000.0));
      const = (fun x -> round16 (f32 (round16 (f32 (x *. c11_up)) +. 1000.0)));
    };
  ]

(* ---- runs ---- *)

let inputs () =
  let inp = Vector.create Vector.float16 count in
  Array.iteri (Vector.set inp) finite_f16 ;
  (Vector.create Vector.float16 count, inp)

let block = 256

let grid = (count + block - 1) / block

let run_device dev sh =
  let out, inp = inputs () in
  Execute.run_vectors
    ~device:dev
    ~ir:sh.ir
    ~args:Execute.[Vec out; Vec inp; Int count]
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d grid)
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let run_interp sh =
  let out, inp = inputs () in
  Execute.run_interpreter_vectors
    ~ir:sh.ir
    ~args:Execute.[Vec out; Vec inp; Int count]
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d grid)
    ~parallel:false ;
  Vector.to_array out

(* ---- comparing ---- *)

let same_bits g e =
  Int64.bits_of_float g = Int64.bits_of_float e || (g <> g && e <> e)

let failures = ref 0

(* [check label ~expect_equal got expected]: [expected] maps an input index to
   the reference value. Prints the count of disagreeing inputs and the first. *)
let check label ~expect_equal got expected =
  let n = ref 0 and first = ref None in
  Array.iteri
    (fun i g ->
      let e = expected i in
      if not (same_bits g e) then begin
        incr n ;
        if !first = None then first := Some (finite_f16.(i), g, e)
      end)
    got ;
  let good = if expect_equal then !n = 0 else !n > 0 in
  if not good then incr failures ;
  Printf.printf
    "  %-4s %-40s %5d/%d differ%s\n"
    (if good then "OK" else "FAIL")
    label
    !n
    count
    (match !first with
    | Some (x, g, e) when !n > 0 ->
        Printf.sprintf "  (first: x=%h got %h reference %h)" x g e
    | _ -> "")

let of_fn f i = f finite_f16.(i)

let host_part sh =
  Printf.printf "%s\n" sh.name ;
  let interp = run_interp sh in
  check "interpreter = model" ~expect_equal:true interp (of_fn sh.model) ;
  check
    "control demoted vs interpreter (red)"
    ~expect_equal:false
    interp
    (of_fn sh.demoted) ;
  check
    "control const vs interpreter (red)"
    ~expect_equal:false
    interp
    (of_fn sh.const) ;
  interp

let device_part dev sh interp =
  Printf.printf "%s\n" sh.name ;
  let got = run_device dev sh in
  check "Metal = model" ~expect_equal:true got (of_fn sh.model) ;
  check "Metal = interpreter" ~expect_equal:true got (Array.get interp) ;
  check
    "control demoted vs Metal (red)"
    ~expect_equal:false
    got
    (of_fn sh.demoted) ;
  check "control const vs Metal (red)" ~expect_equal:false got (of_fn sh.const)

let () =
  let metal =
    Device.init () |> Array.to_list
    |> List.filter (fun d -> d.Device.framework = "Metal")
  in
  let interps = List.map host_part shapes in
  (match metal with
  | [] -> print_endline "[SKIP] no Metal device: the device checks did not run"
  | devs ->
      List.iter
        (fun dev ->
          Printf.printf "=== Metal f16 narrowing: %s ===\n" dev.Device.name ;
          List.iter2 (device_part dev) shapes interps)
        devs) ;
  if !failures > 0 then begin
    Printf.printf "[FAIL] %d check(s)\n" !failures ;
    exit 1
  end
  else print_endline "[PASS]"
