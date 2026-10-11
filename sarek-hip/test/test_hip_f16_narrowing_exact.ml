(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * f32 -> f16 narrowing without the barrier where the narrowed value is exact
 * (specs/f16-narrowing-barrier.md).
 *
 * The narrowing barrier stops the AMDGPU backend from fusing the producing f32
 * multiply into the narrowing (one rounding where the DSL mandates two). When
 * the narrowed f32 value is computed with no rounding at all, the two
 * evaluations agree, so the code generator may drop the barrier. This gate
 * checks that claim on hardware for the dequantisation shapes it was made for:
 *
 *   q8   f16(d * float(int8))                every finite binary16 d x every
 *                                            int8 value: 63488 * 256 outputs
 *   q6   f16(d * float(int8) * float(q6))    every finite binary16 d x every
 *                                            int8 scale, q6 in [-32, 31]
 *   q4   f16(d * float(q4) - m * float(q6))  the barrier must STAY (inexact)
 *
 * For each: the generated source must (q8, q6) or must not (q4) be free of
 * barrier calls; the device must agree BIT FOR BIT with a host reference that
 * rounds to f32 and then to binary16 (the interpreter's semantics); and the
 * interpreter itself must agree with the device on a slice.
 *
 * Skip-clean off ROCm.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

(* One thread per int32 word, four values per word (one per byte, sign-extended
   as Q8_0 does), scale [d] shared by 64 words: every scale meets each of the
   256 byte values exactly once. *)
let k_q8 =
  [%kernel
    fun (out : float16 vector)
        (scales : float16 vector)
        (qs : int32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then begin
        let d = float32_of_float16 scales.(tid / 64l) in
        let w = qs.(tid) in
        let o = tid * 4l in
        out.(o) <- float16_of_float32 (d *. float ((w lsl 24l) asr 24l)) ;
        out.(o + 1l) <- float16_of_float32 (d *. float ((w lsl 16l) asr 24l)) ;
        out.(o + 2l) <- float16_of_float32 (d *. float ((w lsl 8l) asr 24l)) ;
        out.(o + 3l) <- float16_of_float32 (d *. float (w asr 24l))
      end]

(* Q6_K shape: an int8 sub-block scale times a 6-bit quant in [-32, 31]. The
   largest significand is 2047 * 128 * 32 < 2^24, so the f32 value is exact. *)
let k_q6 =
  [%kernel
    fun (out : float16 vector)
        (scales : float16 vector)
        (qs : int32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then begin
        let d = float32_of_float16 scales.(tid / 64l) in
        let w = qs.(tid) in
        let ds = d *. float ((w lsl 24l) asr 24l) in
        let o = tid * 4l in
        out.(o) <- float16_of_float32 (ds *. float (((w asr 8l) land 63l) - 32l)) ;
        out.(o + 1l) <-
          float16_of_float32 (ds *. float (((w asr 14l) land 63l) - 32l)) ;
        out.(o + 2l) <-
          float16_of_float32 (ds *. float (((w asr 20l) land 63l) - 32l)) ;
        out.(o + 3l) <-
          float16_of_float32 (ds *. float (((w asr 26l) land 63l) - 32l))
      end]

(* Q4_K shape: d * q - m * q'. A difference of two products with independent
   exponents is not exact in general, so the barrier must stay. *)
let k_q4 =
  [%kernel
    fun (out : float16 vector)
        (scales : float16 vector)
        (qs : int32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then begin
        let d = float32_of_float16 scales.(tid / 64l) in
        let w = qs.(tid) in
        let dm = d *. 0.5 *. float ((w asr 4l) land 63l) in
        let o = tid * 4l in
        out.(o) <- float16_of_float32 ((d *. float (w land 15l)) -. dm) ;
        out.(o + 1l) <-
          float16_of_float32 ((d *. float ((w asr 8l) land 15l)) -. dm) ;
        out.(o + 2l) <-
          float16_of_float32 ((d *. float ((w asr 16l) land 15l)) -. dm) ;
        out.(o + 3l) <-
          float16_of_float32 ((d *. float ((w asr 24l) land 15l)) -. dm)
      end]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* ---- host model ---- *)

let round16 = Sarek_interp.Sarek_float16.to_float16

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let f16_value_of_bits b =
  let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
  let exp = (b lsr 10) land 0x1f and man = b land 0x3ff in
  if exp = 31 then None
  else if exp = 0 then Some (sign *. float_of_int man *. ldexp 1.0 (-24))
  else Some (sign *. float_of_int (1024 + man) *. ldexp 1.0 (exp - 25))

let scale_values =
  List.init 0x10000 Fun.id |> List.filter_map f16_value_of_bits |> Array.of_list

let n_scales = Array.length scale_values

let n_words = n_scales * 64

(* Word [t]: byte k holds (t mod 64) * 4 + k, so the 64 words of one scale
   carry the 256 byte values once each. *)
let word t =
  let b k = ((t mod 64 * 4) + k) land 0xff in
  Int32.of_int (b 0 lor (b 1 lsl 8) lor (b 2 lsl 16) lor (b 3 lsl 24))

let sbyte w k =
  let v = (Int32.to_int w lsr (8 * k)) land 0xff in
  if v >= 128 then v - 256 else v

let field w shift mask = (Int32.to_int w asr shift) land mask

(* The interpreter's semantics, step for step: every f32 operation rounds to
   f32, the narrowing rounds that to binary16. *)
let ref_q8 d w k = round16 (f32 (d *. float_of_int (sbyte w k)))

let ref_q6 d w k =
  let ds = f32 (d *. float_of_int (sbyte w 0)) in
  let shift = [|8; 14; 20; 26|].(k) in
  round16 (f32 (ds *. float_of_int (field w shift 63 - 32)))

let ref_q4 d w k =
  let dm = f32 (f32 (d *. 0.5) *. float_of_int (field w 4 63)) in
  let q = field w (8 * k) 15 in
  round16 (f32 (f32 (d *. float_of_int q) -. dm))

(* ---- runs ---- *)

let inputs words =
  let scales = Vector.create Vector.float16 ((words + 63) / 64) in
  Array.iteri
    (fun i x -> if i < Vector.length scales then Vector.set scales i x)
    scale_values ;
  let qs = Vector.create Vector.int32 words in
  for t = 0 to words - 1 do
    Vector.set qs t (word t)
  done ;
  let out = Vector.create Vector.float16 (4 * words) in
  (scales, qs, out)

let args (scales, qs, out) words =
  Execute.[Vec out; Vec scales; Vec qs; Int words]

let run_device dev ir words =
  let ((_, _, out) as io) = inputs words in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:(args io words)
    ~block:(Execute.dims1d 256)
    ~grid:(Execute.dims1d ((words + 255) / 256))
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let run_interp ir words =
  let ((_, _, out) as io) = inputs words in
  Execute.run_interpreter_vectors
    ~ir
    ~args:(args io words)
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d ((words + 63) / 64))
    ~parallel:false ;
  Vector.to_array out

let same_bits g e =
  Int64.bits_of_float g = Int64.bits_of_float e || (g <> g && e <> e)

let mismatches ~expected got =
  let bad = ref 0 and first = ref None in
  Array.iteri
    (fun i g ->
      let e = expected i in
      if not (same_bits g e) then begin
        incr bad ;
        if !first = None then first := Some (i, g, e)
      end)
    got ;
  (!bad, !first)

let report label (bad, first) total =
  (match first with
  | Some (i, g, e) ->
      Printf.printf
        "    %-34s %d/%d differ, first [%d]: got %.9g expected %.9g\n"
        label
        bad
        total
        i
        g
        e
  | None -> Printf.printf "    %-34s 0/%d differ\n" label total) ;
  bad = 0

(* ---- the gate ---- *)

(* Narrowings that go through the barrier; the declaration is not counted. *)
let barrier_call = "__float2half(sarek_f32_barrier("

let barrier_calls src =
  let nl = String.length barrier_call and hl = String.length src in
  let rec go i acc =
    if i + nl > hl then acc
    else if String.sub src i nl = barrier_call then go (i + 1) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

type case = {
  name : string;
  ir : Sarek_ir_types.kernel;
  reference : float -> int32 -> int -> float;
  barrier_expected : bool;
}

let cases =
  [
    {name = "q8"; ir = ir_of k_q8; reference = ref_q8; barrier_expected = false};
    {name = "q6"; ir = ir_of k_q6; reference = ref_q6; barrier_expected = false};
    {name = "q4"; ir = ir_of k_q4; reference = ref_q4; barrier_expected = true};
  ]

let check_source c =
  let src =
    Sarek_codegen.Sarek_ir_cuda.generate_with_types
      ~types:c.ir.Sarek_ir_types.kern_types
      c.ir
  in
  let calls = barrier_calls src in
  let ok = if c.barrier_expected then calls = 4 else calls = 0 in
  Printf.printf
    "    %-34s %d barrier call(s), expected %s: %s\n"
    (c.name ^ " source")
    calls
    (if c.barrier_expected then "4" else "0")
    (if ok then "OK" else "FAIL") ;
  ok

let expected_at c i = c.reference scale_values.(i / 256) (word (i / 4)) (i mod 4)

let check_device dev c =
  let got = run_device dev c.ir n_words in
  report
    (c.name ^ " device vs f32-then-f16 model")
    (mismatches ~expected:(expected_at c) got)
    (Array.length got)

(* The interpreter is slow: a slice of 1024 words (16 scales, 4096 values). *)
let check_interp dev c =
  let words = 1024 in
  let dev_out = run_device dev c.ir words in
  let interp_out = run_interp c.ir words in
  report
    (c.name ^ " interpreter vs device (slice)")
    (mismatches ~expected:(fun i -> dev_out.(i)) interp_out)
    (Array.length interp_out)

let () =
  let hip =
    Device.init () |> Array.to_list
    |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  match hip with
  | [] -> print_endline "[SKIP] no HIP device available"
  | dev :: _ ->
      Printf.printf
        "=== f16 narrowing without barrier on exact values: %s ===\n"
        dev.Device.name ;
      let results =
        List.concat_map
          (fun c ->
            let src = check_source c in
            let device = check_device dev c in
            let interp = check_interp dev c in
            [src; device; interp])
          cases
      in
      if List.for_all Fun.id results then print_endline "[PASS]"
      else begin
        print_endline "[FAIL]" ;
        exit 1
      end
