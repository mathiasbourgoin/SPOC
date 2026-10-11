(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * f32 -> f16 narrowings emitted without the barrier
 * (specs/f16-narrowing-barrier.md).
 *
 * The narrowing barrier stops the AMDGPU backend from fusing the producing
 * operation into the narrowing (one rounding where the DSL mandates two). The
 * code generator drops it where it proves the two agree. This gate checks the
 * claim on hardware, bit for bit, over large exhaustive input sets:
 *
 *   sum    f16(d +/- float(int8))          every finite binary16 d x every
 *                                          int8: a binary16 sum (elided)
 *   bytes  f16(float b0 * float b1 +/- b2) every (b0, b1), 64 values of b2:
 *          and f16(float b0 * float b1)    exact, never -0 (elided)
 *   q8     f16(d * float(int8))            exact; d < 0 times 0 is -0, which
 *                                          the fused form gives as +0: emitted
 *                                          as a signed product (no barrier)
 *   q6     f16(ds * float(q6)),            the same, ds a let-bound product
 *          ds = d * float(int8)
 *   q4     f16(d * float(q4) - dm)         inexact (kept)
 *   diff   f16((float b0 - float b1) * k)  exact, -0 when b0 < b1 and k = 0:
 *                                          a signed product (review finding)
 *   bits   f16(float_of_bits (bits_of_float (d * dmin * q))): LLVM folds the
 *          bitcast round trip and fuses the inexact product (kept)
 *
 * For each: the generated source has the expected number of barrier calls;
 * the device agrees BIT FOR BIT with a host model that rounds to f32 and then
 * to binary16 (the interpreter's semantics); the interpreter agrees with the
 * device on a slice. For q8 the source with the sign restoration deleted is
 * also run and its disagreements printed (not asserted): that is the hazard
 * the signed product avoids.
 *
 * Skip-clean off ROCm.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

(* One thread per int32 word, four outputs per word. [d] is shared by 64
   consecutive words. *)
let k_sum =
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
        out.(o) <- float16_of_float32 (d +. float ((w lsl 24l) asr 24l)) ;
        out.(o + 1l) <- float16_of_float32 (d +. float ((w lsl 16l) asr 24l)) ;
        out.(o + 2l) <- float16_of_float32 (d -. float ((w lsl 8l) asr 24l)) ;
        out.(o + 3l) <- float16_of_float32 (d -. float (w asr 24l))
      end]

let k_bytes =
  [%kernel
    fun (out : float16 vector)
        (scales : float16 vector)
        (qs : int32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then begin
        let w = qs.(tid) in
        let p = float (w land 255l) *. float ((w asr 8l) land 255l) in
        let b2 = float ((w asr 16l) land 255l) in
        let o = tid * 4l in
        out.(o) <- float16_of_float32 (p +. b2) ;
        out.(o + 1l) <- float16_of_float32 p ;
        out.(o + 2l) <- float16_of_float32 (p -. b2) ;
        out.(o + 3l) <- float16_of_float32 (b2 -. p)
      end]

let k_diff =
  [%kernel
    fun (out : float16 vector)
        (scales : float16 vector)
        (qs : int32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then begin
        let w = qs.(tid) in
        let b0 = float (w land 255l) in
        let b1 = float ((w asr 8l) land 255l) in
        let k = float ((w asr 16l) land 2l) in
        let o = tid * 4l in
        out.(o) <- float16_of_float32 ((b0 -. b1) *. k) ;
        out.(o + 1l) <- float16_of_float32 ((b1 -. b0) *. k) ;
        out.(o + 2l) <- float16_of_float32 ((0.0 -. b0) *. b1) ;
        out.(o + 3l) <-
          float16_of_float32 ((b0 -. b1) *. float ((w asr 16l) land 255l))
      end]

let k_bits =
  [%kernel
    fun (out : float16 vector)
        (scales : float16 vector)
        (qs : int32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then begin
        let d = float32_of_float16 scales.(tid / 64l) in
        let dmin = float32_of_float16 scales.(tid / 64l lxor 1l) in
        let w = qs.(tid) in
        let o = tid * 4l in
        out.(o) <-
          float16_of_float32
            (Int32.float_of_bits
               (Int32.bits_of_float (d *. dmin *. float (w land 255l)))) ;
        out.(o + 1l) <-
          float16_of_float32
            (Int32.float_of_bits
               (Int32.bits_of_float (d *. dmin *. float ((w asr 8l) land 255l)))) ;
        out.(o + 2l) <-
          float16_of_float32
            (Int32.float_of_bits
               (Int32.bits_of_float
                  (d *. dmin *. float ((w asr 16l) land 255l)))) ;
        out.(o + 3l) <-
          float16_of_float32
            (Int32.float_of_bits
               (Int32.bits_of_float
                  (d *. dmin *. float ((w asr 24l) land 255l))))
      end]

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
   exponents is not exact in general. *)
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

(* Scale cases: byte k of word t holds (t mod 64) * 4 + k, so the 64 words of
   one scale carry the 256 byte values once each. *)
let scale_word t =
  let b k = ((t mod 64 * 4) + k) land 0xff in
  Int32.of_int (b 0 lor (b 1 lsl 8) lor (b 2 lsl 16) lor (b 3 lsl 24))

(* Byte case: every (b0, b1) pair, b2 = 4 * (t lsr 16) + 1 (64 values). *)
let bytes_word t =
  Int32.of_int (t land 0xffff lor ((((t lsr 16) * 4) + 1) lsl 16))

let sbyte w k =
  let v = (Int32.to_int w lsr (8 * k)) land 0xff in
  if v >= 128 then v - 256 else v

let field w shift mask = (Int32.to_int w asr shift) land mask

let fi = float_of_int

(* The interpreter's semantics, step for step: every f32 operation rounds to
   f32, the narrowing rounds that to binary16. *)
let ref_sum i =
  let d = scale_values.(i / 256) and w = scale_word (i / 4) and k = i mod 4 in
  let q = fi (sbyte w k) in
  round16 (f32 (if k < 2 then d +. q else d -. q))

let ref_bytes i =
  let w = bytes_word (i / 4) in
  let p = f32 (fi (field w 0 255) *. fi (field w 8 255))
  and b2 = fi (field w 16 255) in
  round16 (f32 [|p +. b2; p; p -. b2; b2 -. p|].(i mod 4))

let ref_diff i =
  let w = bytes_word (i / 4) in
  let b0 = fi (field w 0 255) and b1 = fi (field w 8 255) in
  let k = fi (field w 16 2) in
  round16
    (f32
       [|
         f32 (b0 -. b1) *. k;
         f32 (b1 -. b0) *. k;
         f32 (0.0 -. b0) *. b1;
         f32 (b0 -. b1) *. fi (field w 16 255);
       |].(i mod 4))

let ref_bits i =
  let d = scale_values.(i / 256) and dmin = scale_values.(i / 256 lxor 1) in
  let q = fi (field (scale_word (i / 4)) (8 * (i mod 4)) 255) in
  round16 (f32 (f32 (d *. dmin) *. q))

let ref_q8 i =
  let d = scale_values.(i / 256) and w = scale_word (i / 4) in
  round16 (f32 (d *. fi (sbyte w (i mod 4))))

let ref_q6 i =
  let d = scale_values.(i / 256) and w = scale_word (i / 4) in
  let ds = f32 (d *. fi (sbyte w 0)) in
  let shift = [|8; 14; 20; 26|].(i mod 4) in
  round16 (f32 (ds *. fi (field w shift 63 - 32)))

let ref_q4 i =
  let d = scale_values.(i / 256) and w = scale_word (i / 4) in
  let dm = f32 (f32 (d *. 0.5) *. fi (field w 4 63)) in
  round16 (f32 (f32 (d *. fi (field w (8 * (i mod 4)) 15)) -. dm))

(* ---- runs ---- *)

type case = {
  name : string;
  ir : Sarek_ir_types.kernel;
  words : int;
  word : int -> int32;
  reference : int -> float;
  barriers : int;  (** expected barrier calls in the generated source *)
}

let scale_words = n_scales * 64

let case name k ?(words = scale_words) ?(word = scale_word) reference barriers =
  {name; ir = ir_of k; words; word; reference; barriers}

let cases =
  [
    case "sum" k_sum ref_sum 0;
    case "bytes" k_bytes ~words:(1 lsl 22) ~word:bytes_word ref_bytes 0;
    case "diff" k_diff ~words:(1 lsl 22) ~word:bytes_word ref_diff 0;
    case "bits" k_bits ref_bits 4;
    case "q8" k_q8 ref_q8 0;
    case "q6" k_q6 ref_q6 0;
    case "q4" k_q4 ref_q4 4;
  ]

let inputs c words =
  let scales = Vector.create Vector.float16 n_scales in
  Array.iteri (fun i x -> Vector.set scales i x) scale_values ;
  let qs = Vector.create Vector.int32 words in
  for t = 0 to words - 1 do
    Vector.set qs t (c.word t)
  done ;
  (scales, qs, Vector.create Vector.float16 (4 * words))

let block = Execute.dims1d 256

let grid words = Execute.dims1d ((words + 255) / 256)

let run_device dev c words =
  let scales, qs, out = inputs c words in
  Execute.run_vectors
    ~device:dev
    ~ir:c.ir
    ~args:Execute.[Vec out; Vec scales; Vec qs; Int words]
    ~block
    ~grid:(grid words)
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let run_source dev c ~source =
  let scales, qs, out = inputs c c.words in
  Execute.run_source
    ~device:dev
    ~source
    ~lang:Execute.CUDA_Source
    ~kernel_name:c.ir.Sarek_ir_types.kern_name
    ~block
    ~grid:(grid c.words)
    Execute.[Vec out; Vec scales; Vec qs; Int32 (Int32.of_int c.words)] ;
  Transfer.flush dev ;
  Vector.to_array out

let run_interp c words =
  let scales, qs, out = inputs c words in
  Execute.run_interpreter_vectors
    ~ir:c.ir
    ~args:Execute.[Vec out; Vec scales; Vec qs; Int words]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d ((words + 63) / 64))
    ~parallel:false ;
  Vector.to_array out

(* ---- comparing ---- *)

let same_bits g e =
  Int64.bits_of_float g = Int64.bits_of_float e || (g <> g && e <> e)

type tally = {bad : int; zero_sign : int; first : (int * float * float) option}

(* [zero_sign]: disagreements that are +0 against -0. *)
let tally ~expected got =
  let bad = ref 0 and zero_sign = ref 0 and first = ref None in
  Array.iteri
    (fun i g ->
      let e = expected i in
      if not (same_bits g e) then begin
        incr bad ;
        if g = 0.0 && e = 0.0 then incr zero_sign ;
        if !first = None then first := Some (i, g, e)
      end)
    got ;
  {bad = !bad; zero_sign = !zero_sign; first = !first}

let report label t total =
  (match t.first with
  | Some (i, g, e) ->
      Printf.printf
        "    %-38s %d/%d differ (%d only in the sign of zero), first [%d]: got \
         %h expected %h\n"
        label
        t.bad
        total
        t.zero_sign
        i
        g
        e
  | None -> Printf.printf "    %-38s 0/%d differ\n" label total) ;
  t.bad = 0

(* ---- the gate ---- *)

let generate c =
  Sarek_codegen.Sarek_ir_cuda.generate_with_types
    ~types:c.ir.Sarek_ir_types.kern_types
    c.ir

let find needle hay =
  let nl = String.length needle and hl = String.length hay in
  let rec go i =
    if i + nl > hl then None
    else if String.sub hay i nl = needle then Some i
    else go (i + 1)
  in
  go 0

let count needle hay =
  let rec go from acc =
    match find needle (String.sub hay from (String.length hay - from)) with
    | None -> acc
    | Some i -> go (from + i + 1) (acc + 1)
  in
  go 0 0

let check_source c =
  let calls = count "__float2half(sarek_f32_barrier(" (generate c) in
  let ok = calls = c.barriers in
  Printf.printf
    "    %-38s %d barrier call(s), expected %d: %s\n"
    (c.name ^ " source")
    calls
    c.barriers
    (if ok then "OK" else "FAIL") ;
  ok

let check_device dev c =
  let got = run_device dev c c.words in
  report
    (c.name ^ " device vs f32-then-f16 model")
    (tally ~expected:c.reference got)
    (Array.length got)

(* The interpreter is slow: a slice of 1024 words (4096 values). *)
let check_interp dev c =
  let words = 1024 in
  let dev_out = run_device dev c words in
  let interp_out = run_interp c words in
  report
    (c.name ^ " interpreter vs device (slice)")
    (tally ~expected:(fun i -> dev_out.(i)) interp_out)
    (Array.length interp_out)

(* Print-only: q8 run from its source with the sign restoration of
   sarek_narrow_exact_product deleted, i.e. a bare narrowing of the product. *)
let show_without_sign dev c =
  let fix = "(__half_as_ushort(h) | s)" in
  let src = generate c in
  match find fix src with
  | None -> print_endline "    (no signed product in the q8 source)"
  | Some i ->
      let n = String.length fix in
      let source =
        String.sub src 0 i ^ "(__half_as_ushort(h))"
        ^ String.sub src (i + n) (String.length src - i - n)
      in
      let got = run_source dev c ~source in
      ignore
        (report
           (c.name ^ " WITHOUT sign fix (not asserted)")
           (tally ~expected:c.reference got)
           (Array.length got))

let check_case dev c =
  let src = check_source c in
  let device = check_device dev c in
  let interp = check_interp dev c in
  if c.name = "q8" then show_without_sign dev c ;
  src && device && interp

let () =
  let hip =
    Device.init () |> Array.to_list
    |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  match hip with
  | [] -> print_endline "[SKIP] no HIP device available"
  | dev :: _ ->
      Printf.printf
        "=== f16 narrowings emitted without barrier: %s ===\n"
        dev.Device.name ;
      let results = List.map (check_case dev) cases in
      if List.for_all Fun.id results then print_endline "[PASS]"
      else begin
        print_endline "[FAIL]" ;
        exit 1
      end
