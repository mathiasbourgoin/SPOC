(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The host proof behind dropping sarek_f32_barrier (specs/f16-narrowing-
 * barrier.md §2).
 *
 * Without the barrier the AMDGPU backend may compute round16(v) from the exact
 * value v of the narrowed expression, where the DSL (and the interpreter)
 * compute round16(round32(v)). The two agree whenever
 *
 *   (T1) v is representable in binary32, so round32(v) = v; or
 *   (T2) v = a + b or a - b with a, b binary16 values: binary32 has
 *        24 >= 2 * 11 + 1 significand bits, so double rounding is innocuous.
 *
 * For a product absorbed into the narrowing the device computes a * b + (+0),
 * which turns a -0 product into +0; T1 is checked together with the sign
 * restoration sarek_narrow_exact_product applies.
 *
 * T1 is what the code generator's exactness analysis establishes; this file
 * checks it, by enumeration, on the shapes it was built for (a binary16 scale
 * times a small integer), and checks T2 on every binary16 [a] against a
 * structured and a random set of [b]. A control shows the harness sees a
 * double-rounding difference when the hypothesis fails.
 ******************************************************************************)

(* Correctly rounded (round to nearest, ties to even) binary64 -> binary16, in
   one step. The reference for "the fused instruction". *)
let round16_once x =
  let ax = Float.abs x in
  if Float.is_nan x || ax = 0.0 || ax = Float.infinity then x
  else
    let _, e = Float.frexp ax in
    let qexp = max (e - 1 - 10) (-24) in
    let scaled = Float.ldexp ax (-qexp) in
    let fl = Float.trunc scaled in
    let frac = scaled -. fl in
    let r =
      if frac > 0.5 then fl +. 1.0
      else if frac < 0.5 then fl
      else if Float.rem fl 2.0 = 0.0 then fl
      else fl +. 1.0
    in
    let v = Float.ldexp r qexp in
    Float.copy_sign (if v > 65504.0 then Float.infinity else v) x

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The interpreter's narrowing: binary32 first, then binary16. *)
let round16_twice x = Sarek_interp.Sarek_float16.to_float16 (round32 x)

let same a b =
  Int64.bits_of_float a = Int64.bits_of_float b
  || (Float.is_nan a && Float.is_nan b)

let f16_values =
  let of_bits b =
    let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
    let exp = (b lsr 10) land 0x1f and man = b land 0x3ff in
    if exp = 31 then None
    else if exp = 0 then Some (sign *. float_of_int man *. ldexp 1.0 (-24))
    else Some (sign *. float_of_int (1024 + man) *. ldexp 1.0 (exp - 25))
  in
  List.init 0x10000 Fun.id |> List.filter_map of_bits |> Array.of_list

let failures = ref 0

(* Counts every failure, prints the first twenty. *)
let fail fmt =
  incr failures ;
  if !failures <= 20 then Printf.printf ("  FAIL " ^^ fmt ^^ "\n")
  else Printf.ifprintf stdout fmt

(* round16_once must agree with the interpreter's narrowing wherever there is
   only one rounding to do: on binary32 inputs. Sampled over every exponent. *)
let check_reference () =
  let rng = Random.State.make [|40|] in
  let n = 1 lsl 20 in
  for i = 0 to n - 1 do
    let hi = i lsl 12 and lo = Random.State.bits rng land 0xfff in
    let x = Int32.float_of_bits (Int32.of_int (hi lor lo)) in
    if Float.is_finite x && not (same (round16_once x) (round16_twice x)) then
      fail "reference: round16_once %h <> interpreter %h" x (round16_twice x)
  done ;
  Printf.printf "reference rounding: %d binary32 samples checked\n" n

(* T1 on [d * n], every finite binary16 [d], every [n] in [lo, hi]. *)
(* The fused narrowing of a product, a * b + (+0) rounded once, with the sign
   of the exact product or-ed in (sarek_narrow_exact_product). *)
let fused_signed a b =
  let r = round16_once ((a *. b) +. 0.0) in
  if Float.sign_bit a <> Float.sign_bit b then -.Float.abs r else r

let check_product ~label ~lo ~hi =
  let checked = ref 0 in
  Array.iter
    (fun d ->
      for n = lo to hi do
        let v = d *. float_of_int n in
        incr checked ;
        if round32 v <> v then
          fail "%s: %h * %d is not exact in binary32" label d n
        else if not (same (round16_once v) (round16_twice v)) then
          fail "%s: %h * %d rounds differently" label d n
        else if not (same (fused_signed d (float_of_int n)) (round16_twice v))
        then fail "%s: %h * %d signed product differs" label d n
      done)
    f16_values ;
  Printf.printf
    "T1 %s: %d products, all exact, one = two roundings, signed fused product \
     = two roundings\n"
    label
    !checked

(* T1 on the Q6_K shape [d * sc * q]: sc int8, q in [-32, 31]. Every d with the
   extreme (sc, q) pairs, then random triples. *)
let check_triple () =
  let one d sc q =
    let v = d *. float_of_int sc *. float_of_int q in
    if round32 v <> v then fail "q6: %h * %d * %d not exact" d sc q
    else if not (same (round16_once v) (round16_twice v)) then
      fail "q6: %h * %d * %d rounds differently" d sc q
  in
  Array.iter
    (fun d ->
      List.iter (fun (sc, q) -> one d sc q) [(-128, -32); (127, 31); (-128, 31)])
    f16_values ;
  let rng = Random.State.make [|6|] in
  let m = Array.length f16_values in
  for _ = 1 to 4_000_000 do
    one
      f16_values.(Random.State.int rng m)
      (Random.State.int rng 256 - 128)
      (Random.State.int rng 64 - 32)
  done ;
  print_endline "T1 q6 triple: extremes x every d, 4M random triples"

(* T2: every binary16 [a] against 64 structured [b] (powers of two and their
   neighbours, both signs) and 64 random [b]; sum and difference. *)
let check_sum () =
  let structured =
    List.concat_map
      (fun k ->
        let p = ldexp 1.0 k in
        List.map
          round16_once
          [p; -.p; p +. ldexp p (-10); -.(p -. ldexp p (-11))])
      (List.init 16 (fun i -> (i * 2) - 16))
  in
  let rng = Random.State.make [|2|] in
  let m = Array.length f16_values in
  let random = List.init 64 (fun _ -> f16_values.(Random.State.int rng m)) in
  let bs = structured @ random in
  let checked = ref 0 in
  Array.iter
    (fun a ->
      List.iter
        (fun b ->
          List.iter
            (fun v ->
              incr checked ;
              if not (same (round16_once v) (round16_twice v)) then
                fail "T2: %h with %h rounds differently" a b)
            [a +. b; a -. b])
        bs)
    f16_values ;
  Printf.printf
    "T2 binary16 sums: %d sums/differences, one = two roundings\n"
    !checked

(* Control: f16(x * 1.1f) — the product is NOT exact in binary32 — must show
   disagreements, or this harness cannot see double rounding at all. *)
let check_control () =
  let c = round32 1.1 in
  let differ =
    Array.fold_left
      (fun acc x ->
        let v = x *. c in
        if same (round16_once v) (round16_twice v) then acc else acc + 1)
      0
      f16_values
  in
  Printf.printf
    "control f16(x * 1.1f): %d/%d differ\n"
    differ
    (Array.length f16_values) ;
  if differ = 0 then fail "control: no disagreement, the harness is blind"

let () =
  check_reference () ;
  check_product ~label:"f16 * int8" ~lo:(-128) ~hi:127 ;
  check_product ~label:"f16 * uint8" ~lo:128 ~hi:255 ;
  check_triple () ;
  check_sum () ;
  check_control () ;
  if !failures > 0 then begin
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1
  end
