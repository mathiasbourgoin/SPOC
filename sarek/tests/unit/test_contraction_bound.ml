(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The bound of epilogue fusion on a contracting backend (specs/epilogue-
 * fusion.md FR-107), checked on the host, no device. For o = a*b + c, the
 * unfused pair computes u = round32(round32(a*b) + c) and a fused kernel whose
 * device compiler contracts computes f = round32(a*b + c) (one contraction).
 * The bound is not "1 ulp of the result": under cancellation (a*b close to -c)
 * the difference is a rounding error of the PRODUCT, many ulps of a small
 * result. What holds, for every input, is
 *
 *   |f - u| <= ulp(a*b)/2 + ulp(max(|f|, |u|))
 *
 * (u - exact <= ulp(a*b)/2 + ulp(u)/2 and f - exact <= ulp(f)/2).
 ******************************************************************************)

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* ulp of a float32 value: 2^(e-24) for x in [2^(e-1), 2^e); the smallest
   subnormal at zero. *)
let ulp x =
  if x = 0.0 then Float.ldexp 1.0 (-149)
  else
    let _, e = Float.frexp (Float.abs x) in
    Float.ldexp 1.0 (max (e - 24) (-149))

let unfused a b c = round32 (round32 (a *. b) +. c)

(* a*b is exact in binary64 (24 + 24 bits); the sum is then rounded to binary32
   (a double rounding through binary64 is possible in principle and only
   shifts the result by far less than the slack of the bound). *)
let fused a b c = round32 ((a *. b) +. c)

let within_bound a b c =
  let f = fused a b c and u = unfused a b c in
  Float.abs (f -. u)
  <= (ulp (a *. b) /. 2.0) +. ulp (Float.max (Float.abs f) (Float.abs u))

let test_cancellation_breaks_one_ulp () =
  let a = round32 (1.0 +. Float.ldexp 1.0 (-15)) in
  let b = a and c = -1.0 in
  let f = fused a b c and u = unfused a b c in
  Alcotest.(check bool)
    "the fused result is far from the unfused one in ulps of the result"
    true
    (Float.abs (f -. u) > 100.0 *. ulp f) ;
  Alcotest.(check bool) "yet within the bound" true (within_bound a b c)

let test_random () =
  let st = Random.State.make [|20261012|] in
  let draw scale =
    round32 (scale *. (Random.State.float st 2.0 -. 1.0) *. 1.5)
  in
  for _ = 1 to 200_000 do
    let a = draw 1.0 and b = draw 1.0 in
    (* c on the scale of the product, and also nearly cancelling it *)
    let c =
      match Random.State.int st 3 with
      | 0 -> draw 1.0
      | 1 -> round32 (-.(a *. b))
      | _ -> round32 (-.(a *. b) *. (1.0 +. Random.State.float st 1e-5))
    in
    if not (within_bound a b c) then
      Alcotest.failf "bound violated for a=%h b=%h c=%h" a b c
  done

let () =
  Alcotest.run
    "contraction bound"
    [
      ( "host",
        [
          Alcotest.test_case
            "cancellation breaks a 1 ulp bound"
            `Quick
            test_cancellation_breaks_one_ulp;
          Alcotest.test_case "the bound holds" `Quick test_random;
        ] );
    ]
