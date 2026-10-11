(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Property test of Sarek_ir_f16_narrowing against a reference evaluation
 * (specs/f16-narrowing-barrier.md §2).
 *
 * Random expressions over binary16 leaves, masked/sign-extended integers and
 * constants. Whenever the analysis claims a bound, every evaluation must be
 * exact in f32 (binary64 and per-operation binary32 evaluation agree) and
 * satisfy the bound. Whenever it drops the barrier, a model of the fused
 * device evaluation — round16 of the exact value, with an absorbed product or
 * negation turning a zero into +0 unless emitted as a signed product — must
 * agree bit for bit with the interpreter's round16(round32(...)).
 ******************************************************************************)

open Sarek_ir_types
module N = Sarek_codegen.Sarek_ir_f16_narrowing

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let facts =
  N.of_kernel
    {
      default_kernel with
      kern_params = [vec "h" TFloat16; vec "g" TFloat16; vec "q" TInt32];
    }

let idx = EConst (CInt32 0l)

let i32 n = EConst (CInt32 (Int32.of_int n))

let ( >>: ) a b = EBinop (Shr, a, b)

(* ---- generator ---- *)

let constants =
  [
    0.0;
    -0.0;
    1.0;
    2.0;
    0.5;
    3.0;
    -7.0;
    1000.0;
    1.1;
    0.1;
    65504.0;
    ldexp 1.0 (-20);
  ]

let int_leaf rng =
  (* Three independent int32 inputs: a leaf reads one of q.(0..2). *)
  let q = EArrayRead ("q", i32 (Random.State.int rng 3)) in
  match Random.State.int rng 4 with
  | 0 ->
      EBinop
        (BitAnd, q, i32 [|15; 63; 255; 2047; 0xffff|].(Random.State.int rng 5))
  | 1 -> EBinop (Shl, q, i32 24) >>: i32 24
  | 2 -> EBinop (Sub, EBinop (BitAnd, q, i32 63), i32 32)
  | _ -> q

let leaf rng =
  match Random.State.int rng 4 with
  | 0 -> ECast (TFloat32, EArrayRead ("h", idx))
  | 1 -> ECast (TFloat32, EArrayRead ("g", idx))
  | 2 -> ECast (TFloat32, int_leaf rng)
  | _ ->
      let c =
        List.nth constants (Random.State.int rng (List.length constants))
      in
      EConst (CFloat32 (Int32.float_of_bits (Int32.bits_of_float c)))

let rec gen rng depth =
  if depth = 0 || Random.State.int rng 3 = 0 then leaf rng
  else
    match Random.State.int rng 4 with
    | 0 -> EBinop (Add, gen rng (depth - 1), gen rng (depth - 1))
    | 1 -> EBinop (Sub, gen rng (depth - 1), gen rng (depth - 1))
    | 2 -> EBinop (Mul, gen rng (depth - 1), gen rng (depth - 1))
    | _ -> EUnop (Neg, gen rng (depth - 1))

(* ---- evaluation ---- *)

type env = {h : float; g : float; q : int32 array}

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

let round16 x = Sarek_interp.Sarek_float16.to_float16 (round32 x)

let int_eval env e =
  let rec go = function
    | EArrayRead ("q", EConst (CInt32 i)) -> env.q.(Int32.to_int i)
    | EConst (CInt32 n) -> n
    | EBinop (BitAnd, a, b) -> Int32.logand (go a) (go b)
    | EBinop (Shl, a, b) -> Int32.shift_left (go a) (Int32.to_int (go b))
    | EBinop (Shr, a, b) -> Int32.shift_right (go a) (Int32.to_int (go b))
    | EBinop (Sub, a, b) -> Int32.sub (go a) (go b)
    | _ -> assert false
  in
  Int32.to_float (go e)

(* [eval ~round env e]: binary64 evaluation, each operation passed to [round]. *)
let rec eval ~round env = function
  | ECast (TFloat32, EArrayRead ("h", _)) -> env.h
  | ECast (TFloat32, EArrayRead ("g", _)) -> env.g
  | ECast (TFloat32, i) -> round (int_eval env i)
  | EConst (CFloat32 c) -> c
  | EBinop (Add, a, b) -> round (eval ~round env a +. eval ~round env b)
  | EBinop (Sub, a, b) -> round (eval ~round env a -. eval ~round env b)
  | EBinop (Mul, a, b) -> round (eval ~round env a *. eval ~round env b)
  | EUnop (Neg, a) -> -.eval ~round env a
  | _ -> assert false

(* The fused device evaluation of a narrowing emitted as [form]: one rounding
   of the exact value; an absorbed product or negation turns a zero into +0,
   except through sarek_narrow_exact_product, which ors in the sign of the
   exact product (that of [v]). *)
let device form env e =
  let v = eval ~round:Fun.id env e in
  match (form, e) with
  | N.Signed_product _, _ -> round16 v
  | _, (EBinop (Mul, _, _) | EUnop (Neg, _)) when v = 0.0 -> 0.0
  | _ -> round16 v

let same a b =
  Int64.bits_of_float a = Int64.bits_of_float b
  || (Float.is_nan a && Float.is_nan b)

let within (b : N.bound) v =
  v = 0.0
  || Float.abs v <= Float.ldexp b.kmax b.qhi
     && Float.is_integer (Float.ldexp v (-b.qlo))
     && ((not b.binary16) || same (round16 v) v)

let flags_hold (b : N.bound) v =
  ((not b.nonzero) || v <> 0.0)
  && ((not b.nonneg) || v > 0.0 || (v = 0.0 && not (Float.sign_bit v)))

(* A random finite binary16 value, both zeros and subnormals included. *)
let rec f16_sample rng =
  let bits = Random.State.int rng 0x10000 in
  let sign = if bits land 0x8000 <> 0 then -1.0 else 1.0 in
  let e = (bits lsr 10) land 0x1f and m = bits land 0x3ff in
  if e = 31 then f16_sample rng
  else if e = 0 then sign *. float_of_int m *. ldexp 1.0 (-24)
  else sign *. float_of_int (1024 + m) *. ldexp 1.0 (e - 25)

(* Small values half the time, so that masked leaves often coincide and
   differences are zero or negative. *)
let int_sample rng =
  if Random.State.bool rng then Int32.of_int (Random.State.int rng 4)
  else Random.State.bits32 rng

let env_sample rng =
  {
    h = f16_sample rng;
    g = f16_sample rng;
    q = Array.init 3 (fun _ -> int_sample rng);
  }

(* ---- the property ---- *)

let check_expr rng e (claims, elided, failures) =
  let bound = N.exact facts e and form = N.form facts e in
  let barrier = form = N.Barrier in
  let bad = ref 0 in
  for _ = 1 to 200 do
    let env = env_sample rng in
    let interp = round16 (eval ~round:round32 env e) in
    (match bound with
    | Some b ->
        let exact = eval ~round:Fun.id env e in
        if
          not
            (same exact (eval ~round:round32 env e)
            && within b exact && flags_hold b exact)
        then incr bad
    | None -> ()) ;
    if (not barrier) && not (same (device form env e) interp) then incr bad
  done ;
  ( (claims + if bound <> None then 1 else 0),
    (elided + if barrier then 0 else 1),
    failures + !bad )

let test_property () =
  let rng = Random.State.make [|1640|] in
  let claims, elided, failures =
    List.fold_left
      (fun acc _ -> check_expr rng (gen rng 3) acc)
      (0, 0, 0)
      (List.init 4000 Fun.id)
  in
  Printf.printf
    "%d expressions with a bound, %d elided, %d failures\n"
    claims
    elided
    failures ;
  Alcotest.(check bool)
    "the property was exercised"
    true
    (claims > 500 && elided > 300) ;
  Alcotest.(check int) "failures" 0 failures

let () =
  Alcotest.run
    "f16_narrowing_property"
    [
      ( "analysis",
        [Alcotest.test_case "random expressions" `Quick test_property] );
    ]
