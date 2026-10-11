(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_eval_int is an int32 evaluator: an int64 value must never be
 * truncated into it (compiler review C-11).
 ******************************************************************************)

open Sarek_ir_types
module E = Sarek_ir_eval_int

let launch : E.launch =
  {grid = (4, 1, 1); block = (16, 16, 1); scalar = (fun _ -> Some 3l)}

let wide = 4_294_967_301L (* 2^32 + 5: Int64.to_int32 gives 5 *)

let check_declined name e =
  match E.eval launch e with
  | None -> ()
  | Some v ->
      Alcotest.failf "%s: evaluated to %ld, an int64 value was truncated" name v

let test_wide_constant () =
  check_declined "2^32 + 5" (EConst (CInt64 wide)) ;
  check_declined
    "cast of the wide constant"
    (ECast (TInt32, EConst (CInt64 wide))) ;
  check_declined
    "wide constant in a sum"
    (EBinop (Add, EConst (CInt32 1l), EConst (CInt64 wide)))

(* The product of two int32 values is an int64 product when cast first: 65536 *
   65536 is 2^32, which int32 arithmetic would wrap to 0. *)
let test_wide_product () =
  let big = EConst (CInt32 65536l) in
  check_declined
    "int64 product"
    (EBinop (Mul, ECast (TInt64, big), ECast (TInt64, big)))

let test_int32_unchanged () =
  Alcotest.(check (option int32))
    "int32 sum"
    (Some 7l)
    (E.eval launch (EBinop (Add, EConst (CInt32 3l), EConst (CInt32 4l)))) ;
  Alcotest.(check (option int32))
    "int32 wraps as on the device"
    (Some 0l)
    (E.eval
       launch
       (EBinop (Mul, EConst (CInt32 65536l), EConst (CInt32 65536l))))

let () =
  Alcotest.run
    "sarek_ir_eval_int"
    [
      ( "int64",
        [
          Alcotest.test_case "wide constant" `Quick test_wide_constant;
          Alcotest.test_case "wide product" `Quick test_wide_product;
          Alcotest.test_case "int32 unchanged" `Quick test_int32_unchanged;
        ] );
    ]
