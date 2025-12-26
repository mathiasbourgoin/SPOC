(** Unit tests for Sarek_fusion *)

open Sarek.Sarek_ir
open Sarek.Sarek_fusion

(** Helper to create a simple variable *)
let mk_var name id typ =
  {var_name = name; var_id = id; var_type = typ; var_mutable = true}

(** Helper to create thread_idx_x intrinsic *)
let thread_idx_x = EIntrinsic (["Gpu"], "thread_idx_x", [])

(** Test: analyze simple kernel with OneToOne access pattern *)
let test_analyze_one_to_one () =
  (* output[thread_idx_x] = input[thread_idx_x] * 2 *)
  let body =
    SAssign
      ( LArrayElem ("output", thread_idx_x),
        EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l)) )
  in
  let kernel =
    {kern_name = "scale"; kern_params = []; kern_locals = []; kern_body = body}
  in
  let info = analyze kernel in
  assert (List.length info.reads = 1) ;
  assert (List.length info.writes = 1) ;
  assert (List.mem_assoc "input" info.reads) ;
  assert (List.mem_assoc "output" info.writes) ;
  assert (not info.has_barriers) ;
  Printf.printf "test_analyze_one_to_one: PASSED\n"

(** Test: analyze kernel with barrier *)
let test_analyze_with_barrier () =
  let body =
    SSeq
      [
        SAssign
          ( LArrayElem ("shared", thread_idx_x),
            EArrayRead ("input", thread_idx_x) );
        SBarrier;
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("shared", thread_idx_x) );
      ]
  in
  let kernel =
    {
      kern_name = "with_barrier";
      kern_params = [];
      kern_locals = [];
      kern_body = body;
    }
  in
  let info = analyze kernel in
  assert info.has_barriers ;
  Printf.printf "test_analyze_with_barrier: PASSED\n"

(** Test: can_fuse returns true for compatible kernels *)
let test_can_fuse_compatible () =
  (* Producer: temp[i] = input[i] * 2 *)
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Consumer: output[i] = temp[i] + 1 *)
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert result ;
  Printf.printf "test_can_fuse_compatible: PASSED\n"

(** Test: can_fuse returns false when producer has barrier *)
let test_can_fuse_with_barrier () =
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SSeq
          [
            SAssign
              ( LArrayElem ("temp", thread_idx_x),
                EArrayRead ("input", thread_idx_x) );
            SBarrier;
          ];
    }
  in
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EArrayRead ("temp", thread_idx_x) );
    }
  in
  let result = can_fuse producer consumer "temp" in
  assert (not result) ;
  Printf.printf "test_can_fuse_with_barrier: PASSED\n"

(** Test: fuse inlines producer into consumer *)
let test_fuse_simple () =
  (* Producer: temp[i] = input[i] * 2 *)
  let producer =
    {
      kern_name = "producer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("temp", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* Consumer: output[i] = temp[i] + 1 *)
  let consumer =
    {
      kern_name = "consumer";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
          );
    }
  in
  let fused = fuse producer consumer "temp" in
  (* Result should be: output[i] = (input[i] * 2) + 1 *)
  assert (fused.kern_name = "consumer_fused") ;
  (* Check that temp is no longer read *)
  let info = analyze fused in
  assert (not (List.mem_assoc "temp" info.reads)) ;
  assert (List.mem_assoc "input" info.reads) ;
  assert (List.mem_assoc "output" info.writes) ;
  Printf.printf "test_fuse_simple: PASSED\n"

(** Test: fuse_pipeline with multiple kernels *)
let test_fuse_pipeline () =
  (* K1: a[i] = input[i] * 2 *)
  let k1 =
    {
      kern_name = "k1";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("a", thread_idx_x),
            EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
          );
    }
  in
  (* K2: b[i] = a[i] + 1 *)
  let k2 =
    {
      kern_name = "k2";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("b", thread_idx_x),
            EBinop (Add, EArrayRead ("a", thread_idx_x), EConst (CInt32 1l)) );
    }
  in
  (* K3: output[i] = b[i] * 3 *)
  let k3 =
    {
      kern_name = "k3";
      kern_params = [];
      kern_locals = [];
      kern_body =
        SAssign
          ( LArrayElem ("output", thread_idx_x),
            EBinop (Mul, EArrayRead ("b", thread_idx_x), EConst (CInt32 3l)) );
    }
  in
  let fused, eliminated = fuse_pipeline [k1; k2; k3] in
  (* Should eliminate both a and b *)
  assert (List.mem "a" eliminated) ;
  assert (List.mem "b" eliminated) ;
  (* Final kernel should read input and write output *)
  let info = analyze fused in
  assert (List.mem_assoc "input" info.reads) ;
  assert (List.mem_assoc "output" info.writes) ;
  assert (not (List.mem_assoc "a" info.reads)) ;
  assert (not (List.mem_assoc "b" info.reads)) ;
  Printf.printf
    "test_fuse_pipeline: PASSED (eliminated: %s)\n"
    (String.concat ", " eliminated)

(** Test: expr_equal *)
let test_expr_equal () =
  let e1 = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 2l)) in
  let e2 = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 2l)) in
  let e3 = EBinop (Add, EConst (CInt32 1l), EConst (CInt32 3l)) in
  assert (expr_equal e1 e2) ;
  assert (not (expr_equal e1 e3)) ;
  assert (expr_equal thread_idx_x thread_idx_x) ;
  Printf.printf "test_expr_equal: PASSED\n"

(** Test: subst_array_read *)
let test_subst_array_read () =
  (* temp[i] + 1  ->  (input[i] * 2) + 1 *)
  let original =
    EBinop (Add, EArrayRead ("temp", thread_idx_x), EConst (CInt32 1l))
  in
  let replacement =
    EBinop (Mul, EArrayRead ("input", thread_idx_x), EConst (CInt32 2l))
  in
  let result = subst_array_read "temp" thread_idx_x replacement original in
  match result with
  | EBinop (Add, inner, EConst (CInt32 1l)) -> (
      match inner with
      | EBinop (Mul, EArrayRead ("input", _), EConst (CInt32 2l)) ->
          Printf.printf "test_subst_array_read: PASSED\n"
      | _ -> failwith "test_subst_array_read: wrong inner expression")
  | _ -> failwith "test_subst_array_read: wrong result structure"

let () =
  Printf.printf "=== Fusion Unit Tests ===\n" ;
  test_expr_equal () ;
  test_subst_array_read () ;
  test_analyze_one_to_one () ;
  test_analyze_with_barrier () ;
  test_can_fuse_compatible () ;
  test_can_fuse_with_barrier () ;
  test_fuse_simple () ;
  test_fuse_pipeline () ;
  Printf.printf "=== All tests passed! ===\n"
