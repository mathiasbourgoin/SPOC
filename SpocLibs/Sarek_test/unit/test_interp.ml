(** Unit tests for Sarek_interp - CPU interpreter *)

open Sarek.Kirc_Ast
open Sarek.Sarek_interp

(** Test: simple thread index intrinsic *)
let test_thread_idx () =
  (* Kernel: output[thread_idx_x] <- thread_idx_x *)
  let body =
    SetV
      ( Acc (Id "output", IntrinsicRef (["Gpu"], "thread_idx_x")),
        IntrinsicRef (["Gpu"], "thread_idx_x") )
  in
  let output = int32_array 4 in
  run_body body ~block:(4, 1, 1) ~grid:(1, 1, 1) [("output", output)] ;
  let results = get_int32s output in
  assert (results.(0) = 0l) ;
  assert (results.(1) = 1l) ;
  assert (results.(2) = 2l) ;
  assert (results.(3) = 3l) ;
  Printf.printf "test_thread_idx: PASSED\n"

(** Test: block index *)
let test_block_idx () =
  (* Kernel: output[global_idx] <- block_idx_x *)
  let body =
    SetV
      ( Acc (Id "output", IntrinsicRef (["Gpu"], "global_idx")),
        IntrinsicRef (["Gpu"], "block_idx_x") )
  in
  let output = int32_array 8 in
  run_body body ~block:(4, 1, 1) ~grid:(2, 1, 1) [("output", output)] ;
  let results = get_int32s output in
  (* First 4 threads in block 0, next 4 in block 1 *)
  assert (results.(0) = 0l) ;
  assert (results.(3) = 0l) ;
  assert (results.(4) = 1l) ;
  assert (results.(7) = 1l) ;
  Printf.printf "test_block_idx: PASSED\n"

(** Test: arithmetic operations *)
let test_arithmetic () =
  (* Kernel: output[i] <- input[i] * 2 + 1 *)
  let idx = IntrinsicRef (["Gpu"], "thread_idx_x") in
  let body =
    SetV
      (Acc (Id "output", idx), Plus (Mul (Acc (Id "input", idx), Int 2), Int 1))
  in
  let input = int32_array 4 in
  let output = int32_array 4 in
  set_int32s input [|1l; 2l; 3l; 4l|] ;
  run_body
    body
    ~block:(4, 1, 1)
    ~grid:(1, 1, 1)
    [("input", input); ("output", output)] ;
  let results = get_int32s output in
  assert (results.(0) = 3l) ;
  (* 1*2+1 *)
  assert (results.(1) = 5l) ;
  (* 2*2+1 *)
  assert (results.(2) = 7l) ;
  (* 3*2+1 *)
  assert (results.(3) = 9l) ;
  (* 4*2+1 *)
  Printf.printf "test_arithmetic: PASSED\n"

(** Test: float arithmetic *)
let test_float_arithmetic () =
  (* Kernel: output[i] <- input[i] * 2.0 + 1.0 *)
  let idx = IntrinsicRef (["Gpu"], "thread_idx_x") in
  let body =
    SetV
      ( Acc (Id "output", idx),
        Plusf (Mulf (Acc (Id "input", idx), Float 2.0), Float 1.0) )
  in
  let input = float32_array 4 in
  let output = float32_array 4 in
  set_float32s input [|1.0; 2.0; 3.0; 4.0|] ;
  run_body
    body
    ~block:(4, 1, 1)
    ~grid:(1, 1, 1)
    [("input", input); ("output", output)] ;
  let results = get_float32s output in
  assert (abs_float (results.(0) -. 3.0) < 0.001) ;
  assert (abs_float (results.(1) -. 5.0) < 0.001) ;
  assert (abs_float (results.(2) -. 7.0) < 0.001) ;
  assert (abs_float (results.(3) -. 9.0) < 0.001) ;
  Printf.printf "test_float_arithmetic: PASSED\n"

(** Test: conditional *)
let test_conditional () =
  (* Kernel: output[i] <- if input[i] > 5 then 1 else 0 *)
  let idx = IntrinsicRef (["Gpu"], "thread_idx_x") in
  let body =
    SetV
      ( Acc (Id "output", idx),
        Ife (GtBool (Acc (Id "input", idx), Int 5), Int 1, Int 0) )
  in
  let input = int32_array 4 in
  let output = int32_array 4 in
  set_int32s input [|3l; 5l; 7l; 10l|] ;
  run_body
    body
    ~block:(4, 1, 1)
    ~grid:(1, 1, 1)
    [("input", input); ("output", output)] ;
  let results = get_int32s output in
  assert (results.(0) = 0l) ;
  (* 3 > 5 = false *)
  assert (results.(1) = 0l) ;
  (* 5 > 5 = false *)
  assert (results.(2) = 1l) ;
  (* 7 > 5 = true *)
  assert (results.(3) = 1l) ;
  (* 10 > 5 = true *)
  Printf.printf "test_conditional: PASSED\n"

(** Test: loop *)
let test_loop () =
  (* Kernel: output[i] <- sum(0..i) using a loop *)
  let idx = IntrinsicRef (["Gpu"], "thread_idx_x") in
  let loop_var = IntVar (100, "j", true) in
  let sum_var = IntVar (101, "sum", true) in
  let body =
    Seq
      ( Decl sum_var,
        Seq
          ( Set (sum_var, Int 0),
            Seq
              ( DoLoop
                  ( loop_var,
                    Int 0,
                    idx,
                    Set (sum_var, Plus (sum_var, IntId ("j", 100))) ),
                SetV (Acc (Id "output", idx), sum_var) ) ) )
  in
  let output = int32_array 5 in
  run_body body ~block:(5, 1, 1) ~grid:(1, 1, 1) [("output", output)] ;
  let results = get_int32s output in
  assert (results.(0) = 0l) ;
  (* sum 0..0 = 0 *)
  assert (results.(1) = 1l) ;
  (* sum 0..1 = 0+1 = 1 *)
  assert (results.(2) = 3l) ;
  (* sum 0..2 = 0+1+2 = 3 *)
  assert (results.(3) = 6l) ;
  (* sum 0..3 = 0+1+2+3 = 6 *)
  assert (results.(4) = 10l) ;
  (* sum 0..4 = 0+1+2+3+4 = 10 *)
  Printf.printf "test_loop: PASSED\n"

(** Test: local variable *)
let test_local_var () =
  (* Kernel: temp = input[i] * 2; output[i] = temp + 1 *)
  let idx = IntrinsicRef (["Gpu"], "thread_idx_x") in
  let temp_var = IntVar (100, "temp", true) in
  let body =
    SetLocalVar
      ( temp_var,
        Mul (Acc (Id "input", idx), Int 2),
        SetV (Acc (Id "output", idx), Plus (temp_var, Int 1)) )
  in
  let input = int32_array 4 in
  let output = int32_array 4 in
  set_int32s input [|1l; 2l; 3l; 4l|] ;
  run_body
    body
    ~block:(4, 1, 1)
    ~grid:(1, 1, 1)
    [("input", input); ("output", output)] ;
  let results = get_int32s output in
  assert (results.(0) = 3l) ;
  assert (results.(1) = 5l) ;
  assert (results.(2) = 7l) ;
  assert (results.(3) = 9l) ;
  Printf.printf "test_local_var: PASSED\n"

(** Test: global_idx intrinsic *)
let test_global_idx () =
  (* Kernel: output[global_idx] <- global_idx *)
  let gidx = IntrinsicRef (["Gpu"], "global_idx") in
  let body = SetV (Acc (Id "output", gidx), gidx) in
  let output = int32_array 8 in
  run_body body ~block:(4, 1, 1) ~grid:(2, 1, 1) [("output", output)] ;
  let results = get_int32s output in
  for i = 0 to 7 do
    assert (results.(i) = Int32.of_int i)
  done ;
  Printf.printf "test_global_idx: PASSED\n"

(** Test: Float32 intrinsics *)
let test_float32_intrinsics () =
  (* Kernel: output[i] <- sqrt(input[i]) *)
  let idx = IntrinsicRef (["Gpu"], "thread_idx_x") in
  let body =
    SetV
      ( Acc (Id "output", idx),
        App (IntrinsicRef (["Float32"], "sqrt"), [|Acc (Id "input", idx)|]) )
  in
  let input = float32_array 4 in
  let output = float32_array 4 in
  set_float32s input [|1.0; 4.0; 9.0; 16.0|] ;
  run_body
    body
    ~block:(4, 1, 1)
    ~grid:(1, 1, 1)
    [("input", input); ("output", output)] ;
  let results = get_float32s output in
  assert (abs_float (results.(0) -. 1.0) < 0.001) ;
  assert (abs_float (results.(1) -. 2.0) < 0.001) ;
  assert (abs_float (results.(2) -. 3.0) < 0.001) ;
  assert (abs_float (results.(3) -. 4.0) < 0.001) ;
  Printf.printf "test_float32_intrinsics: PASSED\n"

(** Test: multi-block execution *)
let test_multi_block () =
  (* Kernel: output[global_idx] <- block_idx_x * block_dim_x + thread_idx_x *)
  let body =
    SetV
      ( Acc (Id "output", IntrinsicRef (["Gpu"], "global_idx")),
        Plus
          ( Mul
              ( IntrinsicRef (["Gpu"], "block_idx_x"),
                IntrinsicRef (["Gpu"], "block_dim_x") ),
            IntrinsicRef (["Gpu"], "thread_idx_x") ) )
  in
  let output = int32_array 16 in
  run_body body ~block:(4, 1, 1) ~grid:(4, 1, 1) [("output", output)] ;
  let results = get_int32s output in
  for i = 0 to 15 do
    assert (results.(i) = Int32.of_int i)
  done ;
  Printf.printf "test_multi_block: PASSED\n"

let () =
  Printf.printf "=== Sarek Interpreter Unit Tests ===\n" ;
  test_thread_idx () ;
  test_block_idx () ;
  test_arithmetic () ;
  test_float_arithmetic () ;
  test_conditional () ;
  test_loop () ;
  test_local_var () ;
  test_global_idx () ;
  test_float32_intrinsics () ;
  test_multi_block () ;
  Printf.printf "=== All interpreter tests passed! ===\n"
