(******************************************************************************
 * Test ktype features with PPX
 *
 * This tests whether the PPX can handle kernel definitions that
 * failed with camlp4 due to Ctypes issues.
 *
 * Note: The PPX uses standard OCaml syntax:
 * - Array access: v.(i) instead of v.[<i>]
 * - Kernel definition: [%kernel fun ...] instead of kern ...
 *
 * This file only verifies the PPX compiles - it does not execute the kernels.
 ******************************************************************************)

(* Test: Simple kernel with vector access *)
let test_simple () =
  let _kernel = [%kernel fun (v : float32 vector) (n : int32) ->
    let tid = thread_idx_x in
    if tid < n then
      v.(tid) <- 1.0
  ] in
  print_endline "Simple kernel: OK"

(* Test: Kernel with multiple vector parameters *)
let test_multi () =
  let _kernel = [%kernel fun (a : float32 vector) (b : float32 vector) (c : float32 vector) (n : int32) ->
    let tid = thread_idx_x in
    if tid < n then
      c.(tid) <- a.(tid) +. b.(tid)
  ] in
  print_endline "Multi-vector kernel: OK"

(* Test: Kernel with arithmetic operations *)
let test_arithmetic () =
  let _kernel = [%kernel fun (v : float32 vector) (scale : float32) (n : int32) ->
    let tid = thread_idx_x in
    if tid < n then begin
      let x = v.(tid) in
      let y = x *. scale +. 1.0 in
      v.(tid) <- y
    end
  ] in
  print_endline "Arithmetic kernel: OK"

(* Test: Kernel with for loop *)
let test_loop () =
  let _kernel = [%kernel fun (v : float32 vector) (n : int32) ->
    let tid = thread_idx_x in
    if tid = 0l then
      for i = 0l to n - 1l do
        v.(i) <- 0.0
      done
  ] in
  print_endline "For loop kernel: OK"

(* Test: Kernel with int32 vector *)
let test_int32 () =
  let _kernel = [%kernel fun (v : int32 vector) (n : int32) ->
    let tid = thread_idx_x in
    if tid < n then
      v.(tid) <- tid
  ] in
  print_endline "Int32 vector kernel: OK"

let () =
  test_simple ();
  test_multi ();
  test_arithmetic ();
  test_loop ();
  test_int32 ();
  print_endline "PPX compilation test passed - all kernels parsed successfully"
