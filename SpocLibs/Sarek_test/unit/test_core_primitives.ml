(** Unit tests for Sarek_core_primitives *)

open Alcotest
open Sarek_ppx_lib.Sarek_core_primitives

(* === Lookup tests === *)

let test_find_thread_idx () =
  let p = find "thread_idx_x" in
  check bool "thread_idx_x exists" true (Option.is_some p)

let test_find_nonexistent () =
  let p = find "nonexistent_primitive" in
  check bool "nonexistent returns None" true (Option.is_none p)

let test_is_core_primitive () =
  check bool "thread_idx_x is core" true (is_core_primitive "thread_idx_x");
  check bool "block_barrier is core" true (is_core_primitive "block_barrier");
  check bool "sin is core" true (is_core_primitive "sin");
  check bool "random is not core" false (is_core_primitive "random")

(* === Variance tests === *)

let test_thread_varying () =
  check bool "thread_idx_x is thread varying" true (is_thread_varying "thread_idx_x");
  check bool "thread_idx_y is thread varying" true (is_thread_varying "thread_idx_y");
  check bool "global_thread_id is thread varying" true (is_thread_varying "global_thread_id");
  check bool "block_dim_x is not thread varying" false (is_thread_varying "block_dim_x");
  check bool "block_idx_x is not thread varying" false (is_thread_varying "block_idx_x")

let test_variance_of () =
  check (option (of_pp pp_variance)) "thread_idx_x variance"
    (Some ThreadVarying) (variance_of "thread_idx_x");
  check (option (of_pp pp_variance)) "block_idx_x variance"
    (Some BlockVarying) (variance_of "block_idx_x");
  check (option (of_pp pp_variance)) "block_dim_x variance"
    (Some Uniform) (variance_of "block_dim_x");
  check (option (of_pp pp_variance)) "nonexistent variance"
    None (variance_of "nonexistent")

let test_join_variance () =
  check (of_pp pp_variance) "Uniform + Uniform" Uniform (join_variance Uniform Uniform);
  check (of_pp pp_variance) "Uniform + ThreadVarying" ThreadVarying (join_variance Uniform ThreadVarying);
  check (of_pp pp_variance) "ThreadVarying + Uniform" ThreadVarying (join_variance ThreadVarying Uniform);
  check (of_pp pp_variance) "BlockVarying + BlockVarying" BlockVarying (join_variance BlockVarying BlockVarying);
  check (of_pp pp_variance) "BlockVarying + ThreadVarying" ThreadVarying (join_variance BlockVarying ThreadVarying);
  check (of_pp pp_variance) "ThreadVarying + BlockVarying" ThreadVarying (join_variance ThreadVarying BlockVarying)

let test_variance_leq () =
  check bool "Uniform <= Uniform" true (variance_leq Uniform Uniform);
  check bool "Uniform <= BlockVarying" true (variance_leq Uniform BlockVarying);
  check bool "Uniform <= ThreadVarying" true (variance_leq Uniform ThreadVarying);
  check bool "BlockVarying <= BlockVarying" true (variance_leq BlockVarying BlockVarying);
  check bool "BlockVarying <= ThreadVarying" true (variance_leq BlockVarying ThreadVarying);
  check bool "ThreadVarying <= ThreadVarying" true (variance_leq ThreadVarying ThreadVarying);
  check bool "ThreadVarying <= Uniform" false (variance_leq ThreadVarying Uniform);
  check bool "BlockVarying <= Uniform" false (variance_leq BlockVarying Uniform)

(* === Convergence tests === *)

let test_convergence_point () =
  check bool "block_barrier is convergence point" true (is_convergence_point "block_barrier");
  check bool "sin is not convergence point" false (is_convergence_point "sin");
  check bool "thread_idx_x is not convergence point" false (is_convergence_point "thread_idx_x")

(* === Purity tests === *)

let test_pure () =
  check bool "sin is pure" true (is_pure "sin");
  check bool "cos is pure" true (is_pure "cos");
  check bool "sqrt is pure" true (is_pure "sqrt");
  check bool "thread_idx_x is pure" true (is_pure "thread_idx_x");
  check bool "block_barrier is not pure" false (is_pure "block_barrier")

(* === Category tests === *)

let test_category () =
  let thread_ids = primitives_in_category "thread_id" in
  check bool "thread_id category has primitives" true (List.length thread_ids >= 4);
  let math = primitives_in_category "math_f32" in
  check bool "math_f32 category has primitives" true (List.length math >= 10);
  let sync = primitives_in_category "sync" in
  check bool "sync category has barrier" true (List.length sync >= 1)

(* === Test suites === *)

let lookup_tests = [
  "find thread_idx", `Quick, test_find_thread_idx;
  "find nonexistent", `Quick, test_find_nonexistent;
  "is_core_primitive", `Quick, test_is_core_primitive;
]

let variance_tests = [
  "thread varying", `Quick, test_thread_varying;
  "variance_of", `Quick, test_variance_of;
  "join variance", `Quick, test_join_variance;
  "variance leq", `Quick, test_variance_leq;
]

let convergence_tests = [
  "convergence point", `Quick, test_convergence_point;
]

let purity_tests = [
  "pure", `Quick, test_pure;
]

let category_tests = [
  "category", `Quick, test_category;
]

let () =
  run "Sarek_core_primitives" [
    "lookup", lookup_tests;
    "variance", variance_tests;
    "convergence", convergence_tests;
    "purity", purity_tests;
    "category", category_tests;
  ]
