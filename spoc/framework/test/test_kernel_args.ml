(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kernel_args Tests - Verify Indexed Argument Container
 ******************************************************************************)

module KA = Spoc_framework.Kernel_args

(** Setting indices out of call order must not affect the extracted order:
    values always come back sorted by index. *)
let test_out_of_order_set () =
  let t = KA.create () in
  KA.set t 2 "c" ;
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  match KA.validate_and_extract t ~expected_count:3 with
  | Ok arr ->
      Alcotest.(check (array string)) "sorted by index" [|"a"; "b"; "c"|] arr
  | Error msg -> Alcotest.fail msg

(** Duplicate idx: the later call wins. *)
let test_duplicate_idx_last_wins () =
  let t = KA.create () in
  KA.set t 1 "first" ;
  KA.set t 1 "second" ;
  KA.set t 0 "zero" ;
  match KA.validate_and_extract t ~expected_count:2 with
  | Ok arr ->
      Alcotest.(check (array string)) "last write wins" [|"zero"; "second"|] arr
  | Error msg -> Alcotest.fail msg

(** A gap in the indices is rejected and named in the error. *)
let test_gap_detection () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 2 "c" ;
  match KA.validate_and_extract t ~expected_count:3 with
  | Ok _ -> Alcotest.fail "expected gap at index 1 to be rejected"
  | Error msg ->
      Alcotest.(check bool)
        "error names missing index 1"
        true
        (Str.string_match (Str.regexp ".*missing indices: \\[1\\].*") msg 0)

(** Setting more indices than expected_count is rejected as unexpected/extra. *)
let test_count_mismatch_extra () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  KA.set t 2 "c" ;
  match KA.validate_and_extract t ~expected_count:2 with
  | Ok _ -> Alcotest.fail "expected extra index 2 to be rejected"
  | Error msg ->
      Alcotest.(check bool)
        "error names unexpected index 2"
        true
        (Str.string_match (Str.regexp ".*2.*") msg 0)

(** An index beyond expected_count (with no gap below it) is also reported as
    unexpected. *)
let test_unexpected_index_beyond_range () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  KA.set t 2 "c" ;
  KA.set t 5 "f" ;
  match KA.validate_and_extract t ~expected_count:3 with
  | Ok _ -> Alcotest.fail "expected unexpected index 5 to be rejected"
  | Error msg ->
      Alcotest.(check bool)
        "error names unexpected index 5"
        true
        (Str.string_match (Str.regexp ".*5.*") msg 0)

(** A negative [expected_count] must return a structured [Error], not raise
    [Invalid_argument] out of the underlying [Array.init]. *)
let test_negative_expected_count () =
  let t = KA.create () in
  match KA.validate_and_extract t ~expected_count:(-1) with
  | Ok _ -> Alcotest.fail "expected negative expected_count to be rejected"
  | Error msg ->
      Alcotest.(check bool)
        "error names invalid expected_count"
        true
        (Str.string_match (Str.regexp ".*invalid expected_count.*") msg 0)

let test_count () =
  let t = KA.create () in
  Alcotest.(check int) "empty" 0 (KA.count t) ;
  KA.set t 0 "a" ;
  KA.set t 3 "d" ;
  Alcotest.(check int) "two slots" 2 (KA.count t)

let test_to_sorted_list () =
  let t = KA.create () in
  KA.set t 3 "d" ;
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  Alcotest.(check (list (pair int string)))
    "sorted by index, no contiguity required"
    [(0, "a"); (1, "b"); (3, "d")]
    (KA.to_sorted_list t)

let expect_error t ~expected_count msg =
  match KA.validate_and_extract t ~expected_count with
  | Ok _ -> Alcotest.fail "expected a validation error"
  | Error got -> Alcotest.(check string) "error text" msg got

(** The exact texts backends wrap into their structured errors. *)
let test_error_texts () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 2 "c" ;
  expect_error
    t
    ~expected_count:4
    "missing indices: [1, 3]; expected 4 args, got 2" ;
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t 1 "b" ;
  KA.set t 3 "d" ;
  KA.set t 7 "h" ;
  expect_error
    t
    ~expected_count:2
    "unexpected indices: [3, 7]; expected contiguous 0..1" ;
  let t = KA.create () in
  KA.set t 1 "b" ;
  KA.set t 4 "e" ;
  expect_error
    t
    ~expected_count:3
    "missing indices: [0, 2]; unexpected index: [4]; expected 3 args, got 2"

(** A negative index is stored, counted and reported as unexpected. *)
let test_negative_index () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t (-1) "m" ;
  Alcotest.(check int) "counted" 2 (KA.count t) ;
  Alcotest.(check (list (pair int string)))
    "listed first"
    [(-1, "m"); (0, "a")]
    (KA.to_sorted_list t) ;
  expect_error
    t
    ~expected_count:1
    "unexpected index: [-1]; expected contiguous 0..0"

(** Sparse, large and repeated indices: count is the number of distinct indices,
    whatever the order and the gaps. *)
let test_sparse_indices () =
  let t = KA.create () in
  KA.set t 1000 "z" ;
  KA.set t 0 "a" ;
  KA.set t 1000 "y" ;
  KA.set t 17 "r" ;
  Alcotest.(check int) "distinct indices" 3 (KA.count t) ;
  Alcotest.(check (list (pair int string)))
    "sorted, last write wins"
    [(0, "a"); (17, "r"); (1000, "y")]
    (KA.to_sorted_list t)

(** Indices far past any real argument count are kept and reported, without
    allocating storage of their size. *)
let test_huge_indices () =
  let t = KA.create () in
  KA.set t 0 "a" ;
  KA.set t max_int "x" ;
  KA.set t 1_000_000 "m" ;
  KA.set t 1_000_000 "n" ;
  Alcotest.(check int) "distinct indices" 3 (KA.count t) ;
  Alcotest.(check (list (pair int string)))
    "sorted"
    [(0, "a"); (1_000_000, "n"); (max_int, "x")]
    (KA.to_sorted_list t) ;
  expect_error
    t
    ~expected_count:2
    (Printf.sprintf
       "missing indices: [1]; unexpected indices: [1000000, %d]; expected 2 \
        args, got 3"
       max_int)

(** Many arguments set in reverse order extract in index order. *)
let test_many_args () =
  let t = KA.create () in
  for i = 63 downto 0 do
    KA.set t i i
  done ;
  match KA.validate_and_extract t ~expected_count:64 with
  | Ok arr -> Alcotest.(check (array int)) "0..63" (Array.init 64 Fun.id) arr
  | Error msg -> Alcotest.fail msg

(** An empty store validates against zero arguments. *)
let test_empty () =
  match KA.validate_and_extract (KA.create ()) ~expected_count:0 with
  | Ok arr -> Alcotest.(check int) "no values" 0 (Array.length arr)
  | Error msg -> Alcotest.fail msg

(** Test suite *)
let () =
  Alcotest.run
    "Kernel_args"
    [
      ( "ordering",
        [
          Alcotest.test_case "out of order set" `Quick test_out_of_order_set;
          Alcotest.test_case "to_sorted_list" `Quick test_to_sorted_list;
          Alcotest.test_case "count" `Quick test_count;
          Alcotest.test_case "sparse indices" `Quick test_sparse_indices;
          Alcotest.test_case "many args" `Quick test_many_args;
          Alcotest.test_case "huge indices" `Quick test_huge_indices;
        ] );
      ( "last-set-wins",
        [
          Alcotest.test_case
            "duplicate idx last wins"
            `Quick
            test_duplicate_idx_last_wins;
        ] );
      ( "validation",
        [
          Alcotest.test_case "gap detection" `Quick test_gap_detection;
          Alcotest.test_case
            "count mismatch extra"
            `Quick
            test_count_mismatch_extra;
          Alcotest.test_case
            "unexpected index beyond range"
            `Quick
            test_unexpected_index_beyond_range;
          Alcotest.test_case
            "negative expected_count"
            `Quick
            test_negative_expected_count;
          Alcotest.test_case "error texts" `Quick test_error_texts;
          Alcotest.test_case "negative index" `Quick test_negative_index;
          Alcotest.test_case "empty" `Quick test_empty;
        ] );
    ]
