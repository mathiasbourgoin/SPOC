(** Unit tests for Kirc_kernel - kernel creation and management *)

open Sarek.Kirc_kernel
open Spoc_framework
open Alcotest

(** {1 Tests for native kernel construction} *)

let test_make_native () =
  (* Create a native-only kernel *)
  let native = fun ~block:_ ~grid:_ _args -> () in
  let k =
    make_native ~name:"native_kernel" ~native_fn:native ~param_types:[] ()
  in
  check string "kernel name" "native_kernel" (name k) ;
  check bool "has native fn" true (has_native k)

let test_native_with_params () =
  let native = fun ~block:_ ~grid:_ _args -> () in
  let types = [Sarek.Sarek_ir.TInt32; Sarek.Sarek_ir.TFloat32] in
  let k = make_native ~name:"typed" ~native_fn:native ~param_types:types () in
  check int "param count" 2 (List.length (param_types k))

(** {1 Tests for extension checking} *)

let test_no_extensions_native () =
  let native = fun ~block:_ ~grid:_ _args -> () in
  let k = make_native ~name:"basic" ~native_fn:native ~param_types:[] () in
  check bool "no fp64" false (requires_fp64 k) ;
  check bool "no fp32" false (requires_fp32 k)

(** {1 Tests for native function invocation} *)

let test_native_function_call () =
  let called = ref false in
  let native = fun ~block:_ ~grid:_ _args -> called := true in
  let k = make_native ~name:"test" ~native_fn:native ~param_types:[] () in
  let fn = native_fn k in
  fn ~block:(Framework_sig.dims_1d 1) ~grid:(Framework_sig.dims_1d 1) [||] ;
  check bool "native fn called" true !called

let test_native_function_args () =
  let arg_count = ref 0 in
  let native = fun ~block:_ ~grid:_ args -> arg_count := Array.length args in
  let k = make_native ~name:"test" ~native_fn:native ~param_types:[] () in
  let fn = native_fn k in
  let dummy_args = [|Framework_sig.EA_Int32 42l; Framework_sig.EA_Int32 99l|] in
  fn ~block:(Framework_sig.dims_1d 1) ~grid:(Framework_sig.dims_1d 1) dummy_args ;
  check int "args passed" 2 !arg_count

(** {1 Test suite} *)

let () =
  run
    "Kirc_kernel"
    [
      ( "native_construction",
        [
          test_case "make_native" `Quick test_make_native;
          test_case "native_with_params" `Quick test_native_with_params;
        ] );
      ( "extension_checking",
        [test_case "no_extensions" `Quick test_no_extensions_native] );
      ( "native_invocation",
        [
          test_case "call_native" `Quick test_native_function_call;
          test_case "pass_args" `Quick test_native_function_args;
        ] );
    ]
