(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Compile_cache Tests - Verify Shared Cache-Key Builder
 *
 * These are key-level (non-hardware) tests: they exercise the pure key
 * builder, not an actual backend compile pipeline. Executable, hardware-
 * verified end-to-end coverage for the Vulkan/OpenCL in-memory pipeline
 * caches lives in sarek-vulkan/test and sarek-opencl/test respectively.
 ******************************************************************************)

module CC = Spoc_framework.Compile_cache

let test_same_inputs_same_key () =
  let key1 = CC.make_key ~device:"0" ~name:"kernel_a" ~source:"code" () in
  let key2 = CC.make_key ~device:"0" ~name:"kernel_a" ~source:"code" () in
  Alcotest.(check string) "identical inputs produce identical keys" key1 key2

(** Non-vacuous regression test: this is precisely the CUDA/Vulkan bug fixed in
    this change. Two kernels compiled from the same source string, on the same
    device, must never share a cache key - before the fix the key was built from
    device + source digest only, so [kernel_b] would silently resolve to
    [kernel_a]'s cached handle. *)
let test_same_source_different_name_different_key () =
  let shared_source = "source defining both kernel_a and kernel_b" in
  let key_a =
    CC.make_key ~device:"0" ~name:"kernel_a" ~source:shared_source ()
  in
  let key_b =
    CC.make_key ~device:"0" ~name:"kernel_b" ~source:shared_source ()
  in
  Alcotest.(check bool)
    "different kernel names in the same source must not collide"
    true
    (key_a <> key_b)

let test_different_device_different_key () =
  let key0 = CC.make_key ~device:"0" ~name:"k" ~source:"code" () in
  let key1 = CC.make_key ~device:"1" ~name:"k" ~source:"code" () in
  Alcotest.(check bool) "different devices must not collide" true (key0 <> key1)

let test_different_source_different_key () =
  let key1 = CC.make_key ~device:"0" ~name:"k" ~source:"code1" () in
  let key2 = CC.make_key ~device:"0" ~name:"k" ~source:"code2" () in
  Alcotest.(check bool) "different sources must not collide" true (key1 <> key2)

let test_options_are_canonicalized () =
  let key1 =
    CC.make_key
      ~device:"0"
      ~name:"k"
      ~source:"code"
      ~options:[("b", "2"); ("a", "1")]
      ()
  in
  let key2 =
    CC.make_key
      ~device:"0"
      ~name:"k"
      ~source:"code"
      ~options:[("a", "1"); ("b", "2")]
      ()
  in
  Alcotest.(check string) "option order does not affect the key" key1 key2

let test_different_options_different_key () =
  let key1 =
    CC.make_key ~device:"0" ~name:"k" ~source:"code" ~options:[("opt", "1")] ()
  in
  let key2 =
    CC.make_key ~device:"0" ~name:"k" ~source:"code" ~options:[("opt", "2")] ()
  in
  Alcotest.(check bool)
    "different option values must not collide"
    true
    (key1 <> key2)

(* Key memo (cheap-repeated-launches FR-007). The oracle recomputes the key from
   the documented format, independently of the memo, so a memo that served a
   stale or crossed entry cannot agree with it by construction. *)
let oracle ~device ~name ~source ~options =
  let hex s = Digest.to_hex (Digest.string s) in
  let opts =
    options
    |> List.sort (fun (a, _) (b, _) -> String.compare a b)
    |> List.map (fun (k, v) -> hex k ^ "=" ^ hex v)
    |> String.concat ","
  in
  Printf.sprintf "%s:%s:%s:%s" (hex device) (hex name) (hex source) (hex opts)

let test_memo_keys_match_oracle () =
  let source = String.concat "\n" (List.init 200 (Printf.sprintf "line %d")) in
  let copy = Bytes.to_string (Bytes.of_string source) in
  Alcotest.(check bool) "the copy is a distinct string" false (copy == source) ;
  let expect = oracle ~device:"0" ~name:"k" ~source ~options:[] in
  for _ = 1 to 3 do
    Alcotest.(check string)
      "same physical source, repeated"
      expect
      (CC.make_key ~device:"0" ~name:"k" ~source ())
  done ;
  Alcotest.(check string)
    "fresh copy of the source"
    expect
    (CC.make_key ~device:"0" ~name:"k" ~source:copy ()) ;
  Alcotest.(check string)
    "empty source"
    (oracle ~device:"0" ~name:"k" ~source:"" ~options:[])
    (CC.make_key ~device:"0" ~name:"k" ~source:"" ())

(* One physical source keyed for several (device, name, options) triples must
   give each triple its own key, in any order of first use. *)
let test_memo_distinguishes_triples () =
  let source = "extern \"C\" __global__ void a() {} void b() {}" in
  let triples =
    [
      ("0", "a", []);
      ("0", "b", []);
      ("1", "a", []);
      ("0", "a", [("O", "3")]);
      ("0", "a", [("O", "2")]);
    ]
  in
  for _ = 1 to 2 do
    List.iter
      (fun (device, name, options) ->
        Alcotest.(check string)
          (Printf.sprintf
             "device=%s name=%s opts=%d"
             device
             name
             (List.length options))
          (oracle ~device ~name ~source ~options)
          (CC.make_key ~device ~name ~source ~options ()))
      triples
  done

let () =
  Alcotest.run
    "Compile_cache"
    [
      ( "key_identity",
        [
          Alcotest.test_case
            "same inputs same key"
            `Quick
            test_same_inputs_same_key;
          Alcotest.test_case
            "same source, different kernel name -> different key"
            `Quick
            test_same_source_different_name_different_key;
          Alcotest.test_case
            "different device -> different key"
            `Quick
            test_different_device_different_key;
          Alcotest.test_case
            "different source -> different key"
            `Quick
            test_different_source_different_key;
        ] );
      ( "options",
        [
          Alcotest.test_case
            "option order canonicalized"
            `Quick
            test_options_are_canonicalized;
          Alcotest.test_case
            "different option values -> different key"
            `Quick
            test_different_options_different_key;
        ] );
      ( "key_memo",
        [
          Alcotest.test_case
            "memoised keys equal the documented format"
            `Quick
            test_memo_keys_match_oracle;
          Alcotest.test_case
            "one source, several triples"
            `Quick
            test_memo_distinguishes_triples;
        ] );
    ]
