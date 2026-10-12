(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Structural check, runnable without a Mac: the Metal binding of
 * [waitUntilCompleted] releases the OCaml runtime lock (R-24, as #471 did for
 * HIP/CUDA/OpenCL/Vulkan), and the plain non-blocking [objc_msgSend] stub does
 * not. libobjc does not exist off macOS, so the binding cannot be built here;
 * the check reads Metal_bindings.ml. The behavioural check (another domain
 * keeps collecting during a long wait) needs a Metal device: alias e2e-metal.
 ******************************************************************************)

let read file = In_channel.with_open_bin file In_channel.input_all

let bindings = read "../Metal_bindings.ml"

let api = read "../Metal_api.ml"

(* Text of the top-level [let name ...] definition: up to the next line that
   starts a new top-level item. *)
let definition ?(source = bindings) name =
  let start =
    try
      Str.search_forward
        (Str.regexp ("^let " ^ Str.quote name ^ "\\b"))
        source
        0
    with Not_found -> Alcotest.failf "no top-level definition of %s" name
  in
  let stop =
    try
      Str.search_forward
        (Str.regexp "^\\(let \\|(\\*\\* \\|(\\* \\)")
        source
        (start + 1)
    with Not_found -> String.length source
  in
  String.sub source start (stop - start)

let contains ~sub s =
  try
    ignore (Str.search_forward (Str.regexp_string sub) s 0) ;
    true
  with Not_found -> false

let test_blocking_stub_releases_lock () =
  Alcotest.(check bool)
    "msg_void_blocking_lazy releases the runtime lock"
    true
    (contains
       ~sub:"~release_runtime_lock:true"
       (definition "msg_void_blocking_lazy"))

let test_plain_stub_keeps_lock () =
  Alcotest.(check bool)
    "msg_void_lazy keeps the runtime lock (short messages)"
    false
    (contains ~sub:"release_runtime_lock" (definition "msg_void_lazy"))

let releases ?source name =
  contains ~sub:"~release_runtime_lock:true" (definition ?source name)

(* [send_void_blocking] must force the blocking stub itself: a textual mention
   of the name elsewhere proves nothing. *)
let test_blocking_sender_forces_blocking_stub () =
  let body = definition "send_void_blocking" in
  Alcotest.(check bool)
    "send_void_blocking forces msg_void_blocking_lazy"
    true
    (contains ~sub:"Lazy.force msg_void_blocking_lazy" body
    && not (contains ~sub:"Lazy.force msg_void_lazy" body))

(* The other blocking calls that take only C memory (#471 parity): the two
   compiles and the host/device copy. *)
let test_compiles_release_lock () =
  Alcotest.(check bool)
    "newLibraryWithSource releases the lock"
    true
    (releases "mtl_device_new_library_with_source") ;
  Alcotest.(check bool)
    "newComputePipelineState releases the lock"
    true
    (releases "mtl_device_new_compute_pipeline_state")

let test_memcpy_releases_lock () =
  Alcotest.(check bool)
    "memcpy releases the lock"
    true
    (releases ~source:api "memcpy_lazy")

let test_wait_uses_blocking_stub () =
  let body = definition "mtl_command_buffer_wait_until_completed" in
  Alcotest.(check bool)
    "waitUntilCompleted goes through send_void_blocking"
    true
    (contains ~sub:"send_void_blocking" body)

let () =
  Alcotest.run
    "metal_wait_binding"
    [
      ( "runtime lock",
        [
          Alcotest.test_case
            "blocking stub"
            `Quick
            test_blocking_stub_releases_lock;
          Alcotest.test_case "plain stub" `Quick test_plain_stub_keeps_lock;
          Alcotest.test_case "wait" `Quick test_wait_uses_blocking_stub;
          Alcotest.test_case
            "blocking sender"
            `Quick
            test_blocking_sender_forces_blocking_stub;
          Alcotest.test_case "compiles" `Quick test_compiles_release_lock;
          Alcotest.test_case "memcpy" `Quick test_memcpy_releases_lock;
        ] );
    ]
