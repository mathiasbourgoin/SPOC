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

let source =
  In_channel.with_open_bin "../Metal_bindings.ml" In_channel.input_all

(* Text of the top-level [let name ...] definition: up to the next line that
   starts a new top-level item. *)
let definition name =
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
        ] );
    ]
