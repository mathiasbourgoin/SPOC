(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The libnvrtc names the CUDA backend dlopens, and the order it tries them.
    Pure: the opener is injected, so this needs no libnvrtc and no device. *)

open Sarek_cuda

let candidates = Cuda_nvrtc.nvrtc_candidates

let index_of name =
  let rec go i = function
    | [] -> Alcotest.failf "%s is not a candidate" name
    | x :: rest -> if x = name then i else go (i + 1) rest
  in
  go 0 candidates

let test_cuda13_listed () =
  Alcotest.(check bool)
    "libnvrtc.so.13 is a candidate (CUDA 13 soname)"
    true
    (List.mem "libnvrtc.so.13" candidates)

let test_order () =
  Alcotest.(check string)
    "the unversioned development symlink comes first"
    "libnvrtc.so"
    (List.hd candidates) ;
  let so13 = index_of "libnvrtc.so.13"
  and so12 = index_of "libnvrtc.so.12"
  and so11 = index_of "libnvrtc.so.11" in
  Alcotest.(check bool) "so.13 before so.12" true (so13 < so12) ;
  Alcotest.(check bool) "so.12 before so.11" true (so12 < so11) ;
  Alcotest.(check bool)
    "CUDA 13 DLL before CUDA 12 DLL"
    true
    (index_of "nvrtc64_130_0.dll" < index_of "nvrtc64_120_0.dll")

let test_no_duplicates () =
  Alcotest.(check int)
    "every candidate appears once"
    (List.length candidates)
    (List.length (List.sort_uniq compare candidates))

(* An opener that only "finds" the names in [present], recording each try. *)
let fake_dlopen present tried name =
  tried := name :: !tried ;
  if List.mem name present then name else failwith ("cannot open " ^ name)

let search present =
  let tried = ref [] in
  let found =
    Cuda_nvrtc.first_loadable ~dlopen:(fake_dlopen present tried) candidates
  in
  (found, List.rev !tried)

let test_cuda13_only_host () =
  let found, tried = search ["libnvrtc.so.13"] in
  Alcotest.(check (option string))
    "a host with only the CUDA 13 soname loads it"
    (Some "libnvrtc.so.13")
    found ;
  Alcotest.(check (list string))
    "tries the unversioned name, then stops at so.13"
    ["libnvrtc.so"; "libnvrtc.so.13"]
    tried

let test_newest_wins () =
  let found, _ =
    search ["libnvrtc.so.11"; "libnvrtc.so.12"; "libnvrtc.so.13"]
  in
  Alcotest.(check (option string))
    "with three sonames present, the newest is taken"
    (Some "libnvrtc.so.13")
    found

let test_none () =
  let found, tried = search [] in
  Alcotest.(check (option string)) "nothing opens -> None" None found ;
  Alcotest.(check (list string)) "every candidate was tried" candidates tried

let () =
  Alcotest.run
    "cuda_nvrtc_candidates"
    [
      ( "candidates",
        [
          Alcotest.test_case "CUDA 13 soname listed" `Quick test_cuda13_listed;
          Alcotest.test_case "newest first" `Quick test_order;
          Alcotest.test_case "no duplicates" `Quick test_no_duplicates;
        ] );
      ( "search",
        [
          Alcotest.test_case "CUDA 13-only host" `Quick test_cuda13_only_host;
          Alcotest.test_case "newest wins" `Quick test_newest_wins;
          Alcotest.test_case "none found" `Quick test_none;
        ] );
    ]
