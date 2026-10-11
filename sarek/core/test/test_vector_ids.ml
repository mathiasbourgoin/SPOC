(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Vector ids are buffer identity (launch windows, epilogue fusion, keep-alive
   ledger): creating vectors from several domains must never repeat an id. *)

open Spoc_core

let domains = 4

let per_domain = 50_000

let create_ids () =
  Array.init per_domain (fun _ -> Vector.id (Vector.create Vector.float32 1))

let () =
  let workers = List.init domains (fun _ -> Domain.spawn create_ids) in
  let all = List.concat_map (fun d -> Array.to_list (Domain.join d)) workers in
  let seen = Hashtbl.create (domains * per_domain) in
  let duplicates =
    List.fold_left
      (fun acc id ->
        if Hashtbl.mem seen id then acc + 1
        else begin
          Hashtbl.add seen id () ;
          acc
        end)
      0
      all
  in
  if duplicates <> 0 then begin
    Printf.eprintf
      "FAIL: %d duplicate vector ids out of %d\n"
      duplicates
      (List.length all) ;
    exit 1
  end ;
  Printf.printf
    "PASS: %d vector ids unique across %d domains\n"
    (List.length all)
    domains
