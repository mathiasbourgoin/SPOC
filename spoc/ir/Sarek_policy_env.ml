(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* INTERNAL: use [Sarek_policy.Env], which re-exports it. The pure text half
   of SAREK_OPT_DISABLE and SAREK_EXPLAIN: what the variables say, and how an
   explain line is written. Reading the environment and acting on it is
   Sarek_policy's. *)

open Sarek_policy_vocab

(* ---- SAREK_OPT_DISABLE ---------------------------------------------------- *)

let vocabulary =
  List.map (fun o -> (optimisation_name o, Disable o)) all_optimisations
  @ List.map (fun g -> (grant_name g, Revoke g)) all_grants

let valid_items = String.concat ", " (List.map fst vocabulary)

let items_of value =
  List.filter_map
    (fun raw ->
      match String.lowercase_ascii (String.trim raw) with
      | "" -> None
      | item -> Some item)
    (String.split_on_char ',' value)

(* The restrictions in order of first appearance, the unknown items likewise;
   a repeated item counts once. *)
let parse value =
  let add_unique x l = if List.mem x l then l else x :: l in
  let restrictions, unknown =
    List.fold_left
      (fun (known, unknown) item ->
        match List.assoc_opt item vocabulary with
        | Some r -> (add_unique r known, unknown)
        | None -> (known, add_unique item unknown))
      ([], [])
      (items_of value)
  in
  (List.rev restrictions, List.rev unknown)

let unknown_message item =
  Printf.sprintf
    "sarek: SAREK_OPT_DISABLE: unknown item %S (valid: %s); ignored"
    item
    valid_items

(* ---- SAREK_EXPLAIN -------------------------------------------------------- *)

let parse_explain = function
  | None | Some "" | Some "0" -> Explain_off
  | Some "1" -> Explain_stderr
  | Some other -> Explain_unknown other

let explain_unknown_message value =
  Printf.sprintf
    "sarek: SAREK_EXPLAIN=%S is not recognised (only 1 turns it on); explain \
     is off"
    value

(* One line per event: no newline survives, spaces in a name become
   underscores, and an epilogue pair ["p + c"] is written ["p+c"]. *)
let flatten s = String.map (function '\n' | '\r' -> ' ' | c -> c) s

let replace_all ~sub ~by s =
  let n = String.length sub in
  let b = Buffer.create (String.length s) in
  let rec go i =
    if i >= String.length s then ()
    else if i + n <= String.length s && String.sub s i n = sub then (
      Buffer.add_string b by ;
      go (i + n))
    else (
      Buffer.add_char b s.[i] ;
      go (i + 1))
  in
  go 0 ;
  Buffer.contents b

let name_field s =
  String.map (function ' ' -> '_' | c -> c) (replace_all ~sub:" + " ~by:"+" s)

let quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"' ;
  String.iter
    (function
      | ('"' | '\\') as c ->
          Buffer.add_char b '\\' ;
          Buffer.add_char b c
      | '\n' | '\r' -> Buffer.add_char b ' '
      | c -> Buffer.add_char b c)
    s ;
  Buffer.add_char b '"' ;
  Buffer.contents b

let decision_text = function
  | Applied (Proof text) -> "applied proof=" ^ flatten text
  | Applied (Granted {grants; reason; lowering}) ->
      Printf.sprintf
        "applied granted=%s reason=%s%s"
        (String.concat "," (List.map grant_name grants))
        (quote reason)
        (match lowering with None -> "" | Some l -> " lowering=" ^ quote l)
  | Applied Exact_by_construction -> "applied exact"
  | Refused why -> "refused " ^ flatten why

let cause_name = function
  | Generated_name -> "grant-on-generated-name"
  | Revoked -> "grant-revoked"

let explain_line = function
  | Decision {kernel; target; optimisation; decision} ->
      Printf.sprintf
        "sarek-explain %s %s %s %s"
        (name_field kernel)
        (name_field target)
        (optimisation_name optimisation)
        (decision_text decision)
  | Gate {kernel; target; decision} ->
      Printf.sprintf
        "sarek-explain %s %s coopmat-gate %s"
        (name_field kernel)
        (name_field target)
        (decision_text decision)
  | Ignored_grant {kernel; grant; cause} ->
      Printf.sprintf
        "sarek-explain %s - policy refused %s %s"
        (name_field kernel)
        (cause_name cause)
        (grant_name grant)
