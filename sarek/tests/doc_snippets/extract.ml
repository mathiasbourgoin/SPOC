(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Documentation snippet extractor.

   [extract.exe FILE.md] prints every ```ocaml block of FILE.md, in order, as
   one OCaml structure. A comment before each block gives its Markdown line
   (a [# line "file"] directive is not used: the Sarek PPX re-reads the
   source file named in the locations, which must stay the generated .ml).
   A block whose preceding line is [<!-- snippet: skip -->] is left out.

   [extract.exe --only NAME FILE.md] prints just the blocks whose preceding
   line is [<!-- snippet: NAME -->]; it builds a page's complete program as an
   executable.

   [extract.exe --check-listed DUNE_FILE FILE.md...] fails if a FILE.md that
   contains an ```ocaml block is not mentioned in DUNE_FILE, so a new page
   cannot escape the check. *)

let marker_prefix = "<!-- snippet: "

let marker_suffix = " -->"

(* The tag of a marker line [<!-- snippet: NAME -->], if it is one. *)
let marker_tag line =
  let line = String.trim line in
  let p = String.length marker_prefix and q = String.length marker_suffix in
  let n = String.length line in
  if
    n > p + q
    && String.starts_with ~prefix:marker_prefix line
    && String.ends_with ~suffix:marker_suffix line
  then Some (String.trim (String.sub line p (n - p - q)))
  else None

let is_ocaml_fence line =
  let line = String.trim line in
  line = "```ocaml" || String.starts_with ~prefix:"```ocaml " line

let is_fence_end line = String.trim line = "```"

let read_lines path =
  let ic = open_in path in
  let rec loop acc =
    match input_line ic with
    | line -> loop (line :: acc)
    | exception End_of_file ->
        close_in ic ;
        List.rev acc
  in
  loop []

type block = {start_line : int; body : string list; tag : string option}

(* Scan the lines of a file; [previous] is the line before the current one. *)
let blocks_of_lines lines =
  let rec scan acc ~line_no ~previous = function
    | [] -> List.rev acc
    | line :: rest when is_ocaml_fence line ->
        let body, after = take_body [] rest in
        let block =
          {start_line = line_no + 1; body; tag = marker_tag previous}
        in
        let consumed = List.length body + 2 in
        scan (block :: acc) ~line_no:(line_no + consumed) ~previous:"" after
    | line :: rest -> scan acc ~line_no:(line_no + 1) ~previous:line rest
  and take_body acc = function
    | [] -> (List.rev acc, [])
    | line :: rest when is_fence_end line -> (List.rev acc, rest)
    | line :: rest -> take_body (line :: acc) rest
  in
  scan [] ~line_no:1 ~previous:"" lines

let print_block path b =
  Printf.printf "(* %s:%d *)\n" path b.start_line ;
  List.iter print_endline b.body ;
  print_newline ()

let emit ?only path =
  let blocks = blocks_of_lines (read_lines path) in
  let wanted b =
    match only with
    | Some name -> b.tag = Some name
    | None -> b.tag <> Some "skip"
  in
  List.iter (fun b -> if wanted b then print_block path b) blocks ;
  let skipped =
    List.length (List.filter (fun b -> b.tag = Some "skip") blocks)
  in
  Printf.printf "(* %d block(s), %d skipped *)\n" (List.length blocks) skipped

let contains_substring ~sub text =
  let n = String.length sub and m = String.length text in
  let rec at i = i + n <= m && (String.sub text i n = sub || at (i + 1)) in
  at 0

(* Dune shortens the paths of its dependencies ([../../core/README.md] for
   [../../../sarek/core/README.md]), so compare the last two components. *)
let last_two_components path =
  match List.rev (String.split_on_char '/' path) with
  | file :: dir :: _ -> dir ^ "/" ^ file
  | _ -> path

let check_listed dune_file files =
  let ic = open_in_bin dune_file in
  let dune_text = really_input_string ic (in_channel_length ic) in
  close_in ic ;
  let unlisted =
    List.filter
      (fun f ->
        blocks_of_lines (read_lines f) <> []
        && not (contains_substring ~sub:(last_two_components f) dune_text))
      files
  in
  List.iter
    (fun f ->
      Printf.eprintf
        "%s has ocaml blocks but no rule in %s (see CONTRIBUTING.md)\n"
        f
        dune_file)
    unlisted ;
  if unlisted <> [] then exit 1

let () =
  match Array.to_list Sys.argv with
  | _ :: "--check-listed" :: dune_file :: files -> check_listed dune_file files
  | [_; "--only"; name; path] -> emit ~only:name path
  | [_; path] -> emit path
  | _ ->
      prerr_endline
        "usage: extract.exe [--only NAME] FILE.md | --check-listed DUNE \
         FILE.md..." ;
      exit 2
