(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kernel_args - Indexed kernel-argument container
 *
 * Shared implementation of "store a value at idx, last write wins" plus
 * strict pre-launch validation, so backend plugins stop hand-rolling
 * accumulate-by-call-order argument lists (see module doc in the .mli).
 ******************************************************************************)

(* Indices [0 .. dense_limit - 1] live in [dense], grown by doubling; any
   other index (negative, or past the limit) lives in [sparse], so a stray
   index never allocates an array of its size. [count] is the number of
   distinct indices stored in both. *)
type 'a t = {
  mutable dense : 'a option array;
  mutable sparse : (int * 'a) list;  (** sorted by index *)
  mutable count : int;
}

let dense_limit = 4096

let create () = {dense = Array.make 8 None; sparse = []; count = 0}

let grow t idx =
  let len = ref (Array.length t.dense) in
  while !len <= idx do
    len := !len * 2
  done ;
  let dense = Array.make (min !len dense_limit) None in
  Array.blit t.dense 0 dense 0 (Array.length t.dense) ;
  t.dense <- dense

let set_sparse t idx v =
  if not (List.mem_assoc idx t.sparse) then t.count <- t.count + 1 ;
  t.sparse <-
    List.merge
      (fun (a, _) (b, _) -> Int.compare a b)
      [(idx, v)]
      (List.remove_assoc idx t.sparse)

let set t idx v =
  if idx < 0 || idx >= dense_limit then set_sparse t idx v
  else begin
    if idx >= Array.length t.dense then grow t idx ;
    (match t.dense.(idx) with None -> t.count <- t.count + 1 | Some _ -> ()) ;
    t.dense.(idx) <- Some v
  end

let count t = t.count

let find t idx =
  if idx >= 0 && idx < Array.length t.dense then t.dense.(idx)
  else List.assoc_opt idx t.sparse

let to_sorted_list t =
  let dense = ref [] in
  for i = Array.length t.dense - 1 downto 0 do
    match t.dense.(i) with Some v -> dense := (i, v) :: !dense | None -> ()
  done ;
  List.merge (fun (a, _) (b, _) -> Int.compare a b) !dense t.sparse

let describe_indices idxs = String.concat ", " (List.map string_of_int idxs)

let index_word n = if n = 1 then "index" else "indices"

(* The indices of [0 .. n - 1] with no value, ascending. *)
let missing_below t n =
  let missing = ref [] in
  for i = n - 1 downto 0 do
    if Option.is_none (find t i) then missing := i :: !missing
  done ;
  !missing

let indices_outside t n =
  List.filter_map
    (fun (idx, _) -> if idx < 0 || idx >= n then Some idx else None)
    (to_sorted_list t)

(* Every index of [0 .. n - 1] is set and [count t = n]: no other index. *)
let extract t n = Array.init n (fun i -> Option.get (find t i))

let validate_and_extract t ~expected_count =
  if expected_count < 0 then
    Error (Printf.sprintf "invalid expected_count: %d" expected_count)
  else
    match missing_below t expected_count with
    | [] when t.count = expected_count -> Ok (extract t expected_count)
    | [] ->
        let extra = indices_outside t expected_count in
        Error
          (Printf.sprintf
             "unexpected %s: [%s]; expected contiguous 0..%d"
             (index_word (List.length extra))
             (describe_indices extra)
             (expected_count - 1))
    | missing -> (
        match indices_outside t expected_count with
        | [] ->
            Error
              (Printf.sprintf
                 "missing indices: [%s]; expected %d args, got %d"
                 (describe_indices missing)
                 expected_count
                 t.count)
        | extra ->
            Error
              (Printf.sprintf
                 "missing indices: [%s]; unexpected %s: [%s]; expected %d \
                  args, got %d"
                 (describe_indices missing)
                 (index_word (List.length extra))
                 (describe_indices extra)
                 expected_count
                 t.count))
