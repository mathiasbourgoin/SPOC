(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Compile_cache - Shared compile-cache key builder
 *
 * See Compile_cache.mli for the rationale: every backend's cache key must
 * include the kernel/entry name, not just device + source digest, or a
 * second kernel compiled from the same source silently aliases the first.
 ******************************************************************************)

(* Each (key, value) pair is digested independently before being joined, for
   the same reason the top-level fields are: without it, an option value
   containing ',' or '=' can shift a byte across the "k=v" boundary and make
   a single pair collide with a distinct multi-pair list, e.g.
   [("a", "1,b=2")] and [("a", "1"); ("b", "2")] would otherwise both
   canonicalize to the literal string "a=1,b=2". *)
let canonicalize_options options =
  options
  |> List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2)
  |> List.map (fun (k, v) ->
      Digest.to_hex (Digest.string k) ^ "=" ^ Digest.to_hex (Digest.string v))
  |> String.concat ","

(* Each free-form component is digested on its own before being joined, so the
   ':'-separated fields below are always exactly 32 hex characters wide. That
   makes the join unambiguous regardless of what bytes appear inside [device],
   [name], or the canonicalized options string (e.g. a device name containing
   ':', or an option value containing ',' or '='). Concatenating the raw
   strings directly (the previous scheme) let two distinct (device, name,
   options) triples collapse to the same key, e.g. device="a:b" ^ name="c"
   digesting identically to device="a" ^ name="b:c". *)
let digest_component s = Digest.string s |> Digest.to_hex

let compute_key ~device ~name ~source ~options =
  let source_digest = digest_component source in
  let opts_digest = digest_component (canonicalize_options options) in
  Printf.sprintf
    "%s:%s:%s:%s"
    (digest_component device)
    (digest_component name)
    source_digest
    opts_digest

(* Key memo, weak in the source string and keyed on its PHYSICAL identity.

   A JIT launch path that re-presents the same source string on every launch
   (Execute's launch memo does exactly that) would otherwise pay four MD5
   digests plus a [sprintf] per launch — 1-3 us for a 0.3-2 KB kernel, measured
   on the HIP path. Physical identity is sound because OCaml strings are
   immutable: the same physical string always has the same contents, so the
   memoised key is the key [compute_key] would return. (A string obtained with
   [Bytes.unsafe_to_string] and mutated afterwards breaks that contract — and
   every other user of the string with it; see the .mli.)

   The hash must not read the whole source (that would cost what the memo
   saves), and it only has to be consistent with [==], which any function of
   the contents is. Length plus a few sampled bytes spreads distinct kernels
   well; collisions only lengthen a bucket scan of physical comparisons.

   The value is a short association list because one source can be keyed for
   several (device, name, options) triples. The ephemeron data never refers to
   the key string, so an entry dies with its source. *)
module Physical_string = struct
  type t = string

  let equal = ( == )

  let hash (s : string) =
    let n = String.length s in
    if n = 0 then 0
    else
      let sample i = Char.code (String.unsafe_get s (i * (n - 1) / 7)) in
      Hashtbl.hash
        ( n,
          sample 0,
          sample 1,
          sample 2,
          sample 3,
          sample 4,
          sample 5,
          sample 6,
          sample 7 )
end

module Key_memo = Ephemeron.K1.Make (Physical_string)

let key_memo :
    ((string * string * (string * string) list) * string) list Key_memo.t =
  Key_memo.create 64

let key_memo_lock = Mutex.create ()

let make_key ~device ~name ~source ?(options = []) () =
  let triple = (device, name, options) in
  let cached =
    Mutex.protect key_memo_lock (fun () ->
        match Key_memo.find_opt key_memo source with
        | Some entries -> List.assoc_opt triple entries
        | None -> None)
  in
  match cached with
  | Some key -> key
  | None ->
      let key = compute_key ~device ~name ~source ~options in
      Mutex.protect key_memo_lock (fun () ->
          let entries =
            Option.value (Key_memo.find_opt key_memo source) ~default:[]
          in
          if not (List.mem_assoc triple entries) then
            Key_memo.replace key_memo source ((triple, key) :: entries)) ;
      key
