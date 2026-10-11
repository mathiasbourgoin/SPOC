(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** A set of array indices in [\[0, capacity)], one bit each, grown on demand:
    the store sets of the epilogue coverage enumeration ({!Launch_epilogue}),
    without hashing or polymorphic comparison per element. *)

(** Largest index + 1 a set accepts (16 MiB of bits). *)
let capacity = 1 lsl 27

type t = {mutable bits : Bytes.t}

exception Out_of_range of int

let create () = {bits = Bytes.make 1024 '\000'}

let grow t byte =
  let n = ref (Bytes.length t.bits) in
  while !n <= byte do
    n := 2 * !n
  done ;
  let bits = Bytes.make !n '\000' in
  Bytes.blit t.bits 0 bits 0 (Bytes.length t.bits) ;
  t.bits <- bits

(** Add [i]; [true] when it was already there. Raises [Out_of_range] outside
    [\[0, capacity)]. *)
let add t i =
  if i < 0 || i >= capacity then raise (Out_of_range i) ;
  let byte = i lsr 3 and bit = 1 lsl (i land 7) in
  if byte >= Bytes.length t.bits then grow t byte ;
  let old = Char.code (Bytes.unsafe_get t.bits byte) in
  Bytes.unsafe_set t.bits byte (Char.unsafe_chr (old lor bit)) ;
  old land bit <> 0

(** Every element of [a] is in [b]. *)
let subset a b =
  let nb = Bytes.length b.bits in
  let ok = ref true and k = ref 0 in
  while !ok && !k < Bytes.length a.bits do
    let x = Char.code (Bytes.unsafe_get a.bits !k) in
    let y = if !k < nb then Char.code (Bytes.unsafe_get b.bits !k) else 0 in
    if x land lnot y <> 0 then ok := false ;
    incr k
  done ;
  !ok
