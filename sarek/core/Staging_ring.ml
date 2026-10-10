(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Staging_ring - offset bookkeeping for a fixed-size staging buffer reused by
 * asynchronous copies (Transfer.stage_to_device, specs/transfer-pitfalls.md).
 *
 * Pure: the caller owns the memory and the fences (events recorded after each
 * copy). An allocation stays live until its fence completes; space is only
 * ever reclaimed from the OLDEST live allocation, so the live ranges are one
 * contiguous arc of the ring and a request either fits beside it or waits for
 * the oldest fence. Not thread-safe: the caller serialises access.
 ******************************************************************************)

type 'fence entry = {start : int; fence : 'fence}

type 'fence t = {
  capacity : int;
  align : int;
  mutable head : int;  (** Where the next allocation starts. *)
  live : 'fence entry Queue.t;  (** Oldest first. *)
}

let create ~(capacity : int) ~(align : int) : 'fence t =
  if align < 1 || capacity < align || capacity mod align <> 0 then
    invalid_arg
      "Staging_ring.create: capacity must be a positive multiple of align" ;
  {capacity; align; head = 0; live = Queue.create ()}

let rounded t size = (size + t.align - 1) / t.align * t.align

(* Hand back every completed fence at the front of the queue. *)
let reclaim t ~is_done ~on_free =
  let continue = ref true in
  while !continue && not (Queue.is_empty t.live) do
    let e = Queue.peek t.live in
    if is_done e.fence then begin
      ignore (Queue.pop t.live) ;
      on_free e.fence
    end
    else continue := false
  done ;
  if Queue.is_empty t.live then t.head <- 0

(* Where [bytes] (already rounded) fit beside the live arc, if they do. The
   comparisons against the oldest start are STRICT where an allocation could
   end exactly on it: [head] reaching the oldest start from below would make a
   full ring look empty. *)
let fits t bytes =
  if Queue.is_empty t.live then if bytes <= t.capacity then Some 0 else None
  else
    let tail = (Queue.peek t.live).start in
    if t.head >= tail then
      (* Live arc is [tail, head): free space is [head, capacity) and
         [0, tail). *)
      if t.head + bytes <= t.capacity then Some t.head
      else if bytes < tail then Some 0
      else None
    else if
      (* Live arc wraps: [tail, capacity) and [0, head). *)
      t.head + bytes < tail
    then Some t.head
    else None

(** Offset of a free range of [size] bytes (rounded up to the alignment).
    Reclaims completed allocations first; when the request still does not fit,
    calls [wait] on the oldest live fence (which must complete it), hands it to
    [on_free], and retries. Raises [Invalid_argument] when [size] exceeds the
    capacity or is not positive. The range is NOT reserved until {!commit}. *)
let reserve (t : 'fence t) ~(size : int) ~(is_done : 'fence -> bool)
    ~(wait : 'fence -> unit) ~(on_free : 'fence -> unit) : int =
  let bytes = rounded t size in
  if size < 1 || bytes > t.capacity then
    invalid_arg
      (Printf.sprintf
         "Staging_ring.reserve: %d bytes outside [1, %d]"
         size
         t.capacity) ;
  let rec go () =
    reclaim t ~is_done ~on_free ;
    match fits t bytes with
    | Some off -> off
    | None ->
        (* Pop only once the wait succeeded: if it raises, the range stays
           live rather than being handed out under a pending copy. *)
        let e = Queue.peek t.live in
        wait e.fence ;
        ignore (Queue.pop t.live) ;
        on_free e.fence ;
        if Queue.is_empty t.live then t.head <- 0 ;
        go ()
  in
  go ()

(** Mark the [size] bytes at [off] live until [fence] completes. [off] must come
    from the {!reserve} immediately before, with the same [size]. *)
let commit (t : 'fence t) ~(off : int) ~(size : int) (fence : 'fence) : unit =
  Queue.push {start = off; fence} t.live ;
  t.head <- off + rounded t size

(** Number of allocations whose fence has not been handed back. *)
let live_count (t : 'fence t) : int = Queue.length t.live
