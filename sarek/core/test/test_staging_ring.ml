(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Staging_ring: the bookkeeping behind Transfer.stage_to_device
 * (specs/transfer-pitfalls.md, AC-2). Pure: fences are integers whose
 * completion this test controls, so wrap-around, waiting and reuse are
 * checked without a device.
 ******************************************************************************)

module R = Spoc_core.Staging_ring

let failures = ref 0

let check name cond =
  if cond then Printf.printf "ok   %s\n%!" name
  else begin
    incr failures ;
    Printf.printf "FAIL %s\n%!" name
  end

(* A fence is an id; [completed] says which ids the "GPU" has finished. *)
let completed = Hashtbl.create 64

let is_done f = Hashtbl.mem completed f

let waited = ref []

let wait f =
  waited := f :: !waited ;
  Hashtbl.replace completed f ()

let freed = ref []

let on_free f = freed := f :: !freed

let reserve r size = R.reserve r ~size ~is_done ~wait ~on_free

let reset () =
  Hashtbl.reset completed ;
  waited := [] ;
  freed := []

let test_basic () =
  reset () ;
  let r = R.create ~capacity:1024 ~align:256 in
  let a = reserve r 10 in
  R.commit r ~off:a ~size:10 1 ;
  let b = reserve r 300 in
  R.commit r ~off:b ~size:300 2 ;
  check "first allocation at 0" (a = 0) ;
  check "sizes are rounded up to the alignment" (b = 256) ;
  check "nothing waited while space remains" (!waited = []) ;
  (* [0,256) [256,768) live; 256 more fit at 768. *)
  let c = reserve r 256 in
  R.commit r ~off:c ~size:256 3 ;
  check "fills to the end" (c = 768) ;
  (* Full: the next reservation waits for the oldest fences, in order. Fence
     1 alone frees [0, 256), but an allocation ending exactly where the oldest
     live one starts is refused (it would make a full ring look empty), so
     fence 2 is waited for too. *)
  let d = reserve r 100 in
  check "full ring waits for the oldest fences, oldest first" (!waited = [2; 1]) ;
  check "and reuses its space after wrapping" (d = 0) ;
  check "completed fences are handed back" (List.mem 1 !freed) ;
  R.commit r ~off:d ~size:100 4 ;
  check
    "oversized request refused"
    (match reserve r 2048 with
    | _ -> false
    | exception Invalid_argument _ -> true)

(* Space freed at the front of a wrapped arc is used without waiting: the ring
   must not wait while a request fits beside the live arc. *)
let test_wrapped_arc () =
  reset () ;
  let r = R.create ~capacity:1024 ~align:256 in
  for f = 1 to 4 do
    let off = reserve r 256 in
    R.commit r ~off ~size:256 f
  done ;
  List.iter (fun f -> Hashtbl.replace completed f ()) [1; 2] ;
  let a = reserve r 256 in
  R.commit r ~off:a ~size:256 5 ;
  check "wraps into freed space at 0 without waiting" (a = 0 && !waited = []) ;
  Hashtbl.replace completed 3 () ;
  let b = reserve r 256 in
  R.commit r ~off:b ~size:256 6 ;
  check
    "allocates inside the wrapped arc without waiting"
    (b = 256 && !waited = [])

let test_reclaim_without_wait () =
  reset () ;
  let r = R.create ~capacity:1024 ~align:256 in
  for i = 1 to 4 do
    let off = reserve r 256 in
    R.commit r ~off ~size:256 i
  done ;
  List.iter (fun f -> Hashtbl.replace completed f ()) [1; 2; 3; 4] ;
  let off = reserve r 512 in
  check "completed entries are reclaimed without waiting" (!waited = []) ;
  check "an empty ring restarts at offset 0" (off = 0) ;
  check "every completed fence handed back" (List.length !freed = 4)

(* Randomised model: live ranges never overlap, offsets stay in bounds, and a
   wait happens only when the request does not fit beside the live ranges. *)
let test_model ~complete_at_random () =
  reset () ;
  Random.init 7 ;
  let capacity = 4096 and align = 64 in
  let r = R.create ~capacity ~align in
  let live = ref [] in
  let overlaps (a, b) (c, d) = a < d && c < b in
  let ok = ref true in
  for f = 1 to 20_000 do
    (* The "GPU" completes outstanding fences at random, in any order (the ring
       must only ever reclaim from the oldest). *)
    List.iter
      (fun (_, _, g) ->
        if complete_at_random && Random.int 3 = 0 && not (is_done g) then
          Hashtbl.replace completed g ())
      (List.rev !live) ;
    let size = 1 + Random.int 1500 in
    let off = reserve r size in
    let rounded = (size + align - 1) / align * align in
    let range = (off, off + rounded) in
    live := List.filter (fun (_, _, g) -> not (is_done g)) !live ;
    if off < 0 || off + rounded > capacity then ok := false ;
    List.iter (fun (a, b, _) -> if overlaps (a, b) range then ok := false) !live ;
    R.commit r ~off ~size f ;
    live := (fst range, snd range, f) :: !live
  done ;
  check
    (Printf.sprintf
       "20000 random reservations (%s): in bounds, never overlapping a live one"
       (if complete_at_random then "random completion"
        else "completion only by waiting, as Transfer uses it"))
    !ok

let () =
  test_basic () ;
  test_reclaim_without_wait () ;
  test_wrapped_arc () ;
  test_model ~complete_at_random:true () ;
  test_model ~complete_at_random:false () ;
  exit (if !failures = 0 then 0 else 1)
