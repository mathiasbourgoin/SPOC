(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E regression test (H6): the launch window must not grow a fused group past
 * Vulkan's 128-byte push-constant block (specs/launch-window.md, "Limits").
 *
 * Each member below has one vector and three int32 scalars: 4 bytes of vector
 * length + 12 bytes of scalars = 16 bytes, plus 8 bytes for the (offset, size)
 * pair the fusion adds = 24 bytes per member. Five members fill 120 bytes; the
 * sixth would make 144. So N independent launches must be issued as
 * ceil(N / 5) backend launches, every result correct, and no launch may be
 * refused by the backend.
 *
 * Vulkan only; skips (prints [SKIP], exits 0) when no Vulkan device is
 * present. Run it with SPOC_VULKAN_FENCE_TIMEOUT_S set so that a fence that
 * never signals fails the test instead of hanging it.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_vulkan.Vulkan_plugin.init ()

let add_three =
  [%kernel
    fun (x : int32 vector) (a : int32) (b : int32) (c : int32) ->
      let i = global_thread_id in
      if i < 64l then x.(i) <- x.(i) + a + b + c]

let ir_of kern =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let launch dev x ~k =
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of add_three)
    ~args:
      [
        Execute.Vec x;
        Execute.Int (k + 1);
        Execute.Int (k + 2);
        Execute.Int (k + 3);
      ]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    ()

let make_vector k =
  let v = Vector.create Vector.int32 64 in
  for i = 0 to 63 do
    Vector.set v i (Int32.of_int (i + k))
  done ;
  v

let result_correct k v =
  let ok = ref true in
  for i = 0 to 63 do
    if Vector.get v i <> Int32.of_int (i + k + (3 * k) + 6) then ok := false
  done ;
  !ok

(* [members] independent launches; returns (window statistics, results
   correct). *)
let run dev ~members =
  let vs = List.init members make_vector in
  Execute.Window.flush () ;
  (* A refused sequence is remembered across cases: start each from nothing. *)
  Execute.Window.clear_fused_cache () ;
  Execute.Window.reset_stats () ;
  List.iteri (fun k v -> launch dev v ~k) vs ;
  Execute.Window.flush () ;
  Transfer.flush dev ;
  let stats = Execute.Window.stats () in
  (stats, List.for_all Fun.id (List.mapi result_correct vs))

let failures = ref 0

let check label ok =
  Printf.printf "  %-64s %s\n%!" label (if ok then "ok" else "FAIL") ;
  if not ok then incr failures

(* Push-constant bytes of a group of [size] launches of [add_three], as the
   backend sees them: the fused block for two or more, the launch's own for
   one. *)
let group_bytes size =
  let params = (ir_of add_three).Sarek_ir_types.kern_params in
  if size = 1 then Sarek_push_constants.block_bytes params
  else Sarek_push_constants.fused_block_bytes (List.init size (fun _ -> params))

(* [sizes] are the launch counts of the groups the window must form. *)
let case dev (members, sizes) =
  let launches = List.length sizes in
  let stats, correct =
    try run dev ~members
    with e ->
      Printf.printf "  raised: %s\n%!" (Printexc.to_string e) ;
      ({(Execute.Window.stats ()) with issued = -1}, false)
  in
  check
    (Printf.sprintf
       "%d members in groups of bytes [%s]: %d launch(es) (got %d), 0 refused \
        (got %d), correct"
       members
       (String.concat
          "; "
          (List.map (fun n -> string_of_int (group_bytes n)) sizes))
       launches
       stats.issued
       stats.fused_refused)
    (stats.issued = launches && stats.fused_refused = 0 && correct)

let () =
  match Array.to_list (Device.by_framework "Vulkan") with
  | [] -> Printf.printf "[SKIP] No Vulkan device available\n%!"
  | dev :: _ ->
      check "Vulkan defers launches" (Execute.Window.defers_device dev) ;
      List.iter
        (case dev)
        [(5, [5]); (6, [5; 1]); (10, [5; 5]); (11, [5; 5; 1])] ;
      if !failures > 0 then exit 1
