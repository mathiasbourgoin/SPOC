(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Epilogue fusion in the launch window (specs/epilogue-fusion.md).

    When an element-wise launch C arrives, it is merged into a pending launch P
    that writes the one vector I it consumes, if all of the following hold. The
    merged launch replaces P in its group and can itself host a further epilogue
    or join a horizontal group. The rewrite is {!Fusion_epilogue}.

    {b Conditions (vectors compared by identity, on one device).}
    - The backend runs epilogues ({!set_backend_enabled}; HIP by default,
      because its compiler is forced to [-ffp-contract=off]).
    - C has the consumer shape ({!Fusion_epilogue.consumer}). C is 1-D with no
      dynamic shared memory.
    - Among the vectors P accesses, C reads exactly one, I. C reads I only at
      its own index. P does not read I, and P binds I to exactly one parameter.
    - C writes nothing P accesses, except I itself.
    - C conflicts with no other launch of P's group, and with no launch of a
      later group.
    - Coverage ({!Index_slice}, for these dimensions and scalar values): P
      stores each element of I at most once, and every element C writes is one
      that P stores.

    The coverage verdict is cached by (P, the producer array, C, their
    dimensions, the values of the scalars the evaluation read). *)

open Sarek_ir_types
module LW = Launch_window

(** {1 Policy} *)

let default_backends = ["HIP"]

(* [SAREK_LAUNCH_EPILOGUE=0] turns epilogue fusion off at start-up. *)
let backends : string list Atomic.t =
  Atomic.make
    (match Sys.getenv_opt "SAREK_LAUNCH_EPILOGUE" with
    | Some ("0" | "false" | "off" | "no") -> []
    | _ -> default_backends)

let enabled (backend : string) = List.mem backend (Atomic.get backends)

(** Turn epilogue fusion on or off for one backend. Off by default except on
    HIP: elsewhere the device compiler may contract a multiply of the producer
    with an add of the consumer once both are in one kernel, which changes the
    result by an ulp. Turning it on there is accepting that. *)
let set_backend_enabled ~(backend : string) (b : bool) =
  let rec update () =
    let old = Atomic.get backends in
    let rest = List.filter (fun x -> x <> backend) old in
    let next = if b then backend :: rest else rest in
    if not (Atomic.compare_and_set backends old next) then update ()
  in
  update ()

(** {1 Per-launch facts} *)

(* (parameter name, bound vector id) for the vector arguments of a launch. *)
let bound (w : 'arg LW.t) (l : 'arg LW.launch) =
  List.concat
    (List.map2
       (fun d a ->
         match (d, w.LW.arg_buf a) with
         | DParam (v, Some _), Some id -> [(v.var_name, id)]
         | _ -> [])
       l.ir.kern_params
       l.args)

let scalars (w : 'arg LW.t) (l : 'arg LW.launch) =
  List.concat
    (List.map2
       (fun d a ->
         match (d, w.LW.arg_int32 a) with
         | DParam (v, None), Some n -> [(v.var_name, n)]
         | _ -> [])
       l.ir.kern_params
       l.args)

let dims_of (l : _ LW.launch) =
  {
    Index_slice.block = (l.block.x, l.block.y, l.block.z);
    grid = (l.grid.x, l.grid.y, l.grid.z);
  }

(* Consumer shapes, by IR id (an IR's shape never changes). *)
let shapes : (int, (Fusion_epilogue.consumer, string) result) Hashtbl.t =
  Hashtbl.create 32

let shape_of (w : _ LW.t) (l : _ LW.launch) =
  match Hashtbl.find_opt shapes l.ir_id with
  | Some r -> r
  | None ->
      let r = Fusion_epilogue.consumer l.ir in
      ignore w ;
      Hashtbl.replace shapes l.ir_id r ;
      r

(** {1 Coverage} *)

let max_threads = 1 lsl 24

(* [true] when P stores each element at most once and covers every element
   C writes. Unknown or too large: [false]. *)
let covers ~(p : _ LW.launch) ~p_array ~p_scalars ~(c : _ LW.launch)
    ~(shape : Fusion_epilogue.consumer) ~c_scalars =
  let threads (l : _ LW.launch) =
    l.grid.x * l.grid.y * l.grid.z * l.block.x * l.block.y * l.block.z
  in
  if threads p > max_threads || threads c > max_threads then (false, [])
  else
    let stored = Hashtbl.create 1024 and dup = ref false in
    let written = Hashtbl.create 1024 in
    match
      let used_p =
        Index_slice.run
          ~is_target:(fun a -> a = p_array)
          ~record:(fun i ->
            if Hashtbl.mem stored i then dup := true
            else Hashtbl.replace stored i ())
          ~scalars:p_scalars
          ~dims:(dims_of p)
          p.ir.kern_body
      in
      let used_c =
        Index_slice.run
          ~is_target:(fun a -> List.mem a shape.written)
          ~record:(fun i -> Hashtbl.replace written i ())
          ~scalars:c_scalars
          ~dims:(dims_of c)
          c.ir.kern_body
      in
      (used_p, used_c)
    with
    | used_p, used_c ->
        let ok =
          (not !dup)
          && Hashtbl.fold
               (fun i () ok -> ok && Hashtbl.mem stored i)
               written
               true
        in
        ( ok,
          List.map (fun n -> ("p", n)) used_p
          @ List.map (fun n -> ("c", n)) used_c )
    | exception Index_slice.Unknown _ -> (false, [])

type verdict = {used : (string * string) list; values : int32 list; ok : bool}

(* (p ir, the producer array evaluated, c ir, p dims, c dims) -> verdicts for
   the scalar valuations seen. The array is part of the key: one producer can
   store two outputs under different guards. *)
let coverage_cache :
    ( int * string * int * Index_slice.dims * Index_slice.dims,
      verdict list )
    Hashtbl.t =
  Hashtbl.create 32

let lookup_value ~p_scalars ~c_scalars (side, n) =
  List.assoc_opt n (if side = "p" then p_scalars else c_scalars)

let cached_covers ~(p : _ LW.launch) ~p_array ~p_scalars ~(c : _ LW.launch)
    ~shape ~c_scalars =
  let key = (p.LW.ir_id, p_array, c.LW.ir_id, dims_of p, dims_of c) in
  let entries =
    Option.value ~default:[] (Hashtbl.find_opt coverage_cache key)
  in
  let matches v =
    List.map (lookup_value ~p_scalars ~c_scalars) v.used
    = List.map Option.some v.values
  in
  match List.find_opt matches entries with
  | Some v -> v.ok
  | None ->
      let ok, used = covers ~p ~p_array ~p_scalars ~c ~shape ~c_scalars in
      let values =
        List.map
          (fun u ->
            Option.value ~default:0l (lookup_value ~p_scalars ~c_scalars u))
          used
      in
      let entries = if List.length entries >= 64 then [] else entries in
      Hashtbl.replace coverage_cache key ({used; values; ok} :: entries) ;
      ok

(** {1 Merging} *)

let fatal = function Out_of_memory | Stack_overflow -> true | _ -> false

(* (producer ir id, consumer ir id, shared mapping) -> fused IR, or refused. *)
let fused_cache :
    (int * string * int * (string * string) list, kernel option) Hashtbl.t =
  Hashtbl.create 32

let counter = Atomic.make 0

let fused_ir ~(p : _ LW.launch) ~p_array ~(c : _ LW.launch) ~shared =
  let key = (p.LW.ir_id, p_array, c.LW.ir_id, shared) in
  match Hashtbl.find_opt fused_cache key with
  | Some k -> k
  | None ->
      let k =
        match
          Fusion_epilogue.fuse
            ~name:
              (Printf.sprintf
                 "sarek_efused_%d"
                 (Atomic.fetch_and_add counter 1))
            ~producer:p.ir
            ~p_array
            ~consumer:c.ir
            ~shared
        with
        | k -> Some k
        | exception
            (Fusion_epilogue.Incompatible _ | Fusion_horizontal.Incompatible _)
          ->
            None
      in
      if Hashtbl.length fused_cache >= 4096 then Hashtbl.reset fused_cache ;
      Hashtbl.replace fused_cache key k ;
      k

(* The merged launch: P's place, the fused kernel, P's grid and block. If the
   fused kernel fails when issued, P and C are issued as submitted, and this
   pair is never merged again. *)
let merged (w : 'arg LW.t) (p : 'arg LW.launch) (c : 'arg LW.launch) ~fused
    ~p_array ~shared ~tc =
  let c_args =
    List.concat
      (List.map2
         (fun d a ->
           match d with
           | DParam (v, Some _) when List.mem_assoc v.var_name shared -> []
           | _ -> [a])
         c.ir.kern_params
         c.args)
  in
  let args = p.args @ c_args @ [w.LW.int32_arg (Int32.of_int tc)] in
  let info = LW.ir_info_locked w fused in
  let key = (p.ir_id, p_array, c.ir_id, shared) in
  let issue () =
    match p.issue_fused fused args p.grid with
    | () -> ()
    | exception e when not (fatal e) ->
        Hashtbl.replace fused_cache key None ;
        p.issue () ;
        c.issue ()
  in
  {
    p with
    LW.ir = fused;
    ir_id = info.ir_id;
    shared_bytes = Result.to_option info.fusable;
    args;
    accesses = p.accesses @ c.accesses;
    issue;
  }

(* Under [w.lock]: the producer [m] in [g] that [c] can be the epilogue of,
   with what the rewrite needs. *)
let bufs ~pick accesses =
  List.filter_map (fun a -> if pick a then Some a.LW.buf else None) accesses

(* The intermediate: the one vector C reads that [m] writes. *)
let intermediate (m : _ LW.launch) (c : _ LW.launch) =
  let reads = bufs ~pick:(fun a -> a.LW.read) c.accesses in
  let m_writes = bufs ~pick:(fun a -> a.LW.write) m.accesses in
  match
    List.sort_uniq compare (List.filter (fun b -> List.mem b m_writes) reads)
  with
  | [i_buf] -> Some i_buf
  | _ -> None

(* Under [w.lock]: the access conditions of the module comment, for C as the
   epilogue of [m] in [g]. Returns [m]'s parameter bound to the intermediate. *)
let merge_point w g (m : _ LW.launch) (c : _ LW.launch) shape i_buf =
  let c_bound = bound w c and p_bound = bound w m in
  let m_touches b = List.exists (fun a -> a.LW.buf = b) m.accesses in
  let written_bufs =
    List.filter_map
      (fun n -> List.assoc_opt n c_bound)
      shape.Fusion_epilogue.written
  in
  let ok =
    m.device_id = c.device_id
    && (not (List.exists (fun a -> a.LW.buf = i_buf && a.LW.read) m.accesses))
    && List.for_all
         (fun (n, b) -> b <> i_buf || List.mem n shape.reads_at_index)
         c_bound
    && List.for_all (fun b -> b = i_buf || not (m_touches b)) written_bufs
    && List.for_all (fun o -> o == m || not (LW.conflicts o c)) g.LW.rev_members
  in
  match List.filter (fun (_, b) -> b = i_buf) p_bound with
  | [(p_array, _)] when ok ->
      let shared =
        List.filter_map
          (fun (n, b) ->
            Option.map
              (fun (pn, _) -> (n, pn))
              (List.find_opt (fun (_, pb) -> pb = b) p_bound))
          c_bound
      in
      Some (p_array, shared)
  | _ -> None

(* Under [w.lock]: [m] merged with its epilogue [c], when every condition and
   the coverage hold. *)
let try_member w g (m : _ LW.launch) (c : _ LW.launch) shape =
  match intermediate m c with
  | None -> None
  | Some i_buf -> (
      match merge_point w g m c shape i_buf with
      | None -> None
      | Some (p_array, shared) ->
          if
            cached_covers
              ~p:m
              ~p_array
              ~p_scalars:(scalars w m)
              ~c
              ~shape
              ~c_scalars:(scalars w c)
          then
            Option.map
              (fun fused ->
                merged w m c ~fused ~p_array ~shared ~tc:(c.grid.x * c.block.x))
              (fused_ir ~p:m ~p_array ~c ~shared)
          else None)

let one_dimensional (l : _ LW.launch) =
  l.grid.y = 1 && l.grid.z = 1 && l.block.y = 1 && l.block.z = 1

(** The window's epilogue hook ({!install}). *)
let join (w : 'arg LW.t) (c : 'arg LW.launch) : bool =
  enabled c.backend && c.shared_mem = 0 && one_dimensional c
  &&
  match shape_of w c with
  | Error _ -> false
  | Ok shape ->
      let rec scan = function
        | [] -> false
        | g :: older -> (
            let hit =
              List.find_map
                (fun m ->
                  if LW.conflicts m c then
                    Option.map
                      (fun merged -> (m, merged))
                      (try_member w g m c shape)
                  else None)
                g.LW.rev_members
            in
            match hit with
            | Some (m, merged) ->
                g.rev_members <-
                  List.map (fun x -> if x == m then merged else x) g.rev_members ;
                g.size <- g.size + 1 ;
                true
            | None -> if LW.group_conflicts g c then false else scan older)
      in
      scan (List.rev w.LW.groups)

(** Enable epilogue fusion in [w] (on the backends {!enabled} allows). An
    unexpected exception while deciding means no merge: the launch then takes
    the ordinary path. *)
let install (w : _ LW.t) =
  w.LW.epilogue <-
    (fun w c ->
      try join w c with
      | (Out_of_memory | Stack_overflow) as e -> raise e
      | _ -> false)

(** Forget every cached consumer shape, coverage verdict and merged kernel. *)
let clear () =
  Hashtbl.reset shapes ;
  Hashtbl.reset coverage_cache ;
  Hashtbl.reset fused_cache
