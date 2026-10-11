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
      because every kernel it compiles has [-ffp-contract=off]).
    - C has the consumer shape ({!Fusion_epilogue.consumer}). C is 1-D with no
      dynamic shared memory.
    - Among the vectors P writes, C reads exactly one, I. C reads I only at its
      own index. P does not read I, and P binds I to exactly one parameter.
    - C writes nothing P accesses, except I itself.
    - C conflicts with no other launch of P's group, and with no launch of a
      later group.
    - Coverage (for these dimensions and scalar values): P stores each element
      of I at most once, at a non-negative index, and every element C writes is
      one that P stores. Proven by {!Coverage_affine} when the indices are
      affine, else by enumerating both launches with {!Index_slice}, within
      [max_enumerated] threads.

    The coverage verdict is cached by (P, the producer array, C, their
    dimensions, the values of the scalars the evaluation read). *)

open Sarek_ir_types
module LW = Launch_window

(** {1 Policy} *)

(* HIP only: the fused kernel equals the unfused pair only if no kernel of the
   backend is contracted, since the device compiler may share a subexpression
   of P and C (or forward P's stored value) in the fused kernel that the
   unfused C contracts. HIP compiles every kernel with [-ffp-contract=off].
   Metal does too ([fp contract(off)]), by construction, but has not been
   executed on Apple hardware. CUDA, OpenCL and Vulkan contract inside kernels
   today (specs/epilogue-fusion.md, docs/fp-contraction-policy.md section 14),
   so they are opt-in. *)
let default_backends = ["HIP"]

(* [SAREK_LAUNCH_EPILOGUE=0] turns epilogue fusion off at start-up. *)
let backends : string list Atomic.t =
  Atomic.make
    (match Sys.getenv_opt "SAREK_LAUNCH_EPILOGUE" with
    | Some ("0" | "false" | "off" | "no") -> []
    | _ -> default_backends)

let enabled (backend : string) = List.mem backend (Atomic.get backends)

(** Turn epilogue fusion on or off for one backend. Off by default except on
    HIP: elsewhere the device compiler may contract in the fused kernel what it
    does not contract in the unfused pair, or the reverse, which changes the
    result by an ulp. Turning it on there is a precision grant. *)
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

(* Enumeration bound: threads of P plus threads of C. Each run also stops
   after twice that many threads plus loop iterations. Beyond it the check
   answers "not proven": the launches stay unfused. *)
let max_enumerated = 1 lsl 20

(** How coverage verdicts were reached, since start-up: by the affine proof
    ({!Coverage_affine}), by enumeration, or refused as over the enumeration
    bound. *)
type proof_counts = {affine : int; enumerated : int; over_budget : int}

let n_affine = Atomic.make 0

let n_enumerated = Atomic.make 0

let n_over_budget = Atomic.make 0

let proof_counts () =
  {
    affine = Atomic.get n_affine;
    enumerated = Atomic.get n_enumerated;
    over_budget = Atomic.get n_over_budget;
  }

let threads (l : _ LW.launch) =
  l.grid.x * l.grid.y * l.grid.z * l.block.x * l.block.y * l.block.z

exception Stored_twice

let tag used_p used_c =
  List.map (fun n -> ("p", n)) used_p @ List.map (fun n -> ("c", n)) used_c

let slice (l : _ LW.launch) ~is_target ~record ~scalars =
  Index_slice.run
    ~budget:(2 * max_enumerated)
    ~is_target
    ~record
    ~scalars
    ~dims:(dims_of l)
    l.ir.kern_body

(* Every thread of P and C, on the host. A negative or huge index, an element
   stored twice or an unknown index: not proven. *)
let enumerate ~(p : _ LW.launch) ~p_array ~p_scalars ~(c : _ LW.launch)
    ~(shape : Fusion_epilogue.consumer) ~c_scalars =
  if threads p + threads c > max_enumerated then begin
    Atomic.incr n_over_budget ;
    (* Reached because the affine proof failed, possibly for these scalar
       values only (a negative dividend, an int32 overflow): the refusal holds
       for these values of every scalar parameter. *)
    (false, tag (List.map fst p_scalars) (List.map fst c_scalars))
  end
  else
    let stored = Index_bitset.create () and written = Index_bitset.create () in
    let store i =
      if Index_bitset.add stored (Int32.to_int i) then
        raise_notrace Stored_twice
    in
    let write i = ignore (Index_bitset.add written (Int32.to_int i)) in
    match
      let used_p =
        slice
          p
          ~is_target:(String.equal p_array)
          ~record:store
          ~scalars:p_scalars
      in
      let is_target a = List.mem a shape.written in
      (used_p, slice c ~is_target ~record:write ~scalars:c_scalars)
    with
    | used_p, used_c ->
        Atomic.incr n_enumerated ;
        (Index_bitset.subset written stored, tag used_p used_c)
    | exception
        (Index_slice.Unknown _ | Index_bitset.Out_of_range _ | Stored_twice) ->
        (* Stopped part-way, possibly on a value-dependent condition (an
           index, a division by zero, a loop bound): the refusal holds for
           these values of every scalar parameter, not for all values. *)
        (false, tag (List.map fst p_scalars) (List.map fst c_scalars))

(* [true] when P stores each element at most once, at a non-negative index,
   and covers every element C writes. Unknown or too large: [false]. Also
   returns the scalars the verdict read, tagged ["p"] or ["c"]. *)
let covers ~(p : _ LW.launch) ~p_array ~p_scalars ~(c : _ LW.launch)
    ~(shape : Fusion_epilogue.consumer) ~c_scalars =
  match
    Coverage_affine.covers
      ~p_body:p.ir.kern_body
      ~p_array
      ~p_scalars
      ~p_dims:(dims_of p)
      ~c_body:c.ir.kern_body
      ~c_written:shape.written
      ~c_scalars
      ~c_dims:(dims_of c)
  with
  | verdict ->
      Atomic.incr n_affine ;
      verdict
  | exception Coverage_affine.Unknown _ ->
      enumerate ~p ~p_array ~p_scalars ~c ~shape ~c_scalars

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
            ~name:""
            ~producer:p.ir
            ~p_array
            ~consumer:c.ir
            ~shared
        with
        | k -> (
            match Fusion_horizontal.content_named ~prefix:"sarek_efused_" k with
            | Some k -> Some k
            | None ->
                let n = Atomic.fetch_and_add counter 1 in
                Some {k with kern_name = Printf.sprintf "sarek_efused_%d" n})
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
  (* P and C as submitted, like the window issues a group singly: both are
     attempted even if P raises, and the first exception is re-raised after C
     (as [Launch_window.issue_singly]). The window counts the merged launch as one backend launch when
     this returns, and not at all when it raises, so the launches really made
     are added here: all but that one on return, all of them on a raise. *)
  let issue_unfused () =
    let made = ref 0 in
    let count ~counted_by_caller =
      for _ = 1 to !made - counted_by_caller do
        LW.bump w.LW.c_issued
      done
    in
    match
      LW.each
        (fun (l : _ LW.launch) ->
          l.issue () ;
          incr made)
        [p; c]
    with
    | () -> count ~counted_by_caller:1
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        count ~counted_by_caller:0 ;
        Printexc.raise_with_backtrace e bt
  in
  let issue () =
    match p.issue_fused fused args p.grid with
    | () -> ()
    | exception e when not (fatal e) ->
        Hashtbl.replace fused_cache key None ;
        issue_unfused ()
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
                    Option.bind (try_member w g m c shape) (fun merged ->
                        let after =
                          List.map
                            (fun x -> if x == m then merged else x)
                            g.LW.rev_members
                        in
                        if LW.signature_ok w (List.rev after) then
                          Some (m, merged)
                        else None)
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
