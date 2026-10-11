(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Deferred launch window with horizontal fusion (specs/launch-window.md).

    [Execute] submits JIT launches here instead of issuing them. Pending
    launches are kept in groups, in submission order. A new launch joins the
    newest pending group it is compatible with, provided it conflicts with no
    launch in that group or in any later one. Otherwise it opens a group of its
    own. A group of several launches is issued as one horizontally fused kernel
    ({!Fusion_horizontal}); a group of one is the original launch.

    {b Conflicts.} Each launch carries the buffers its vector arguments bind and
    whether the kernel reads or writes each one ({!Kernel_access}). A buffer is
    named by its vector's id: every device buffer belongs to exactly one vector
    (only [Transfer.ensure_buffer] installs one, in that vector's own table), so
    on one device two arguments share memory exactly when they are the same
    vector. Device addresses are not used: Vulkan reports none and OpenCL
    reports object handles. Two launches on the same device conflict when they
    bind the same buffer and at least one of the two accesses writes. A launch
    is never moved before a launch it conflicts with, so every conflicting pair
    is issued in submission order.

    {b Flushing.} {!flush} issues every pending group. [Execute] installs it as
    the {!Spoc_core.Launch_hooks} flusher, so every synchronisation point
    flushes before it acts. At most [max_groups] groups are pending: a new group
    beyond that issues the oldest one first.

    {b Extension point.} A launch is a plain record. A later stage that
    specialises a launch (for instance a launch-time kernel rewrite) can map
    {!launch} values before {!submit} without touching grouping or flushing.

    This module depends only on the IR and Stdlib, so it builds for jsoo too. *)

open Sarek_ir_types
module Dims = Spoc_framework.Framework_sig

(** {1 Launch records} *)

(** One bound vector argument: its buffer (the vector's id) and how the kernel
    uses it. *)
type access = {buf : int; read : bool; write : bool}

type 'arg launch = {
  device_id : int;
  backend : string;
  ir : kernel;
  ir_id : int;  (** {!ir_info}[.ir_id] of [ir] *)
  fusable : bool;  (** {!ir_info}[.fusable] of [ir] is [Ok ()] *)
  args : 'arg list;
  block : Dims.dims;
  grid : Dims.dims;
  shared_mem : int;
  accesses : access list;
  issue : unit -> unit;  (** issue this launch now, unfused *)
  issue_fused : kernel -> 'arg list -> Dims.dims -> unit;
      (** issue a fused kernel on this launch's device with this launch's block,
          the given arguments and grid *)
}

(** {1 Per-kernel facts} *)

module Physical_kernel = struct
  type t = kernel

  let equal = ( == )

  let hash (k : t) = Hashtbl.hash (k.kern_name, k.kern_params)
end

module Ir_table = Ephemeron.K1.Make (Physical_kernel)

type ir_info = {
  ir_id : int;
  param_accesses : (string * Kernel_access.access) list;
  fusable : (unit, string) result;
}

(** {1 Window state} *)

type 'arg group = {
  g_device : int;
  mutable rev_members : 'arg launch list;  (** newest first *)
  mutable size : int;
}

let members g = List.rev g.rev_members

let key_of g = List.rev_map (fun (m : _ launch) -> m.ir_id) g.rev_members

type stats = {
  submitted : int;  (** JIT launches submitted through [Execute] *)
  issued : int;  (** backend launches actually made *)
  fused_groups : int;  (** groups of two or more launches issued fused *)
  fused_built : int;  (** distinct fused kernels built *)
  fused_refused : int;  (** member sequences refused (build or launch failed) *)
}

type 'arg t = {
  lock : Mutex.t;
  mutable groups : 'arg group list;  (** oldest first *)
  mutable n_groups : int;
  mutable n_pending : int;
  int32_arg : int32 -> 'arg;
  set_pending : int -> unit;
  guard : (unit -> unit) -> unit;
  infos : ir_info Ir_table.t;
  mutable next_ir_id : int;
  cache : (int list, kernel) Hashtbl.t;  (** member ir_ids -> fused kernel *)
  refused : (int list, unit) Hashtbl.t;  (** member sequences never fused *)
  mutable stats : stats;
}

let zero_stats =
  {
    submitted = 0;
    issued = 0;
    fused_groups = 0;
    fused_built = 0;
    fused_refused = 0;
  }

(** [create ~int32_arg ~set_pending ~guard] makes an empty window. [int32_arg]
    builds a scalar argument for the fused kernel's position parameters.
    [set_pending] is told the number of pending launches after each change.
    [guard f] runs [f] (the issuing of pending groups) with the synchronisation
    hooks disabled on this domain. *)
let create ~int32_arg ~set_pending ~guard : 'arg t =
  {
    lock = Mutex.create ();
    groups = [];
    n_groups = 0;
    n_pending = 0;
    int32_arg;
    set_pending;
    guard;
    infos = Ir_table.create 64;
    next_ir_id = 0;
    cache = Hashtbl.create 64;
    refused = Hashtbl.create 8;
    stats = zero_stats;
  }

(** {1 Configuration} *)

let max_groups = ref 8

let max_members = ref 8

let env_enabled () =
  match Sys.getenv_opt "SAREK_LAUNCH_WINDOW" with
  | Some ("0" | "false" | "off" | "no") -> false
  | _ -> true

let enabled = Atomic.make (env_enabled ())

(** Backends whose launches are deferred by default: the ones whose window and
    fused kernels are exercised by the test suites on this project's hardware.
    Others stay immediate unless {!set_backend_enabled} opts them in. *)
let default_backends = ["HIP"; "OpenCL"; "Vulkan"]

(* Read on every launch: an immutable list behind an atomic, no lock. *)
let backends : string list Atomic.t = Atomic.make default_backends

(** [true] when launches on [backend] are deferred. *)
let defers (backend : string) : bool =
  Atomic.get enabled && List.mem backend (Atomic.get backends)

(** {1 Per-IR facts} *)

let ir_info (w : _ t) (ir : kernel) : ir_info =
  Mutex.protect w.lock (fun () ->
      match Ir_table.find_opt w.infos ir with
      | Some i -> i
      | None ->
          let i =
            {
              ir_id = w.next_ir_id;
              param_accesses = Kernel_access.param_accesses ir;
              fusable = Kernel_access.horizontal_verdict ir;
            }
          in
          w.next_ir_id <- w.next_ir_id + 1 ;
          Ir_table.replace w.infos ir i ;
          i)

(** {1 Grouping} *)

let accesses_conflict a b = a.buf = b.buf && (a.write || b.write)

let conflicts (a : _ launch) (b : _ launch) =
  a.device_id = b.device_id
  && List.exists
       (fun x -> List.exists (accesses_conflict x) b.accesses)
       a.accesses

let group_conflicts g l = List.exists (conflicts l) g.rev_members

let same_dims (a : Dims.dims) (b : Dims.dims) =
  a.x = b.x && a.y = b.y && a.z = b.z

(* A member of a fused group: fusable IR, no dynamic shared memory, a
   non-empty grid. *)
let joinable (l : _ launch) = l.fusable && l.shared_mem = 0 && l.grid.x > 0

(* Under [w.lock]. [l] may join [g] as its last member. Every member of a
   group of two or more is [joinable] with the same block and grid y/z, so
   comparing with the newest member is enough. *)
let compatible w g (l : _ launch) =
  match g.rev_members with
  | [] -> false
  | newest :: _ ->
      g.g_device = l.device_id && g.size < !max_members && joinable l
      && joinable newest
      && same_dims l.block newest.block
      && l.grid.y = newest.grid.y && l.grid.z = newest.grid.z
      && (Hashtbl.length w.refused = 0
         || not (Hashtbl.mem w.refused (key_of g @ [l.ir_id])))

(* Newest group [l] may join without crossing a conflicting launch. *)
let target w (l : _ launch) =
  let rec scan = function
    | [] -> None
    | g :: older ->
        if group_conflicts g l then None
        else if compatible w g l then Some g
        else scan older
  in
  scan (List.rev w.groups)

(** {1 Issuing} *)

let count_issued w n = w.stats <- {w.stats with issued = w.stats.issued + n}

let refuse w key =
  Hashtbl.replace w.refused key () ;
  w.stats <- {w.stats with fused_refused = w.stats.fused_refused + 1}

let fused_kernel w key members =
  match Hashtbl.find_opt w.cache key with
  | Some k -> Some k
  | None -> (
      match
        Fusion_horizontal.fuse
          ~name:(Printf.sprintf "sarek_hfused_%d" w.stats.fused_built)
          (List.map (fun m -> m.ir) members)
      with
      | k ->
          w.stats <- {w.stats with fused_built = w.stats.fused_built + 1} ;
          Hashtbl.replace w.cache key k ;
          Some k
      | exception Fusion_horizontal.Incompatible _ ->
          refuse w key ;
          None)

(* Arguments of the fused kernel: every member's own, then (off, gdx) per
   member, matching {!Fusion_horizontal.fuse}'s parameter list. *)
let fused_args w members =
  let rec positions off = function
    | [] -> ([], off)
    | m :: rest ->
        let tail, total = positions (off + m.grid.x) rest in
        ( w.int32_arg (Int32.of_int off)
          :: w.int32_arg (Int32.of_int m.grid.x)
          :: tail,
          total )
  in
  let pos, gx = positions 0 members in
  (List.concat_map (fun m -> m.args) members @ pos, gx)

let issue_singly w members =
  List.iter
    (fun m ->
      m.issue () ;
      count_issued w 1)
    members

let issue_fused w key members first =
  match fused_kernel w key members with
  | None -> issue_singly w members
  | Some k -> (
      let args, gx = fused_args w members in
      match first.issue_fused k args {first.grid with x = gx} with
      | () ->
          count_issued w 1 ;
          w.stats <- {w.stats with fused_groups = w.stats.fused_groups + 1}
      | exception _ ->
          (* Codegen, compilation or launch of the fused kernel failed before
             it ran: issue the members as submitted and never fuse this
             sequence again (FR-013). *)
          Hashtbl.remove w.cache key ;
          refuse w key ;
          issue_singly w members)

(* Under [w.lock]. *)
let issue_group w g =
  match members g with
  | [] -> ()
  | [m] -> issue_singly w [m]
  | first :: _ as ms -> issue_fused w (key_of g) ms first

let set_counts w =
  w.n_groups <- List.length w.groups ;
  w.n_pending <- List.fold_left (fun n g -> n + g.size) 0 w.groups ;
  w.set_pending w.n_pending

(* Under [w.lock]: issue [groups] in order. Every group is issued even if an
   earlier one raises; the first exception is re-raised at the end (FR-006). *)
let issue_all w groups =
  let first_exn = ref None in
  w.guard (fun () ->
      List.iter
        (fun g ->
          match issue_group w g with
          | () -> ()
          | exception e ->
              if !first_exn = None then
                first_exn := Some (e, Printexc.get_raw_backtrace ()))
        groups) ;
  match !first_exn with
  | Some (e, bt) -> Printexc.raise_with_backtrace e bt
  | None -> ()

(** {1 Public operations} *)

(** Issue every pending launch, in order. *)
let flush (w : _ t) : unit =
  Mutex.protect w.lock (fun () ->
      if w.groups <> [] then begin
        let groups = w.groups in
        w.groups <- [] ;
        set_counts w ;
        issue_all w groups
      end)

(** Defer [l]. Its IR must have been registered with {!ir_info}. *)
let submit (w : 'arg t) (l : 'arg launch) : unit =
  Mutex.protect w.lock (fun () ->
      w.stats <- {w.stats with submitted = w.stats.submitted + 1} ;
      (match target w l with
      | Some g ->
          g.rev_members <- l :: g.rev_members ;
          g.size <- g.size + 1
      | None ->
          w.groups <-
            w.groups @ [{g_device = l.device_id; rev_members = [l]; size = 1}]) ;
      let overflow =
        if List.length w.groups > !max_groups then
          match w.groups with
          | oldest :: rest ->
              w.groups <- rest ;
              [oldest]
          | [] -> []
        else []
      in
      set_counts w ;
      issue_all w overflow)

(** [true] when a pending launch on [device_id] binds vector [buf]. *)
let touches (w : _ t) ~device_id ~buf : bool =
  let probe = {buf; read = true; write = true} in
  Mutex.protect w.lock (fun () ->
      List.exists
        (fun g ->
          g.g_device = device_id
          && List.exists
               (fun m -> List.exists (accesses_conflict probe) m.accesses)
               g.rev_members)
        w.groups)

(** Count a launch that bypassed the window (issued immediately). *)
let count_immediate (w : _ t) : unit =
  Mutex.protect w.lock (fun () ->
      w.stats <-
        {
          w.stats with
          submitted = w.stats.submitted + 1;
          issued = w.stats.issued + 1;
        })

let stats (w : _ t) : stats = Mutex.protect w.lock (fun () -> w.stats)

let reset_stats (w : _ t) : unit =
  Mutex.protect w.lock (fun () -> w.stats <- zero_stats)

(** Forget every fused kernel and refusal (the next fused group rebuilds). *)
let clear_cache (w : _ t) : unit =
  Mutex.protect w.lock (fun () ->
      Hashtbl.reset w.cache ;
      Hashtbl.reset w.refused)

let set_enabled (w : _ t) (b : bool) : unit =
  flush w ;
  Atomic.set enabled b

let set_backend_enabled (w : _ t) ~(backend : string) (b : bool) : unit =
  flush w ;
  let rec update () =
    let old = Atomic.get backends in
    let rest = List.filter (fun x -> x <> backend) old in
    let next = if b then backend :: rest else rest in
    if not (Atomic.compare_and_set backends old next) then update ()
  in
  update ()
