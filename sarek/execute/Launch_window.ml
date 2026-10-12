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
  shared_bytes : int option;
      (** shared memory the kernel declares when it can be fused horizontally
          ({!ir_info}[.fusable]); [None] when it cannot *)
  max_grid_x : int;  (** the device's grid-size limit in x *)
  max_shared : int;  (** the device's shared memory per work-group *)
  max_arg_slots : int;
      (** the backend's kernel argument slots ({!arg_slot_limit}); [max_int]
          when unbounded *)
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

type ir_info = {
  ir_id : int;
  param_accesses : (string * Kernel_access.access) list;
  fusable : (int, string) result;  (** shared bytes, or why not fusable *)
}

(** {1 Window state} *)

type 'arg group = {
  g_device : int;
  mutable rev_members : 'arg launch list;  (** newest first *)
  mutable size : int;
  mutable grid_x : int;  (** sum of the members' grid x *)
  mutable shared : int;  (** sum of the members' shared memory *)
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
  mutable epilogue : 'arg t -> 'arg launch -> bool;
      (** called under [lock] before grouping: [true] when the launch was merged
          into a pending launch as its epilogue ({!Launch_epilogue}) *)
  int32_arg : int32 -> 'arg;
  arg_buf : 'arg -> int option;  (** the vector id a vector argument binds *)
  arg_int32 : 'arg -> int32 option;  (** the value of an int32 scalar *)
  set_pending : int -> unit;
  guard : (unit -> unit) -> unit;
  info_key : ir_info Ir_memo.key;  (** this window's slot of each IR's facts *)
  mutable next_ir_id : int;
  cache : (int list, kernel) Hashtbl.t;  (** member ir_ids -> fused kernel *)
  refused : (int list, unit) Hashtbl.t;  (** member sequences never fused *)
  c_submitted : int Atomic.t;
  c_issued : int Atomic.t;
  c_groups : int Atomic.t;
  c_built : int Atomic.t;
      (** never reset: it also names fused kernels without a content name *)
  c_refused : int Atomic.t;
}

(** [create ~int32_arg ~set_pending ~guard] makes an empty window. [int32_arg]
    builds a scalar argument for the fused kernel's position parameters.
    [set_pending] is told the number of pending launches after each change.
    [guard f] runs [f] (the issuing of pending groups) with the synchronisation
    hooks disabled on this domain. Each window has its own slot in the
    {!Ir_memo} record of every IR it sees; the slot lives as long as that IR,
    not as long as the window. *)
let create ~int32_arg ~arg_buf ~arg_int32 ~set_pending ~guard : 'arg t =
  {
    lock = Mutex.create ();
    groups = [];
    epilogue = (fun _ _ -> false);
    int32_arg;
    arg_buf;
    arg_int32;
    set_pending;
    guard;
    info_key = Ir_memo.new_key ();
    next_ir_id = 0;
    cache = Hashtbl.create 64;
    refused = Hashtbl.create 8;
    c_submitted = Atomic.make 0;
    c_issued = Atomic.make 0;
    c_groups = Atomic.make 0;
    c_built = Atomic.make 0;
    c_refused = Atomic.make 0;
  }

(** {1 Configuration} *)

let max_groups = ref 8

let max_members = ref 8

let max_cached = 4096

let env_enabled () =
  match Sys.getenv_opt "SAREK_LAUNCH_WINDOW" with
  | Some ("0" | "false" | "off" | "no") -> false
  | _ -> true

let enabled = Atomic.make (env_enabled ())

(** Backends deferred by default (specs/launch-window.md); Native, Interpreter
    and WebGPU stay immediate. {!set_backend_enabled} opts in or out. *)
let default_backends =
  ["HIP"; "OpenCL"; "Vulkan"; "CUDA/PTX"; "CUDA/C"; "Metal"]

(* Read on every launch: an immutable list behind an atomic, no lock. *)
let backends : string list Atomic.t = Atomic.make default_backends

include Launch_limits

(** [true] when launches on [backend] are deferred. *)
let defers (backend : string) : bool =
  Atomic.get enabled
  && List.mem backend (Atomic.get backends)
  && not (Sarek_policy.restricted Sarek_policy.Launch_window)

(** {1 Per-IR facts} *)

(** As {!ir_info}, for a caller that already holds [w.lock]. *)
let ir_info_locked (w : _ t) (ir : kernel) : ir_info =
  let facts = Ir_memo.facts ir in
  match Ir_memo.find facts w.info_key with
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
      Ir_memo.set facts w.info_key i ;
      i

let ir_info (w : _ t) (ir : kernel) : ir_info =
  Mutex.protect w.lock (fun () -> ir_info_locked w ir)

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
let joinable (l : _ launch) =
  Option.is_some l.shared_bytes && l.shared_mem = 0 && l.grid.x > 0

(** Slots one launch's arguments take in a kernel signature: a vector is a
    buffer plus its length, a scalar one slot. *)
let arg_slots w (l : _ launch) =
  List.fold_left
    (fun n a -> n + if Option.is_some (w.arg_buf a) then 2 else 1)
    0
    l.args

(** Slots of the signature of the fused kernel of [members]: every member's
    arguments plus (offset, grid size) per member. *)
let fused_slots w members =
  List.fold_left (fun n m -> n + arg_slots w m + 2) 0 members

(** [true] when [members], issued as one group, stay within the backend's
    argument slots: the fused signature for two or more, the launch's own
    signature for one. *)
let slots_ok w members =
  match members with
  | [] -> true
  | [m] -> arg_slots w m <= m.max_arg_slots
  | m :: _ -> fused_slots w members <= m.max_arg_slots

let params_of (l : _ launch) = l.ir.kern_params

(** The group's push-constant block fits the backend ({!push_bytes_ok}). *)
let push_ok members =
  match members with
  | [] -> true
  | m :: _ -> push_bytes_ok ~backend:m.backend (List.map params_of members)

(** The backend can issue [members] as one group: {!slots_ok} and {!push_ok}. *)
let signature_ok w members = slots_ok w members && push_ok members

(* The fused grid and shared memory must stay within the device's limits, work-
   group offsets within int32 (they are passed as int32 arguments), and the
   fused signature within the backend's argument slots and push-constant
   bytes. *)
let fits w g (l : _ launch) =
  let gx = g.grid_x + l.grid.x in
  gx <= l.max_grid_x
  && gx <= Int32.to_int Int32.max_int
  && g.shared + Option.value ~default:0 l.shared_bytes <= l.max_shared
  && fused_slots w (l :: g.rev_members) <= l.max_arg_slots
  && push_ok (List.rev (l :: g.rev_members))

(* Under [w.lock]. [l] may join [g] as its last member. Every member of a
   group of two or more is [joinable] with the same block and grid y/z, so
   comparing with the newest member is enough. *)
let compatible w g (l : _ launch) =
  match g.rev_members with
  | [] -> false
  | newest :: _ ->
      g.g_device = l.device_id && g.size < !max_members && joinable l
      && joinable newest && fits w g l
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

let bump counter = Atomic.incr counter

let refuse w key =
  Hashtbl.replace w.refused key () ;
  bump w.c_refused

let fused_kernel w key members =
  match Hashtbl.find_opt w.cache key with
  | Some k -> Some k
  | None -> (
      match
        let k =
          Fusion_horizontal.fuse ~name:"" (List.map (fun m -> m.ir) members)
        in
        match Fusion_horizontal.content_named ~prefix:"sarek_hfused_" k with
        | Some k -> k
        | None ->
            {
              k with
              kern_name =
                Printf.sprintf "sarek_hfused_%d" (Atomic.get w.c_built);
            }
      with
      | k ->
          (* Its budget is the meet over the members' (Sarek_policy). *)
          Sarek_policy.register_fused
            ~name:k.kern_name
            ~members:(List.map (fun m -> m.ir.kern_name) members) ;
          bump w.c_built ;
          (* Bounded: launch sequences of IRs built at run time would
             otherwise accumulate. A dropped entry is rebuilt on demand. *)
          if Hashtbl.length w.cache >= max_cached then Hashtbl.reset w.cache ;
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

(* Exceptions a fallback must not swallow. *)
let fatal = function Out_of_memory | Stack_overflow -> true | _ -> false

(* Run every [f x] even if some raise; re-raise the first exception after the
   last one has run (FR-006). *)
let each f xs =
  let first = ref None in
  List.iter
    (fun x ->
      match f x with
      | () -> ()
      | exception e when not (fatal e) ->
          if !first = None then first := Some (e, Printexc.get_raw_backtrace ()))
    xs ;
  Option.iter (fun (e, bt) -> Printexc.raise_with_backtrace e bt) !first

let issue_singly w members =
  each
    (fun m ->
      m.issue () ;
      bump w.c_issued)
    members

let issue_fused w key members first =
  match fused_kernel w key members with
  | None -> issue_singly w members
  | Some k -> (
      let args, gx = fused_args w members in
      match first.issue_fused k args {first.grid with x = gx} with
      | () ->
          bump w.c_issued ;
          bump w.c_groups
      | exception e when not (fatal e) ->
          (* Codegen, compilation or the driver's launch call of the fused
             kernel failed. Backends raise before enqueueing, so nothing of it
             ran: issue the members as submitted and never fuse this sequence
             again (FR-013). *)
          Hashtbl.remove w.cache key ;
          refuse w key ;
          issue_singly w members)

(* Under [w.lock]. *)
let issue_group w g =
  match members g with
  | [] -> ()
  | [m] -> issue_singly w [m]
  | first :: _ as ms -> issue_fused w (key_of g) ms first

let count_pending groups = List.fold_left (fun n g -> n + g.size) 0 groups

(* Under [w.lock]: issue [groups] in order, with the synchronisation hooks
   disabled on this domain. Every group is issued even if an earlier one
   raises; the first exception is re-raised at the end (FR-006). *)
let issue_all w groups = w.guard (fun () -> each (issue_group w) groups)

(** {1 Public operations} *)

(** Issue every pending launch, in order. The pending count stays non-zero until
    they are all issued, so a synchronisation point reached meanwhile on another
    domain waits for this flush (on [w.lock]) instead of going ahead (FR-007).
*)
let flush (w : _ t) : unit =
  Mutex.protect w.lock (fun () ->
      if w.groups <> [] then begin
        let groups = w.groups in
        w.groups <- [] ;
        Fun.protect
          ~finally:(fun () -> w.set_pending (count_pending w.groups))
          (fun () -> issue_all w groups)
      end)

(** Defer [l]. When [l] needs a new group and the window is full, the oldest
    group is issued first; if that raises, [l] is not submitted and the
    exception propagates (the launching call fails before its own launch is
    accepted, so its vectors' state is unchanged). *)
let submit (w : 'arg t) (l : 'arg launch) : unit =
  Mutex.protect w.lock (fun () ->
      (if not (w.epilogue w l) then
         match target w l with
         | Some g ->
             g.rev_members <- l :: g.rev_members ;
             g.size <- g.size + 1 ;
             g.grid_x <- g.grid_x + l.grid.x ;
             g.shared <- g.shared + Option.value ~default:0 l.shared_bytes
         | None ->
             (match w.groups with
             | oldest :: rest when List.length w.groups >= !max_groups ->
                 w.groups <- rest ;
                 Fun.protect
                   ~finally:(fun () -> w.set_pending (count_pending w.groups))
                   (fun () -> issue_all w [oldest])
             | _ -> ()) ;
             w.groups <-
               w.groups
               @ [
                   {
                     g_device = l.device_id;
                     rev_members = [l];
                     size = 1;
                     grid_x = l.grid.x;
                     shared = Option.value ~default:0 l.shared_bytes;
                   };
                 ]) ;
      bump w.c_submitted ;
      w.set_pending (count_pending w.groups))

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
  bump w.c_submitted ;
  bump w.c_issued

let stats (w : _ t) : stats =
  {
    submitted = Atomic.get w.c_submitted;
    issued = Atomic.get w.c_issued;
    fused_groups = Atomic.get w.c_groups;
    fused_built = Atomic.get w.c_built;
    fused_refused = Atomic.get w.c_refused;
  }

let reset_stats (w : _ t) : unit =
  List.iter
    (fun c -> Atomic.set c 0)
    [w.c_submitted; w.c_issued; w.c_groups; w.c_refused]

(** Forget every fused kernel and refusal (the next fused group rebuilds), and
    run [also] in the same critical section. *)
let clear_cache ?(also = fun () -> ()) (w : _ t) : unit =
  Mutex.protect w.lock (fun () ->
      Hashtbl.reset w.cache ;
      Hashtbl.reset w.refused ;
      also ())

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
