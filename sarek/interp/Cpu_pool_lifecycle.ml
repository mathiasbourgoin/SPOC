(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* See Cpu_pool_lifecycle.mli. [registry_lock] and a slot's mutex are
   never held together. *)

(** {1 Configuration} *)

let env_positive_int name =
  match Sys.getenv_opt name with
  | None -> None
  | Some s -> (
      match int_of_string_opt (String.trim s) with
      | Some n when n > 0 -> Some n
      | _ -> None)

let max_domains_override : int option Atomic.t = Atomic.make None

let idle_timeout_override : float option Atomic.t = Atomic.make None

let set_max_domains n =
  if n < 1 then invalid_arg "Cpu_pool_lifecycle.set_max_domains: n < 1" ;
  Atomic.set max_domains_override (Some n)

let set_idle_timeout s =
  if not (s > 0.) then
    invalid_arg "Cpu_pool_lifecycle.set_idle_timeout: not positive" ;
  Atomic.set idle_timeout_override (Some s)

let default_idle_timeout = 1.0

let idle_timeout () =
  match Atomic.get idle_timeout_override with
  | Some s -> s
  | None -> (
      match env_positive_int "SAREK_CPU_POOL_IDLE_MS" with
      | Some ms -> float_of_int ms /. 1000.
      | None -> default_idle_timeout)

let capped n =
  let cap =
    match Atomic.get max_domains_override with
    | Some c -> Some c
    | None -> env_positive_int "SAREK_CPU_POOL_MAX_DOMAINS"
  in
  max 1 (match cap with Some c -> min c n | None -> n)

let domain_count ~default () =
  let wanted =
    match env_positive_int "SAREK_DOMAIN_COUNT" with
    | Some n -> n
    | None -> ( try Domain.recommended_domain_count () with _ -> default)
  in
  capped wanted

(** {1 Registry and reaper} *)

let registry_lock = Mutex.create ()

(* Each returns [true] iff its resource is still live after the poll. *)
let reapables : (now:float -> idle_after:float -> bool) list ref = ref []

let reaper_running = ref false

(* Bumped by [note_live]: a reaper that saw nothing live keeps going if a
   resource became live meanwhile. *)
let activity = ref 0

let register_reapable f =
  Mutex.protect registry_lock (fun () -> reapables := f :: !reapables)

let sweep ~idle_after =
  let all = Mutex.protect registry_lock (fun () -> !reapables) in
  let now = Unix.gettimeofday () in
  List.fold_left
    (fun live poll ->
      let still = poll ~now ~idle_after in
      still || live)
    false
    all

let rec reaper_loop () =
  let seen = Mutex.protect registry_lock (fun () -> !activity) in
  let timeout = idle_timeout () in
  Unix.sleepf (Float.max 0.01 (timeout /. 4.)) ;
  let live = sweep ~idle_after:timeout in
  let stop =
    Mutex.protect registry_lock (fun () ->
        if live || !activity <> seen then false
        else begin
          reaper_running := false ;
          true
        end)
  in
  if not stop then reaper_loop ()

(* A failed spawn (the process is at OCaml's domain limit) leaves the pools
   alive, as they were before idle retirement existed; the next pool creation
   tries again. *)
let note_live () =
  Mutex.protect registry_lock (fun () ->
      incr activity ;
      if not !reaper_running then
        match Domain.spawn reaper_loop with
        | (_ : unit Domain.t) -> reaper_running := true
        | exception Failure _ -> ())

let retire_idle_now () = ignore (sweep ~idle_after:0.)

let any_live () =
  Mutex.protect registry_lock (fun () -> !reaper_running)
  || sweep ~idle_after:infinity

let () = at_exit retire_idle_now

(** {1 Pool slots} *)

type 'p slot = {
  create : unit -> 'p;
  retire : 'p -> unit;
  mutex : Mutex.t;
  mutable pool : 'p option;
  mutable users : int;  (** Launches holding [pool]. *)
  mutable last_used : float;
}

(* The join happens under the slot mutex, so a launch cannot create a second
   pool while the first one's workers still count against the domain limit. *)
let retire_if_idle s ~now ~idle_after =
  Mutex.protect s.mutex (fun () ->
      match s.pool with
      | None -> false
      | Some p ->
          if s.users = 0 && now -. s.last_used >= idle_after then begin
            s.pool <- None ;
            s.retire p ;
            false
          end
          else true)

let slot ~create ~retire =
  let s =
    {
      create;
      retire;
      mutex = Mutex.create ();
      pool = None;
      users = 0;
      last_used = 0.;
    }
  in
  register_reapable (retire_if_idle s) ;
  s

let acquire s =
  let created = ref false in
  let p =
    Mutex.protect s.mutex (fun () ->
        let p =
          match s.pool with
          | Some p -> p
          | None ->
              let p = s.create () in
              s.pool <- Some p ;
              created := true ;
              p
        in
        s.users <- s.users + 1 ;
        p)
  in
  if !created then note_live () ;
  p

let release s =
  Mutex.protect s.mutex (fun () ->
      s.users <- s.users - 1 ;
      s.last_used <- Unix.gettimeofday ())

let with_pool s f =
  let p = acquire s in
  Fun.protect ~finally:(fun () -> release s) (fun () -> f p)
