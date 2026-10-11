(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Debug Logging
 *
 * Configurable logging by component and level.
 * Components can be enabled/disabled at runtime.
 ******************************************************************************)

(** Log levels *)
type level = Debug | Info | Warn | Error

(** Components that can be logged *)
type component = Transfer | Kernel | Device | Memory | Execute | All

(** Level to int for comparison *)
let level_to_int = function Debug -> 0 | Info -> 1 | Warn -> 2 | Error -> 3

(* Messages at or above this level pass the level gate. Read on every log
   call, before anything else, so a disabled call is one int compare. *)
let threshold = Atomic.make (level_to_int Warn)

let bit = function
  | Transfer -> 1
  | Kernel -> 2
  | Device -> 4
  | Memory -> 8
  | Execute -> 16
  | All -> 32

let all_bits = 63

(* Enabled components, one bit each. The [All] bit enables every component;
   enabling or disabling [All] sets or clears every bit. *)
let mask = Atomic.make 0

let rec update_mask f =
  let old = Atomic.get mask in
  if not (Atomic.compare_and_set mask old (f old)) then update_mask f

(** Enable a component *)
let enable comp =
  let b = if comp = All then all_bits else bit comp in
  update_mask (fun m -> m lor b)

(** Disable a component *)
let disable comp =
  let b = if comp = All then all_bits else bit comp in
  update_mask (fun m -> m land lnot b)

(** Set log level *)
let set_level lvl = Atomic.set threshold (level_to_int lvl)

let component_of_string = function
  | "transfer" -> Some Transfer
  | "kernel" -> Some Kernel
  | "device" -> Some Device
  | "memory" -> Some Memory
  | "execute" -> Some Execute
  | "all" | "*" -> Some All
  | _ -> None

let level_of_string = function
  | "debug" -> Some Debug
  | "info" -> Some Info
  | "warn" -> Some Warn
  | "error" -> Some Error
  | _ -> None

(** Enable from environment variable SAREK_DEBUG=transfer,kernel,... Use "*" or
    "all" to enable all components. A level ([debug], [info], [warn], [error])
    sets the level; naming a component without a level sets it to [debug], so
    [SAREK_DEBUG=kernel] prints the kernel component's debug messages. Unknown
    words are ignored. *)
let init_from_env () =
  match Sys.getenv_opt "SAREK_DEBUG" with
  | None -> ()
  | Some s -> (
      let words =
        List.map
          (fun p -> String.lowercase_ascii (String.trim p))
          (String.split_on_char ',' s)
      in
      let comps = List.filter_map component_of_string words in
      List.iter enable comps ;
      let levels = List.filter_map level_of_string words in
      match List.rev levels with
      | last :: _ -> set_level last
      | [] -> if comps <> [] then set_level Debug)

(** Initialize on module load *)
let () = init_from_env ()

(** Check if component is enabled *)
let is_enabled comp = Atomic.get mask land (bit All lor bit comp) <> 0

(** [would_log level comp]: a message at [level] on [comp] would be printed.
    Guard a call whose arguments cost something to compute with it. *)
let would_log level comp =
  level_to_int level >= Atomic.get threshold && is_enabled comp

(** Level to string *)
let level_to_string = function
  | Debug -> "DEBUG"
  | Info -> "INFO"
  | Warn -> "WARN"
  | Error -> "ERROR"

(** Component to string *)
let component_to_string = function
  | Transfer -> "Transfer"
  | Kernel -> "Kernel"
  | Device -> "Device"
  | Memory -> "Memory"
  | Execute -> "Execute"
  | All -> "All"

(** Core logging function - printf style *)
let logf level comp fmt =
  if would_log level comp then begin
    Printf.printf "[%s][%s] " (level_to_string level) (component_to_string comp) ;
    Printf.kfprintf (fun oc -> Printf.fprintf oc "\n%!") stdout fmt
  end
  else Printf.ifprintf stdout fmt

(** String-based logging (simpler, no format issues) *)
let log level comp msg =
  if would_log level comp then
    Printf.printf
      "[%s][%s] %s\n%!"
      (level_to_string level)
      (component_to_string comp)
      msg

(** [log_any_component level msg]: [msg] tagged [All], gated by the level only.
    For callers with their own switch ([Spoc_core.Error.log]), so they need not
    enable every component. *)
let log_any_component level msg =
  if level_to_int level >= Atomic.get threshold then
    Printf.printf
      "[%s][%s] %s\n%!"
      (level_to_string level)
      (component_to_string All)
      msg

(** Convenience functions - string based *)
let debug comp msg = log Debug comp msg

let info comp msg = log Info comp msg

let warn comp msg = log Warn comp msg

let error comp msg = log Error comp msg

(** Convenience functions - printf style *)
let debugf comp fmt = logf Debug comp fmt

let infof comp fmt = logf Info comp fmt

let warnf comp fmt = logf Warn comp fmt

let errorf comp fmt = logf Error comp fmt
