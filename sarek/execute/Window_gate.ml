(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Whether the launch window defers a launch (specs/launch-window.md): the
   switches, and the one function every caller derives its answer from, so that
   the launch path, [Execute.Window.defers_device] and the explain report
   cannot drift apart. Re-exported by {!Launch_window}. *)

(** Why a launch is not deferred, in the order the reasons are checked. *)
type refusal =
  | Switched_off  (** [Execute.Window.set_enabled false] *)
  | Restricted  (** [SAREK_OPT_DISABLE=window] *)
  | Cpu_device
      (** host-pointer buffers alias host memory, and distinct vectors can alias
          each other through it, so buffer identity is no longer the vector's *)
  | Not_on of string  (** the backend (framework name) is not in {!backends} *)
  | Streamed  (** a launch on a stream *)
  | Soa_params  (** structure-of-arrays arguments *)

let enabled = Atomic.make true

(** Backends deferred by default (specs/launch-window.md); Native, Interpreter
    and WebGPU stay immediate. [set_backend_enabled] opts in or out. *)
let default_backends =
  ["HIP"; "OpenCL"; "Vulkan"; "CUDA/PTX"; "CUDA/C"; "Metal"]

(* Read on every launch: an immutable list behind an atomic, no lock. *)
let backends : string list Atomic.t = Atomic.make default_backends

(** [refusal ~framework ~is_cpu ~streamed ~soa_params]: [None] when a JIT launch
    on a device of [framework] is deferred. *)
let refusal ~framework ~is_cpu ~streamed ~soa_params =
  if not (Atomic.get enabled) then Some Switched_off
  else if Sarek_policy.restricted Sarek_policy.Launch_window then
    Some Restricted
  else if is_cpu then Some Cpu_device
  else if not (List.mem framework (Atomic.get backends)) then
    Some (Not_on framework)
  else if streamed then Some Streamed
  else if soa_params <> [] then Some Soa_params
  else None

(** [true] when launches on [backend] are deferred. *)
let defers (backend : string) : bool =
  refusal ~framework:backend ~is_cpu:false ~streamed:false ~soa_params:[] = None

(** [true] when JIT launches on a device of [framework] are deferred: its
    backend's window is on and it is not a CPU device. *)
let defers_device ~framework ~is_cpu =
  refusal ~framework ~is_cpu ~streamed:false ~soa_params:[] = None
