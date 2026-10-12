(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX ISA versions: which one an SM needs and which one a driver accepts.

    [Cuda_api.Kernel.load_from_ptx] rewrites [.target] to the device's own SM,
    so [.version] must follow: PTX ISA 8.0 does not know sm_120 (issue #506).
    Pure: no libcuda, no device. *)

(** A PTX ISA version, [(major, minor)]; ordered by structural comparison. *)
type t = int * int

let to_string ((major, minor) : t) = Printf.sprintf "%d.%d" major minor

(** [of_string "8.7"] is [Some (8, 7)]; [None] on anything else. *)
let of_string s : t option =
  match String.split_on_char '.' (String.trim s) with
  | [a; b] -> (
      match (int_of_string_opt a, int_of_string_opt b) with
      | Some a, Some b when a >= 0 && b >= 0 -> Some (a, b)
      | _ -> None)
  | _ -> None

(** The lowest PTX ISA version that admits [.target sm_<major><minor>]. sm_88
    (also what ZLUDA reports), sm_110 and SMs not listed get 9.0, the newest
    version here; ZLUDA parses it. Pre-Ampere SMs get 7.0. *)
let required_version ~major ~minor : t =
  match (major, minor) with
  | 8, 0 -> (7, 0)
  | 8, 6 -> (7, 1)
  | 8, 7 -> (7, 4)
  | 8, 9 | 9, 0 -> (7, 8)
  | 10, 0 | 10, 1 -> (8, 6)
  | 12, 0 -> (8, 7)
  | 10, 3 | 12, 1 -> (8, 8)
  | _ -> if major < 8 then (7, 0) else (9, 0)

(** [driver_max_version (major, minor)] is the highest PTX ISA the driver's CUDA
    version (from [cuDriverGetVersion], e.g. 13.0) accepts, per the PTX ISA
    release notes. [None] when the driver is newer than this table (or older
    than CUDA 11.8): unknown, so not gated here, and the driver's own error
    stands. *)
let driver_max_version ((major, minor) : int * int) : t option =
  let table =
    [
      ((11, 8), (7, 8));
      ((12, 0), (8, 0));
      ((12, 1), (8, 1));
      ((12, 2), (8, 2));
      ((12, 3), (8, 3));
      ((12, 4), (8, 4));
      ((12, 5), (8, 5));
      ((12, 6), (8, 5));
      ((12, 8), (8, 7));
      ((12, 9), (8, 8));
    ]
  in
  if (major, minor) >= (13, 0) || (major, minor) < (11, 8) then None
  else
    List.fold_left
      (fun acc (cuda, isa) -> if cuda <= (major, minor) then Some isa else acc)
      None
      table

(** [check_driver ~driver ~required] is [Error msg] naming both versions when
    the driver is known to accept less than [required]. *)
let check_driver ~driver:(dmaj, dmin) ~required : (unit, string) result =
  match driver_max_version (dmaj, dmin) with
  | Some max when max < required ->
      Error
        (Printf.sprintf
           "PTX ISA %s is required, but the CUDA driver (CUDA %d.%d) supports \
            PTX ISA up to %s; update the NVIDIA driver"
           (to_string required)
           dmaj
           dmin
           (to_string max))
  | _ -> Ok ()

let starts_with ~prefix line =
  String.length line >= String.length prefix
  && String.sub line 0 (String.length prefix) = prefix

(** Rewrite [.target] to [sm_<major><minor>] and raise [.version] to at least
    {!required_version} for that SM. A [.version] already at or above it, or one
    that does not parse, is left alone: the version is never lowered. *)
let with_sm_target ~major ~minor ptx =
  let required = required_version ~major ~minor in
  let version_prefix = ".version " in
  let rewrite line =
    if starts_with ~prefix:".target " line then
      Printf.sprintf ".target sm_%d%d" major minor
    else if starts_with ~prefix:version_prefix line then
      let rest =
        String.sub
          line
          (String.length version_prefix)
          (String.length line - String.length version_prefix)
      in
      match of_string rest with
      | Some present when present >= required -> line
      | Some _ -> version_prefix ^ to_string required
      | None -> line
    else line
  in
  String.split_on_char '\n' ptx |> List.map rewrite |> String.concat "\n"

(** The [.version] the PTX declares, [None] if absent or unparseable. *)
let declared_version ptx : t option =
  let prefix = ".version " in
  String.split_on_char '\n' ptx
  |> List.find_map (fun line ->
      if starts_with ~prefix line then
        of_string
          (String.sub
             line
             (String.length prefix)
             (String.length line - String.length prefix))
      else None)
