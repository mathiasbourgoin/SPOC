(******************************************************************************
 * Sarek Framework - Plugin Interface (Stub)
 *
 * This is a placeholder for the future plugin system.
 * Full implementation pending ctypes integration.
 ******************************************************************************)

(** 3D dimensions for grid and block *)
type dims = {x : int; y : int; z : int}

let dims_1d x = {x; y = 1; z = 1}

let dims_2d x y = {x; y; z = 1}

let dims_3d x y z = {x; y; z}

(** Minimal framework signature - will be expanded later *)
module type S = sig
  val name : string

  val version : int * int * int

  val is_available : unit -> bool
end
