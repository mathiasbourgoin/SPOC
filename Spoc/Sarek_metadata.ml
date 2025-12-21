(** Runtime registry for Sarek metadata embedded in compiled modules. *)

let type_blobs : string list ref = ref []

let module_blobs : string list ref = ref []

let register_type_blob blob = type_blobs := blob :: !type_blobs

let register_module_blob blob = module_blobs := blob :: !module_blobs

let get_type_blobs () = List.rev !type_blobs

let get_module_blobs () = List.rev !module_blobs

let clear () =
  type_blobs := [] ;
  module_blobs := []
