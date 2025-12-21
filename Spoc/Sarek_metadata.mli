(** Runtime registry for Sarek metadata embedded in compiled modules. The PPX
    stores marshalled representations of Sarek AST values as opaque strings. At
    runtime (and during PPX execution, since the PPX links against [spoc]),
    these blobs can be registered and later retrieved by the kernel expander. *)

(** Register a marshalled [Sarek_ast.type_decl] blob. *)
val register_type_blob : string -> unit

(** Register a marshalled [Sarek_ast.module_item] blob. *)
val register_module_blob : string -> unit

(** Return all registered type blobs (most recent last). *)
val get_type_blobs : unit -> string list

(** Return all registered module item blobs (most recent last). *)
val get_module_blobs : unit -> string list

(** Clear all registries (useful in tests). *)
val clear : unit -> unit
