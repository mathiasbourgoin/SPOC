val profile_vect : (int64, Bigarray.int64_elt) Spoc.Vector.vector ref

val space : int -> string

val indent : int -> string

module type CodeGenerator = sig
  val target_name : string

  val global_function : string

  val device_function : string

  val host_function : string

  val global_parameter : string

  val global_variable : string

  val local_variable : string

  val shared_variable : string

  val kern_start : string

  val kern_end : string

  val parse_intrinsics : Sarek.Kirc_Ast.intrinsics -> string

  val default_parser : bool

  val parse_fun :
    int -> Sarek.Kirc_Ast.k_ext -> string -> string -> Spoc.Devices.device -> string

  val parse : int -> Sarek.Kirc_Ast.k_ext -> Spoc.Devices.device -> string
end

module Generator : functor (M : CodeGenerator) -> sig
  val global_funs : (Sarek.Kirc_Ast.k_ext, string * string) Hashtbl.t

  val return_v : (string * string) ref

  val global_fun_idx : int ref

  val protos : string list ref

  val parse_fun :
    ?profile:bool ->
    int ->
    Sarek.Kirc_Ast.k_ext ->
    string ->
    string ->
    Spoc.Devices.device ->
    string

  val profiler_counter : int ref

  val get_profile_counter : unit -> int

  val parse :
    ?profile:bool -> int -> Sarek.Kirc_Ast.k_ext -> Spoc.Devices.device -> string

  val parse_int :
    ?profile:bool -> int -> Sarek.Kirc_Ast.k_ext -> Spoc.Devices.device -> string

  val parse_float :
    ?profile:bool -> int -> Sarek.Kirc_Ast.k_ext -> Spoc.Devices.device -> string

  val parse_vect : Sarek.Kirc_Ast.kvect -> int
end
