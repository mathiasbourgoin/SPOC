(** Lightweight external kernel declaration transformer. Replaces
    [%kernel external ... = "file" "symbol"] structure items with value bindings
    that carry the external kernel metadata (file, symbol, params). This is a
    simplified port of the legacy SPOC external kernel PPX. *)

open Ppxlib

let loc = Location.none

(* Parse the function type of an external kernel and build its parameter list
   as Spoc.Kernel.param values. *)
let rec parse_typ t =
  let loc = t.ptyp_loc in
  let dummy = [%expr Obj.magic ()] in
  match t with
  | [%type: char] -> [%expr Spoc.Kernel.Char [%e dummy]]
  | [%type: int] | [%type: int32] -> [%expr Spoc.Kernel.Int32 [%e dummy]]
  | [%type: int64] -> [%expr Spoc.Kernel.Int64 [%e dummy]]
  | [%type: float] | [%type: float32] -> [%expr Spoc.Kernel.Float32 [%e dummy]]
  | [%type: float64] -> [%expr Spoc.Kernel.float64 [%e dummy]]
  | [%type: Spoc.Vector.vint32] ->
      [%expr Spoc.Kernel.VInt32 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.vint64] ->
      [%expr Spoc.Kernel.VInt64 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.vfloat32] ->
      [%expr Spoc.Kernel.VFloat32 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.vfloat64] ->
      [%expr Spoc.Kernel.VFloat64 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.vchar] ->
      [%expr Spoc.Kernel.VChar (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.localvfloat32] ->
      [%expr Spoc.Kernel.LocalFloat32 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.localvfloat64] ->
      [%expr Spoc.Kernel.LocalFloat64 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.vcomplex32] ->
      [%expr Spoc.Kernel.VComplex32 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.vcomplex64] ->
      [%expr Spoc.Kernel.VComplex64 (Spoc.Kernel.relax_vector [%e dummy])]
  | [%type: Spoc.Vector.vbool] | [%type: [%t? _] Spoc.Vector.vcustom] ->
      [%expr Spoc.Kernel.VCustom (Spoc.Kernel.relax_vector [%e dummy])]
  | {ptyp_desc = Ptyp_constr ({txt = Lident _; _}, [t2]); _} -> parse_typ t2
  | [%type: [%t? _]] -> dummy

let rec flatten_arrow acc = function
  | {ptyp_desc = Ptyp_arrow (Nolabel, t1, t2); _} ->
      flatten_arrow (t1 :: acc) t2
  | t -> (List.rev acc, t)

let build_params types =
  let exprs = List.map parse_typ types in
  Ast_builder.Default.elist ~loc exprs

let transform_external_structure_item item =
  match item.pstr_desc with
  | Pstr_extension
      ( ( {txt = "kernel"; loc = ext_loc},
          PStr [{pstr_desc = Pstr_primitive p; _}] ),
        _attrs ) -> (
      match p.pval_prim with
      | [file_name; fun_name] ->
          let args, _ret = flatten_arrow [] p.pval_type in
          let params = build_params args in
          let obj_expr =
            [%expr
              object
                method file =
                  [%e Ast_builder.Default.estring ~loc:ext_loc file_name]

                method kern =
                  [%e Ast_builder.Default.estring ~loc:ext_loc fun_name]

                method params = [%e params]
              end]
          in
          let vb =
            Ast_builder.Default.value_binding
              ~loc:ext_loc
              ~pat:(Ast_builder.Default.ppat_var ~loc:ext_loc p.pval_name)
              ~expr:obj_expr
          in
          Some (Ast_builder.Default.pstr_value ~loc:ext_loc Nonrecursive [vb])
      | _ -> None)
  | _ -> None

let impl str =
  List.concat_map
    (fun si ->
      match transform_external_structure_item si with
      | Some s -> [s]
      | None -> [si])
    str

let () = Driver.register_transformation "sarek_external_kernel" ~impl
