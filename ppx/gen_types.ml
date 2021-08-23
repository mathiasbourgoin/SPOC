[@@@ocaml.warning "-26"]
open Ppxlib
open Sarek_types




let gen_ctype t1 ident t3 loc =
  let open Ast_builder.Default in
  let field_name  = ident^"_"^t3 in
  pstr_value ~loc Nonrecursive
    [(value_binding ~loc
        ~pat:(ppat_var ~loc {txt=field_name; loc})
        ~expr:[%expr let open Ctypes in Ctypes.field
            [%e evar ~loc ident] [%e estring ~loc field_name]
            [%e evar ~loc @@ get_sarek_name (string_of_ctyp t1)]])]


let gen_ctype_repr t name =
  let field_name  = name^"_t" in
  ((ctype_of_sarek_type (string_of_ctyp t))^" "^field_name)

(* begin
 *
 *   [%stri:
 *     let field_name =
 *       let open Ctypes in
 *       Ctypes.field
 *       $lid:string_of_ident t2$
 *            $str:field_name$
 *                 $lid:get_sarek_name ((string_of_ctyp t1))$
 *                      <:str_item< let $lid:field_name$ =
 *                                  let open Ctypes in
 *                                  Ctypes.field
 *                                  $lid:string_of_ident t2$
 *                                  $str:field_name$
 *                                  $lid:get_sarek_name ((string_of_ctyp t1))$
 *                                  ;;
 *                      >>
 * end *)


let gen_variant loc k name =
  let gen_repr t =
    let managed_ktypes = Hashtbl.create 5 in
    let sarek_type_name = name^"_sarek" in
    Hashtbl.add sarek_types_tbl name sarek_type_name;
    let fieldsML =
      match t with
      | KVariant l ->
        List.rev @@ List.fold_left
          (fun acc (label,typ)  ->
             match typ with
             | Some t ->
               let open Ast_builder.Default in
               let name = {txt=sarek_type_name^"_"^label; loc} in
               let ct = gen_ctype t sarek_type_name "" loc in
               let ctr = gen_ctype_repr t name.txt in
               (* let () = Ctypes.seal $lid:name$ ;; >> *)
               [%stri let () = Ctypes.seal [%e evar ~loc name.txt]]::
               ct::
               (* let $lid:name$ : $lid:name$ Ctypes.structure Ctypes.typ = Ctypes.structure  $str:name$ *)
               [%stri let [%p ppat_var ~loc name] : [%t ptyp_constr ~loc {txt=(lident name.txt); loc}
                   []] Ctypes.structure Ctypes.typ =
                        Ctypes.structure [%e estring ~loc name.txt]  ]              ::
               (* type $lid:name$ ;; *)
               (
                 pstr_type ~loc Nonrecursive
                   [
                     type_declaration ~loc ~name ~params:[]
                       ~cstrs:[] ~kind:Ptype_abstract ~private_:Public ~manifest:None])::
               acc
             | None -> [[%stri let lol = "loooool"]]) [] l;

        (*           begin
         *             let gen_mlstruct accML (cstr,ty) =
         *               let name = sarek_type_name^"_"^cstr in
         *               begin
         *                 match ty with
         *                 | Some t ->
         *                   begin
         *                     let ct = gen_ctype t Pa(_loc,name))  sarek_type_name "" _loc in
         * let ctr = gen_ctype_repr
         *     t
         *     (IdLid(_loc,name))  sarek_type_name  in
         * Hashtbl.add managed_ktypes cstr ctr;
         * <:str_item<
         *                                   $accML$ ;;
         *                                   type $lid:name$ ;;
         *                                   let $lid:name$ : $lid:name$ Ctypes.structure Ctypes.typ =
         *                                   Ctypes.structure  $str:name$ ;;
         *                                   $ct$ ;;
         *                                   let () = Ctypes.seal $lid:name$ ;; >>
         * end
         * | None ->
         *   begin
         *
         *     <:str_item<
         *                         $accML$
         *                       >>
         *   end
         * end
         *
         * in *)


      | _ -> assert false
    in fieldsML in
  gen_repr k
(* [%stri let _ =
 *          let s =
 *            ("I should have generated a ctype for "^
 *             [%e Ast_builder.Default.estring ~loc name]^"\n")in
 *          failwith s ] *)
