open Ppxlib
open Sarek_types
open Gen_types

class int_kernel_mapper =
  object (_self)
    inherit Ast_traverse.map as super
    method! structure tp =
      let str = super#structure tp in
      List.fold_left (fun a str ->
          let _loc = str.pstr_loc in
          match  str.pstr_desc with
          | Pstr_extension
              (({txt = "kernel"; _  },
                PStr
                  [{pstr_desc =
                      Pstr_type (Recursive,
                                 [{ptype_name = {txt ; loc}; ptype_params = [];
                                   ptype_cstrs = _;
                                   ptype_kind ; _}]); _}]), _) ->
            (match ptype_kind with
             | Ptype_abstract -> assert false
             | Ptype_variant cstrs ->
               let cstrs = (List.map
                              (fun t ->
                                 match t.pcd_res, t.pcd_args with
                                 | None, Pcstr_tuple [] -> (Loc.txt t.pcd_name, None)
                                 | None,Pcstr_tuple [ct] -> (Loc.txt t.pcd_name, Some ct)
                                 | _ -> failwith "Tuple not implemented in record type"
                              ) cstrs) in
               let k = KVariant cstrs in
               Hashtbl.add custom_types txt k;
               (gen_variant loc k txt)@a
             | Ptype_record _lbls -> assert false
             | Ptype_open -> assert false)
          | _ -> (str::a)
        ) [] str


  end

let _ =
  Driver.register_transformation
    "internal_kernel"
    ~impl:(new int_kernel_mapper)#structure
