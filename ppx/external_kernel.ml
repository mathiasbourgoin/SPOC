open Ppxlib

(*

[{pstr_desc =
   Pstr_extension
    (({txt = "kernel"},
      PStr
       [{pstr_desc =
          Pstr_primitive
           {pval_name = {txt = "test"};
            pval_type =
             {ptyp_desc =
               Ptyp_arrow (Nolabel,
                {ptyp_desc = Ptyp_constr ({txt = Lident "int"}, []);
                 ptyp_loc_stack = []},
                {ptyp_desc =
                  Ptyp_arrow (Nolabel,
                   {ptyp_desc = Ptyp_constr ({txt = Lident "int"}, []);
                    ptyp_loc_stack = []},
                   {ptyp_desc =
                     Ptyp_constr ({txt = Lident "vector"},
                      [{ptyp_desc = Ptyp_constr ({txt = Lident "int"}, []);
                        ptyp_loc_stack = []}]);
                    ptyp_loc_stack = []});
                 ptyp_loc_stack = []});
              ptyp_loc_stack = []};
            pval_prim = ["test_file"; "test_fun"]}}]),
    ...)};

*)

type parsed_type = {
  constraint_vars : expression list;
  arg_list : string list;
  id_list : core_type list;
}

let id = ref 0

let idx = ref 1

let arg_string = ref ""

let arg_string2 = ref ""

let new_fresh_var loc =
  let v = "spoc_var" ^ string_of_int !id in
  let a = Ast_builder.Default.estring ~loc v in
  incr id ;
  (a, v)

let rec parse_ctyp = function
  (* a -> b *)
  | {ptyp_desc = Ptyp_arrow (Nolabel, typ1, typ2); _} ->
      let l1 = parse_ctyp typ1 and l2 = parse_ctyp typ2 in
      {
        constraint_vars = l1.constraint_vars @ l2.constraint_vars;
        arg_list = l1.arg_list @ l2.arg_list;
        id_list = l1.id_list @ l2.id_list;
      }
  (* a b *)
  | {ptyp_desc = Ptyp_constr ({loc; txt = Lident type_ident}, _); _} as typ ->
      let e, v = new_fresh_var loc in
      let _typ_ident =
        match type_ident with
        | "int32" | "int64" -> [%expr int]
        | "float32" | "float64" -> [%expr float]
        | a -> [%expr [%e Ast_builder.Default.estring ~loc a]]
      in
      {
        constraint_vars = [[%expr ([%e e] : [%e typ_ident])]];
        arg_list = [v];
        id_list = [typ];
      }
  (* a *)
  | [%type: [%t? a]] as t ->
      let loc = t.ptyp_loc in
      let e, v = new_fresh_var loc in
      let typ_ident =
        match a.ptyp_desc with
        | Ptyp_var "int32" | Ptyp_var "int64" -> [%type: int]
        | Ptyp_var "float32" | Ptyp_var "float64" -> [%type: float]
        | _ -> a
      in
      {
        constraint_vars = [[%expr ([%e e] : [%t typ_ident])]];
        arg_list = [v];
        id_list = [t];
      }

let rec concrete_type_to_type t =
  let loc = t.ptyp_loc in
  match t with
  | [%type: float64] -> [%type: float]
  | [%type: Spoc.Vector.vfloat64] ->
      [%type: (float, Bigarray.float64_elt) Vector.vector]
  | [%type: Spoc.Vector.vfloat32] ->
      [%type: (float, Bigarray.float32_elt) Vector.vector]
  | [%type: Spoc.Vector.localvfloat32] ->
      [%type: (float, Bigarray.float32_elt) Vector.vector]
  | [%type: Spoc.Vector.localvfloat64] ->
      [%type: (float, Bigarray.float64_elt) Vector.vector]
  | [%type: Spoc.Vector.vchar] ->
      [%type: (char, Bigarray.int8_unsigned_elt) Vector.vector]
  | [%type: Spoc.Vector.vint32] ->
      [%type: (int32, Bigarray.int32_elt) Vector.vector]
  | [%type: Spoc.Vector.vint64] ->
      [%type: (float, Bigarray.int64_elt) Vector.vector]
  | [%type: Spoc.Vector.vbool] -> [%type: (bool, bool) Vector.vector]
  | [%type: Spoc.Vector.vcustom] -> [%type: Vector.vector]
  | [%type: Spoc.Vector.vcomplex32] ->
      [%type: (Complex.t, Bigarray.complex32_elt) Vector.vector]
  | [%type: Spoc.Vector.vcomplex64] ->
      [%type: (Complex.t, Bigarray.complex64_elt) Vector.vector]
  (* t1 txt  *)
  | {ptyp_desc = Ptyp_constr ({loc; txt}, [t1]); _} as typ -> (
      let t1 = concrete_type_to_type t1 in
      let t0 =
        concrete_type_to_type
          {typ with ptyp_desc = Ptyp_constr ({loc; txt}, [])}
      in
      match t0 with
      | {ptyp_desc = Ptyp_constr ({loc; txt}, []); _} ->
          {typ with ptyp_desc = Ptyp_constr ({loc; txt}, [t1; t1])}
      | _ -> assert false)
  | [%type: [%t? a]] -> a
(* TODO * maybe this function can be removed *)

let rec type_to_type t =
  let loc = t.ptyp_loc in
  match t with
  | [%type: float64] -> [%type: float]
  | [%type: Spoc.Vector.vfloat64] -> [%type: ('spoc_a, 'spoc_b) Vector.vector]
  | [%type: Spoc.Vector.vfloat32] -> [%type: ('spoc_c, 'spoc_d) Vector.vector]
  | [%type: Spoc.Vector.localvfloat32] ->
      [%type: ('spoc_e, 'spoc_f) Vector.vector]
  | [%type: Spoc.Vector.localvfloat64] ->
      [%type: ('spoc_g, 'spoc_h) Vector.vector]
  | [%type: Spoc.Vector.vchar] -> [%type: ('spoc_i, 'spoc_j) Vector.vector]
  | [%type: Spoc.Vector.vint32] -> [%type: ('spoc_k, 'spoc_l) Vector.vector]
  | [%type: Spoc.Vector.vint64] -> [%type: ('spoc_n, 'spoc_m) Vector.vector]
  | [%type: Spoc.Vector.vbool] -> [%type: ('spoc_o, 'spoc_p) Vector.vector]
  | [%type: Spoc.Vector.vcustom] -> [%type: ('spoc_q, 'spoc_r) Vector.vector]
  | [%type: Spoc.Vector.vcomplex32] -> [%type: ('spoc_s, 'spoc_t) Vector.vector]
  | [%type: Spoc.Vector.vcomplex64] -> [%type: ('spoc_u, 'spoc_v) Vector.vector]
  (* t1 txt  *)
  | {ptyp_desc = Ptyp_constr ({loc; txt}, [_t1]); _} as typ ->
      type_to_type {typ with ptyp_desc = Ptyp_constr ({loc; txt}, [])}
  | [%type: [%t? a]] -> a
(* TODO * maybe this function can be removed *)

let rec parseTyp t var =
  incr idx ;
  let loc = t.ptyp_loc in
  let i = Ast_builder.Default.evar ~loc var in
  match t with
  | [%type: char] -> [%expr Spoc.Kernel.Char [%e i]]
  | [%type: int] | [%type: int32] -> [%expr Spoc.Kernel.Int32 [%e i]]
  | [%type: int64] -> [%expr Spoc.Kernel.Int64 [%e i]]
  | [%type: float] | [%type: float32] -> [%expr Spoc.Kernel.Float32 [%e i]]
  | [%type: float64] -> [%expr Spoc.Kernel.float64 [%e i]]
  | [%type: Spoc.Vector.vint32] ->
      [%expr Spoc.Kernel.VInt32 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.vint64] ->
      [%expr Spoc.Kernel.VInt64 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.vfloat32] ->
      [%expr Spoc.Kernel.VFloat32 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.vfloat64] ->
      [%expr Spoc.Kernel.VFloat64 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.vchar] ->
      [%expr Spoc.Kernel.VChar (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.localvfloat32] ->
      [%expr Spoc.Kernel.LocalFloat32 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.localvfloat64] ->
      [%expr Spoc.Kernel.LocalFloat64 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.vcomplex32] ->
      [%expr Spoc.Kernel.VComplex32 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.vcomplex64] ->
      [%expr Spoc.Kernel.VComplex64 (Spoc.Kernel.relax_vector [%e i])]
  | [%type: Spoc.Vector.vbool] | [%type: [%t? _] Spoc.Vector.vcustom] ->
      [%expr Spoc.Kernel.VCustom (Spoc.Kernel.relax_vector [%e i])]
  | {ptyp_desc = Ptyp_constr ({txt = Lident _t1; _}, [t2]); _} ->
      parseTyp t2 var
  | [%type: [%t? _]] -> i

let gen_args loc type_list =
  id := 0 ;
  arg_string := "" ;
  let var_list = ref [] in
  let l = List.tl (List.rev type_list) in
  let l = List.rev l in
  let translate t =
    let e, v = new_fresh_var loc in
    var_list := !var_list @ [(e, v)] ;
    let t2 = type_to_type t in
    let t1 = Ast_builder.Default.pvar ~loc v in
    [%pat? ([%p t1] : [%t t2])]
  in
  let tuple =
    match l with
    | [] -> assert false
    | [x] ->
        idx := !idx + 1 ;
        translate x
    | a :: _tl ->
        let l = List.map translate l in
        Ast_builder.Default.ppat_tuple ~loc:a.ptyp_loc l
  in
  (* List.fold_left
   *   (fun a b ->
   *      [%pat? [%p a] ,  [%p b]]) (List.hd l) (List.tl l) in *)
  arg_string := "" ;
  let array =
    let f t =
      arg_string := "" ;
      parseTyp t (snd (List.nth !var_list !idx))
      (* ;
       * Ast_builder.Default.evar ~loc !arg_string  *)
    in
    (* (
     *   Ast_builder.Default.evar ~loc (String.sub !arg_string 12 ((String.length !arg_string - 15)))) in *)
    (* let arg =
     *   Ast_builder.Default.ppat_var ~loc {txt = arg; loc} in *)

    idx := 0 ;
    let l = List.map f l in
    Ast_builder.Default.pexp_array ~loc l
  in
  idx := 1 ;
  ((match l with [] -> assert false | [_x] -> tuple | _ -> tuple), array)

let first_vector = ref false

let relaxed loc nv =
  if not !first_vector then (
    first_vector := true ;
    [%expr [%e Ast_builder.Default.evar ~loc nv]])
  else [%expr Spoc.Kernel.relax_vector [%e Ast_builder.Default.evar ~loc nv]]

let rec gen_inv_id t loc nv =
  match t with
  | [%type: Spoc.Vector.vfloat64]
  | [%type: Spoc.Vector.vfloat32]
  | [%type: Spoc.Vector.localvfloat32]
  | [%type: Spoc.Vector.localvfloat64]
  | [%type: Spoc.Vector.vchar]
  | [%type: Spoc.Vector.vint32]
  | [%type: Spoc.Vector.vint64]
  | [%type: Spoc.Vector.vbool]
  | [%type: Spoc.Vector.vcustom]
  | [%type: [%t? _] Spoc.Vector.vcustom]
  | [%type: Spoc.Vector.vcomplex32]
  | [%type: Spoc.Vector.vcomplex64] ->
      relaxed loc nv
  (* t1 txt  *)
  | {ptyp_desc = Ptyp_constr (_, [t1]); _} -> gen_inv_id t1 loc nv
  | _ -> [%expr [%e Ast_builder.Default.evar ~loc nv]]

let rec parseInvTyp t i =
  incr idx ;
  arg_string2 := i ^ " " ^ !arg_string2 ;
  match t with
  | [%type: char] -> Longident.Lident "Char"
  | [%type: int] | [%type: int32] -> Longident.Lident "Int32"
  | [%type: int64] -> Longident.Lident "Int64"
  | [%type: float] | [%type: float32] -> Longident.Lident "Float32"
  | [%type: Spoc.Vector.vint32] -> Longident.Lident "VInt32"
  | [%type: Spoc.Vector.vint64] -> Longident.Lident "VInt64"
  | [%type: Spoc.Vector.vfloat32] -> Longident.Lident "VFloat32"
  | [%type: Spoc.Vector.vfloat64] -> Longident.Lident "VFloat64"
  | [%type: Spoc.Vector.vchar] -> Longident.Lident "VChar"
  | [%type: Spoc.Vector.localvfloat32] -> Longident.Lident "LocalFloat32"
  | [%type: Spoc.Vector.localvfloat64] -> Longident.Lident "LocalFloat64"
  | [%type: Spoc.Vector.vcomplex32] -> Longident.Lident "VComplex32"
  | [%type: Spoc.Vector.vcomplex64] -> Longident.Lident "VComplex64"
  | [%type: Spoc.Vector.vbool] | [%type: Spoc.Vector.vcustom] ->
      Longident.Lident "VCustom"
  | [%type: [%t? _] Spoc.Vector.vcustom] -> Longident.Lident "VCustom"
  | {ptyp_desc = Ptyp_constr ({txt = Lident t1; loc}, _ :: _); _} ->
      parseInvTyp (Ast_builder.Default.ptyp_var ~loc t1) i
  | [%type: [%t? _]] -> Longident.Lident (i ^ " " ^ !arg_string)

let gen_inv_args loc type_list =
  id := 0 ;
  arg_string := "" ;
  let var_list = ref [] in
  first_vector := false ;
  let l = List.tl (List.rev type_list) in
  let l = List.rev l in
  let translate t =
    let e, v = new_fresh_var loc in
    var_list := !var_list @ [(e, v)] ;
    [%expr ([%e gen_inv_id t loc v] : [%t concrete_type_to_type t])]
  in
  let tuple =
    match l with
    | [] -> assert false
    | [x] -> translate x
    | a :: _tl ->
        let l = List.map translate l in
        Ast_builder.Default.pexp_tuple ~loc:a.ptyp_loc l
  in
  (* List.fold_left
   *   (fun a b ->
   *      let loc = a.pexp_loc in [%expr [%e a] *  [%e b]]) (List.hd l) (List.tl l) in *)
  arg_string := "" ;
  let array =
    let f t =
      arg_string := "" ;
      let arg = parseInvTyp t " " in

      (* (
       *   String.sub !arg_string 12 ((String.length !arg_string - 15))) in *)
      (* let arg =
       *   Ast_builder.Default.ppat_var ~loc {txt = arg; loc} in *)
      let arg2 =
        Ast_builder.Default.ppat_var
          ~loc
          {txt = snd (List.nth !var_list (!idx - 1)); loc}
      in
      Ast_builder.Default.ppat_construct ~loc {txt = arg; loc} (Some arg2)
    in
    idx := 0 ;
    let l = List.map f l in
    Ast_builder.Default.ppat_array ~loc l
  in
  idx := 1 ;
  ((match l with [] -> assert false | [_x] -> tuple | _ -> tuple), array)

let gen_ktyp id_list =
  let ktyp = List.map concrete_type_to_type id_list in
  let ktyp = List.tl (List.rev ktyp) in
  let ktyp = List.rev ktyp in
  match ktyp with
  | [] -> assert false
  | [x] -> x
  | a :: _tl -> Ast_builder.Default.ptyp_tuple ~loc:a.ptyp_loc ktyp
(* List.fold_left
 *   (fun (a:core_type) b ->
 *      let loc = a.ptyp_loc in [%type: [%t a] *  [%t b]]) (List.hd ktyp) (List.tl ktyp) *)

(*
      idx := 1 ;
    (match l with
     | [] -> assert false
     | [x] -> tuple
     | _ ->
       ExTup(loc, tuple)),array




in
ktyp *)

class ext_kernel_mapper =
  object (_self)
    inherit Ast_traverse.map as super

    method! structure tp =
      let str = super#structure tp in
      List.fold_left
        (fun a str ->
          let loc = str.pstr_loc in
          match str.pstr_desc with
          | Pstr_extension
              ( ( {txt = "kernel"; _},
                  PStr
                    (({
                        pstr_desc =
                          Pstr_primitive
                            {
                              pval_name;
                              pval_type;
                              pval_prim = [filename; funcname];
                              pval_loc;
                              _;
                            };
                        _;
                      } as _external_v)
                    :: _) ),
                _ ) ->
              let ctyp = parse_ctyp pval_type in
              let k_typ = gen_ktyp ctyp.id_list in
              let id = "kernel_" ^ pval_name.txt in
              let inv_args = gen_inv_args loc ctyp.id_list in
              let args = gen_args loc ctyp.id_list in
              let k_fun =
                Ast_builder.Default.ppat_var ~loc:pval_loc {txt = id; loc}
              in
              let exec_fun =
                let expr =
                  Ast_builder.Default.pexp_fun
                    ~loc
                    Nolabel
                    None
                    (fst args)
                    [%expr Spoc.Kernel.exec [%e snd args]]
                in
                Ast_builder.Default.pstr_value
                  ~loc
                  Nonrecursive
                  [Ast_builder.Default.value_binding ~loc ~pat:k_fun ~expr]
              in
              let class_name = "class_kernel_" ^ pval_name.txt
              and filename = Ast_builder.Default.estring ~loc filename in
              let _exec = id in
              let inherit_kernel =
                Ast_builder.Default.pcf_inherit
                  ~loc
                  Fresh
                  (Ast_builder.Default.pcl_apply
                     ~loc
                     (* class_epxr *)
                     (Ast_builder.Default.pcl_constr
                        ~loc
                        {txt = Lident "spoc_kernel"; loc}
                        [k_typ; [%type: ('a, 'b) Kernel.kernelArgs array]])
                     [
                       (Nolabel, filename);
                       (Nolabel, Ast_builder.Default.estring ~loc funcname);
                     ])
                  None
              in
              let method_exec =
                Ast_builder.Default.pcf_method
                  ~loc
                  ( {txt = "exec"; loc},
                    Public,
                    Cfk_concrete (Fresh, Ast_builder.Default.evar ~loc id) )
              in
              let method_args_to_list =
                Ast_builder.Default.pcf_method
                  ~loc
                  ( {txt = "args_to_list"; loc},
                    Public,
                    Cfk_concrete
                      ( Fresh,
                        Ast_builder.Default.pexp_fun
                          ~loc
                          Nolabel
                          None
                          (fst args)
                          [%expr
                            let open Spoc.Kernel in
                            [%e snd args]] ) )
              in
              let method_list_to_args =
                Ast_builder.Default.pcf_method
                  ~loc
                  ( {txt = "list_to_args"; loc},
                    Public,
                    Cfk_concrete
                      ( Fresh,
                        let arg_name = {txt = "spoc_args"; loc} in
                        let arg_expr =
                          Ast_builder.Default.evar ~loc "spoc_args"
                        in
                        Ast_builder.Default.pexp_fun
                          ~loc
                          Nolabel
                          None
                          (Ast_builder.Default.ppat_var ~loc arg_name)
                          (Ast_builder.Default.pexp_match
                             ~loc
                             arg_expr
                             [
                               Ast_builder.Default.case
                                 ~lhs:(snd inv_args)
                                 ~guard:None
                                 ~rhs:(fst inv_args);
                               Ast_builder.Default.case
                                 ~lhs:[%pat? _]
                                 ~guard:None
                                 ~rhs:
                                   [%expr
                                     failwith "spoc_kernel_extension error"];
                             ]) ) )
              in
              let class_structure =
                Ast_builder.Default.class_structure
                  ~self:[%pat? self]
                  ~fields:
                    [
                      inherit_kernel;
                      method_exec;
                      method_args_to_list;
                      method_list_to_args;
                    ]
              in
              let class_expr =
                Ast_builder.Default.pcl_structure ~loc class_structure
              in
              let generated_class =
                Ast_builder.Default.class_infos
                  ~loc
                  ~virt:Concrete
                  ~params:
                    [
                      ([%type: 'a], (NoVariance, NoInjectivity));
                      ([%type: 'b], (NoVariance, NoInjectivity));
                    ]
                  ~name:{txt = class_name; loc}
                  ~expr:class_expr
              in

              let instantiate =
                Ast_builder.Default.pstr_value
                  ~loc
                  Nonrecursive
                  [
                    Ast_builder.Default.value_binding
                      ~loc
                      ~pat:
                        (Ast_builder.Default.ppat_var ~loc:pval_loc pval_name)
                      ~expr:
                        (Ast_builder.Default.pexp_new
                           ~loc
                           {txt = Lident class_name; loc});
                  ]
              in
              a
              @ [
                  [%stri open Spoc.Kernel];
                  exec_fun;
                  Ast_builder.Default.pstr_class ~loc [generated_class];
                  instantiate;
                ]
          | _ -> a @ [str])
        []
        str
  end

let _ =
  Driver.register_transformation
    "external_kernel"
    ~impl:(new ext_kernel_mapper)#structure
