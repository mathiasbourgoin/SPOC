open Camlp4.PreCast
open Syntax
open Ast


open Sarek_types
open Debug

let rec basic_check l current_type expected_type loc =
  if expected_type <> current_type && not (is_unknown current_type) then
    ( assert (not debug); raise (TypeError (expected_type, current_type, loc)) );
  List.iter (fun e -> typer e expected_type) l

and elt_check body t l =
  if body.t <> t && not (is_unknown body.t) then
    (assert (not debug); raise (TypeError (t, body.t, l)) )
  else
    update_type body t;

and equal_sum acc l1 l2 =
  match l1,l2 with
  | [],[] -> acc
  | t1::q1,t2::q2 ->
    equal_sum (acc && t1 = t2) q1 q2
  | _ -> false

and equal_rec acc lct1 lid1 lct2 lid2 =
  my_eprintf "Equalrec\n";
  match lct1,lid1,lct2,lid2 with
  | c1::qc1,i1::qi1,
    c2::qc2,i2::qi2 ->
    equal_rec (acc && (string_of_ctyp c1) = (string_of_ctyp c2) && (string_of_ident i1) = (string_of_ident i2))  qc1 qi1 qc2 qi2
  | [],[],[],[] -> acc
  | _ -> false

and check_custom c1 c2 =
  match c1,c2 with
  | KSum l1, KSum l2 -> equal_sum true l1 l2
  | KRecord (c1,i1,_) , KRecord (c2,i2,_)  -> equal_rec true c1 i1 c2 i2
  | KRecord _ , KSum _
  | KSum _ , KRecord _ -> false

and equal_types t1 t2 =
  match t1,t2 with
  | Custom (x,_), Custom (y,_) -> check_custom x y
  | Custom (_,_), _ | _, Custom (_,_) -> false
  | _ ->
    if t1 = t2 then
      true
    else
      match t1,t2 with
      | TArr (t1_, _), TArr (t2_, _) ->
        equal_types  t1_ t2_
      | _ -> false

and check t1 t2 l =
  if (not (equal_types t1 t2)) && (not (is_unknown t1)) && (not (is_unknown t2)) then
    (assert (not debug); raise (TypeError (t1, t2, l)) )



and gen_app_from_constr t cstr =
  match t.typ with
  | KRecord _ -> assert false
  | KSum l ->
     begin
       let rec aux = function
         | (c,Some s)::q ->
          (* for now Constructors only applies to single arguments *)
            let rec aux2 = function
              | <:ctyp< int >> | <:ctyp< int32 >> -> TInt32
	      | <:ctyp< float >> | <:ctyp< float32 >> -> TFloat32 (*missing many types *)
              | a -> Custom (t.typ, string_of_ctyp a)

            in
            TApp ((aux2 s),
                  ( Custom (t.typ,t.name)))
         | _::q -> aux q
         | [] -> assert false
       in aux l
     end



and  typer_id body t tt f =
  match body.e with
  | Id (l, s) ->
    (try
       let var = Hashtbl.find !current_args (string_of_ident s) in
       my_eprintf ((string_of_ident s)^ " of type " ^(ktyp_to_string t)^"\n");
       if not (is_unknown t) then
         if is_unknown var.var_type  then
           (var.var_type <- t;
            retype := true;
            update_type body t)
         else
           check var.var_type t l;
       tt := var.var_type
     with Not_found ->
     try
       let c_const = Hashtbl.find !intrinsics_const (string_of_ident s) in
       if t = TUnknown then
         ( update_type body c_const.typ;)
       else if t <> c_const.typ then
         (assert (not debug); raise (TypeError (c_const.typ, t, l)))
     with _ ->
     try ignore(Hashtbl.find !intrinsics_fun (string_of_ident s) )
     with _ ->
     try
       let cstr = Hashtbl.find !constructors (string_of_ident s) in
       my_eprintf ("Found a constructor : "^(string_of_ident s)
                   ^" of type "^cstr.name^" with "
                   ^(string_of_int cstr.nb_args)^" arguments\n");
       tt :=
         if cstr.nb_args <> 0 then
           gen_app_from_constr cstr s
         else
           Custom (cstr.typ, cstr.name);
     with
     | _ -> f ())
  | _ -> assert false
    
and typer body t =
  my_eprintf (Printf.sprintf"(* %s ############# typ %s *)\n%!" (k_expr_to_string body.e) (ktyp_to_string t)) ;
  (match body.e with
   | Id (l, s) ->
     let tt = ref t in
     typer_id body t tt (fun () ->
         assert (not debug); raise (Unbound_value (string_of_ident s, l)));

     update_type body !tt
   | ArrSet (l, e1, e2) ->
     check t TUnit l;
     typer e1 e2.t;
     typer e2 e1.t;
     update_type body TUnit
   | ArrGet (l, e1, e2) ->
     typer e1 (TArr (t, Any));
     typer e2 TInt32;
     update_type body (
       match e1.t with
       | TArr (tt,_) -> tt
       | _ -> my_eprintf (ktyp_to_string e1.t);
         assert false;)
   | VecSet (l, e1, e2) ->
     check t TUnit l;
     typer e1 e2.t;
     typer e2 e1.t;
     update_type body TUnit
   | VecGet (l, e1, e2) ->
     typer e1 (TVec t);
     typer e2 TInt32;
     update_type body (
       match e1.t with
       | TVec tt -> tt
       | _ -> TVec TUnknown)
   (*my_eprintf ((k_expr_to_string e1.e) ^" ++ "^(ktyp_to_string e1.t)^"\n");
     assert false;)*)
   | Seq (l, e1, e2) ->
     typer e1 TUnit;
     typer e2 t;
     update_type body e2.t
   | Int32 (l,s) ->
     elt_check body TInt32 l
   | Int64 (l,s) ->
     elt_check body TInt64 l
   | Float32 (l,s) ->
     elt_check body TFloat32 l
   | Float64 (l,s) ->
     elt_check body TFloat64 l
   | Bind (_loc, var,y, z, is_mutable)  ->
     (match var.e with
      | Id (_loc,s)  ->
         (match y.e with
          | Fun (_loc,stri,tt,funv,lifted) ->
             my_eprintf ("ADDDD: "^(string_of_ident s)^"\n%!");
             Hashtbl.add !local_fun (string_of_ident s)
			         (funv, (<:str_item<let $id:s$ = $stri$>>), lifted);
             update_type y tt;
          | _ ->
            try
	           let v = Hashtbl.find !current_args (string_of_ident s)
           in
	          typer y v.var_type;
	     with
	     | Not_found ->
		(*Printf.eprintf "Looking for %s in [" (string_of_ident s);
		Hashtbl.iter (fun a b -> Printf.eprintf " %s;" a) !current_args;
		Printf.eprintf " ]\n";*)
		typer y TUnknown;
  (incr arg_idx;
   my_eprintf ("AD: "^(string_of_ident s)^"\n%!");
		Hashtbl.add !current_args (string_of_ident s)
			     {n = !arg_idx; var_type = y.t;
			      is_mutable = is_mutable;
			      read_only = false;
			      write_only = false;
			      is_global = false;}
             )
         );

      | _ -> assert false
     );
     update_type var y.t;
     typer z t;
     update_type body z.t;
   | Plus32 (l,e1,e2) | Min32 (l,e1,e2)
   | Mul32 (l,e1,e2) | Div32 (l,e1,e2)
   | Mod (l, e1, e2) ->
     basic_check [e1;e2] t TInt32 l;
     update_type body TInt32;
   | PlusF32 (l,e1,e2) | MinF32 (l,e1,e2) | MulF32 (l,e1,e2) | DivF32 (l,e1,e2) ->
     basic_check [e1;e2] t TFloat32 l;
     update_type body TFloat32;
   | If (l, e1, e2) ->
     typer e1 TBool;
     basic_check [e1] e1.t TBool l;
     typer e2 TUnit;
     update_type body TUnit
   | Ife (l, e1, e2, e3) ->
     typer e1 TBool;
     basic_check [e1] e1.t TBool l;
     typer e2 e3.t;
     typer e3 e2.t;
     if e2.t <> e3.t then
       ( assert (not debug); raise (TypeError (e2.t, e3.t, l)) );
     update_type body e2.t
   | Noop ->
     if t <> TUnit && t <> TUnknown then
       assert false
     else
       update_type body TUnit
   | Open (l, m_ident, e2) ->
     let rec _open = function
       | IdAcc (l,a,b) -> _open a; _open b
       | IdUid (l,s) -> open_module s l
       | _ -> assert false
     and _close = function
       | IdAcc (l,a,b) -> _close b; _close a
       | IdUid (l,s) -> close_module s
       | _ -> assert false
     in
     _open m_ident;
     typer e2 t;
     _close m_ident;
     update_type body e2.t;
   | While (l, cond, loop_body) ->
     typer cond TBool;
     basic_check [cond] cond.t TBool l;
     basic_check [loop_body] t TUnit l;
     typer cond TBool;
     typer loop_body TUnit;
     update_type body TUnit;
   | DoLoop (l, var, y, z, loop_body) ->
     (match var.e with
      | Id (_loc,s)  ->
        (incr arg_idx;
         Hashtbl.add !current_args (string_of_ident s)
           {n = !arg_idx; var_type = TInt32;
            is_mutable = false;
            read_only = false;
            write_only = false;
            is_global = false;}
        )
      | _ -> assert false
     );
     typer var TInt32;
     typer y TInt32;
     typer z TInt32;
     typer loop_body TUnit;
     update_type body TUnit;
   | App (l, e1, e2) ->
     let t = typer_app e1 e2 t in
     update_type body t
   | BoolEq32 (l, e1, e2)
   | BoolLt32 (l, e1, e2)
   | BoolLtE32 (l, e1, e2)
   | BoolGt32 (l, e1, e2)
   | BoolGtE32 (l, e1, e2) ->
     typer e1 TInt32;
     typer e2 TInt32;
     update_type body TBool
   | BoolEqF32 (l, e1, e2)
   | BoolLtF32 (l, e1, e2)
   | BoolLtEF32 (l, e1, e2)
   | BoolGtF32 (l, e1, e2)
   | BoolGtEF32 (l, e1, e2) ->
     typer e1 TFloat32;
     typer e2 TFloat32;
     update_type body TBool
   | BoolOr (l,e1,e2)
   | BoolAnd (l,e1,e2) ->
     typer e1 TBool;
     typer e2 TBool;
     update_type body TBool
   | Ref (l, ({e = Id (ll, s); _} as e)) ->
        let body = e in
        let tt = ref t in
        typer_id e t tt (fun () ->
            (incr arg_idx;
	     Hashtbl.add !current_args (string_of_ident s) {n = -1; var_type = t;
                                                     is_mutable = false;
                                                     read_only = true;
                                                     write_only = false;
                                                     is_global = true;};
             tt := t;);
            update_type body !tt);
        update_type body e.t;
        decr unknown;
   | Acc (l, e1, e2) ->
      typer e2 TUnknown;
      typer e1 TUnknown;
      typer e2 e1.t;
      typer e1 e2.t;
      update_type body TUnit
   | Match (l,e,mc) ->
     let get_patt_typ = function
       | Constr (s,of_) ->
          let cstr =
            my_eprintf ("--------- type case :  "^s^"\n");
            try
              Hashtbl.find !constructors s
            with | _ -> failwith "error in pattern matching"
          in
          (Custom (cstr.typ,cstr.name))
     in
     let type_patt f=
       let t = get_patt_typ f in
       typer e t;
       check t e.t l;
     in
     (match mc with
      | (_loc,(Constr (s,of_) as p), ee)::_ ->
         (type_patt p;
	  (*(match of_ with
           |Some id ->
	     incr arg_idx;
             Hashtbl.add !current_args (string_of_ident id)
			 {n = !arg_idx; var_type = ktyp_of_typ (TyId(_loc,IdLid(_loc,type_of_patt p)));
			  is_mutable = false;
			  read_only = false;
			  write_only = false;
			  is_global = false;};
             typer ee t;
             Hashtbl.remove !current_args (string_of_ident id);
           | None -> typer ee t;
          )
          );*)
	 )
      | _ -> failwith "No match cases in patern matching");
     let rec aux = function
       | (ll,(Constr (s,of_) as p),ee)::q ->
          check (get_patt_typ p) e.t ll;
	  (match of_ with
           |Some id ->
	     incr arg_idx;
             Hashtbl.replace !current_args (string_of_ident id)
			     {n = !arg_idx; var_type = ktyp_of_typ (TyId(ll,IdLid(ll,type_of_patt p)));
			      is_mutable = false;
			      read_only = false;
			      write_only = false;
			      is_global = false;};
	     typer ee t;
           Hashtbl.remove !current_args (string_of_ident id);
          | None -> typer ee t;
         );
         check ee.t t l;
         aux  q
       | [] -> ();
     in
     aux (mc);
     let ttt =
       let (_,_,e) = List.hd mc in
       e.t in
     update_type body ttt
   | Record(l,fl) ->
      let seed : string list =
	let (_loc, id, _) = (List.hd fl) in
	try
          (Hashtbl.find !rec_fields (string_of_ident id)).ctyps
	with
	| _ ->
         (assert (not debug);
          raise (FieldError (string_of_ident id, "\"\"", _loc)))
     in

     (* get custom_type corresponding to the record*)
     let rec_typ =
       try
         Hashtbl.find custom_types (List.hd seed)
       with
       | _ ->
         assert false;
     in
     let field_typ_list =
       match rec_typ with
       | KRecord (typ, id, _) ->
         List.combine  (List.map string_of_ident id) typ
       | _ -> assert false
     in

     let t =
       let rec aux (acc:string list) (flds : field list) : string list =
         match flds with
         | (_loc, id, e)::q ->
           let rec_fld : recrd_field =
             try
                typer e (ktyp_of_typ (List.assoc (string_of_ident id) field_typ_list));
               Hashtbl.find !rec_fields (string_of_ident id)
             with
             | Not_found->
               (assert (not debug);
                raise (FieldError (string_of_ident id, List.hd acc, _loc)))
           in
           aux
             (let rec aux2 (res:string list) (acc_:string list) (flds_:string list) =
                match acc_,flds_ with
                | (t1::q1),(t2::q2) ->
                  if t1 = t2 then
                    aux2 (t1::acc_) acc q2
                  else
                    aux2 (t1::acc_) q1 (t2::q2)

                | _,[] -> res
                | [],q ->
                  aux2 res acc q
              in aux2 [] acc rec_fld.ctyps) q
         | [] -> acc
       in

       let r : string list =
         aux seed fl
       in List.hd r
     in
     let _loc = Loc.ghost in
     update_type body (ktyp_of_typ (TyId(_loc,IdLid(_loc,t))));
     my_eprintf ("record_type : "^(ktyp_to_string body.t^"\n"))
   | RecSet (l, e1, e2) ->
     check t TUnit l;
     typer e1 e2.t;
     typer e2 e1.t;
     update_type body TUnit
   | RecGet (_loc, e1, e2) ->
     let t =
       try Hashtbl.find !rec_fields (string_of_ident e2)
       with
       | Not_found ->
         (    match e1.e with
              | Id(_,i) ->
                assert (not debug);
                raise (FieldError ((string_of_ident i), (string_of_ident e2), _loc))
              | _ -> assert false
         )
     in
     typer e1 (Custom (Hashtbl.find custom_types t.name, t.name));
     let tt =
       match e1.t with
       | TUnknown | TVec TUnknown | TArr (TUnknown,_) -> TUnknown;
       | Custom (KRecord  (l1,l2,_),n) ->
         let rec aux = function
           |t1::q1,t2::q2 ->
             if (string_of_ident t2) = (string_of_ident e2) then
               (my_eprintf (string_of_ctyp t1);
                ktyp_of_typ t1)
             else
               (my_eprintf ("N : "^(string_of_ctyp t1));
                aux (q1,q2))
           | [],[] ->
             (assert (not debug);
              raise (FieldError (string_of_ident e2, n, _loc)))
           | _ -> assert false
         in
         aux (l1,l2)
       | _ ->
         my_eprintf (ktyp_to_string e1.t);
         assert false in
     update_type body tt
   | False _ | True _ ->
		update_type  body TBool
   | BoolNot (_loc,e1) ->
      typer e1 TBool;
      check e1.t TBool _loc;
      update_type body TBool;
   | BoolEq (_loc,e1,e2) ->
     typer e1 TUnknown;
     typer e2 e1.t;
     check e1.t e2.t _loc;
     update_type body TBool;
   | ModuleAccess (l,m,e) ->
      open_module  m l;
      typer e t;
      close_module m;
      update_type body e.t;
   | TypeConstraint (l, x, tc) ->
     typer x tc;
     check t tc;
     update_type body tc;
   | Nat _ ->
     update_type body TUnit
   | _ -> my_eprintf  ((k_expr_to_string body.e)^"\n"); assert false);
  if is_unknown body.t then
    (my_eprintf  (("UNKNOWN : "^k_expr_to_string body.e)^"\n");
     incr unknown
    )



and typer_app e1 (e2 : kexpr list) t =
  let  typ, loc  =
    let rec aux e1 =
      match e1.e with
      | Id (_l, s) ->
        (try (Hashtbl.find !intrinsics_fun (string_of_ident s)).typ , _l
         with |_ ->
         try (Hashtbl.find !global_fun (string_of_ident s)).typ , _l
         with |_ ->
           try
             let (f,_,lifted) =
               (Hashtbl.find !local_fun (string_of_ident s))
             in

             f.typ,_l
           with
	   |_ ->
      try
        let cstr = Hashtbl.find !constructors (string_of_ident s) in
        my_eprintf ("App : Found a constructor : "^(string_of_ident s)
                    ^" of type "^cstr.name^" with "
                    ^(string_of_int cstr.nb_args)^" arguments\n");
        (if cstr.nb_args <> 0 then
           gen_app_from_constr cstr s
         else
           Custom (cstr.typ, cstr.name)),_l
      with
      | Not_found  ->
        assert (not debug); raise (Unbound_value (string_of_ident s,_l) ))

      | ModuleAccess (_l, s, e) ->
        open_module s _l;
        let typ, loc = aux e in
        close_module s;
        typ, loc
      | _ ->  typer e1 t; e1.t, e1.loc
    in
    aux e1
  in
  let ret = ref TUnit in
  let rec aux2 typ expr =
    match typ, expr with
    | TApp (t1, t2), e::[] -> typer e t1; ret := t2
    | _ , [] -> assert false
    | _ -> assert false
  in

  let rec aux typ1 e =
    match typ1,e with
    | (TApp (t1, (TApp (_,_) as t2)),
       App ( l , e1, (t::(tt::qq) as e2))) ->
      aux2 t2 e2;
      typer e1 t1;
      if e1.t <> t1  then ( assert (not debug); raise (TypeError (t1, e1.t, l)) );
      update_type e1  t1;
    | ((TApp (TApp(t1, t3) as t, t2)),
       (App (l, e1, e2::[]))) ->
      assert (t3 = t2);
      ret := t2;
      typer e2 t1;
      if e2.t <> t1 && e2.t <> TUnknown then ( assert (not debug); raise (TypeError (t1, e2.t, l)) );
      update_type e2 t1;
      update_type e1 t;
    | (TApp(t1, t2), App(_, _, e2::[] ) )->
      my_eprintf (Printf.sprintf"(* typ %s +++-> %s *)\n%!" (k_expr_to_string e2.e) (ktyp_to_string e2.t)) ;
      ret := e2.t;
      typer e2 t1;
    | (t1 , App(_, _, e2::[] ) )->
      my_eprintf (Printf.sprintf"(* typ %s +++-> %s *)\n%!" (k_expr_to_string e2.e) (ktyp_to_string e2.t)) ;
      ret := t1;
      typer e2 t1;
    | _ -> assert (not debug);
  in aux typ (App (loc,e1,e2));

  let rec aux typ =
    match typ with
    | TApp(t1,t2) -> t2
    | t -> t
  in
  my_eprintf (Printf.sprintf"(* >>>>>>>>>>>> typ %s *)\n%!" (ktyp_to_string typ)) ;
  let t = aux typ  in
  my_eprintf (Printf.sprintf"(* <<<<<<<<<<<< typ %s *)\n%!" (ktyp_to_string t)) ;
  t
