open Camlp4.PreCast
open Syntax
open Ast


open Types


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

and check_custom c1 c2 =
  match c1,c2 with
  | KSum l1, KSum l2 -> equal_sum true l1 l2
  | KSum _, _ | _, KSum _ -> false
  | _ -> failwith "unimplemented yet\n"

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
            | _ -> assert false
          in
          TApp ((aux2 s), 
                ( Custom (t.typ,t.name)))
        | _::q -> aux q
        | [] -> assert false
      in aux l
    end




and typer body t =
  my_eprintf (Printf.sprintf"(* %s ############# typ %s *)\n%!" (k_expr_to_string body.e) (ktyp_to_string t)) ;  
  (match body.e with
   | Id (l, s) ->
     let tt = ref t in
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
            | _ ->
              (Hashtbl.add !current_args (string_of_ident s) 
                 {n = -1; var_type = t;
                  is_mutable = false;
                  read_only = true;
                  write_only = false;
                  is_global = true;};
               tt := t;)
              
     );
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
	 | Fun (_loc,stri,tt,funv) ->
	   my_eprintf ("ADDDD: "^(string_of_ident s));
	   Hashtbl.add !local_fun (string_of_ident s) 
	     (funv,<:str_item<
let $id:s$ = 
$stri$>>);
	   update_type y tt;
	 | _ -> typer y TUnknown;
	   (incr arg_idx;
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
       | IdAcc (l,a,b) -> _close a; _close b
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
   | BoolEqF (l, e1, e2) 
   | BoolLtF (l, e1, e2) 
   | BoolLtEF (l, e1, e2) 
   | BoolGtF (l, e1, e2) 
   | BoolGtEF (l, e1, e2) -> 
     typer e1 TFloat32;
     typer e2 TFloat32;
     update_type body TBool
   | BoolOr (l,e1,e2) 
   | BoolAnd (l,e1,e2) ->
     typer e1 TBool;
     typer e2 TBool;
     update_type body TBool
   | Ref (l, e) ->
     typer e t;
     update_type body e.t;
     decr unknown;
   | Acc (l, e1, e2) ->
     typer e2 e1.t;
     typer e1 e2.t;
     update_type body TUnit
   | Match (l,e,mc) ->
     let rec aux tt = function
       | (ll,p,t)::q ->
         (match p with
          | Constr (s,_) ->
            let cstr =
             try
               Hashtbl.find !constructors s
             with | _ -> failwith "error in pattern matching"
            in
           typer e (Custom (cstr.typ,cstr.name)););
         typer t tt; 
         check tt t.t l;
         aux t.t q
       | [] -> ();
     in
     aux t mc;
     let ttt = let (_,_,e) = List.hd mc in
       e.t in
     update_type body ttt
   | Record(l,fl) ->
     let t = 
       let rec aux (acc:string list) (flds : field list) : string list = 
         match flds with
         | (_loc,t,_)::q -> 
           let rec_fld : recrd_field = 
             try Hashtbl.find !rec_fields (string_of_ident t) 
             with 
             | _ -> 
               (assert (not debug); 
                raise (FieldError (string_of_ident t, List.hd acc, _loc)))
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
       let start : string list = 
         let (_loc,t,_) = (List.hd fl) in
         try 
           (Hashtbl.find !rec_fields (string_of_ident t)).ctyps 
         with 
         | _ -> 
           (assert (not debug);
            raise (FieldError (string_of_ident t, "\"\"", _loc)))
       in
       let r : string list = 
         aux start fl
       in List.hd r
     in
     let _loc = Loc.ghost in
     body.t <- ktyp_of_typ (TyId(_loc,IdLid(_loc,t)))
   | RecSet (l, e1, e2) ->
     check t TUnit l;
     typer e1 e2.t;
     typer e2 e1.t;
     update_type body TUnit
   | RecGet (_loc, e1, e2) ->
     typer e1 TUnknown;
     let tt =
       match e1.t with
       | TUnknown | TVec TUnknown | TArr (TUnknown,_) -> TUnknown;
       | Custom (KRecord (l1,l2,_),n) -> 
         let rec aux = function
           |t1::q1,t2::q2 ->
             if (string_of_ident t2) = (string_of_ident e2) then
               (my_eprintf (string_of_ctyp t1); 
                ktyp_of_typ t1)
             else
               aux (q1,q2)
           | [],[] ->   
             (assert (not debug);
              raise (FieldError (string_of_ident e2, n, _loc)))
           | _ -> assert false
         in
         aux (l1,l2)
       | _ -> my_eprintf (ktyp_to_string e1.t); 
         assert false 
     in
     update_type body tt
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
      try (fst(Hashtbl.find !local_fun (string_of_ident s))).typ , _l
      with |_ ->
        try 
          let cstr = Hashtbl.find !constructors (string_of_ident s) in
          my_eprintf ("App : Found a constructor : "^(string_of_ident s)
                      ^" of type "^cstr.name^" with "
                      ^(string_of_int cstr.nb_args)^" arguments\n");
          (if cstr.nb_args <> 0 then
             gen_app_from_constr cstr s
           else
             Custom (cstr.typ, cstr.name)),_l
        with | _  -> 
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
  
