open Camlp4.PreCast
open Syntax
open Ast

open Types

let rec parse_int i t= 
  match i.e with
  | Id (_loc,s)  -> 
    (
      let is_mutable = ref false in
      ( try 
          let var = 
            Hashtbl.find !current_args (string_of_ident s)  in
          (
            is_mutable := var.is_mutable;
            match var.var_type with
            | TUnknown -> var.var_type <- t
            | x when x = t  -> ()
            | _  -> assert (not debug); 
              raise (TypeError (t, var.var_type, _loc)));
        with Not_found -> 
          (
            try 
              let c_const = Hashtbl.find !intrinsics_const 
                  (string_of_ident s) in
              (match c_const.typ with
               | x when x = t -> ()
               | _  -> assert (not debug); 
                 raise (TypeError (t, c_const.typ, _loc)))
            with Not_found ->
              (assert (not debug); 
               raise (Unbound_value ((string_of_ident s), _loc)))));
      (match i.t with
       | x when x = t -> () 

       | TUnknown  ->  i.t <- t; 
       | _ -> assert (not debug); 
         raise (TypeError (t, i.t, i.loc)));
      if !is_mutable then
        <:expr< $(ExId (_loc, s))$.contents>>
      else
        <:expr< $(ExId (_loc, s))$>>)
  | Int (_loc, s) -> 
    ( match i.t with 
      | TInt32 -> <:expr< $(ExInt (_loc, s))$>>
      | _ -> assert (not debug); 
        raise (TypeError (t, i.t, i.loc)))
  | Int32 (_loc, s) -> 
    ( match i.t with 
      |  TInt32 -> <:expr< $(ExInt32 (_loc, s))$>>
      | _ -> assert (not debug); 
        raise (TypeError (t, i.t, i.loc)))

  | Int64 (_loc, s) -> 
    ( match i.t with 
      | TInt64  -> <:expr< $(ExInt64 (_loc, s))$>>
      | _ -> assert (not debug); 
        raise (TypeError (t, i.t, i.loc)))
  | Plus32 (_loc, a, b) | Plus64 (_loc, a, b)  -> 
    parse_body i
  | Min32 (_loc, a, b) | Min64 (_loc, a, b)  -> 
    parse_body i
  | Mul32 (_loc, a, b) | Mul64 (_loc, a, b)  -> 
    parse_body i
  | Div32 (_loc, a, b) | Div64 (_loc, a, b)  -> 
    parse_body i
  | Mod (_loc, a, b) -> 
    parse_body i
  | Bind (_loc, var,y, z, is_mutable)  ->
    (
      let gen_z = parse_int z t in
      match var.e with
      | Id (_loc,s) ->  
        (<:expr<let $PaId(_loc,s)$ = $(parse_body y)$ in $gen_z$>>)
      | _ -> failwith "error parse_body Bind")
  | Ref (_loc, id) -> 
    <:expr< ! $parse_body id$>>
  | VecGet (_loc, vector, index)  -> 
    ( match i.t with
      | x when x = t -> ()
      | x when x = TUnknown -> i.t <- TVec t 
      | _  -> assert (not debug); 
        raise (TypeError (t, i.t , _loc)));
    (match vector.e with
     | Id (_loc,s)  -> 
       let var = (Hashtbl.find !current_args (string_of_ident s)) in
       let type_constraint = 
         let vec_typ_to_e k =
           match k with 
           | TInt32  -> 
             <:ctyp<(int32, Bigarray.int32_elt) Spoc.Vector.vector>>
           | TInt64  -> 
             <:ctyp<(int64, Bigarray.int64_elt) Spoc.Vector.vector>>
           |  _  ->  assert false
         in
         (match var.var_type with
          | TVec k when k = t  -> (
              vec_typ_to_e k
            )
          | TVec TUnknown -> var.var_type <- TVec t;
            vec_typ_to_e t
          | _  -> assert (not debug); 
            raise (TypeError (TVec t, var.var_type, _loc)))
       in
       <:expr<Spoc.Mem.get ($ExId(_loc,s)$:$type_constraint$) 
              (Int32.to_int $parse_body index$)>>
     | _  ->  assert (not debug); 
       failwith "Unknwown vector");
  | ArrGet _ -> parse_body i
  | RecGet _ -> parse_body i
  | App (_loc, e1, e2) -> parse_body i
  | _ -> my_eprintf (k_expr_to_string i.e); 
    assert (not debug);  
    raise (TypeError (t, i.t, i.loc))

and parse_float f t = 
  match f.e with
  | App (_loc, e1, e2) ->
    parse_body f
  | Id (_loc,s)  ->
    (let is_mutable = ref false in
     ( let var = (
        ( try Hashtbl.find !current_args (string_of_ident s) 
          with _ -> assert (not debug); 
            raise (Unbound_value ((string_of_ident s),_loc)))) in
       is_mutable := var.is_mutable;
       match var.var_type with
       | TUnknown ->
         ( try 
             (Hashtbl.find !current_args 
                (string_of_ident s)).var_type <- t  
           with _ -> assert (not debug); 
             raise (Unbound_value ((string_of_ident s),_loc)))
       | x when x = t  -> ()
       | _  -> assert (not debug); 
         raise (TypeError (t, var.var_type, _loc)));
     (match f.t with
      | x when x = t -> ()
      | TUnknown  ->  f.t <- t
      | _ -> assert (not debug); 
        raise (TypeError (t, f.t, f.loc))) ;
     if !is_mutable then
       <:expr< $(ExId (_loc, s))$.contents>>
     else
       <:expr< $(ExId (_loc, s))$>>)
  | Ref (_loc, id) -> 
    <:expr< ! $parse_body id$>>
  | Float (_loc, s)  -> 
    ( match f.t with 
      | TFloat32  -> <:expr<$(ExFlo(_loc, s))$>>
      | _ -> assert (not debug); 
        raise (TypeError (t, f.t, f.loc)))
  | Float32 (_loc, s) -> 
    ( match f.t with 
      | TFloat32  -> <:expr< $(ExFlo (_loc, s))$>>
      | _ -> assert (not debug); 
        raise (TypeError (t, f.t, f.loc)))
  | Float64 (_loc, s) -> 
    ( match f.t with 
      | TFloat64  -> <:expr< $(ExFlo (_loc, s))$>>
      | _ -> assert (not debug); 
        raise (TypeError (t, f.t, f.loc)))
  | PlusF (_loc, a, b) | PlusF32 (_loc, a, b) | PlusF64  (_loc, a, b)-> 
    parse_body f 
  | MinF (_loc, a, b) | MinF32 (_loc, a, b) | MinF64  (_loc, a, b)-> 
    parse_body f 
  | MulF _ | MulF32 _  | MulF64 _-> 
    parse_body f 
  | DivF (_loc, a, b) | DivF32 (_loc, a, b) | DivF64  (_loc, a, b)-> 
    parse_body f 

  | VecGet (_loc, vector, index)  -> 
    ( match f.t with
      | x when x = t -> ()
      | TUnknown ->f.t <-  t
      | _  ->  assert (not debug); 
        raise (TypeError (t, f.t , _loc)));
    (match vector.e with
     | Id (_loc,s)  -> 
       let var = (Hashtbl.find !current_args (string_of_ident s)) in
       let type_constraint =
         let rec aux () =
           (match var.var_type with
            | TUnknown ->
              (var.var_type <- TVec t;
               aux ())
            | TVec k when k = t  -> (
                match k with 
                | TFloat32 -> 
                  <:ctyp<(float, Bigarray.float32_elt) Spoc.Vector.vector>>
                | TFloat64 -> 
                  <:ctyp<(float, Bigarray.float64_elt) Spoc.Vector.vector>>
                |  _  ->  assert false
              )
            | _  ->
              assert (not debug); 
              raise (TypeError (TVec t, var.var_type, _loc))

           ) in 
         aux () in
       begin
         <:expr<Spoc.Mem.get ($ExId(_loc,s)$:$type_constraint$) 
                (Int32.to_int $parse_body index$)>>
       end
     | _  ->
       assert (not debug); 
       failwith "Unknwown vector");
  | ModuleAccess _ -> parse_body f
  | Acc _ -> parse_body f
  | _ ->   
    my_eprintf (Printf.sprintf "(* val %s *)\n%!" (k_expr_to_string f.e));
    assert (not debug); 
    raise (TypeError (t, f.t, f.loc))

and parse_modacc m =
  let res = ref m in
  let rec aux m = 
    match m.e with
    | ModuleAccess (_loc, e1, e2) -> ExAcc(_loc, <:expr< $uid:e1$>>, aux e2)
    | Id _ -> parse_body m
    | App (_loc, e1, e2::[]) -> res:=e2; aux e1
    | _ -> assert false
  in
  (aux m), !res

and parse_app_ml a modu =
  match a.e with
  | Id (_loc,s) -> ExAcc(_loc, modu,ExId(_loc, s))
  | ModuleAccess (_loc, s, e) ->
    let e = parse_app_ml e (ExAcc(_loc, modu, ExId(_loc, IdUid(_loc,s)))) in e
  | App (_loc, e1, e2::[]) ->
    let rec aux a modu =
      match a.e with
      | Id (_loc,s) -> ExApp(_loc, ExAcc(_loc, modu,ExId(_loc, s)), parse_body e2)
      | App (l, e1, e2::[]) -> ExApp (_loc, aux e1 modu, parse_body e2)
      | ModuleAccess (_loc, s, e) -> 
        ExAcc(_loc, modu, (parse_body a))
      | _ -> my_eprintf (Printf.sprintf "(* app %s *)\n%!" (k_expr_to_string a.e)); 
        assert false 
    in
    ExApp(_loc, aux e1 modu, parse_body e2)
  | _ -> parse_body a 

and parse_case (_loc,patt,e) =
  match patt with 
  | Constr (s,Some i) ->
    let patt = 
      <:patt<$uid:s$ $lid:string_of_ident i$>> in
    <:match_case< $patt$   -> $parse_body e$ >>
  | Constr (s,None) ->
    <:match_case< $uid:s$ -> $parse_body e$ >>

and parse_body body = 
  my_eprintf (Printf.sprintf"(* val %s *)\n%!" (k_expr_to_string body.e));
  match body.e with
  | Bind (_loc, var,y, z, is_mutable)  ->
    (
      match var.e with
      | Id (_loc,s)  ->
	(match y.e with
         | Fun (_loc,stri,tt,funv) ->
           parse_body z;
         | _ ->
           let y = parse_body y in
           let gen_z = parse_body z in
           if is_mutable then
             (<:expr<let $PaId(_loc,s)$ = ref $y$ in $gen_z$>>)
           else
             (<:expr<let $PaId(_loc,s)$ = $y$ in $gen_z$>>)
        )
      | _ -> failwith "error parse_body Bind"
    )
  | Plus32 (_loc, a,b) -> 
    ( <:expr<Int32.add $(parse_int a TInt32)$  $(parse_int b TInt32)$>>)
  | Plus64 (_loc, a, b) ->
    ( <:expr<Int64.add $(parse_int a TInt64)$  $(parse_int b TInt64)$>>)
  | PlusF (_loc, a,b) -> 
    ( <:expr<$(parse_float a TFloat32)$ +. $(parse_float b TFloat32)$>>)
  | PlusF32 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat32) in
    let b_ = (parse_float b TFloat32) in
    ( <:expr<$a_$ +. $b_$>>)
  | PlusF64 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat64) in
    let b_ = (parse_float b TFloat64) in
    ( <:expr<$a_$ +. $b_$>>)
  | Min32 (_loc, a,b) -> 
    ( <:expr<Int32.sub $(parse_int a TInt32)$  $(parse_int b TInt32)$>>)
  | Min64 (_loc, a, b) ->
    ( <:expr<Int64.sub $(parse_int a TInt64)$  $(parse_int b TInt64)$>>)
  | MinF (_loc, a,b) -> 
    ( <:expr<$(parse_float a TFloat32)$ -. $(parse_float b TFloat32)$>>)
  | MinF32 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat32) in
    let b_ = (parse_float b TFloat32) in
    ( <:expr<$a_$ -. $b_$>>)
  | MinF64 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat64) in
    let b_ = (parse_float b TFloat64) in
    ( <:expr<$a_$ -. $b_$>>)
  | Mul32 (_loc, a,b) -> 
    ( <:expr<Int32.mul $(parse_int a TInt32)$  $(parse_int b TInt32)$>>)
  | Mul64 (_loc, a, b) ->
    ( <:expr<Int64.mul $(parse_int a TInt64)$  $(parse_int b TInt64)$>>)
  | MulF (_loc, a,b) -> 
    ( <:expr<$(parse_float a TFloat32)$ *. $(parse_float b TFloat32)$>>)
  | MulF32 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat32) in
    let b_ = (parse_float b TFloat32) in
    ( <:expr<$a_$ *. $b_$>>)
  | MulF64 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat64) in
    let b_ = (parse_float b TFloat64) in
    ( <:expr<$a_$ *. $b_$>>)
  | Div32 (_loc, a,b) -> 
    ( <:expr<Int32.div $(parse_int a TInt32)$  $(parse_int b TInt32)$>>)
  | Div64 (_loc, a, b) ->
    ( <:expr<Int64.div $(parse_int a TInt64)$  $(parse_int b TInt64)$>>)
  | DivF (_loc, a,b) -> 
    ( <:expr<$(parse_float a TFloat32)$ /. $(parse_float b TFloat32)$>>)
  | DivF32 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat32) in
    let b_ = (parse_float b TFloat32) in
    ( <:expr<$a_$ /. $b_$>>)
  | DivF64 (_loc, a,b) -> 
    let a_ = (parse_float a TFloat64) in
    let b_ = (parse_float b TFloat64) in
    ( <:expr<$a_$ /. $b_$>>)
  | Mod (_loc, a,b) -> 
    ( <:expr<Int32.rem $(parse_int a TInt32)$ $(parse_int b TInt32)$>>)
  | Id (_loc,s) -> 
    begin
      try
        let var = (Hashtbl.find !current_args (string_of_ident s)) in
        if var.is_mutable then
          <:expr< $ExId(_loc,s)$.contents>>
        else
          <:expr<$ExId(_loc,s)$>>
      with 
        Not_found ->
        <:expr<$ExId(_loc,s)$>>
    end
  | Int (_loc, i)  -> <:expr<$ExInt(_loc, i)$>>
  | Int32 (_loc, i)  -> <:expr<$ExInt32(_loc, i)$>>
  | Int64 (_loc, i)  -> <:expr<$ExInt64(_loc, i)$>>
  | Float (_loc, f)
  | Float32 (_loc,f)
  | Float64 (_loc,f)  -> <:expr<$ExFlo(_loc, f)$>>
  | Seq (_loc, x, y) -> 
    let x = parse_body x in
    let y = parse_body y in
    <:expr<$x$; $y$>>
  | End(_loc, x)  -> 
    let res = parse_body x in
    <:expr<$res$>> 
  | VecSet (_loc, vector, value)  -> 								
    let gen_value = parse_body value in
    (match vector.e with
     | VecGet (_, v,idx)  -> 
       let vec_typ_to_e t = 
         (match t with 
          | TInt32  -> <:ctyp<(int32, Bigarray.int32_elt) Spoc.Vector.vector>>
          | TInt64  -> <:ctyp<(int64, Bigarray.int64_elt) Spoc.Vector.vector>>
          | TFloat32 -> <:ctyp<(float, Bigarray.float32_elt) Spoc.Vector.vector>>
          | TFloat64 -> <:ctyp<(float, Bigarray.float64_elt) Spoc.Vector.vector>>
          | Custom (_,name)  ->  
            let name = TyId(_loc,IdLid(_loc,name)) 
            and sarek_name = TyId(_loc, IdLid(_loc,name^"_sarek")) in
            <:ctyp<($name$,$sarek_name$) Spoc.Vector.vector >> 
          | _ -> assert false
         ) in
       (match v.e with
        | Id (_loc,s)  -> 
          let var = (Hashtbl.find !current_args (string_of_ident s)) in
          let type_constaint = 
            (match var.var_type with
             | TVec k when k = value.t  -> 
               vec_typ_to_e k
             | _ ->   
               (match value.t, var.var_type with
                | TUnknown, TUnknown ->
                  (assert (not debug); raise (TypeError (TVec value.t, var.var_type, _loc)))
                | TUnknown, TVec tt ->
                  (var.var_type <- TVec tt;
                   vec_typ_to_e tt)
                | tt, TVec TUnknown ->
                  (var.var_type <- TVec tt;
                   vec_typ_to_e tt)
                | _ ->
                  (assert (not debug); raise (TypeError (TVec value.t, var.var_type, _loc))))
            )
          in

          <:expr<Spoc.Mem.set ($parse_body v$:$type_constaint$)
                 (Int32.to_int $parse_body idx$) $gen_value$>>
        | _  ->  failwith "Unknwown vector");
     | _  -> failwith (Printf.sprintf "erf %s" (k_expr_to_string vector.e)) ); 

  | ArrSet (_loc, array, value)  -> 								
    let gen_value = parse_body value in
    (match array.e with
     | ArrGet (_, a,idx)  -> 
       (match a.e with
        | Id (_loc,s)  -> 
          <:expr<($parse_body a$).(Int32.to_int $parse_body idx$) <- $gen_value$>>
        | _  ->  failwith "Unknwown array");
     | _  -> failwith (Printf.sprintf "erf_arr %s" (k_expr_to_string array.e)) ); 


  | VecGet(_loc, vector, index)  -> 
    ignore(parse_body vector);
    (match vector.e with
     | Id (_loc, s)-> 
       let var = (Hashtbl.find !current_args (string_of_ident s)) in
       let type_constraint = 
         (match var.var_type with
          | TVec k -> 
            (match k with 
             | TInt32  -> <:ctyp<(int32, Bigarray.int32_elt) Spoc.Vector.vector>>
             | TInt64  -> <:ctyp<(int64, Bigarray.int64_elt) Spoc.Vector.vector>>
             | TFloat32 -> <:ctyp<(float, Bigarray.float32_elt) Spoc.Vector.vector>>
             | TFloat64 -> <:ctyp<(float, Bigarray.float64_elt) Spoc.Vector.vector>>
             | Custom (_,name)  ->  
               let name = TyId(_loc,IdLid(_loc,name)) 
               and sarek_name = TyId(_loc, IdLid(_loc,name^"_sarek")) in
               <:ctyp<($name$,$sarek_name$) Spoc.Vector.vector >> 
             | TBool | TVec _  | TUnknown | TUnit | TArr _ 
             | TApp _   ->  assert false
            )
          | _  -> assert (not debug); 
            failwith (Printf.sprintf "strange vector %s" (ktyp_to_string var.var_type ))
         )in
       <:expr<Spoc.Mem.get ($ExId(_loc,s)$:$type_constraint$) (Int32.to_int $parse_body index$)>>
     | _  -> assert (not debug); 
       failwith (Printf.sprintf "strange vector %s" (k_expr_to_string body.e )))

  | ArrGet(_loc, array, index)  -> 
    ignore(parse_body array);
    (match array.e with
     | Id (_loc, s)-> 
       let var = (Hashtbl.find !current_args (string_of_ident s)) in
       let type_constraint = 
         (match var.var_type with
          | TArr k -> 
            (match k with 
             | TInt32,_  -> <:ctyp<int32 array>>
             | TInt64,_  -> <:ctyp<int64 array>>
             | TFloat32,_ -> <:ctyp<float array>>
             | TFloat64,_ -> <:ctyp<float array>>
             |	_  ->  assert false
            )
          | _  -> (assert (not debug); 
                   raise (TypeError (TArr (TUnknown, Any), var.var_type, _loc)));
         )in
       <:expr<($ExId(_loc,s)$:$type_constraint$).(Int32.to_int $parse_body index$)>>
     | _  -> assert (not debug); 
       failwith "strange array")
  | BoolEq32 (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ = $parse_int b TInt32$>>)
  | BoolEq64 (_loc, a, b) ->
    (<:expr<$parse_int a TInt64$ = $parse_int b TInt64$>>)
  | BoolEqF (_loc, a, b) ->
    (<:expr<$parse_float a TFloat32$ = $parse_float b TFloat32$>>)
  | BoolEqF64 (_loc, a, b) ->
    (<:expr<$parse_float a TFloat64$ = $parse_float b TFloat64$>>)
  | BoolLt (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ < $parse_int b TInt32$>>)
  | BoolLt32 (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ < $parse_int b TInt32$>>)
  | BoolLt64 (_loc, a, b) ->
    (<:expr<$parse_int a TInt64$ < $parse_int b TInt64$>>)
  | BoolLtF (_loc, a, b) ->
    (<:expr<$parse_float a TFloat32$ < $parse_float b TFloat32$>>)
  | BoolLtF64 (_loc, a, b) ->
    (<:expr<$parse_float a TFloat64$ < $parse_float b TFloat64$>>)
  | BoolGt (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ > $parse_int b TInt32$>>)
  | BoolGt32 (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ > $parse_int b TInt32$>>)
  | BoolGt64 (_loc, a, b) ->
    (<:expr<$parse_int a TInt64$ > $parse_int b TInt64$>>)
  | BoolGtF (_loc, a, b) ->
    (<:expr<$parse_float a TFloat32$ > $parse_float b TFloat32$>>)
  | BoolGtF64 (_loc, a, b) ->
    (<:expr<$parse_float a TFloat64$ > $parse_float b TFloat64$>>)
  | BoolLtE (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ <= $parse_int b TInt32$>>)
  | BoolLtE32 (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ <= $parse_int b TInt32$>>)
  | BoolLtE64 (_loc, a, b) ->
    (<:expr<$parse_int a TInt64$ <= $parse_int b TInt64$>>)
  | BoolLtEF (_loc, a, b) ->
    (<:expr<$parse_float a TFloat32$ <= $parse_float b TFloat32$>>)
  | BoolLtEF64 (_loc, a, b) ->
    (<:expr<$parse_float a TFloat64$ <= $parse_float b TFloat64$>>)
  | BoolGtE (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ >= $parse_int b TInt32$>>)
  | BoolGtE32 (_loc, a, b) ->
    (<:expr<$parse_int a TInt32$ >= $parse_int b TInt32$>>)
  | BoolGtE64 (_loc, a, b) ->
    (<:expr<$parse_int a TInt64$ >= $parse_int b TInt64$>>)
  | BoolGtEF (_loc, a, b) ->
    (<:expr<$parse_float a TFloat32$ >= $parse_float b TFloat32$>>)
  | BoolGtEF64 (_loc, a, b) ->
    (<:expr<$parse_float a TFloat64$ >= $parse_float b TFloat64$>>)
  | BoolOr (_loc, a, b) ->
    (<:expr<$parse_body a$ || $parse_body b$>>)
  | BoolAnd (_loc, a, b) ->
    (<:expr<$parse_body a$ &&  $parse_body b$>>)
  | Ife (_loc, cond, cons1, cons2) ->
    let cond  = parse_body cond in 
    let cons1 = parse_body cons1 in 
    let cons2 = parse_body cons2 in 
    (<:expr<if $cond$ then $cons1$ else $cons2$>>)
  | If (_loc, cond, cons1) ->
    let cond  = parse_body cond in 
    let cons1 = parse_body cons1 in 
    (<:expr<if $cond$ then $cons1$>>)
  | DoLoop (_loc, ({t=_;e=Id(_,s) ;loc=_} as id), min, max, body) ->
    let var = (Hashtbl.find !current_args (string_of_ident s)) in
    var.var_type <- TInt32;
    let min = (parse_body min) in
    let max = (parse_body max) in
    let body = parse_body body in
    let string_of_id id =
      match id with
      | Id (_,s ) -> string_of_ident s
      | _ ->  assert false
    in
    (<:expr<for 
            $PaId(_loc,
            IdLid(_loc, (string_of_id id.e)))$ = 
            (Int32.to_int $min$) to (Int32.to_int  $max$) do 
            let $(PaId (_loc, IdLid(_loc, (string_of_id id.e))))$ = 
            (Int32.of_int $(ExId (_loc, IdLid(_loc, (string_of_id id.e))))$) in
            $body$ 
            done>>)
  | While (_loc, cond, body) ->
    let cond = parse_body cond in
    let body = parse_body body in
    (<:expr< while $cond$ do $body$ done>>)
  | App (_loc, e1, e2) ->
    let rec aux e2 = 
      match e2 with
      | t::[] -> parse_body t
      | t::q -> ExApp(_loc, (aux q), (parse_body t))
      | [] -> assert false
    in 
    let e2 = aux e2 in
    let e1 = 
      match e1.e with
      | Id (_loc, s) ->
        (try 
	   ignore(Hashtbl.find !global_fun (string_of_ident s)); 
           <:expr<$parse_body e1$.ml_fun>>
         with
         | Not_found ->
           (try 
	      ignore(Hashtbl.find !local_fun (string_of_ident s));
	      <:expr<$parse_body e1$.ml_fun>>
            with
            | Not_found ->
	      parse_body e1) )
      | _ -> parse_body e1
    in
    <:expr<$e1$ $e2$>>
  | Open (_loc, id, e) ->
    let rec aux = function
      | IdAcc (l,a,b) -> aux a; aux b
      | IdUid (l,s) -> open_module s l
      | _ -> assert (not debug)
    in
    aux id;
    let e = <:expr<let open $id$ in $parse_body e$>> in
    let rec aux = function
      | IdAcc (l,a,b) -> aux a; aux b
      | IdUid (l,s) -> close_module s
      | _ -> assert (not debug)
    in
    aux id;
    e
  | Noop -> 
    let _loc = body.loc in
    <:expr< ()>>
  | Acc (_loc, var, value) ->
    (match var.t with
     | TVec _ ->
       parse_body {
         t = body.t; 
         e = (VecSet (_loc, var, value));
         loc = body.loc} 
     | _ -> 
       <:expr<$parse_body var$ <- $parse_body value$>>
    )
  | Ref (_, {loc=_; e=Id(_loc,s); t=_}) ->
    <:expr< ! $ExId(_loc, s)$>>
  | ModuleAccess (_loc, s, e) -> 
    let e = parse_app_ml e <:expr< $uid:s$>> in
    e
  | Match (_loc, e, mc) ->
    let e = parse_body e and
    mc = List.map parse_case mc in
    <:expr< match $e$ with $list:mc$ >>
  | Record (_loc,fl) ->
    let fl = List.map (
        fun (_loc,id,e) ->
            <:rec_binding< $lid:string_of_ident id$ =  $parse_body e$>>)
        fl  in
    let recb = List.fold_left (fun a b -> <:rec_binding< $a$; $b$>>) 
        (List.hd fl) (List.tl fl) in
    <:expr< { $recb$ } >>
  | RecGet (_loc,e,fld) ->
    <:expr< $parse_body e$.$lid:string_of_ident fld$>>
  | RecSet (_loc,e1,e2) ->
    <:expr< $parse_body e1$ <- $parse_body e2$>>
  | True _loc ->  <:expr< true >>
  | False _loc ->  <:expr< false >>
  | BoolEq (_loc,e1,e2) -> <:expr< $parse_body e1$ = $parse_body e2$>>
  | _ -> assert (not debug); failwith "unimplemented yet"
