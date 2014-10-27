(******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2013)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or 
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
 *******************************************************************************)

open Camlp4.PreCast
open Syntax
open Ast

open Types

let remove_int_var var = 
  match var.e with 
  | Id (_loc, s)  -> 
    Hashtbl.remove !current_args (string_of_ident s);
  | _ -> failwith "error new_var"


let rec  parse_int2 i t= 
  match i.e with
  | Id (_loc,s) -> 
    (try 
       let var = (Hashtbl.find !current_args (string_of_ident s)) in
       if var.is_global then
         <:expr<global_int_var $ExId(_loc,s)$>>
       else
         <:expr<var  $ExInt(_loc, string_of_int var.n)$>>
     with
     | Not_found ->
       try 
         let c_const = Hashtbl.find !intrinsics_const 
             (string_of_ident s) in
         match c_const.typ with
         | x when x = t -> 
           <:expr< intrinsics 
                   $ExStr(_loc, c_const.cuda_val)$ 
                   $ExStr(_loc, c_const.opencl_val)$>>
         | _  -> assert (not debug); 
           raise (TypeError (t, c_const.typ, _loc))
       with Not_found ->
         (assert (not debug); 
          raise (Unbound_value ((string_of_ident s),_loc))))
  | Ref (_, {loc=_; e=Id(_loc,s); t=_}) ->
    <:expr<global_int_var (fun () -> ! $ExId(_loc, s)$)>>
  | Int (_loc, s)  -> <:expr<spoc_int32 $(ExInt32 (_loc, s))$>>
  | Int32 (_loc, s)  -> <:expr<spoc_int32 $(ExInt32 (_loc, s))$>>
  | Int64 (_loc, s)  -> <:expr<spoc_int64 $(ExInt64 (_loc, s))$>>
  | Plus32 (_loc, a, b)| Plus64 (_loc, a, b)  -> 
    parse_body2 i false
  | Min32 (_loc, a, b)| Min64 (_loc, a, b)  -> 
    parse_body2 i false
  | Mul32 (_loc, a, b)| Mul64 (_loc, a, b)  -> 
    parse_body2 i false
  | Mod (_loc, a, b) -> 
    parse_body2 i false
  | Div32 (_loc, a, b)| Div64 (_loc, a, b)  -> 
    parse_body2 i false
  | Bind (_loc, var, y, z, is_mutable)  -> parse_body2 i false
  | VecGet (_loc, vector, index)  -> 
    <:expr<get_vec $parse_int2 vector (TVec t)$ 
           $parse_int2 index TInt32$>>
  | ArrGet (_loc, array, index)  -> 
    <:expr<get_arr $parse_int2 array (TVec t)$ 
           $parse_int2 index TInt32$>>
  | App _ -> parse_body2 i false
  | RecGet _ -> parse_body2 i false
  | _ -> (my_eprintf (Printf.sprintf "--> (*** val2 %s *)\n%!" (k_expr_to_string i.e));
          raise (TypeError (t, i.t, i.loc));)

and  parse_float2 f t= 
  match f.e with
  | App (_loc, e1, e2) ->
    parse_body2 f false
  | Id (_loc,s)  -> 
    (try 
       let var = (Hashtbl.find !current_args (string_of_ident s)) in
       if var.is_global then
         <:expr<global_float_var $ExId(_loc,s)$>>
       else
         <:expr<var  $ExInt(_loc, string_of_int var.n)$>>
     with
     | Not_found ->
       try 
         let c_const = Hashtbl.find !intrinsics_const (string_of_ident s) in
         match c_const.typ with
         | x when x = t -> 
           <:expr< intrinsics $ExStr(_loc, c_const.cuda_val)$ $ExStr(_loc, c_const.opencl_val)$>>
         | _  -> assert (not debug); 
           raise (TypeError (t, c_const.typ, _loc))
       with Not_found ->
         (assert (not debug); 
          raise (Unbound_value ((string_of_ident s),_loc))))
  | Ref (_, {loc=_; e=Id(_loc,s); t=_}) ->
    <:expr<global_float_var (fun () -> ! $ExId(_loc, s)$)>>
  | Float (_loc, s)  -> <:expr<spoc_float $(ExFlo(_loc, s))$>>
  | Float32 (_loc, s)  -> <:expr<spoc_float $(ExFlo(_loc, s))$>>
  | Float64 (_loc, s)  -> <:expr<spoc_double $(ExFlo(_loc, s))$>>
  | PlusF (_loc, a, b) |  PlusF32 (_loc, a, b) | PlusF64 (_loc, a, b)  -> 
    parse_body2 f false
  | MinF (_loc, a, b) |  MinF32 (_loc, a, b) | MinF64 (_loc, a, b)  -> 
    parse_body2 f false
  | MulF (_loc, a, b) |  MulF32 (_loc, a, b) | MulF64 (_loc, a, b)  -> 
    parse_body2 f false
  | DivF (_loc, a, b) |  DivF32 (_loc, a, b) | DivF64 (_loc, a, b)  -> 
    parse_body2 f false
  | VecGet (_loc, vector, index)  -> 
    <:expr<get_vec $parse_float2 vector (TVec t)$ $parse_int2 index TInt32$>>
  | ModuleAccess _ -> parse_body2 f false
  | RecGet _ ->  parse_body2 f false
  | _  -> ( my_eprintf (Printf.sprintf "(*** val2 %s *)\n%!" (k_expr_to_string f.e));
            raise (TypeError (t, f.t, f.loc));)

and parse_app a =
  my_eprintf (Printf.sprintf "(* val2 parse_app %s *)\n%!" (k_expr_to_string a.e));
  match a.e with
  | App (_loc, e1, e2::[]) ->
    let res = ref [] in
    let constr = ref false in
    let rec aux app =
      my_eprintf (Printf.sprintf "(* val2 parse_app_app %s *)\n%!" (k_expr_to_string app.e));
      match app.e with
      | Id (_loc, s) ->
        (try 
           let intr = Hashtbl.find !intrinsics_fun (string_of_ident s) in
           <:expr< intrinsics $ExStr(_loc, intr.cuda_val)$ $ExStr(_loc, intr.opencl_val)$>> 
         with Not_found -> 
           try 
	     ignore(Hashtbl.find !global_fun (string_of_ident s));
              (<:expr< global_fun $id:s$>> )
           with Not_found -> 
             try 
	       ignore(Hashtbl.find !local_fun (string_of_ident s));
               <:expr< global_fun $id:s$>> 
             with Not_found -> 
               try
                 let t = Hashtbl.find !constructors (string_of_ident s) in
                 constr := true;
                 <:expr< spoc_constr $str:t.name$ $str:string_of_ident s$ [$parse_body2 e2 false$]>> 
               with _ -> 
                 parse_body2 e1 false;)
      | App (_loc, e3, e4::[]) ->
        let e = aux e3 in
        res := <:expr< ($parse_body2 e4 false$)>> :: !res;
        e
      | ModuleAccess (_loc, s, e3) ->
        open_module s _loc; 
        let e = aux e3 in
        close_module s;
        e
      | _  -> assert false;
    in 
    let intr = aux e1 in
    if !constr then
      <:expr< $intr$ >>
    else
      (
        res := (parse_body2 e2 false) :: !res ; 
        (match !res with
         | [] -> assert false
         | t::[] -> 
           <:expr< app $intr$ [| ($t$) |]>>
         | t::q ->
           <:expr< app $intr$ [| $exSem_of_list (List.rev !res)$ |]>>)
      )
  | _ -> parse_body2 a false



and expr_of_app t _loc gen_var y =
  match t with
  | TApp (t1,((TApp (t2,t3)) as tt)) ->
    expr_of_app tt _loc gen_var y
  | TApp (t1,t2) -> 
    (match t2 with
     | TInt32 -> <:expr<(new_int_var $`int:gen_var.n$)>>, (parse_body2 y false)
     | TInt64 -> <:expr<(new_int_var $`int:gen_var.n$)>>, (parse_body2 y false)
     | TFloat32 -> <:expr<(new_float_var $`int:gen_var.n$)>>,(parse_body2 y false)
     | TFloat64 -> <:expr<(new_double_var $`int:gen_var.n$)>>, (parse_body2 y false)
     | _  ->  failwith "unknown var type")
  | _ -> assert false


and parse_case2 mc _loc =
  let aux (_loc,patt,e) = 
       match patt with
       | Constr (_,None) ->
         <:expr< spoc_case $`int:ident_of_patt _loc patt$ None $parse_body2 e false$>> 
       | Constr (s,Some id) ->
         incr arg_idx;
         incr arg_idx;
         Hashtbl.add !current_args (string_of_ident id) 
           {n = !arg_idx; var_type = ktyp_of_typ (TyId(_loc,IdLid(_loc,type_of_patt patt)));
            is_mutable = false;
            read_only = false;
            write_only = false;
            is_global = false;};
         let e = 
           <:expr< spoc_case $`int:ident_of_patt _loc patt$ 
                         (Some ($str:ctype_of_sarek_type (type_of_patt patt)$,$str:s$,$`int:!arg_idx$)) $parse_body2 e false$>> in
         Hashtbl.remove !current_args (string_of_ident id);
         e 
  in 
  let l = List.map aux mc
  in <:expr< [| $exSem_of_list l$ |]>>

and parse_body2 body bool = 
  let rec aux ?return_bool:(r=false) body =
    my_eprintf (Printf.sprintf "(* val2 %s *)\n%!" (k_expr_to_string body.e));
    match body.e with
    | Bind (_loc, var,y, z, is_mutable)  ->             
      (match var.e with 
       | Id (_loc, s)  -> 
	 (match y.e with
	  | Fun _ -> parse_body2 z bool;
	  | _ ->
	    (let gen_var = 
	      try (Hashtbl.find !current_args (string_of_ident s)) 
	      with _ -> assert false in
	     let rec f () = 
	       match var.t with
	       | TInt32 -> <:expr<(new_int_var $`int:gen_var.n$)>>, (aux y)
	       | TInt64 -> <:expr<(new_int_var $`int:gen_var.n$)>>, (aux y)
	       | TFloat32 -> <:expr<(new_float_var $`int:gen_var.n$)>>,(aux y)
	       | TFloat64 -> <:expr<(new_double_var $`int:gen_var.n$)>>,(aux y)
	       | TApp _ -> expr_of_app var.t _loc gen_var y
               | Custom (t,n) ->
                 <:expr<(new_custom_var $str:n$ $`int:gen_var.n$)>>,(aux y)
	       | TUnknown -> if gen_var.var_type <> TUnknown then 
		   ( var.t <- gen_var.var_type;
		     f ();)
		 else
		   (assert (not debug); 
                    raise (TypeError (TUnknown, gen_var.var_type , _loc));)
	       | TArr (t,s) -> 
                 let elttype = 
                   match t with
                   | TInt32 -> <:expr<eint32>>
                   | TInt64 -> <:expr<eint64>>
                   | TFloat32 -> <:expr<efloat32>>
                   | TFloat64 -> <:expr<efloat64>>
		   | _ -> assert false
                 and memspace =
                   match s with
                   | Local -> <:expr<local>>
                   | Shared -> <:expr<shared>>
                   | Global -> <:expr<global>> 
		   | _ -> assert false 
		 in
                 <:expr<(new_array $`int:gen_var.n$) ($aux y$) $elttype$ $memspace$>>,(aux y)
	      | _  ->  ( assert (not debug); 
                          raise (TypeError (TUnknown, gen_var.var_type , _loc));)
             in
             let ex1, ex2 = f () in
	     arg_list := <:expr<(spoc_declare $ex1$)>>:: !arg_list;
	     (let var_ = parse_body2 var false in
	      let y = aux y in
	      let z = aux z in
	      let res = 
		match var.t with
		  TArr _ ->  <:expr< $z$>>
		| _ -> <:expr< seq (spoc_set $var_$ $y$) $z$>>
	      in remove_int_var var;
	      res)))
       | _  ->  failwith "strange binding");
    | Plus32 (_loc, a,b) -> body.t <- TInt32; 
      let p1 = (parse_int2 a TInt32) 
      and p2 = (parse_int2 b TInt32) in
      if not r then 
        return_type := TInt32;
      ( <:expr<spoc_plus $p1$ $p2$>>) 
    | Plus64 (_loc, a,b) -> body.t <- TInt64; 
      let p1 = (parse_int2 a TInt64) 
      and p2 = (parse_int2 b TInt64) in
      if not r then 
        return_type := TInt64;
      ( <:expr<spoc_plus $p1$ $p2$>>) 
    | PlusF (_loc, a,b) -> 
      let p1 = (parse_float2 a TFloat32) 
      and p2 = (parse_float2 b TFloat32) in
      if not r then 
        return_type := TFloat32;
      ( <:expr<spoc_plus_float $p1$ $p2$>>) 
    | PlusF32 (_loc, a,b) -> 
      let p1 = (parse_float2 a TFloat32) 
      and p2 = (parse_float2 b TFloat32) in
      if not r then 
        return_type := TFloat32;
      ( <:expr<spoc_plus_float $p1$ $p2$>>) 
    | PlusF64 (_loc, a,b) -> 
      let p1 = (parse_float2 a TFloat64) 
      and p2 = (parse_float2 b TFloat64) in
      if not r then 
        return_type := TFloat64;
      ( <:expr<spoc_plus_float $p1$ $p2$>>) 
    | Min32 (_loc, a,b) -> body.t <- TInt32; 
      ( <:expr<spoc_min $(parse_int2 a TInt32)$ $(parse_int2 b TInt32)$>>)
    | Min64 (_loc, a,b) -> body.t <- TInt64; 
      ( <:expr<spoc_min $(parse_int2 a TInt64)$ $(parse_int2 b TInt64)$>>)
    | MinF (_loc, a,b) -> 
      ( <:expr<spoc_min_float $(parse_float2 a TFloat32)$ $(parse_float2 b TFloat32)$>>)
    | MinF32 (_loc, a,b) -> 
      ( <:expr<spoc_min_float $(parse_float2 a TFloat32)$ $(parse_float2 b TFloat32)$>>)
    | MinF64 (_loc, a,b) -> 
      ( <:expr<spoc_min_float $(parse_float2 a TFloat64)$ $(parse_float2 b TFloat64)$>>)
    | Mul32 (_loc, a,b) -> 
      if not r then 
        return_type := TInt32;
      ( <:expr<spoc_mul $(parse_int2 a TInt32)$ $(parse_int2 b TInt32)$>>)
    | Mul64 (_loc, a,b) -> body.t <- TInt64; 
      ( <:expr<spoc_mul $(parse_int2 a TInt64)$ $(parse_int2 b TInt64)$>>)
    | MulF (_loc, a,b) -> 
      ( <:expr<spoc_mul_float $(parse_float2 a TFloat32)$ $(parse_float2 b TFloat32)$>>)
    | MulF32 (_loc, a,b) -> 
      if not r then 
        return_type := TFloat32;
      ( <:expr<spoc_mul_float $(parse_float2 a TFloat32)$ $(parse_float2 b TFloat32)$>>)
    | MulF64 (_loc, a,b) -> 
      ( <:expr<spoc_mul_float $(parse_float2 a TFloat64)$ $(parse_float2 b TFloat64)$>>)
    | Div32 (_loc, a,b) -> body.t <- TInt32; 
      ( <:expr<spoc_div $(parse_int2 a TInt32)$ $(parse_int2 b TInt32)$>>)
    | Div64 (_loc, a,b) -> body.t <- TInt64; 
      ( <:expr<spoc_div $(parse_int2 a TInt64)$ $(parse_int2 b TInt64)$>>)
    | DivF (_loc, a,b) -> 
      ( <:expr<spoc_div_float $(parse_float2 a TFloat32)$ $(parse_float2 b TFloat32)$>>)
    | DivF32 (_loc, a,b) -> 
      ( <:expr<spoc_div_float $(parse_float2 a TFloat32)$ $(parse_float2 b TFloat32)$>>)
    | DivF64 (_loc, a,b) -> 
      ( <:expr<spoc_div_float $(parse_float2 a TFloat64)$ $(parse_float2 b TFloat64)$>>)
    | Mod (_loc, a,b) -> body.t <- TInt32; 
      let p1 = (parse_int2 a TInt32) 
      and p2 = (parse_int2 b TInt32) in
      if not r then 
        return_type := TInt32;
      ( <:expr<spoc_mod $p1$ $p2$>>) 
    | Id (_loc,s)  -> 
      (try 
         let var = 
           (Hashtbl.find !current_args (string_of_ident s))  in
         ( match var.var_type with 
           | TUnit -> <:expr< Unit>>
           | _ -> 
             body.t <- var.var_type;
             if var.is_global then
               match var.var_type with
               | TFloat32 ->
                 <:expr<global_float_var (fun () -> $ExId(_loc,s)$)>>
               | TInt32 -> <:expr<global_int_var (fun () -> $ExId(_loc,s)$)>>
               | _ -> assert false
             else
               <:expr<var  $ExInt(_loc, string_of_int var.n)$>>)
       with _  ->  
         try 
           let c_const = (Hashtbl.find !intrinsics_const (string_of_ident s))  in
           if body.t <> c_const.typ then
             if body.t = TUnknown then
               body.t <- c_const.typ
             else
               (assert (not debug); raise (TypeError (c_const.typ, body.t, _loc)));
           <:expr<intrinsics $ExStr(_loc, c_const.cuda_val)$ $ExStr(_loc, c_const.opencl_val)$>>
         with _ -> 
           (try 
              let intr = 
                Hashtbl.find !intrinsics_fun (string_of_ident s) in
              <:expr< intrinsics $ExStr(_loc, intr.cuda_val)$ $ExStr(_loc, intr.opencl_val)$>>
            with Not_found -> 
              try 
                ignore(Hashtbl.find !global_fun (string_of_ident s));
                <:expr< global_fun $id:s$>>
              with Not_found -> 
		try 
                  ignore(Hashtbl.find !local_fun (string_of_ident s)); 
                  <:expr< global_fun $id:s$>>
  with Not_found -> 
    try 
      let t = Hashtbl.find !constructors (string_of_ident s) in
      <:expr< spoc_constr $str:t.name$ $str:(string_of_ident s)$ [] >>
with 
| _  ->
  (assert (not debug); 
                   raise (Unbound_value ((string_of_ident s), _loc)))));
    | Int (_loc, i)  -> <:expr<spoc_int $ExInt(_loc, i)$>>
    | Int32 (_loc, i)  -> <:expr<spoc_int32 $ExInt32(_loc, i)$>>
    | Int64 (_loc, i)  -> <:expr<spoc_int64 $ExInt64(_loc, i)$>>
    | Float (_loc, f)  -> <:expr<spoc_float $ExFlo(_loc, f)$>>
    | Float32 (_loc, f) -> <:expr<spoc_float $ExFlo(_loc, f)$>>
    | Float64 (_loc, f) -> <:expr<spoc_double $ExFlo(_loc, f)$>>
    | Seq (_loc, x, y) -> 
      (match y.e with
       | Seq _ ->
         let x = parse_body2 x false in
         let y = parse_body2 y false in
         <:expr<seq $x$ $y$>>
       | _ -> 
         let e1 = parse_body2 x false in
         let e2 = aux y
         in  <:expr<seq $e1$ $e2$>> 
      )
    | End (_loc, x)  -> 
      let res = <:expr< $aux x$>> 
      in 
      <:expr<$res$>> 
    | VecSet (_loc, vector, value)  -> 
      let gen_value = aux (~return_bool:true) value in
      let gen_value = 
        match vector.t, value.e with
        | TInt32, (Int32 _) -> <:expr<( $gen_value$)>> 
        | TInt64, (Int64 _) -> <:expr<( $gen_value$)>> 
        | TFloat32, (Float32 _) -> <:expr<( $gen_value$)>> 
        | TFloat64, (Float64 _) -> <:expr<( $gen_value$)>> 
        | _ -> gen_value
      in
      let v = aux  (~return_bool:true) vector in
      let e = <:expr<set_vect_var $v$ $gen_value$>> in
      return_type := TUnit; 
      e
    | VecGet(_loc, vector, index)  -> 
      let e =
        <:expr<get_vec $aux vector$ $parse_int2 index TInt32$>> in
      (match vector.t with
       | TVec ty->
         ();
       | _ ->
         assert (not debug));
      e

    | ArrSet (_loc, array, value)  -> 
      let gen_value = aux (~return_bool:true) value in
      let gen_value = 
        match array.t, value.e with
        | TInt32, (Int32 _) -> <:expr<( $gen_value$)>> 
        | TInt64, (Int64 _) -> <:expr<( $gen_value$)>> 
        | TFloat32, (Float32 _) -> <:expr<( $gen_value$)>> 
        | TFloat64, (Float64 _) -> <:expr<( $gen_value$)>> 
        | _ -> gen_value
      in
      let v = aux  (~return_bool:true) array in
      let e = <:expr<set_arr_var $v$ $gen_value$>> in
      return_type := TUnit; 
      e
    | ArrGet(_loc, array, index)  -> 
      let e =
        <:expr<get_arr $aux array$ $parse_int2 index TInt32$>> in
      (match array.t with
       | TArr ty->
         ();
       | _ ->
         assert (not debug));
      e
    | True _loc  -> 
      if not r then 
        return_type := TBool;
      <:expr<spoc_int32 $(ExInt32 (_loc, "1"))$>>
    | False _loc  -> 
      if not r then 
        return_type := TBool;
      <:expr<spoc_int32 $(ExInt32 (_loc, "0"))$>>

    | BoolOr(_loc, a, b) ->
      if not r then 
        return_type := TBool;
      <:expr< b_or $aux a$ $aux b$>>
    | BoolAnd(_loc, a, b) ->
      if not r then 
        return_type := TBool;
      <:expr< b_and $aux a$ $aux b$>>
    | BoolEq(_loc, a, b) ->
      if not r then 
        return_type := TBool;
      (match a.t with
       | Custom (_,n)  ->
         <:expr< equals_custom $str:"spoc_custom_compare_"^n^"_sarek"$
                 $aux a$ $aux b$>>
       | _ -> <:expr< equals32 $aux a$ $aux b$>>
      )
    | BoolEq32 (_loc, a, b) ->
      if not r then 
        return_type := TBool;
      <:expr< equals32 $aux a$ $aux b$>>
    | BoolEq64(_loc, a, b) ->
      <:expr< equals64 $aux a$ $aux b$>>
    | BoolEqF(_loc, a, b) ->
      <:expr< equalsF $aux a$ $aux b$>>
    | BoolEqF64(_loc, a, b) ->
      <:expr< equalsF64 $aux a$ $aux b$>>
    | BoolLt(_loc, a, b) ->
      let p1 = (parse_int2 a TInt32) 
      and p2 = (parse_int2 b TInt32) in
      if not r then 
        return_type := TInt32;
      ( <:expr<lt $p1$ $p2$>>) 
    | BoolLt32(_loc, a, b) ->
      let p1 = (parse_int2 a TInt32) 
      and p2 = (parse_int2 b TInt32) in
      if not r then 
        return_type := TInt32;
      ( <:expr<lt32 $p1$ $p2$>>) 
    | BoolLt64(_loc, a, b) ->
      <:expr< lt64 $aux a$ $aux b$>>
    | BoolLtF(_loc, a, b) ->
      <:expr< ltF $aux a$ $aux b$>>
    | BoolLtF64(_loc, a, b) ->
      <:expr< ltF64 $aux a$ $aux b$>>
    | BoolGt(_loc, a, b) ->
      let p1 = (parse_int2 a TInt32) 
      and p2 = (parse_int2 b TInt32) in
      if not r then 
        return_type := TInt32;
      ( <:expr<gt $p1$ $p2$>>) 
    | BoolGt32(_loc, a, b) ->
      let p1 = (parse_int2 a TInt32) 
      and p2 = (parse_int2 b TInt32) in
      if not r then 
        return_type := TInt32;
      ( <:expr<gt32 $p1$ $p2$>>) 
    | BoolGt64(_loc, a, b) ->
      <:expr< gt64 $aux a$ $aux b$>>
    | BoolGtF(_loc, a, b) ->
      <:expr< gtF $aux a$ $aux b$>>
    | BoolGtF64(_loc, a, b) ->
      <:expr< gtF64 $aux a$ $aux b$>>
    | BoolLtE(_loc, a, b) ->
      <:expr< lte $aux a$ $aux b$>>
    | BoolLtE32(_loc, a, b) ->
      <:expr< lte32 $aux a$ $aux b$>>
    | BoolLtE64(_loc, a, b) ->
      <:expr< lte64 $aux a$ $aux b$>>
    | BoolLtEF(_loc, a, b) ->
      <:expr< lteF $aux a$ $aux b$>>
    | BoolLtEF64(_loc, a, b) ->
      <:expr< lteF64 $aux a$ $aux b$>>

    | BoolGtE(_loc, a, b) ->
      <:expr< gte $aux a$ $aux b$>>
    | BoolGtE32(_loc, a, b) ->
      <:expr< gte32 $aux a$ $aux b$>>
    | BoolGtE64(_loc, a, b) ->
      <:expr< gte64 $aux a$ $aux b$>>
    | BoolGtEF(_loc, a, b) ->
      <:expr< gteF $aux a$ $aux b$>>
    | BoolGtEF64(_loc, a, b) ->
      <:expr< gteF64 $aux a$ $aux b$>>
    | Ife (_loc, cond, cons1, cons2) ->
      let p1 = aux cond
      and p2 = aux cons1
      and p3 = aux cons2 
      in
      if  r then 
        return_type := cons2.t;
      (<:expr< spoc_ife $p1$ $p2$ $p3$>>)
    | If (_loc, cond, cons1) ->
      (<:expr< spoc_if $aux cond$ $aux cons1$>>)
    | DoLoop (_loc, id, min, max, body) ->
      (<:expr<spoc_do $aux id$ $aux min$ $aux max$ $aux body$>>)
    | While (_loc, cond, body) ->
      let cond = aux cond in
      let body = aux body in
      (<:expr<spoc_while $cond$ $body$>>)
    | App (_loc, e1, e2) ->
      let e = <:expr< $parse_app body$>> in
      e
    | Open (_loc, id, e) ->
      let rec aux2 = function
        | IdAcc (l,a,b) -> aux2 a; aux2 b
        | IdUid (l,s) -> open_module s l
        | _ -> assert (not debug)
      in
      aux2 id;
      let ex = <:expr< $aux e$>> in
      let rec aux2 = function
        | IdAcc (l,a,b) -> aux2 a; aux2 b
        | IdUid (l,s) -> close_module s
        | _ -> assert (not debug)
      in
      aux2 id;
      ex
    | ModuleAccess (_loc, s, e) ->
      open_module s _loc;
      let ex = <:expr< $aux e$>> in
      close_module s;
      ex
    | Noop ->
      let _loc = body.loc in
      <:expr< spoc_unit () >>
    | Acc (_loc, e1, e2) ->
      let e1 = parse_body2 e1 false 
      and e2 = parse_body2 e2 false
      in
      if not r then 
        return_type := TUnit;
      <:expr< spoc_acc $e1$ $e2$>>
    | Ref (_loc, {t=_; e=Id(_loc2, s);loc=_}) ->
      let var = 
        Hashtbl.find !current_args (string_of_ident s)  in
      body.t <- var.var_type;
      if var.is_global then
        match var.var_type with
        | TFloat32 ->
          <:expr<global_float_var (fun _ -> ! $ExId(_loc,s)$)>>
        | TInt32 -> <:expr<global_int_var (fun _ -> ! $ExId(_loc,s)$)>>
        | _ -> assert false
      else
        assert false
    | Match(_loc,e,
            ((_,Constr (n,_),ec)::q as mc )) ->
      let e = parse_body2 e false 
      and mc = parse_case2 mc _loc in
      let name = (Hashtbl.find !constructors n).name in
      if not r then
        return_type := ec.t;
      <:expr< spoc_match $str:name$ $e$ $mc$ >>
    | Match _ -> assert false
    | Record (_loc,fl) ->
      (*get Krecord from field list *)
      let t,name = 
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
       in ktyp_of_typ (TyId(_loc,IdLid(_loc,List.hd r))),(List.hd r)
     in
     
     (* sort fields *)
     (match t with
      | Custom (KRecord (l1,l2,_),n) ->
        let fl = List.map
            (fun x -> List.find (fun (_,y,_) -> 
                 (string_of_ident y) = (string_of_ident x)) fl) l2 in 
        let r = List.map 
            (fun  (_,_,b) -> <:expr< $parse_body2 b false$>>)
            fl
        in <:expr< spoc_record $str:name$ [$Ast.exSem_of_list r$] >>
      | _ -> assert false)
    | RecGet (_loc,r,fld) -> <:expr< spoc_rec_get $parse_body2 r false$ $str:string_of_ident fld$>>
    | RecSet (_loc,e1,e2) -> <:expr< spoc_rec_set $parse_body2 e1 false$ $parse_body2 e2 false$>>
    | _ -> assert (not debug); failwith "pb2 : unimplemented yet"
  in
  let _loc = body.loc in
  if bool then 
    (
      my_eprintf (Printf.sprintf"(* val2 %s *)\n%!" (k_expr_to_string body.e));
      match body.e with
      | Bind (_loc, var,y, z, is_mutable)  -> 
        (
          (match var.e with 
           | Id (_loc, s) ->
	     (match y.e with
	      | Fun _ -> ()
	      | _ ->
		(let gen_var = (
		   try Hashtbl.find !current_args (string_of_ident s)
		   with _ -> assert false) in
		 let ex1,ex2 = 
                   match var.t with
                   | TInt32 -> <:expr<(new_int_var $`int:gen_var.n$)>>, (aux y)
                   | TInt64 -> <:expr<(new_int_var $`int:gen_var.n$)>>, (aux y)
                   | TFloat32 -> <:expr<(new_float_var $`int:gen_var.n$)>>,(aux y)
                   | TFloat64 -> <:expr<(new_double_var $`int:gen_var.n$)>>,(aux y)
                   | TUnit -> <:expr<Unit>>,aux y; 
                   | _  -> assert (not debug); 
                     failwith "unknown var type"
		 in
		 match var.t with
		 |  TUnit ->
                   arg_list := <:expr<Seq ($ex1$, $ex2$)>>::!arg_list
		 | _ -> arg_list := <:expr<spoc_set $ex1$ $ex2$>>:: !arg_list);
	     );
             let res = <:expr<$parse_body2 z true$>>
             in remove_int_var var;
	     res
	   | _ -> assert false))
      | Seq (a,b,c)  -> aux body
      | _  -> 
        let e = {t=body.t; e =End(_loc, body); loc = _loc} in 
        match body.t with 
        | TUnit ->
          let res = aux e in
          return_type := TUnit;
          <:expr< $res$ >>
        |_ -> 
          <:expr<spoc_return $aux e$>>
    )
  else
    aux body
