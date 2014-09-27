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

open Typer

let remove_int_var var = 
  match var.e with 
  | Id (_loc, s)  -> 
    Hashtbl.remove !current_args (string_of_ident s);
  | _ -> failwith "error new_var"

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

and  parse_int2 i t= 
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
  | _ -> (my_eprintf (Printf.sprintf "--> (* val %s *)\n%!" (k_expr_to_string i.e));
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
  | _  -> ( my_eprintf (Printf.sprintf "(* val %s *)\n%!" (k_expr_to_string f.e));
            raise (TypeError (t, f.t, f.loc));)
          
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
    ( <:expr<Int32.min $(parse_int a TInt32)$  $(parse_int b TInt32)$>>)
  | Min64 (_loc, a, b) ->
    ( <:expr<Int64.min $(parse_int a TInt64)$  $(parse_int b TInt64)$>>)
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
    ( <:expr<$(parse_int a TInt32)$ mod $(parse_int b TInt32)$>>)
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
          |  _  ->  assert false
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
             | TBool | TVec _  | TUnknown | TUnit | TArr _ 
             | TApp _  | Custom _ ->  assert false
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
  | _ -> assert (not debug); failwith "unimplemented yet"






and parse_app a =
  my_eprintf (Printf.sprintf "(* val2 parse_app %s *)\n%!" (k_expr_to_string a.e));
  match a.e with
  | App (_loc, e1, e2::[]) ->
    let res = ref [] in
    let rec aux app =
      match app.e with
      | 	Id (_loc, s) ->
        (try 
           let intr = Hashtbl.find !intrinsics_fun (string_of_ident s) in
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
    res := (parse_body2 e2 false) :: !res ; 
    (match !res with
     | [] -> assert false
     | t::[] -> 
       <:expr< app $intr$ [| ($t$) |]>>
     | t::q ->
       <:expr< app $intr$ [| $exSem_of_list (List.rev !res)$ |]>>)
  | _ -> parse_body2 a false



and expr_of_app t _loc gen_var y =
  match t with
  | TApp (t1,((TApp (t2,t3)) as tt)) ->
    expr_of_app tt _loc gen_var y
  | TApp (t1,t2) -> 
    (match t2 with
     | TInt32 -> <:expr<(new_int_var $ExInt(_loc,string_of_int gen_var.n)$)>>, (parse_body2 y false)
     | TInt64 -> <:expr<(new_int_var $ExInt(_loc,string_of_int gen_var.n)$)>>, (parse_body2 y false)
     | TFloat32 -> <:expr<(new_float_var $ExInt(_loc,string_of_int gen_var.n)$)>>,(parse_body2 y false)
     | TFloat64 -> <:expr<(new_double_var $ExInt(_loc,string_of_int gen_var.n)$)>>, (parse_body2 y false)
     | _  ->  failwith "unknown var type")
  | _ -> assert false

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
	       | TInt32 -> <:expr<(new_int_var $ExInt(_loc,string_of_int gen_var.n)$)>>, (aux y)
	       | TInt64 -> <:expr<(new_int_var $ExInt(_loc,string_of_int gen_var.n)$)>>, (aux y)
	       | TFloat32 -> <:expr<(new_float_var $ExInt(_loc,string_of_int gen_var.n)$)>>,(aux y)
	       | TFloat64 -> <:expr<(new_double_var $ExInt(_loc,string_of_int gen_var.n)$)>>,(aux y)
	       | TApp _ -> expr_of_app var.t _loc gen_var y
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
                 <:expr<(new_array $ExInt(_loc,string_of_int gen_var.n)$) ($aux y$) $elttype$ $memspace$>>,(aux y)
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
    | BoolOr(_loc, a, b) ->
      <:expr< b_or $aux a$ $aux b$>>
    | BoolAnd(_loc, a, b) ->
      <:expr< b_and $aux a$ $aux b$>>
    | BoolEq32 (_loc, a, b) ->
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

    | _ -> assert (not debug); failwith "unimplemented yet"
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
                   | TInt32 -> <:expr<(new_int_var $ExInt(_loc,string_of_int gen_var.n)$)>>, (aux y)
                   | TInt64 -> <:expr<(new_int_var $ExInt(_loc,string_of_int gen_var.n)$)>>, (aux y)
                   | TFloat32 -> <:expr<(new_float_var $ExInt(_loc,string_of_int gen_var.n)$)>>,(aux y)
                   | TFloat64 -> <:expr<(new_double_var $ExInt(_loc,string_of_int gen_var.n)$)>>,(aux y)
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



let parse_args params body= 
  let _loc = Loc.ghost in
  let rec aux acc = function
    | []  ->  acc
    | t::q -> aux <:expr<fun $t$ -> $acc$>> q
  in aux body
    (List.rev params)





let gen_arg_from_patt p = 
  match p with
  | PaId (_loc, x)   -> <:expr<(spoc_id $ExId(_loc,x)$)>>
  | _  -> failwith "error gen_arg_from_patt"


let gen_arg_from_patt2 p =  

  match p with
  | PaId (_loc, IdLid (_loc2, x) ) -> 

    (let var = (Hashtbl.find !current_args x) in   
     match var.var_type with
     | TUnknown  -> <:expr<(new_unknown_var $ExInt(_loc2,  string_of_int var.n)$)>>
     | TInt32 | TInt64 ->                                                 
       <:expr<(new_int_var $ExInt(_loc2,  string_of_int var.n)$)>>
     | TFloat32 -> 
       <:expr<(new_float_var $ExInt(_loc2,  string_of_int var.n)$)>>
     | TFloat64  -> 
       <:expr<(new_float64_var $ExInt(_loc2,  string_of_int var.n)$)>>
     | TVec k -> 
       (match k with
        | TInt32 | TInt64  ->   <:expr<(new_int_vec_var $ExInt(_loc2,  string_of_int var.n)$)>>
        | TFloat32 ->   <:expr<(new_float_vec_var $ExInt(_loc2,  string_of_int var.n)$)>>
        | TFloat64 ->   <:expr<(new_double_vec_var $ExInt(_loc2,  string_of_int var.n)$)>>
        | _  -> failwith "Forbidden vector  type in kernel declaration")
     | _ ->  failwith "unimplemented yet"
    )
  | _  -> failwith "error gen_arg_form_patt2"




let patt_is_vector p =
  match p with
  | PaId (_loc, IdLid (_loc2, x) ) -> 
    let var = (Hashtbl.find !current_args x) in   
    begin
      match var.var_type with
      | TVec _ ->true
      | _ -> false
    end
  | _ -> assert false


let gen_arg_from_patt3 p =  
  match p with
  | PaId (_loc, IdLid (_loc2, x) ) -> 
    (let var = (Hashtbl.find !current_args x) in   
     let s = <:ident< $lid:("spoc_var"^string_of_int var.n)$>> in
     let v = <:patt< $id:s$ >> in
     let e1, e2, e3, e4, e5 =
       match var.var_type with
       | TUnknown  -> 
         <:ctyp< 'a>>, 
         <:expr< Spoc.Kernel.abs>>,
         <:ctyp< 'a>>,
         IdAcc(_loc, 
               IdUid(_loc, "Spoc"), 
               IdAcc(_loc, 
                     IdUid (_loc, "Kernel"), 
                     IdLid(_loc, "abs"))),
         <:ctyp< 'a>>
       | TInt32 ->  
         <:ctyp< int>>, 
         <:expr< Spoc.Kernel.Int32 >>,
         <:ctyp< int>>,
         IdAcc(_loc, 
               IdUid(_loc, "Spoc"), 
               IdAcc(_loc, 
                     IdUid (_loc, "Kernel"), 
                     IdUid(_loc, "Int32"))),
         <:ctyp< int32>>
       | TInt64 ->   
         <:ctyp< int64>>, 
         <:expr< Spoc.Kernel.Int64>>,
         <:ctyp< int32>>,
         IdAcc(_loc, 
               IdUid(_loc, "Spoc"), 
               IdAcc(_loc, 
                     IdUid (_loc, "Kernel"), 
                     IdUid(_loc, "Int64"))),
         <:ctyp< int64>>
       | TFloat32 | TFloat64 ->  
	 <:ctyp< float>>,
         <:expr< Spoc.Kernel.Float32>>,
         <:ctyp< float>>,
         IdAcc(_loc, 
               IdUid(_loc, "Spoc"), 
               IdAcc(_loc, 
                     IdUid (_loc, "Kernel"), 
                     IdUid(_loc, "Float32"))),
         <:ctyp< float>>
       | TVec k -> 
         (match k with
          | TInt32 -> 
            <:ctyp< (('spoc_a, 'spoc_b) Vector.vector)>>, 
            <:expr< Spoc.Kernel.VInt32>>,
            <:ctyp< ((int32, Bigarray.int32_elt) Vector.vector)>> ,
            IdAcc(_loc, 
                  IdUid(_loc, "Spoc"), 
                  IdAcc(_loc, 
                        IdUid (_loc, "Kernel"), 
                        IdUid(_loc, "VInt32"))),
            <:ctyp< Spoc.Vector.vint32>>
          | TInt64  -> 
            <:ctyp< (('spoc_e, 'spoc_f) Vector.vector)>>, 
            <:expr< Spoc.Kernel.VInt64>>,
            <:ctyp< ((int64, Bigarray.int64_elt) Vector.vector)>>,
            IdAcc(_loc, 
                  IdUid(_loc, "Spoc"), 
                  IdAcc(_loc, 
                        IdUid (_loc, "Kernel"), 
                        IdUid(_loc, "VInt64"))),
            <:ctyp< Spoc.Vector.vint64>>
          | TFloat32 -> 
            <:ctyp< (('spoc_i, 'spoc_j) Vector.vector)>>, 
            <:expr< Spoc.Kernel.VFloat32>>,
            <:ctyp< ((float, Bigarray.float32_elt) Vector.vector)>>,
            IdAcc(_loc, 
                  IdUid(_loc, "Spoc"), 
                  IdAcc(_loc, 
                        IdUid (_loc, "Kernel"), 
                        IdUid(_loc, "VFloat32"))),
            <:ctyp< Spoc.Vector.vfloat32>>
          | TFloat64 ->  
            <:ctyp< (('spoc_k, 'spoc_l) Vector.vector)>>, 
            <:expr< Spoc.Kernel.VFloat64>>,
            <:ctyp< ((float, Bigarray.float64_elt) Vector.vector)>>,
            IdAcc(_loc, 
                  IdUid(_loc, "Spoc"), 
                  IdAcc(_loc, 
                        IdUid (_loc, "Kernel"), 
                        IdUid(_loc, "VFloat64"))),
            <:ctyp< Spoc.Vector.vfloat64>>
          | _  -> failwith "Forbidden vector  type in kernel declaration")
       | _ -> assert (not debug); 
         failwith "unimplemented yet"
     in
     match var.var_type with
     | TVec _ ->
       PaTyc (_loc, v, e1), 
       <:expr< $e2$ (Spoc.Kernel.relax_vector $id:s$)>>, 
       e3, 
       PaApp (_loc,
              PaId(_loc, e4),
              PaId(_loc,s)), 
       (if !new_kernel then
          ( new_kernel := false;
            <:expr< ($id:s$: $e3$)>>)
        else
          <:expr< ( Spoc.Kernel.relax_vector $id:s$: $e3$)>>), e5
     | _ ->
       PaTyc (_loc, v, e1), 
       <:expr< $e2$ ($id:s$)>>, 
       e3, 
       PaApp (_loc,
              PaId(_loc, e4),
              PaId(_loc,s)), 
       <:expr< ($id:s$: $e3$)>> , e5
    )
  | _  -> failwith "error gen_arg_form_patt2"




let rec float32_expr f = 
  let rec f32_typer f =
    (match f.t with
     | TUnknown -> f.t <- TFloat32
     | TFloat32 -> f.t <- TFloat32
     | TVec TFloat32 -> f.t <- TVec TFloat32
     | _ -> assert (not debug); raise (TypeError (TFloat32, f.t, f.loc)))
  in f32_typer f;

  (match f.e with
   | PlusF (l, a,b) 
   | PlusF32 (l, a,b) -> 
     f.e <- PlusF32 (l, float32_expr a, float32_expr b)
   | Float (l,s) 
   | Float32 (l,s) -> f.e<- Float32 (l, s)

   | VecGet (l,a,b) -> ()
   | Seq (l, a, b) -> f.e <- Seq (l,a, float32_expr b)
   | Id  (l, id) -> ()
   | _ -> assert (not debug); raise (TypeError (TFloat32, f.t, f.loc)));
  f

let rec float64_expr f = 
  let rec f64_typer f =
    (match f.t with
     | TUnknown -> ()
     | TVec TFloat64 -> f.t <- TVec TFloat64
     | TFloat64-> ()
     | _ ->() ) 
  in f64_typer f;
  (match f.e with
   | PlusF (l, a,b) 
   | PlusF64 (l, a,b) -> 
     (
       f.e <- PlusF64 (l, float64_expr a, float64_expr b)
     )
   | MinF (l, a,b) 
   | MinF64 (l, a,b) -> 
     (
       f.e <- MinF64 (l, float64_expr a, float64_expr b)
     )

   | MulF (l, a,b) 
   | MulF64 (l, a,b) -> 
     (
       f.e <- MulF64 (l, float64_expr a, float64_expr b)
     )

   | DivF (l, a,b) 
   | DivF64 (l, a,b) -> 
     (
       f.e <- DivF64 (l, float64_expr a, float64_expr b)
     )
   | App (l, a, b) ->
     f.e <- App (l, float64_expr a, (List.map float64_expr b))
   | Float32 (l,s) 
   | Float64 (l,s) -> 
     (
       f.e<- Float64 (l, s))
   | Bind (l, e1, e2, e3, is_mutable) -> 
     f.e <- Bind (l, float64_expr e1, float64_expr e2, float64_expr e3, is_mutable)
   | VecGet (l,a,b) -> ()
   | VecSet (l,a,b) -> 
     f.e <- VecSet (l,float64_expr a, float64_expr b)
   | Seq (l, a, b) -> f.e <- Seq (l,a, float64_expr b)
   | Id  (l, id) -> f.e <-  Id  (l, id)
   | Int _ | BoolEq32 _ | BoolEq64 _ -> () 
   | BoolLt _ | BoolLt32 _ | BoolLt64 _ 
   (*   | Plus _ | Min _ | Mul _ | Div  _ *) ->() 
   | BoolEqF (l,a,b) ->
     f.e <- BoolEqF64 (l, float64_expr a, float64_expr b)
   | BoolLtF (l,a,b) -> 
     f.e <- BoolLtF64 (l, float64_expr a, float64_expr b)
   | Ife (l,cond,cons1,cons2) -> 
     f.e <- Ife (l, float64_expr cond, float64_expr cons1, float64_expr cons2)
   | If (l,cond,cons1) -> 
     f.e <- If (l, float64_expr cond, float64_expr cons1)
   | DoLoop (l, id, min, max, body) ->
     f.e <- DoLoop (l, id, min, max, 	float64_expr (body))
   | Open (l, m, e) -> 
     f.e <- Open (l, m, float64_expr e)
   | _ -> assert (not debug); raise (TypeError (TFloat64, f.t, f.loc));
  );
  f


let nb_ker = ref 0

let ctype_of_sarek_type  = function
  | "int32" -> "int"
  | "int64" -> "long"
  | "int" -> "int"
  | a -> a 


let rec string_of_ctyp = function
  | Ast.TyArr (_loc, t1, t2) -> (string_of_ctyp t1)^
				" -> "^(string_of_ctyp t2)
  | (Ast.TyId (_, Ast.IdLid (_, s ))) -> s
  | TyCol (l,t1,t2)-> string_of_ctyp t2
  | _ -> failwith "error in string_of_ctyp"


let sarek_types_tbl = Hashtbl.create 10

let get_sarek_name t =
  match t with
  | "int" | "float" | "int32" | "int64" | "float32" | "float64" -> t
  | _ ->
    try 
      Hashtbl.find sarek_types_tbl t
    with
    | _ -> print_endline t; assert false

let gen_ctype t1 t2 name _loc = 
  let field_name  = (string_of_ident t2^"_t") in
  begin
    <:str_item< let $lid:field_name$ = 
		let open Ctypes in
		Ctypes.field 
		$lid:string_of_ident t2$ 
		$str:field_name$
		$lid:get_sarek_name (string_of_ctyp t1)$ 
                ;; 
    >> 
  end

let gen_ctype_repr t1 t2 name : string =
  let field_name  = (string_of_ident t2^"_t") in
  ((get_sarek_name (string_of_ctyp t1))^" "^field_name)


type ktyp_repr = {
  type_id : int;
  name : string;
  typ : ktyp;
  ml_typ : str_item;
  ctype : str_item;
  crepr : string;
  ml_to_c : expr;
  c_to_ml : expr;
}

let type_id = ref 0


let gen_mltyp _loc name t =
  begin
    match t with 
    | Custom (KRecord (ctypes,idents)) -> 
      begin
	<:str_item< 
	            type $lid:name$ = {
	            $List.fold_left 
	            (fun elt liste_elt -> 
	            <:ctyp< $elt$; $liste_elt$ >>) 
<:ctyp< >> ctypes$
} >>
end
| Custom (KSum l) -> 
  begin
    <:str_item< 
	        type $lid:name$ = 
	        $let type_of_string s =
		TyId (_loc, (IdUid(_loc,s)) )
	        in
	        let typeof t1 t2 =
		TyOf(_loc, t1, t2)
	        in
	        List.fold_left 
		(fun elt (liste_elt,t) ->
		match t with
		| Some (s:ctyp) -> 
		let t  =
		typeof (type_of_string liste_elt) s 
		in 
		<:ctyp<  $elt$ | $t$ >>
  | None ->
    <:ctyp<  $elt$ | $type_of_string liste_elt$ >> )
<:ctyp<  >> 
  l$
>>
end
| TInt32 -> 
  <:str_item< int >>
| _ -> 
  <:str_item< >>
end


type managed_ktyp = 
{
  mk_name : string;
  mk_crepr : string;
}



let gen_ctypes _loc kt name =
  let rec gen_repr _loc t name =
    let managed_ktypes = Hashtbl.create 5 in
    let sarek_type_name = name^"_sarek" in
    Hashtbl.add sarek_types_tbl name sarek_type_name;
    let ctype =
      begin
	let fieldsML =
   match t with
   | Custom (KRecord (ctypes,idents)) -> 
     begin
       let rec content acc l1 = 
	 match (l1 : ctyp list) with
  | [] -> 
    acc
  | t1::q1 -> 
    content 
		      (<:str_item< $acc$
		     $gen_ctype t1 (IdLid(_loc,sarek_type_name)) sarek_type_name _loc$
		     >>) q1
       in
       content (<:str_item< >>) ctypes
     end
   | Custom (KSum l) -> 
     begin
       let gen_mlstruct accML (cstr,ty) = 
	 let name = sarek_type_name^"_"^cstr in
  begin
		    match ty with 
		    | Some t -> 
	begin
   let ct = gen_ctype t (IdLid(_loc,name))  sarek_type_name _loc in 
   let ctr = gen_ctype_repr 
       t 
       (IdLid(_loc,name))  sarek_type_name  in  
   Hashtbl.add managed_ktypes cstr ctr;
   <:str_item< 
			            $accML$ ;; 
			            type $lid:name$ ;;
			            let $lid:name$ : $lid:name$ Ctypes.structure Ctypes.typ = 
			            Ctypes.structure 
			            $str:name$;;
     $ct$;;
			            let () = Ctypes.seal $lid:name$ ;; >>
		      end  
		    | None -> 
		      begin
			<:str_item< >>
		      end
		  end 

		in	  
		let rec content (accML)  = function
		  | t::q -> content (gen_mlstruct accML t) q
		  | [] -> accML
		in
		content (<:str_item< >>) l
	      end
	    | _ -> assert false

	  in
	  begin
	    <:str_item< 
	                (** open ctype **)
	                type $lid:sarek_type_name$ ;;
	                let ($lid:sarek_type_name$ :($lid:sarek_type_name$ 
			Ctypes.structure) 
			Ctypes.typ) = 
	                Ctypes.structure $str:sarek_type_name$ ;;
	                (** fill its fields **)
	                $fieldsML$ ;;
	                (** close ctype **)
	                let () = Ctypes.seal $lid:sarek_type_name$ ;;
		        >>
	  end ;
	end;
    in 
    {
      type_id = (incr type_id; !type_id);
      name = name;
      typ = t;
      ml_typ = gen_mltyp _loc name t;
      ctype = 
        ctype;
      crepr = 
        begin
          match t with 
          | Custom (KRecord _) -> "int"
          | Custom (KSum l) ->
            let rec has_of = function
              | (_,Some t)::q -> true
              | (_,None)::q -> has_of q
              | [] -> false 
            in
            let gen_cstruct accC = function
              | cstr,Some t -> accC ^ "\t\tstruct "^sarek_type_name^"_"^cstr^" {\n\t\t\t"^ 
                               (Hashtbl.find managed_ktypes cstr)^"\n\t\t};\n"
                               
              | cstr,None -> accC ^ ""
            in
            let rec contents accC  = function
              | t::q -> contents (gen_cstruct accC t) q
              | [] -> accC
            in
            let fieldsC  =
              let b = contents "" l in    
              ("struct " ^ sarek_type_name ^ "= {\n\tint "^
               sarek_type_name ^ "_tag;\n"^
               (if has_of l then
                 "\tunion "^sarek_type_name^"_union {\n"
               ^b ^"\t}"
                else "" )^
               "\n}") 
            in fieldsC 
          | _ -> "int" ;
        end;
      ml_to_c = <:expr< >>;
      c_to_ml = <:expr< >>;
    }
  in
  let t = gen_repr _loc (Custom kt) name in
  begin
    <:str_item<
      $t.ml_typ$ ;;
      $t.ctype$ ;;
      let $lid:t.name^"_c_repr"$ = $str:t.crepr$ ;;
      >>
  end





(*		  

let gen_ctypes _loc kt name  =
  let sarek_type_name = name^"_sarek" in
  match kt with
  | KRecord (l1,l2) ->
     begin
       let t1 = List.fold_left 
(fun elt liste_elt -> 
<:ctyp< $elt$; $liste_elt$ >>) 
<:ctyp< >> l1  in
       (*let ident_of_string s =
IdLid (_loc,s)
       in   *)

       let rec content (acc,ctype) (l1,l2)= 
match (l1 : ctyp list), (l2 :ident list) with
| [],[] -> 
(acc,ctype)
| t1::q1, t2::q2 -> 
            let nextsML,nextsC = content (acc,ctype) (q1,q2) in

             ((<:str_item< 
$gen_ctype t1 t2 sarek_type_name _loc$
$nextsML$ >>),
("\n\t"^(ctype_of_sarek_type (string_of_ctyp t1)) ^ 
"  " ^ (string_of_ident t2) ^ ";" ^ nextsC
))
| _ -> assert false
       in

       let fieldsML,fieldsC =
let a,b = content (<:str_item< >>,"") (l1,l2) in    
(a, 			
("struct " ^ sarek_type_name ^ "= {"
^b ^"\n}") ) in

       let copy_to_caml = 
List.fold_left 
(fun a b -> 
let field_name = 
(sarek_type_name^"_"^(string_of_ident b)) in
<:rec_binding< $a$; $b$ = 
Ctypes.getf x $lid:field_name$ >>) <:rec_binding< >> l2
       and copy_from_caml = 
List.fold_left 
(fun a b -> 
let field_name = 
(sarek_type_name^"_"^(string_of_ident b)) in
<:expr< $a$; Ctypes.setf x
$lid:field_name$ elt.$ExId(_loc,b)$ >>) <:expr< >> l2

       in
       <:str_item< 
       type $lid:name$ = { $t1$ } ;;
       type $lid:sarek_type_name$ ;;


       let $lid:sarek_type_name$ : 
$lid:sarek_type_name$ Ctypes.structure Ctypes.typ = 
Ctypes.structure $str:sarek_type_name$ ;;
         $fieldsML$ ;;
let () = Ctypes.seal $lid:sarek_type_name$ ;;
let $lid:"custom_"^name$ = 
{
size = Ctypes.sizeof $lid:sarek_type_name$ ; 
get = (fun c i -> 
let x = Ctypes.Array.get c i in
{$copy_to_caml$}) ;
set = (fun c i elt -> 
let x = 
Ctypes.make $lid:sarek_type_name$ in 
$copy_from_caml$ ;
Ctypes.Array.set c i x)
} ;;
let _ = Hashtbl.add ctypes 
$lid:sarek_type_name$ $str:fieldsC$ ;;
>>
     end


  | KSum l ->
     begin
       let type_of_string s =
TyId (_loc, (IdUid(_loc,s)) )
       in
       let typeof t1 t2 =
TyOf(_loc, t1, t2)
       in
       let t1 = List.fold_left 
(fun elt (liste_elt,t) ->
match t with
| Some (s:ctyp) -> 
let t  =
typeof (type_of_string liste_elt) s in 
<:ctyp<  $elt$ | $t$ >>

| None ->
<:ctyp<  $elt$ | $type_of_string liste_elt$ >> )
<:ctyp<  >> 
l  in
       let gen_mlstruct accML (cstr,ty) = 
let name = sarek_type_name^"_"^cstr in
begin
match ty with 
| Some t -> 
begin

<:str_item< 
$accML$ ;; 
type $lid:name$ ;;
let $lid:name$ : $lid:name$ Ctypes.structure Ctypes.typ = 
Ctypes.structure $str:name$;;
$gen_ctype t (IdLid(_loc,name))  sarek_type_name _loc$ ;;
let () = Ctypes.seal $lid:name$ ;; >>
end  
| None -> 
begin
<:str_item< >>
end
end ;
       in
       let gen_cstruct accC = function
| cstr,Some t -> accC ^ "struct"
| cstr,None -> accC ^ ""
       in
       let rec content (accML,accC)  = function
| t::q -> content ((gen_mlstruct accML t), (gen_cstruct accC t)) q
| [] -> accML,accC
       in
       let rec gen_union acc  = function
| (cstr,Some o) :: q -> 
gen_union 
(	 let name = sarek_type_name^"_"^cstr in
let union_name = sarek_type_name^"_union_" in
<:str_item<
$acc$ ;;
let $lid:(name^"_val")$ = 
Ctypes.field $lid:union_name$ $str:(name^"_val")$ $lid:name$;;
>>)
q
| (cstr,None):: q ->
gen_union acc q
| [] -> acc
       in


       let gen_sum_rep  l = 
let rec aux acc tag = function
| (cstr,of_) :: q ->
aux ((cstr,tag,of_)::acc) (tag+1) q
| [] -> acc
in
aux [] 0 l
       in

       let copy_content sarek_type_name cstr of_ =
let copy_of _of = 
<:expr< let union = 
Ctypes.getf x $lid:sarek_type_name^"_union"$ in
let $lid:"val"^cstr$ = 
Ctypes.getf union 
$lid:sarek_type_name^"_"^cstr^"_val"$ in
Ctypes.getf $lid:"val"^cstr$ $lid:sarek_type_name^"_t"$
>>
in 
match of_ with
| Some x -> 
<:expr< $ExId(_loc,IdUid(_loc,cstr))$ ($copy_of x$) >>
| None ->
ExId(_loc,IdUid(_loc,cstr))
in
let repr_ = gen_sum_rep l 	
       in
       let copy_to_caml = 
let match_cases = 
List.map 
(fun (cstr,tag,of_) ->
(*let field_name =
(sarek_type_name^"_"^cstr) in*)
begin
<:match_case< 
$int:string_of_int tag$ -> 
$copy_content sarek_type_name cstr of_$
>>
end
) 
repr_
in
<:expr< let tag = Ctypes.getf x $lid:sarek_type_name^"_tag"$ in
match tag with 
$list:(List.rev match_cases)$ >>


       in
       let fieldsML,fieldsC =
let a,b = content (<:str_item< >>,"") l in    
(a, 			
("struct " ^ sarek_type_name ^ "= {\nint "^
sarek_type_name ^ "_tag;\n"
^b ^"\n}") ) 
       in

       let tag = sarek_type_name^"_tag" in
       let union_name = sarek_type_name^"_union_" in
       <:str_item< 
       type $lid:name$ = $t1$ ;;
         $fieldsML$ ;;
type $lid:union_name$ ;;
let $lid:union_name$ : ($lid:union_name$ Ctypes.union Ctypes.typ) =
Ctypes.union $str:union_name$ ;;
$gen_union <:str_item< >> l$ ;;
let () = Ctypes.seal $lid:union_name$ ;;
type $lid:sarek_type_name$ ;;


let $lid:sarek_type_name$ : $lid:sarek_type_name$ 
Ctypes.structure Ctypes.typ = 
Ctypes.structure $str:sarek_type_name$ ;;
let $lid:tag$ = 
Ctypes.field $lid:sarek_type_name$ $str:tag$ Ctypes.int ;;
let $lid:(sarek_type_name^"_union")$ = 
Ctypes.field $lid:sarek_type_name$ $str:sarek_type_name^"union"$ 
$lid:sarek_type_name$ ;;
             let () = Ctypes.seal $lid:sarek_type_name$ ;;
let _ = Hashtbl.add ctypes 
$lid:sarek_type_name$ $str:fieldsC$ ;;

let $lid:"custom_"^name$ = 
{
size = Ctypes.sizeof $lid:sarek_type_name$ ; 
get = (fun c i -> 		    
let x = Ctypes.Array.get c i in
$copy_to_caml$
) ;

set = (fun c i elt -> ()) ;
} ;;
>>
     end
 *)

let gen_labels _loc (t1 : ident * ctyp) 
    (t2 : (ctyp list * ident list) option) 
  : ctyp list * ident list =
  let s,t = t1 in
  let t1 = 
    TyCol (_loc, (TyId (_loc, s)), t)
  in
  match t2 with 
  | Some (t2,s2) -> 
    t1::t2, s::s2
  | None ->
    t1::[], s::[]


let gen_constructors _loc 
    (t1 : string * ctyp option) 
    (t2 : (string * ctyp option) list option) 
  : (string *ctyp option) list =
  match t2 with 
  | Some t -> 
    t1::t
  | None ->
    t1::[]
