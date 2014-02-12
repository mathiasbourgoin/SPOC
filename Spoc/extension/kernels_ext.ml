(******************************************************************************
 * © Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow GPU 
 * programming with the OCaml language.
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
 ******************************************************************************)

open Camlp4.PreCast
open Syntax
open Ast

exception Unauthorized_Recursion of string
exception Unauthorized_String_Access of string
exception Unauthorized_Array_Access of string
exception Unauthorized_Type of (string*string)
exception Unbound_value of (Loc.t*string)
exception Unbound_value2 of ((Loc.t*string)*Loc.t)

(******************** debug tools **********************)

let debug = false
let print i = 
  if debug then
    Printf.printf "%s\n" i; Pervasives.flush stdout
      
(******************** external kernels **********************)

let expr_of_string = Syntax.Gram.parse_string Syntax.expr_eoi;;


let id = ref 0



let arg_string = ref ""
let arg_string2 = ref ""

let idx = ref 1

let create_new_var loc =
  let v = "spoc_var"^(string_of_int !id) 
  in 
  let a = expr_of_string loc (v) in
  incr id;
  a,v


let rec string_of_ident = function
  | <:ident< $lid:s$ >> -> s
  | <:ident< $uid:s$ >> -> s
  | <:ident< $i1$.$i2$ >> -> "acc_" ^ (string_of_ident i1) ^ "_" ^ (string_of_ident i2)
  | <:ident< $i1$($i2$) >> -> "app_" ^ (string_of_ident i1) ^ "_" ^ (string_of_ident i2)
  | <:ident< $anti:_$ >> -> assert false

let fst3 = fun (x,y,z) -> x
let snd3 = fun (x,y,z) -> y
let thrd3 = fun (x,y,z) -> z
  
let rec parse_ctyp = function
  | TyNil _ -> failwith "TyNil";
  | TyAli _ -> failwith "TyAli";
  | TyAny _ -> failwith "TyAny";
  | TyApp (l, c1, c2) as t -> 
    let e,v =create_new_var l in
    (*let l1 = parse_ctyp c1 
      and l2 =parse_ctyp c2 in*)
    [ExTyc (l, e, t)],[v],[t] 
  (*failwith "TyApp";*)
  | TyArr (l,a,b) -> 
    (*Printf.printf "TyArr\n";*) 
    let l1 = parse_ctyp a 
    and l2 =parse_ctyp b in
    ((fst3 l1)@(fst3 l2),(snd3 l1)@(snd3 l2),(thrd3 l1)@(thrd3 l2))
  | TyCls _ -> failwith "TyCls";
  | TyLab _ -> failwith "TyLab";
  | (TyId (l,id)) as t -> let e,v =create_new_var l in
			  begin
			    match t with
			    | TyId(_, id) when ((string_of_ident id) = "int32") 
				-> ([ExTyc (l,e,TyId(l, ident_of_expr (expr_of_string l "int")))],[v],[t])
			    | TyId(_, id) when ((string_of_ident id) = "int64")
				-> ([ExTyc (l,e,TyId(l, ident_of_expr (expr_of_string l "int")))],[v],[t])
			    | TyId(_, id) when ((string_of_ident id) = "float32")
				-> ([ExTyc (l,e,TyId(l, ident_of_expr (expr_of_string l "float")))],[v],[t])
			    | TyId(_, id) when ((string_of_ident id) = "float64")
				-> ([ExTyc (l,e,TyId(l, ident_of_expr (expr_of_string l "float")))],[v],[t])		
			    | _ ->  ([ExTyc (l,e,t)],[v],[t])
			  end;
  | TyMan _ -> failwith "TyMan"
  | TyDcl _ -> failwith "TyDcl"
  | TyObj _ -> failwith "TyObj"
  | TyOlb _ -> failwith "TyOlb"
  | TyPol _ -> failwith "TyPol"
  | TyTypePol _ -> failwith "TyTypePol"
  | TyQuo _ -> failwith "TyQuo"
  | TyQuP _ -> failwith "TyQup"
  | TyQuM _ -> failwith "TyQum"
  | TyAnP _ -> failwith "TyAnP"
  | TyAnM _ -> failwith "TyAnM"
  | TyVrn _ -> failwith "TyVrn"
  | TyRec _ -> failwith "TyRec"
  | TyCol _ -> failwith "TyCol"
  | TySem _ -> failwith "TySem"
  | TyCom _ -> assert false
  | TySum _ -> assert false
  | TyOf _ -> assert false
  | TyAnd _ -> assert false
  | TyOr _ -> assert false
  | TyPrv _ -> assert false
  | TyMut _ -> assert false
  | TyTup _ -> assert false
  | TySta _ -> assert false
  | TyVrnEq _ -> assert false
  | TyVrnSup _ -> assert false
  | TyVrnInf _ -> assert false
  | TyVrnInfSup _ -> assert false
  | TyAmp _ -> assert false
  | TyPkg _ -> assert false
  | (TyAnt (l,s)) as t  -> failwith ("TyAnt : "^s)
  | _ -> failwith "parse_ctyp OUHLALA"
 
    
let rec parseTyp t i= 
  match t with
  | TyId (_, id) when ((string_of_ident id)  = "char") -> 
    incr idx;
    arg_string := "Spoc.Kernel.Char "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)
  | TyId (_, id) when ((string_of_ident id) = "int")  -> 
    incr idx;
    arg_string := "Spoc.Kernel.Int32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)
  | TyId (_, id) when ((string_of_ident id) = "int32") -> 
    incr idx;
    arg_string := "Spoc.Kernel.Int32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "int64") -> 
    incr idx;
    arg_string := "Spoc.Kernel.Int64 "^i^" "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "float") -> 
    incr idx;
    arg_string := "Spoc.Kernel.Float32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "float32") -> 
    incr idx;
    arg_string := "Spoc.Kernel.Float32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "float64") -> 
    incr idx;
    arg_string := "Spoc.Kernel.Float64 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vfloat32") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VFloat32 (Spoc.Kernel.relax_vector "^i^") "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)
      
  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vchar") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VChar (Spoc.Kernel.relax_vector "^i^") "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_localvfloat32") -> 
    incr idx;
    arg_string := "Spoc.Kernel.LocalFloat32 (Spoc.Kernel.relax_vector "^i^") "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vfloat64") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VFloat64 (Spoc.Kernel.relax_vector "^i^") "^(!arg_string);  
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vcomplex32") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VComplex32 (Spoc.Kernel.relax_vector "^i^") "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vint32") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VInt32 (Spoc.Kernel.relax_vector "^i^") "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vint64") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VInt64 (Spoc.Kernel.relax_vector "^i^") "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vbool") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VCustom (Spoc.Kernel.relax_vector "^i^") "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vcustom") -> 
    incr idx;
    arg_string := "Spoc.Kernel.VCustom (Spoc.Kernel.relax_vector "^i^") "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) -> 
    arg_string := i^" "^(!arg_string);
    arg_string2 := i^" "^(!arg_string2)
  | TyApp (l, t1, t2)  -> 
    begin
      match t2 with 
      | TyId (_, id) ->
	(parseTyp t1 i );
      | _  -> failwith "erf"
    end
  | _ -> 
    failwith "apply_of_constraints Not TyId "				
      
let rec parseInvTyp t i= 
  match t with
  | TyId (_, id) when (String.compare (string_of_ident id) "char") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.Char "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)
  | TyId (_, id) when (String.compare (string_of_ident id) "int") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.Int32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)
  | TyId (_, id) when (String.compare (string_of_ident id) "int32") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.Int32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "int64") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.Int64 "^i^" "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "float") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.Float32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "float32") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.Float32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "float64") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.Float64 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vfloat32") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VFloat32 "^i^" "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)
      
  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vchar") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VChar "^i^" "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_localvfloat32") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.LocalFloat32 "^i^" "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vfloat64") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VFloat64 "^i^" "^(!arg_string);  
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vcomplex32") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VComplex32  "^i^" "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vint32") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VInt32 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vint64") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VInt64 "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vbool") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VCustom "^i^" "^(!arg_string);
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vcustom") = 0 -> 
    incr idx;
    arg_string := "Spoc.Kernel.VCustom "^i^" "^(!arg_string); 
    arg_string2 :=i^" "^(!arg_string2)

  | TyId (_, id) -> 
								(*Printf.printf "--- %s\n" (string_of_ident id); 
								  flush stdout;  *)
    arg_string := i^" "^(!arg_string);
    arg_string2 := i^" "^(!arg_string2)

							(*failwith "apply_of_constraints TyId "*)
  | TyApp (l, t1, t2)  -> 
    begin
      match t2 with 
      | TyId (_, id) ->
										(*incr idx;*)
	(parseInvTyp t1 i );(*arg_string:= (string_of_ident id)^" "^ !arg_string;*)
      | _  -> failwith "erf"
    end
  | _ -> 
    failwith "apply_of_constraints Not TyId "			
      
let rec type_to_type t = 
  match t with
  | TyId (l, id) when (String.compare (string_of_ident id) "float64") = 0 -> 
    let _loc = l in
    (<:ctyp< float>>)
  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vfloat32") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_c, 'spoc_d) Vector.vector>>)
      
  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vchar") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_e, 'spoc_f) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_localvfloat32") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_g, 'spoc_h) Vector.vector>>)
      
  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vfloat64") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_i, 'spoc_j) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vcomplex32") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_j, 'spoc_k) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vint32") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_l, 'spoc_m) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vint64") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_n, 'spoc_o) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vbool") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_p,'spoc_q) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vcustom") = 0 -> 
    let _loc = l in
    (<:ctyp< ('spoc_r,'spoc_s) Vector.vector>>)
  | TyId (_, id) -> 
    t
  | TyApp (l, t1, t2)  -> 
    type_to_type t1
  | _ -> 
    failwith "apply_of_constraints Not TyId "				
      
let rec concrete_type_to_type t = 
  match t with
  | TyId (l, id) when (String.compare (string_of_ident id) "float64") = 0 -> 
    let _loc = l in
    (<:ctyp< float>>)
      
  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vfloat32") = 0 -> 
    let _loc = l in
    (<:ctyp< (float, Bigarray.float32_elt)  Vector.vector>>)
      
  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vchar") = 0 -> 
    let _loc = l in
    (<:ctyp< (char, Bigarray.int8_unsigned_elt) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_localvfloat32") = 0 -> 
    let _loc = l in
    (<:ctyp< (float, Bigarray.float32_elt) Vector.vector>>)
      
  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vfloat64") = 0 -> 
    let _loc = l in
    (<:ctyp< (float, Bigarray.float64_elt) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vcomplex32") = 0 -> 
    let _loc = l in
    (<:ctyp< (Complex.t, Bigarray.complex32_elt) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vint32") = 0 -> 
    let _loc = l in
    (<:ctyp< (int32, Bigarray.int32_elt) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vint64") = 0 -> 
    let _loc = l in
    (<:ctyp< (float, Bigarray.float64_elt) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vbool") = 0 -> 
    let _loc = l in
    (<:ctyp< (bool, bool) Vector.vector>>)

  | TyId (l, id) when (String.compare (string_of_ident id) "acc_acc_Spoc_Vector_vcustom") = 0 -> 
    let _loc = l in
    (<:ctyp< Vector.vector>>)
  | TyId (_, id) -> 
								(*Printf.printf "--- %s\n" (string_of_ident id); 
								  flush stdout;  *)
    t

							(*failwith "apply_of_constraints TyId "*)
  | TyApp (_loc, t1, t2)  -> 
    <:ctyp< ($concrete_type_to_type t2$,$concrete_type_to_type t2$) $concrete_type_to_type t1$>>
  | _ -> 
    failwith "apply_of_constraints Not TyId "				
      
      
let  gen_ktyp loc type_list = 
  let l = List.map concrete_type_to_type type_list in
  let l = List.tl (List.rev l) in
  let l = List.rev l in
  TyTup (loc,	 Ast.tySta_of_list l)


let ident_of_string loc s= IdLid (loc, s)

let gen_args loc type_list =
  id := 0;
  arg_string :=  "";
  let var_list = ref [] in
  let _loc = loc in
  let l = List.tl (List.rev type_list) in
  let l = List.rev l in
  let translate t = 
    let nv = create_new_var loc in
    var_list :=!var_list @ [nv];
    PaTyc (loc, 
	   (PaId (loc,
		  (ident_of_string loc (snd	nv))
	    )), 
	   (type_to_type t)
    ) 
  in
  let tuple =  Ast.paCom_of_list (List.map (translate) l) in
  arg_string := "";				
  let array =	
    let arr_content =
      let f t = 
	arg_string := "";
 	parseTyp t (*(snd (List.nth !var_list !idx))*) (snd (List.nth !var_list (!idx))) ;
  	expr_of_string loc (!arg_string^"; ")
      in 
      idx := 0 ;
      let l = (List.map f l) 
      in	
      Ast.exSem_of_list (l)
    in
    ExArr(loc, arr_content)
  in
  idx := 1 ;
  PaTup(loc, tuple),array	

let first_vector = ref false

let relaxed _loc nv =  
  if not !first_vector then
    (first_vector := true;
     <:expr<$(ExId (_loc,
		    (ident_of_string _loc (nv))
     ))$>>)
  else
    <:expr<Spoc.Kernel.relax_vector $(ExId (_loc,
					    (ident_of_string _loc (nv))
    ))$>>        
      
      let rec gen_inv_id t _loc nv =         
	match t with
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vfloat32") -> relaxed _loc nv
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vfloat64") -> relaxed _loc nv
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vint32") -> relaxed _loc nv
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vint64") -> relaxed _loc nv
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_localvfloat32") -> relaxed _loc nv
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vchar") -> relaxed _loc nv
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vbool") -> relaxed _loc nv
	| TyId (l, id) when ((string_of_ident id) = "acc_acc_Spoc_Vector_vcustom") -> relaxed _loc nv
	| TyApp   (_loc, t1, t2)  -> 
	  gen_inv_id t1 _loc nv
	| _   ->  (ExId (_loc,
			 (ident_of_string _loc (nv))
	))    

let gen_inv_args loc type_list =
  id := 0;
  arg_string :=  "";
  let var_list = ref [] in
  first_vector := false;
  let _loc = loc in
  let l = List.tl (List.rev type_list) in
  let l = List.rev l in
  let translate t = 
    let nv = create_new_var loc in
    var_list :=!var_list @ [nv];
    ExTyc (loc, 
	   gen_inv_id t loc (snd nv), 
	   (concrete_type_to_type t)
    ) 
  in
  let tuple =  Ast.exCom_of_list (List.map (translate) l) in
  arg_string := "";
  let array =	
    let arr_content =
      let f t = 
	arg_string := "";
 	parseInvTyp t " " (*(snd (List.nth !var_list !idx))*) ;
	let arg = 
	  ((*print_endline ("arg: "^ !arg_string);*) 
	    String.sub !arg_string 12 ((String.length !arg_string - 15))) in
	      (*Printf.printf "arg_string = %s\n" !arg_string;
		Printf.printf "arg_ = %s\n" (String.sub !arg_string 12 ((String.length !arg_string - 15)));
	      *)
  	PaApp (loc,
	       PaId (loc, IdUid (_loc, arg)), 
	       PaId (loc, ident_of_string loc (snd (List.nth !var_list (!idx-1)))))
      in 
      idx := 0 ;					
      Ast.paSem_of_list (List.map f l)
    in
    PaArr(loc, arr_content)
  in
  idx := 1 ;
  ExTup(loc, tuple),array
    
    
let bigarray_set _loc var newval =
  match var with
    <:expr< Bigarray.Array1.get $arr$ $c1$ >> ->
      Some <:expr< Bigarray.Array1.set $arr$ $c1$ $newval$ >>
  | <:expr< Bigarray.Array2.get $arr$ $c1$ $c2$ >> ->
    Some <:expr< Bigarray.Array2.set $arr$ $c1$ $c2$ $newval$ >>
  | <:expr< Bigarray.Array3.get $arr$ $c1$ $c2$ $c3$ >> ->
    Some <:expr< Bigarray.Array3.set $arr$ $c1$ $c2$ $c3$ $newval$ >>
  | <:expr< Bigarray.Genarray.get $arr$ [| $coords$ |] >> ->
    Some <:expr< Bigarray.Genarray.set $arr$ [| $coords$ |] $newval$ >>
  | <:expr< Spoc.Mem.get $arr$ $c1$ >>  -> 
    Some <:expr< Spoc.Mem.set $arr$ $c1$ $newval$ >>
  | _ -> None 

    
    DELETE_RULE Gram expr: SELF; "."; "("; SELF; ")" END
      DELETE_RULE Gram expr: SELF; "."; "["; SELF; "]" END
	DELETE_RULE Gram expr: SELF; "<-"; expr LEVEL "top" END


	  

	  EXTEND Gram
	  GLOBAL: str_item  sequence do_sequence expr ctyp opt_rec;
    
      (******************** external kernels **********************)
    expr: LEVEL "."
      [ 
	[ e1 = SELF; ":="; e2 = expr LEVEL "top" ->
        <:expr< $e1$ := $e2$ >>
	|
	    e1 = SELF; "<-"; e2 = expr LEVEL "top" ->
            begin
              match bigarray_set _loc e1 e2 with
	        Some e -> e
              | None ->  ExAss (_loc, e1, e2)
            end
(*         |e1 = SELF; "<~"; e2 = expr LEVEL "top" ->
          begin
            match e1 with
            | ExAcc(_loc, e1_, e2_) -> 
              (* TODO : bypass useless copy *)
              <:expr< 
                      let sarek_temp = $e1_$  in 
                      sarek_tmp.x <- $e2$;
                      $e1_$ <- sarek_tmp
              >>
            | _ -> failwith "Wrong use of \"<~\""
          end *)
	] ];

    expr:  LEVEL "."
      [
	LEFTA
	  [	e0 = SELF; "."; "[<"; e1 = SELF; ">]" ->
	  let e = expr_of_string _loc "Spoc.Mem.get" in
	  <:expr< $e$ $e0$ $e1$ >>
	  |e0 = SELF; "."; "("; e1 = SELF; ")" -> 
	  <:expr< $e0$ .( $e1$ ) >>
	  |e0 = SELF; "."; "["; e1 = SELF; "]" ->
	  <:expr< $e0$ .[ $e1$ ] >>
	  ]
      ];
    
    str_item: 
      [ [ "kernel"; lid = ident; ":";  typ = ctyp; "=";  file_name = STRING; func_name = STRING  -> 
      let constraint_var, arg_list, id_list  = parse_ctyp typ 
      and id =  IdLid  (_loc, "kernel_"^(string_of_ident lid) )in
      
      let p1 = PaId (_loc, lid) in 
      
      
      let k_typ = gen_ktyp _loc id_list in
      let inv_args = gen_inv_args _loc id_list  in 
      let args = gen_args _loc id_list in
      let k_fun = <:patt<
	$id:id$
	>> 
      in
      let exec_fun =
	<:binding< 
	  $k_fun$ = 	
	fun  $fst args$ ->
	  Spoc.Kernel.exec $snd args$ 
	  >> in
      
      let class_name = "class_kernel_"^(string_of_ident lid)
      and	filename =ExStr (_loc, file_name)
      and funcname =ExStr (_loc, func_name) in
      let exec = ExId(_loc, id) in
      
      
      let gen_object = 
	<:str_item< 
	  
	  class ['a, 'b] $lid:class_name$  = object (self)
	    inherit [$k_typ$, ('a,'b) Kernel.kernelArgs array ] 
	      Spoc.Kernel.spoc_kernel $filename$ $funcname$
	    method exec  = 
	      $exec$
		
	    method args_to_list  = fun  $fst args$ -> $snd args$
	      
	    method list_to_args = function
	    | $snd inv_args$ -> $fst inv_args$
            | _  -> failwith "spoc_kernel_extension error"
	      
	  end
	  let $p1$ = new $lid:class_name$;;
	  >> in
	  
	  
	  arg_string := "";
	  StSem(_loc, <:str_item< open Spoc.Kernel >>,
		
		StSem(_loc,
		      StVal	(_loc, ReNil,exec_fun),				
		      gen_object))
	]];
    
    END
      

