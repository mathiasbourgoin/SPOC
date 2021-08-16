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

open Sarek_types
open Debug

let parse_args params body=
  let _loc = Loc.ghost in
  let nargs = List.length params in

  let rec aux acc i a =
    if i > nargs - !n_lifted_vals then
      acc
    else
      match a with
      | []  ->  acc
      | t::q ->
        match t with
        | PaTyc (_,x, <:ctyp< $e$ vector>>) ->
          (match e with
           | <:ctyp< float32 >> ->
             aux <:expr<fun ($x$:(float,Bigarray.float32_elt) Vector.vector) -> $acc$>> (i+1) q
           | <:ctyp< float64 >> ->
             aux <:expr<fun ($x$:(float,Bigarray.float64_elt) Vector.vector) -> $acc$>> (i+1) q
           | <:ctyp< int32 >> ->
             aux <:expr<fun ($x$:(int32,Bigarray.int32_elt) Vector.vector) -> $acc$>> (i+1) q
           | <:ctyp< int64 >> ->
             aux <:expr<fun ($x$:(int64,Bigarray.int64_elt) Vector.vector) -> $acc$>> (i+1) q
           | _ ->
             aux <:expr<fun ($x$:$e$ Vector.vector) -> $acc$>> (i+1) q
          )
        | PaId(_loc, IdLid(_,x)) as p ->
          let var = (Hashtbl.find !current_args x) in
          (match var.var_type with
           | TVec k -> (
               match k with
               |TInt32 ->
                 aux <:expr<fun ($p$:(int32,Bigarray.int32_elt) Vector.vector) -> $acc$>> (i+1) q
               |TInt64 ->
                 aux <:expr<fun ($p$:(int64,Bigarray.int64_elt) Vector.vector) -> $acc$>> (i+1) q
               |TFloat32 ->
                 aux <:expr<fun ($p$:(float,Bigarray.float32_elt) Vector.vector) -> $acc$>> (i+1) q
               |TFloat64 ->
                 aux <:expr<fun ($p$:(float,Bigarray.float64_elt) Vector.vector) -> $acc$>> (i+1) q
               |_ ->
                 aux <:expr<fun $p$ -> $acc$>> (i+1) q
             )
           | _ ->
             aux <:expr<fun $p$ -> $acc$>> (i+1) q)
        | x -> aux <:expr<fun $x$ -> $acc$>> (i+1) q
  in aux body 0
    (List.rev params)





let gen_arg_from_patt p =
  match p with
  | PaId (_loc, x)   -> <:expr<(spoc_id $ExId(_loc,x)$)>>
  | _  -> failwith "error gen_arg_from_patt"


let rec gen_arg_from_patt2 p =
  match p with
  | PaId (_loc, IdLid (_loc2, x) ) ->
    (let var = (Hashtbl.find !current_args x) in
     match var.var_type with
     | TUnknown  -> <:expr<(new_unknown_var $`int:var.n$ $str:x$)>>
     | TInt32 | TInt64 | TBool ->
       <:expr<(new_int_var $`int:var.n$  $str:x$)>>
     | TFloat32 ->
       <:expr<(new_float_var $`int:var.n$  $str:x$)>>
     | TFloat64  ->
       <:expr<(new_float64_var $`int:var.n$  $str:x$)>>
     | Custom (t,n) ->
       <:expr<(new_custom_var $str:n$ $`int:var.n$  $str:x$)>>
     | TVec k ->
       (match k with
        | TInt32 | TInt64 | TBool ->   <:expr<(new_int_vec_var $`int:var.n$ $str:x$)>>
        | TFloat32 ->   <:expr<(new_float_vec_var $`int:var.n$ $str:x$)>>
        | TFloat64 ->   <:expr<(new_double_vec_var $`int:var.n$  $str:x$)>>
        | Custom (t,n) ->
          <:expr<(new_custom_vec_var $str:n$ $`int:var.n$  $str:x$)>>
        | _  -> failwith (Printf.sprintf "Forbidden vector  type (%s) in kernel declaration" (ktyp_to_string k)))
     | _ ->  failwith "gap : unimplemented yet"
    )
  | PaTyc (_loc, x, _) ->
    gen_arg_from_patt2 x
  | _  -> failwith "error gen_arg_form_patt2"




let rec patt_is_vector p =
  match p with
  | PaId (_loc, IdLid (_loc2, x) ) ->
    let var = (Hashtbl.find !current_args x) in
    begin
      match var.var_type with
      | TVec _ ->true
      | _ -> false
    end
  | PaTyc (_loc, x, _) ->
    patt_is_vector x
  | _ -> assert false


let rec gen_arg_from_patt3 p =
  match p with
  | PaId (_loc, IdLid (_loc2, x) ) ->
    (let var = (Hashtbl.find !current_args x) in
     let s = <:ident< $lid:("spoc_var"^string_of_int var.n)$>> in
     let v = <:patt< $id:s$ >> in
     let e1, e2, e3, e4, e5 =
       match var.var_type with
       | TUnknown  ->
	 failwith ("Could not infer type for argument "^ x);
         (* <:ctyp< 'a>>,
          * <:expr< Spoc.Kernel.abs>>,
          * <:ctyp< 'a>>,
          * IdAcc(_loc,
          *       IdUid(_loc, "Spoc"),
          *       IdAcc(_loc,
          *             IdUid (_loc, "Kernel"),
          *             IdLid(_loc, "abs"))),
          * <:ctyp< 'a>> *)
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
         <:ctyp< int64>>,
         IdAcc(_loc,
               IdUid(_loc, "Spoc"),
               IdAcc(_loc,
                     IdUid (_loc, "Kernel"),
                     IdUid(_loc, "Int64"))),
         <:ctyp< int64>>
       | TFloat32 ->
         <:ctyp< float>>,
         <:expr< Spoc.Kernel.Float32>>,
         <:ctyp< float>>,
         IdAcc(_loc,
               IdUid(_loc, "Spoc"),
               IdAcc(_loc,
                     IdUid (_loc, "Kernel"),
                     IdUid(_loc, "Float32"))),
         <:ctyp< float>>
       | TFloat64 ->
         <:ctyp< float>>,
         <:expr< Spoc.Kernel.Float64>>,
         <:ctyp< float>>,
         IdAcc(_loc,
               IdUid(_loc, "Spoc"),
               IdAcc(_loc,
                     IdUid (_loc, "Kernel"),
                     IdUid(_loc, "Float32"))),
         <:ctyp< float>>
       | Custom (t,name) ->

         let sarek_namet = TyId(_loc, IdLid(_loc,name^"_sarek")) in
         <:ctyp< $sarek_namet$>>,
         <:expr< Spoc.Kernel.Custom >>,
         <:ctyp< $lid:name$ >>,
         <:ident< Spoc.Kernel.Custom >>,
         <:ctyp<$sarek_namet$>>
       | TVec k ->
         (match k with
          | TInt32 | TBool ->
            <:ctyp< (('spoc_a, 'spoc_b) Vector.vector)>>,
            <:expr< Spoc.Kernel.VInt32>>,
            <:ctyp< ((int32, Bigarray.int32_elt) Vector.vector)>> ,
            <:ident< Spoc.Kernel.VInt32>>,
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
          | Custom (t,name) ->
            let name = TyId(_loc,IdLid(_loc,name))
            and sarek_name = TyId(_loc, IdLid(_loc,name^"_sarek")) in
            <:ctyp< (($name$, $sarek_name$) Vector.vector)>>,
            <:expr< Spoc.Kernel.VCustom>>,
            <:ctyp< (($name$, $sarek_name$) Vector.vector)>>,
            <:ident< Spoc.Kernel.VCustom>>,
            <:ctyp< Spoc.Vector.custom>>
          | _  -> failwith "Forbidden vector type in kernel declaration")
       | _ ->
         assert (not debug);
         failwith "gap3 : unimplemented yet"
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
            <:expr< ($id:s$:$e3$)>>)
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
  | PaTyc (_loc, x, _) ->
    gen_arg_from_patt3 x
  | _  -> failwith "error gen_arg_form_patt3"




let rec float32_expr f =
  let f32_typer f =
    (match f.t with
     | TUnknown -> f.t <- TFloat32
     | TFloat32 -> f.t <- TFloat32
     | TVec TFloat32 -> f.t <- TVec TFloat32
     | _ -> assert (not debug); raise (TypeError (TFloat32, f.t, f.loc)))
  in f32_typer f;

  (match f.e with
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
  let f64_typer f =
    (match f.t with
     | TUnknown -> ()
     | TVec TFloat64 -> f.t <- TVec TFloat64
     | TFloat64-> ()
     | _ ->() )
  in f64_typer f;
  (match f.e with
   | PlusF64 (l, a,b) ->
     (
       f.e <- PlusF64 (l, float64_expr a, float64_expr b)
     )
   | MinF64 (l, a,b) ->
     (
       f.e <- MinF64 (l, float64_expr a, float64_expr b)
     )

   | MulF64 (l, a,b) ->
     (
       f.e <- MulF64 (l, float64_expr a, float64_expr b)
     )

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
   | BoolEqF32 (l,a,b) ->
     f.e <- BoolEqF64 (l, float64_expr a, float64_expr b)
   | BoolLtF32 (l,a,b) ->
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


let gen_ctype t1 t2 t3 name _loc =
  let field_name  = (string_of_ident t2^"_"^t3) in
  begin
    <:str_item< let $lid:field_name$ =
                let open Ctypes in
                Ctypes.field
                $lid:string_of_ident t2$
                $str:field_name$
                $lid:get_sarek_name ((string_of_ctyp t1))$
                ;;
    >>
  end

let gen_ctype_repr t1 t2 name : string =
  let field_name  = (string_of_ident t2^"_t") in
  ((ctype_of_sarek_type (string_of_ctyp t1))^" "^field_name)


type ktyp_repr = {
  type_id : int;
  name : string;
  typ : ktyp;
  ml_typ : str_item;
  ctype : str_item;
  crepr : string;
  ml_to_c : expr;
  c_to_ml : expr;
  build_c : string list;
  compare : string;
}

let type_id = ref 0


let gen_mltyp _loc name t =
  begin
    match t with
    | Custom (KRecord (ctypes,idents,muts),_) ->
      begin
        let aux list_elt list_mut =
          let idfield =
            match list_elt with
            | TyCol(_loc,TyId(_,IdLid(_,t)),_) -> mltype_of_sarek_type t
            | _ -> assert false
          in
          (try
             let rcr = Hashtbl.find !rec_fields idfield in
             rcr.ctyps <- name::rcr.ctyps;
           with
           | Not_found ->
             (let recrd_field =
                { id = (let i = !type_id in incr type_id; i);
                  field = idfield;
                  name = name;
                  ctyps = [name];
                }
              in
              my_eprintf ("Add field : "^(idfield)^"\n");
              Hashtbl.add !rec_fields idfield recrd_field));
          if list_mut then
            let list_elt = TyMut (_loc,list_elt)in
            <:ctyp< $list_elt$ >>
          else
            <:ctyp< $list_elt$ >> in
        <:str_item<
        type $lid:name$ = {

          $List.fold_left2
             (fun elt list_elt list_mut ->
              <:ctyp< $elt$; $aux list_elt list_mut$ >> )
(aux (List.hd ctypes) (List.hd muts)) (List.tl ctypes) (List.tl muts)$} >>
end
| Custom ((KSum l),_) ->
  begin
    let type_of_string s =
      TyId (_loc, (IdUid(_loc,s)) )
    in
    let typeof t1 t2 =
      TyOf(_loc, t1, t2)
    in
    let id = ref 0 in
    let aux (list_elt,t) =
      Hashtbl.add !constructors (list_elt) (
	{id = (let i = !id in incr id; i);
	 name = name;
	 nb_args = 0;
	 ctyp = "";
	 typ = KSum l});
      match t with
      | Some (s:ctyp) ->
        let c = (Hashtbl.find !constructors list_elt) in
        c.nb_args <- 1;
        c.ctyp <-  (string_of_ctyp s);
        let t  =
          typeof (type_of_string list_elt) s
        in
        begin
          <:ctyp<  $t$ >>end
      | None ->
	<:ctyp< $type_of_string list_elt$ >>
    in
    let ct =
      <:ctyp< $Ast.tyOr_of_list (List.map aux l)$>>
    in
    <:str_item< type $lid:name$ = $Ast.TySum (_loc, ct)$>>
  end
| TInt32 ->
  <:str_item< int >>
| _ ->
  <:str_item<  >>
end


type managed_ktyp =
  {
    mk_name : string;
    mk_crepr : string;
  }


let type_repr = Hashtbl.create 10

let rec has_of = function
  | (_,Some t)::q -> true
  | (_,None)::q -> has_of q
  | [] -> false



let gen_ctypes _loc kt name =
  let gen_repr _loc t name =
    let managed_ktypes = Hashtbl.create 5 in
    let sarek_type_name = name^"_sarek" in
    Hashtbl.add sarek_types_tbl name sarek_type_name;
    let ctype =
      begin
        let fieldsML =
          match t with
          | Custom (KRecord (ctypes,idents,_),_) ->
            begin
              let rec content acc l1 l2 =
                match (l1 ,l2) with
                | [],[] ->
                  acc
                | t1::q1, t2::q2 ->
                  content
                    (<:str_item< $acc$
                                 $gen_ctype t1 (IdLid(_loc,sarek_type_name))
                                 (string_of_ident t2) sarek_type_name _loc$
                     >>) q1 q2
                | _ -> assert false
              in
              content (<:str_item< >>) ctypes idents
            end
          | Custom ((KSum l),_) ->
            begin
              let gen_mlstruct accML (cstr,ty) =
                let name = sarek_type_name^"_"^cstr in
                begin
                  match ty with
                  | Some t ->
                    begin
                      let ct = gen_ctype t (IdLid(_loc,name))  sarek_type_name "" _loc in
                      let ctr = gen_ctype_repr
                          t
                          (IdLid(_loc,name))  sarek_type_name  in
                      Hashtbl.add managed_ktypes cstr ctr;
                      <:str_item<
                                  $accML$ ;;
                                  type $lid:name$ ;;
                                  let $lid:name$ : $lid:name$ Ctypes.structure Ctypes.typ =
                                  Ctypes.structure  $str:name$ ;;
                                  $ct$ ;;
                                  let () = Ctypes.seal $lid:name$ ;; >>
                    end
                  | None ->
                    begin

                      <:str_item<
                        $accML$
                      >>
                    end
                end

              in
              let rec content (accML)  = function
                | t::q -> content (gen_mlstruct accML t) q
                | [] -> accML
              in
              let tag = sarek_type_name^"_tag" in
              let fields = content (<:str_item<let $lid:tag$ =
                                               Ctypes.field $lid:sarek_type_name$ $str:tag$ Ctypes.int ;;>>) l
              in

              if has_of l then
                let union = sarek_type_name^"_union" in
                let union_name = union^"_" in
                let rec aux acc = function
                  | (cstr, Some x)::q ->
                    let field_name_c = sarek_type_name^"_"^cstr in
                    let field_name = field_name_c^"_val" in
                    aux <:str_item<$acc$;;
                                   let $lid:field_name$ = let open Ctypes in
                                   Ctypes.field $lid:union_name$ $str:field_name_c$ $lid:field_name_c$;; >> q
                  | (_,None)::q -> aux acc q
                  | [] -> <:str_item< $acc$;;
                                      let () = Ctypes.seal $lid:union_name$;;
                                      let $lid:sarek_type_name^"_union"$ =
                                      Ctypes.field
                                      $lid:sarek_type_name$
                                      $str:union$
                                      $lid:union_name$;;
                          >>
                in
                aux <:str_item< $fields$;;
                                type $lid:union_name$ ;;
                                let $lid:union_name$ : ($lid:union_name$ Ctypes.union Ctypes.typ) =
                                Ctypes.union $str:union_name$ ;;
                    >> l
              else
                fields
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
      ctype =  ctype;
      crepr =
        begin
          match t with
          | Custom (KRecord (l1,l2,_),_) ->
            let rec content (ctype) (l1,l2)=
              match (l1 : ctyp list), (l2 :ident list) with
              | [],[] -> ctype
              | t1::q1, t2::q2 ->
                content (ctype^"\n\t"^(ctype_of_sarek_type (string_of_ctyp t1)) ^
                         "  " ^ (string_of_ident t2) ^ ";") (q1,q2)
              | _ -> assert false
            in

            let fieldsC =
              let a = content "" (l1,l2) in
              ("struct " ^ sarek_type_name ^ " {"
               ^a ^"\n};")  in
            fieldsC^";\n"
          | Custom ((KSum l),_) ->
            let gen_cstruct accC = function
              | cstr,Some t ->
                accC ^ "\t\tstruct "^sarek_type_name^"_"^cstr^" {\n\t\t\t"^
                (Hashtbl.find managed_ktypes cstr)^";\n\t\t} "^
                sarek_type_name^"_"^cstr^";\n"
              | cstr,None -> accC ^ ""
            in
            let rec contents accC  = function
              | t::q -> contents (gen_cstruct accC t) q
              | [] -> accC
            in
            let fieldsC  =
              let b = contents "" l in
              ("struct " ^ sarek_type_name ^ " {\n\tint "^
               sarek_type_name ^ "_tag;\n"^
               (if has_of l then
                  "\tunion "^sarek_type_name^"_union {\n"
                  ^b ^"\t} "^sarek_type_name^"_union;"
                else "" )^
               "\n}")
            in fieldsC^";\n"
          | _ -> "int" ;
        end;
      compare =
        begin
          match t with
          | Custom (KRecord (l1,l2,_),_) ->
            let rec content (ctype) (l1,l2)=
              match (l1 : ctyp list), (l2 :ident list) with
              | [],[] -> ctype
              | t1::q1, t2::q2 ->
                (try
                   let tt = Hashtbl.find sarek_types_tbl (string_of_ctyp t1) in
                   content (ctype^ "&& spoc_custom_compare_"^tt^
                            " ( a."^(string_of_ident t2)^", b."^
                            (string_of_ident t2)^")") (q1,q2)
                 with
                 | _ ->
                   content (ctype^ " && ( a."^(string_of_ident t2)^
                            " ==  b."^(string_of_ident t2)^")") (q1,q2)
                )
              | _ -> assert false
            in

            let fieldsC =
              let a = content "true " (l1,l2) in
              ("int spoc_custom_compare_"^sarek_type_name^"(struct " ^ sarek_type_name ^
               " a, struct "^sarek_type_name ^" b) {\n\treturn ("
               ^a ^");\n}")  in
            fieldsC
          | Custom ((KSum l),_) ->
            let c = ref (-1) in
            let gen_cstruct accC = function
              | cstr,Some t ->
                incr c;
                accC^"\tcase "^string_of_int !c^" :\n"^
                let field = sarek_type_name^"_union."^
                            sarek_type_name^"_"^cstr^"."^
                            sarek_type_name^"_"^cstr^"_t" in
                (try
                   let tt = Hashtbl.find sarek_types_tbl (string_of_ctyp t) in
                   ("\t\treturn spoc_custom_compare_"^tt^
                    "( a."^field^", b."^field^");\n\t\tbreak;\n")
                 with _ ->
                   "\t\treturn (a."^field^" == b."^field^");\n\t\tbreak;\n"
                )
              | cstr,None ->
                incr c;
                accC^"\tcase "^string_of_int !c^" :\n"^"\t\treturn 1;\n\t\tbreak;\n"
            in
            let rec contents accC  = function
              | t::q -> contents (gen_cstruct accC t) q
              | [] -> accC
            in
            let fieldsC  =
              let b = contents "" l in
              ("int spoc_custom_compare_"^sarek_type_name^"(struct " ^ sarek_type_name ^
               " a, struct "^sarek_type_name ^" b) {\n"^
               (if has_of l then
                  "\tif (a."^sarek_type_name^"_tag != b."^sarek_type_name^"_tag)\n"^
                  "\t\treturn 0;\n"^
                  ("\tswitch (a."^sarek_type_name^"_tag) {\n"^
                   b ^"}\n\treturn 0;\n}")
                else "\treturn (a."^sarek_type_name^"_tag == b."^sarek_type_name^"_tag);\n}"))
            in fieldsC
          | _ -> "int" ;
        end;
      ml_to_c =
        begin
          match t with
          | Custom (KRecord (l1,l2,_),_) ->
            let copy_to_c =
              let aux b c =
                let field_name =
                  (sarek_type_name^"_"^(string_of_ident c)) in
                try
                  let get = (Hashtbl.find type_repr (string_of_ctyp b)).ml_to_c in
                  <:expr< Ctypes.setf tmp $lid:field_name$
                          ($get$ x.$lid:string_of_ident c$);>>
                with
                | _ ->
                  <:expr< Ctypes.setf tmp $lid:field_name$ x.$lid:string_of_ident c$;>>
              in
              List.fold_left2
                (fun a b c ->
                   <:expr< $a$; $aux b c$>>)
                (aux (List.hd l1) (List.hd l2)) (List.tl l1) (List.tl l2)
            in
            begin
              <:expr< fun x ->
                      let tmp =
                      Ctypes.make $lid:sarek_type_name$ in
                      $copy_to_c$ ;
                      tmp
              >>;
            end

          | Custom ((KSum l),_) ->
            let copy_to_c =
              let gen_sum_rep  l =
                let rec aux acc tag = function
                  | (cstr,of_) :: q ->
                    aux ((cstr,tag,of_)::acc) (tag+1) q
                  | [] -> acc
                in
                aux [] 0 l
              in
              let repr_ = gen_sum_rep l
              in
              let copy_content cstr (of_ :ctyp option) =
                let copy_of _of =
                  try
                    let get =
                      (Hashtbl.find type_repr (string_of_ctyp _of)).ml_to_c in
                    <:expr<
                     let  union =
                     Ctypes.make $lid:sarek_type_name^"_union_"$ in
                     let str =
                     Ctypes.make $lid:sarek_type_name^"_"^cstr$ in
                     Ctypes.setf str $lid:sarek_type_name^"_"^cstr^"_"^sarek_type_name$ ($get$ sarek_tmp);
                     Ctypes.setf union
                     $lid:sarek_type_name^"_"^cstr^"_val"$ str;
                     Ctypes.setf tmp
                     $lid:sarek_type_name^"_union"$ union ;
                     >>
                  with
                  | _ -> <:expr<
                          let  union =
                          Ctypes.make $lid:sarek_type_name^"_union_"$ in
                          let str =
                          Ctypes.make $lid:sarek_type_name^"_"^cstr$ in
                          Ctypes.setf str $lid:sarek_type_name^"_"^cstr^"_"^sarek_type_name$ sarek_tmp;
                          Ctypes.setf union
                          $lid:sarek_type_name^"_"^cstr^"_val"$ str;
                          Ctypes.setf tmp
                          $lid:sarek_type_name^"_union"$ union ;
                          >>
                in
                match of_ with
                | Some x -> copy_of x
                | None -> <:expr< >>
              in
              let match_cases =
                List.map
                  (fun (cstr,tag,of_) ->
                     begin
                       <:match_case<
$match of_ with
                       | Some _ ->
                          <:patt< $uid:cstr$ sarek_tmp>>
		     | None -> <:patt< $uid:cstr$ >>$
			       ->
			       begin
			         Ctypes.setf tmp $lid:sarek_type_name^"_tag"$
						      $int:string_of_int tag$ ;
                                                           $copy_content cstr of_$;
			       end
		       >>
                     end
                  )
                  repr_
              in
	      let l =
		List.rev match_cases in
	      <:expr< match x with
		      $list:l$  >>
	    in
            begin
	      <:expr< fun x ->
                                                                              let tmp =
                                                                              Ctypes.make $lid:sarek_type_name$ in
                                                                              $copy_to_c$ ;
                                                                              tmp
		      >>;
            end

          | _ -> assert false
        end;
      c_to_ml =
        begin
          match t with
          | Custom (KRecord (l1,l2,_),_) ->
            let copy_to_caml =
              let aux b c =
                let field_name =
                  (sarek_type_name^"_"^(string_of_ident c)) in
                try
                  let get = (Hashtbl.find type_repr (string_of_ctyp b)).c_to_ml in
                  <:rec_binding<
                                  $c$ =
                                  $get$
                                  (Ctypes.getf x $lid:field_name$) >>

                with | _ ->
                  <:rec_binding<
                                 $c$ =
                                 Ctypes.getf x $lid:field_name$ >>
              in
              List.fold_left2
                (fun a b c ->
                   <:rec_binding< $a$; $aux b c$>>)
                (aux (List.hd l1) (List.hd l2)) (List.tl l1) (List.tl l2)
            in

            begin
              <:expr< fun x -> $ExRec(_loc, copy_to_caml, Ast.ExNil _loc)$>>;
            end
          (*              <:expr< fun x -> {$copy_to_caml$} ;
                          >>;
                          end*)
          | Custom ((KSum l),_)  ->
            let gen_sum_rep  l =
              let rec aux acc tag = function
                | (cstr,of_) :: q ->
                  aux ((cstr,tag,of_)::acc) (tag+1) q
                | [] -> acc
              in
              aux [] 0 l

            in

            let copy_content sarek_type_name cstr (of_:ctyp option) =
              let copy_of _of =
                try
                  let get = (Hashtbl.find type_repr (string_of_ctyp _of)).c_to_ml in

                  <:expr<
                          $get$ (let union =
                          Ctypes.getf x $lid:sarek_type_name^"_union"$ in
                          let $lid:"val"^cstr$ =
                          Ctypes.getf union
                          $lid:sarek_type_name^"_"^cstr^"_val"$ in
                          Ctypes.getf $lid:"val"^cstr$ $lid:sarek_type_name^"_"^cstr^"_"^sarek_type_name$)
                  >>
                with
                |_ ->
                  <:expr< let union =
                          Ctypes.getf x $lid:sarek_type_name^"_union"$ in
                          let $lid:"val"^cstr$ =
                          Ctypes.getf union
                          $lid:sarek_type_name^"_"^cstr^"_val"$ in
                          Ctypes.getf $lid:"val"^cstr$ $lid:sarek_type_name^"_"^cstr^"_"^sarek_type_name$
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
                (<:match_case< a -> failwith ("Sarek error : c to caml failed for type "^
                               $str:name$^" : "^(string_of_int a) )>>)::
                (List.map
                   (fun (cstr,tag,of_) ->
                      begin
                        <:match_case<
                                      $int:string_of_int tag$ ->
                                      $copy_content sarek_type_name cstr of_$
                        >>
                      end
                   )
                   repr_)
              in
              <:expr< let tag = Ctypes.getf x $lid:sarek_type_name^"_tag"$ in
                      match tag with
                      $list:(List.rev match_cases)$ >>

            in
            <:expr< fun x -> $copy_to_caml$>>
          | _ ->
            assert false
        end;
      build_c =
        match t with
        | Custom (KRecord (l1,l2,_),n) ->
          let params =
            let i = ref false in
            List.fold_left2 (fun a b c ->
                a ^( if !i then "," else (i := true; "")) ^
                (ctype_of_sarek_type (string_of_ctyp b)) ^" "^(string_of_ident c))
              "" l1 l2 in
          let content =
            List.fold_left (fun a b ->
                let i = (string_of_ident b) in
                a^"\tsarek_tmp."^i^ " = " ^i^";\n")
              "" l2
          in
          ["struct "^sarek_type_name^" build_"^n^" ("^params^"){\n\t"^
           "struct "^sarek_type_name^" sarek_tmp;\n"^
           content^"\treturn sarek_tmp;\n}"];
        | Custom ((KSum l),n) ->
          let content i = function
            |cstr,None ->
              "struct "^sarek_type_name^" build_"^n^"_"^cstr^"(){\n\t"^
              "struct "^sarek_type_name^" sarek_tmp;\n"^
              "\tsarek_tmp."^sarek_type_name^"_tag = "^
              (string_of_int i)^";\n\treturn sarek_tmp;\n}"

            |cstr,Some of_ ->
              let params =
                ctype_of_sarek_type (string_of_ctyp of_) in
              "struct "^sarek_type_name^" build_"^n^"_"^cstr^"("
              ^params^" "^(String.uncapitalize_ascii cstr)^"){\n\t"^
              "struct "^sarek_type_name^" sarek_tmp;\n"^
              "\tsarek_tmp."^sarek_type_name^"_tag = "^
              (string_of_int i)^";\n"^
              "\tstruct "^sarek_type_name^"_"^cstr^" t"^cstr^";\n"^
              "\tt"^cstr^"."^sarek_type_name^"_"^cstr^"_t = "^
              (String.uncapitalize_ascii cstr)^";\n"^
              "\tsarek_tmp."^sarek_type_name^"_union."^
              sarek_type_name^"_"^cstr^" = "^"t"^cstr^";\n"^
              "\treturn sarek_tmp;\n}"
          in
          List.mapi content l
        | _ -> assert false;
    }
  in
  let t = gen_repr _loc (Custom (kt,name)) name in
  Hashtbl.add type_repr name t;
  let sarek_type_name = name^"_sarek" in

  let custom =
    let l = [
      <:rec_binding<size = Ctypes.sizeof $lid:sarek_type_name$>>;
      <:rec_binding<get =
                    (fun c i ->
                    $t.c_to_ml$
                    (let open Ctypes in
                    let cr = Obj.repr c in
                    let ptrcr =
                    (Ctypes.from_voidp $lid:sarek_type_name$
                    (Ctypes.ptr_of_raw_address
                    (Obj.magic cr))) in
                    !@(ptrcr +@ i)))>>;
      <:rec_binding<set = (fun c i v ->
		    let open Ctypes in
		    let cr = Obj.repr c in
		    let ptrcr =
		    (Ctypes.from_voidp $lid:sarek_type_name$
		    (Ctypes.ptr_of_raw_address
		    (Obj.magic cr))) in
		    Ctypes.(<-@) (Ctypes.(+@) ptrcr  i)
		    ($t.ml_to_c$ v)) >>]
    in
    ExRec(_loc, rbSem_of_list l, (Ast.ExNil _loc))
  in
  begin
    Ast.stSem_of_list
      ([
	(<:str_item<open Vector>>) ;
	t.ml_typ ;
	t.ctype ;
	(<:str_item<
let $lid:"custom"^(String.capitalize_ascii name)$ : (($lid:name$,$lid:sarek_type_name$) Vector.custom) =
$custom$>>) ;
	(<:str_item<let $lid:t.name^"_c_repr"$ = $str:t.crepr$>>) ;
	(<:str_item<Kirc.constructors := $str:t.crepr$ :: !Kirc.constructors>>) ;
	(<:str_item<Kirc.constructors := $str:t.compare$ :: !Kirc.constructors>>)
      ]
        @
	(List.map (fun a ->
             (<:str_item<Kirc.constructors := $str:a^"\n"$ :: !Kirc.constructors>>)
	   ) t.build_c
	)
      )
  end



let gen_labels _loc (t1 : ident * ctyp * bool)
    (t2 : (ctyp list * ident list * bool list) option)
  : ctyp list * ident list * bool list =
  let s,t,m = t1 in
  let t1 =
    TyCol (_loc, (TyId (_loc, s)), t)
  in
  match t2 with
  | Some (t2,s2,m2) ->
    t1::t2, s::s2, m::m2
  | None ->
    t1::[], s::[],m::[]


let gen_constructors _loc
    (t1 : string * ctyp option)
    (t2 : (string * ctyp option) list option)
  : (string *ctyp option) list =
  match t2 with
  | Some t ->
    t1::t
  | None ->
    t1::[]
