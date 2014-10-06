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
type kernel


type var = 
  | Var of string

(*and kint =*)


and kvect =
  | IntVect of int
  | Floatvect of int

type intrinsics = string*string

type elttype = 
  | EInt32
  | EInt64
  | EFloat32
  | EFloat64

type memspace =
  | LocalSpace
  | Global
  | Shared

type  k_ext =
  | Kern of  k_ext* k_ext
  | Params of  k_ext
  | Plus of  k_ext* k_ext
  | Plusf of  k_ext *  k_ext
  | Min of  k_ext* k_ext
  | Minf of  k_ext *  k_ext
  | Mul of  k_ext* k_ext
  | Mulf of  k_ext *  k_ext
  | Div of  k_ext* k_ext
  | Divf of  k_ext *  k_ext
  | Mod of  k_ext *  k_ext
  | Id of string
  | IdName of string
  | GlobalFun of k_ext*string
  | IntVar of int
  | FloatVar of int
  | UnitVar of int
  | CastDoubleVar of int
  | DoubleVar of int
  | Arr of int*k_ext*elttype*memspace
  | VecVar of  k_ext*int
  | Concat of  k_ext* k_ext
  | Constr of string * string * k_ext list
  | Record of string * k_ext list
  | RecGet of k_ext * string
  | RecSet of k_ext * k_ext
  | Empty
  | Seq of  k_ext *  k_ext
  | Return of  k_ext
  | Set of  k_ext *  k_ext 
  | Decl of  k_ext
  | SetV of  k_ext*  k_ext
  | SetLocalVar of  k_ext *  k_ext *  k_ext
  | Intrinsics of intrinsics
  | IntId of string*int
  | Int of int
  | Float of float
  | Double of float
  | Custom of string*int
  | IntVecAcc of  k_ext *  k_ext
  | Local of  k_ext *  k_ext
  | Acc of  k_ext *  k_ext
  | Ife of  k_ext *  k_ext  *  k_ext   
  | If of  k_ext *  k_ext
  | Match of string* k_ext * case list
  | Or of  k_ext *  k_ext
  | And of  k_ext *  k_ext
  | EqBool of  k_ext *  k_ext
  | LtBool of  k_ext *  k_ext
  | GtBool of  k_ext *  k_ext
  | LtEBool of  k_ext *  k_ext
  | GtEBool of  k_ext *  k_ext
  | DoLoop of  k_ext *  k_ext *  k_ext *  k_ext
  | While of  k_ext *  k_ext
  | App of  k_ext *  k_ext array
  | GInt of (unit -> int32)
  | GFloat of (unit -> float)
  | Unit


and case =  int * (string*string*int) option * k_ext

type  kfun = 
  | KernFun of  k_ext* k_ext



let print_ast a =
  let print i s =
    for j = 0 to i - 1 do
      Printf.printf "  ";
    done; 
    Printf.printf "%s\n" s
  in  
  let rec aux i = function
    | Kern (a,b) ->
      print i "Kern"; 
      (aux (i + 1) a);
      (aux (i + 1) b)
    | Params p ->
      print i "Params";
      (aux (i+1) p)
    | Plus (a,b) ->
      print i "Plus";
      aux (i+1) a;
      aux (i+1) b;
    | Plusf (a,b) ->
      print i "Plusf";
      aux (i+1) a;
      aux (i+1) b;
    | Min (a,b) ->
      print i "Min";
      aux (i+1) a;
      aux (i+1) b;
    | Minf (a,b) ->
      print i "Minf";
      aux (i+1) a;
      aux (i+1) b;
    | Mul (a,b) ->
      print i "Mul";
      aux (i+1) a;
      aux (i+1) b;
    | Mulf (a,b) ->
      print i "Mulf";
      aux (i+1) a;
      aux (i+1) b;
    | Div (a,b) ->
      print i "Div";
      aux (i+1) a;
      aux (i+1) b;
    | Divf (a,b) ->
      print i "Divf";
      aux (i+1) a;
      aux (i+1) b;
    | Mod (a,b) ->
      print i "Mod";
      aux (i+1) a;
      aux (i+1) b;
    | Id (s) ->
      print i ("Id "^s)
    | IdName s -> 
      print i ("IdName "^s);
    | IntVar ii ->
      print i ("IntVar "^(string_of_int ii))
    | FloatVar ii ->
      print i ("FloatVar "^(string_of_int ii))
    | CastDoubleVar ii ->
      print i ("CastDoubleVar "^(string_of_int ii))
    | DoubleVar ii ->
      print i ("DoubleVar "^(string_of_int ii))
    | UnitVar ii ->
      print i ("UnitVar "^(string_of_int ii))
    | VecVar (t,ii) ->
      print i ("VecVar "^(string_of_int ii))
    | Concat (a,b) ->
      print i "Concat";
      aux (i+1) a;
      aux (i+1) b;
    | Empty ->
      print i "Empty"
    | Seq (a,b) ->
      print i "Seq";
      aux (i+1) a;
      aux (i+1) b;
    | Return a ->
      print i "Return";
      aux (i+1) a;
    | Set (a,b) ->
      print i "Set";
      aux (i+1) a;
      aux (i+1) b;
    | Decl (a) ->
      print i "Decl";
      aux (i+1) a;
    | Acc (a,b) ->
      print i "Acc";
      aux (i+1) a;
      aux (i+1) b;
    | SetV (a,b) ->
      print i "SetV";
      aux (i+1) a;
      aux (i+1) b;
    | SetLocalVar (a,b,c) ->
      print i "SetLocalVar";
      aux (i+1) a;
      aux (i+1) b;
      aux (i+1) c;
    | Intrinsics _ ->
      print i "Intrinsics"
    | IntId (s,ii) ->
      print i ("IntId "^s^" "^(string_of_int ii));
    | Int ii ->
      print i ("Int "^(string_of_int ii));
    | Float f 
    | Double f ->
      print i ("Float "^(string_of_float f));
    | IntVecAcc (a,b) ->
      print i "IntVecAcc";
      aux (i+1) a;
      aux (i+1) b;
    | Local (a,b) ->
      print i "Local";
      aux (i+1) a;
      aux (i+1) b;
    | Ife (a,b,c) ->
      print i "Ife";
      aux (i+1) a;
      aux (i+1) b;
      aux (i+1) c;
    | If (a,b) ->
      print i "If";
      aux (i+1) a;
      aux (i+1) b;
    | EqBool (a,b) ->
      print i "EqBool";
      aux (i+1) a;
      aux (i+1) b;
    | Or (a,b) ->
      print i "Or";
      aux (i+1) a;
      aux (i+1) b;
    | And (a,b) ->
      print i "And";
      aux (i+1) a;
      aux (i+1) b;
    | LtBool (a,b) ->
      print i "LtBool";
      aux (i+1) a;
      aux (i+1) b;
    | GtBool (a,b) ->
      print i "GtBool";
      aux (i+1) a;
      aux (i+1) b;
    | LtEBool (a,b) ->
      print i "LtEBool";
      aux (i+1) a;
      aux (i+1) b;
    | GtEBool (a,b) ->
      print i "GtEBool";
      aux (i+1) a;
      aux (i+1) b;
    | DoLoop (a,b,c,d) ->
      print i "DoLoop";
      aux (i+1) a;
      aux (i+1) b;
      aux (i+1) c;
      aux (i+1) d;
    | While (a,b) ->
      print i "While";
      aux (i+1) a;
      aux (i+1) b;
    | Arr (s,l,t,m) ->
      let memspace = 
        match m with 
        | LocalSpace -> "__private"
        | Shared -> "__local"
        | Global -> "__global"
      and elttype = 
        match t with
        | EInt32 -> "int"
        | EInt64 -> "long"
        | EFloat32 -> "float"
        | EFloat64 -> "double" 
      in
      print i ("Arr" ^ memspace^" "^elttype);
    | App (a,b) ->
      print i "App";
      aux (i+1) a;
      Array.iter (aux (i+1)) b;
    | GInt a ->
      print i "GInt"
    | GFloat a ->
      print i "GFloat"
    | Unit ->
      print i "Unit"
    | GlobalFun (e,s) ->
      print i ("Global Fun " ^s);
      aux (i+1) e;
    | Constr (s1,s2,l) ->
      print i ("Constr "^s1^" "^s2);
      List.iter (fun a -> aux (i+1) a) l
    | Record (s,l) ->
      print i ("Record "^s);
      List.iter (fun a -> aux (i+1) a) l
    | RecGet (r,s) ->
      print i ("RecGet");
      aux (i+1) r
    | RecSet (r,v) ->
      print i ("RecGet");
      aux (i+1) r;
      aux (i+1) v;
    | Custom (s,_) -> 
      print i ("Custom "^s)
    | Match (s,e1,l) ->
      print i ("Match "^s);
      aux (i+1) e1;
      List.iter (fun (_,_,a) -> aux (i+1) a) l
  in aux 0 a;;
