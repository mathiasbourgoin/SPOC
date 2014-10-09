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
open Kirc_Ast

let space i =
  String.make i ' '

let global_funs = Hashtbl.create 0

let return_v = ref ("","")

let global_fun_idx = ref 0 

let rec parse_fun i a b = 
 let rec aux name a =
 let rec aux2 i a =
  match a with 
 | Kern  (args,body) -> 
    (let pargs = aux2 i args  in
     let pbody = 
       match body with 
       | Seq (_,_)  -> (aux2 i body )
       | _  ->  ((aux2 i body )^"\n"^(space i))
     in
     (pargs ^ pbody)^ "}")
  | Params k -> 
    ("__device__ "^b^" "^name^"  ( "^
     (if (fst !return_v) <> "" then
        (fst !return_v)^", " else "")^(parse i k)^" ) {")
  | a -> parse  i a
in 
aux2 i a in

let name = 
try snd (Hashtbl.find global_funs a) with
| Not_found ->
(let gen_name =  ("spoc_fun__"^(string_of_int !global_fun_idx)) in
     let fun_src = aux gen_name a in
     incr global_fun_idx;
     Hashtbl.add global_funs a (fun_src,gen_name) ;
     gen_name)
in 
name

and parse i = function
  | Kern (args,body) -> 
    (let pargs = parse i args in
     let pbody = 
       match body with 
       | Seq (_,_)  -> (parse i body)
       | _  ->  ((parse i body)^"\n"^(space i))
     in
     (pargs ^ pbody)^ "\n}\n#ifdef __cplusplus\n}\n#endif")
  | Local (x,y)  -> (space (i))^"{"^
                    (parse i x)^";\n"^
                    (space (i))^(parse i y)^
                    ";\n"^(space (i))^"}\n"
  | VecVar (t,i)  -> 
    (match t with
     | Int _ ->"int"
     | Float _ -> "float"
     | Double _ -> "double"
     | _ -> assert false
    )^("* spoc_var"^(string_of_int (i)))
  | Block b -> (space i)^"{\n"^parse (i+1) b^"\n"^(space i)^"}"
  | IdName s  ->  s
  | IntVar s -> ("int spoc_var"^(string_of_int s))
  | FloatVar s -> ("float spoc_var"^(string_of_int s))
  | UnitVar v -> assert false
  | CastDoubleVar s -> ("(double) spoc_var"^(string_of_int s))
  | DoubleVar s -> ("double spoc_var"^(string_of_int s))
  | Arr (s,l,t,m) -> 
    let memspace = 
      match m with 
      | LocalSpace -> ""
      | Shared -> "__shared__"
      | Global -> "__device__"
    and elttype = 
      match t with
      | EInt32 -> "int"
      | EInt64 -> "long"
      | EFloat32 -> "float"
      | EFloat64 -> "double" 
    in
        (memspace^" "^elttype^" spoc_var"^
                       (string_of_int s)^"["^
                       (parse i l)^"]")
  | Params k -> 
    ("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n__global__ void spoc_dummy ( "^
     (if (fst !return_v) <> "" then
        (fst !return_v)^", " else "")^(parse i k)^" ) {\n")
  (*    let rec aux acc = function
        	| t::[]  ->  acc^", " ^(parse t)
        	| t::q  -> aux (acc^","^parse t) q
        	in
        	"f (" ^(List.fold_left (fun a b -> (a^(parse b))) "" l) ^ ")\n{"*)
  |Concat (a,b)  ->  
    (match b with 
     | Empty  -> (parse i a)
     | Concat (c,d)  ->  ((parse i a)^", "^(parse i b))
     | _  -> failwith "parse concat"
    )
  | Plus  (a,b) -> ("("^(parse_int i a)^" + "^(parse_int i b)^")")
  | Plusf  (a,b) -> ("("^(parse_float i a)^" + "^(parse_float i b)^")")
  | Min  (a,b) -> ("("^(parse_int i a)^" - "^(parse_int i b)^")")
  | Minf  (a,b) -> ("("^(parse_float i a)^" - "^(parse_float i b)^")")
  | Mul  (a,b) -> ("("^(parse_int i a)^" * "^(parse_int i b)^")")
  | Mulf  (a,b) -> ("("^(parse_float i a)^" * "^(parse_float i b)^")")
  | Div  (a,b) -> ("("^(parse_int i a)^" / "^(parse_int i b)^")")
  | Divf  (a,b) -> ("("^(parse_float i a)^" / "^(parse_float i b)^")")
  | Mod  (a,b) -> ("("^(parse_int i a)^" % "^(parse_int i b)^")")

  | Id (s)  -> s
  | Set (var,value) 
  | Acc (var,value) -> 
    ((parse i var)^" = "^
     (parse i value)^";")
  | Decl (var) ->
    (parse i var)
  | SetLocalVar (v,gv,k) -> 
    ((parse i v)^" = "^
     (match gv with
      | Intrinsics i -> ((parse_intrinsics i)^";")
      | _ -> (parse i gv))^";\n"^(space i)^(parse i k))
  | Return k -> 
    (if (snd !return_v) <> "" then
       snd !return_v
     else
       (space i)^"return ")^
    (parse i k)
  | Unit  -> ""
  | IntVecAcc (vec,idx)  -> (parse i vec)^"["^(parse i idx)^"]"
  | SetV (vecacc,value)  -> (
      (parse i vecacc)^" = "^(parse i value)^";") 
  | Int  a  -> string_of_int a
  | GInt  a  -> Int32.to_string (a ())
  | GFloat  a  -> (string_of_float (a ()))^"f"
  | Float f -> (string_of_float f)^"f"
  | Double f -> string_of_float f
  | IntId (s,_) -> s
  | Intrinsics gv -> parse_intrinsics gv
  | Seq (a,b)  -> (parse i a)^" ;\n"^(space i)^(parse i b)
  | Ife (a,b,c) -> "if ("^(parse i a)^"){\n"^(space (i+2))^(parse (i+2) b)^";}\n"^(space i)^"else{\n"^(space (i+2))^(parse (i+2) c)^";}\n"^(space i)
  | If (a,b) -> "if ("^(parse i a)^"){\n"^(space (i+2))^(parse (i+2) b)^";}\n"^(space i)
  | Or (a,b) -> (parse i a)^" || "^(parse i b)
  | And (a,b) -> (parse i a)^" && "^(parse i b)
  | EqBool (a,b) -> (parse i a)^" == "^(parse i b)
  | EqSum (n,a,b,l) -> (parse i a)^" == "^(parse i b)
  | LtBool (a,b) -> (parse i a)^" < "^(parse i b)
  | GtBool (a,b) -> (parse i a)^" > "^(parse i b)
  | LtEBool (a,b) -> (parse i a)^" <= "^(parse i b)
  | GtEBool (a,b) -> (parse i a)^" >= "^(parse i b)
  | DoLoop (a,b,c,d) -> 
    let id = parse i a in
    let min = parse i b in
    let max = parse i c in
    let body = parse (i+2) d in
    "for (int "^id^" = "^min^"; "^id^" <= "^max^"; "^id^"++){\n"^(space (i+2))^body^"}"
  | While (a,b) ->
    let cond = parse i a in
    let body = parse (i+2) b in
    "while ("^cond^"){\n"^(space (i+2))^body^"}"
  | App (a,b) ->
    let f = parse i a in
    let rec aux = function
      | t::[] -> parse i t
      | t::q -> (parse i t)^","^(aux q)
      | [] -> assert false
    in 
    (match a with 
     | Intrinsics ("return","return") -> f^" "^(aux (Array.to_list b))^" "
     |  _ -> f^" ("^(aux (Array.to_list b))^") ")
  | Empty  -> ""
  | GlobalFun (a,b) -> 
     let s = (parse_fun i a b) in
     s
  | Constr (t,s,l) ->
    "build_"^t^"_"^s^"("^(List.fold_left (fun a b -> a^parse i b) "" l)^")"
  | Record (s,l) ->
    let params = 
      match l with
      | t::q -> (parse i t)^(List.fold_left (fun a b -> a^parse i b) "," q)
      | [] -> assert false in
    "build_"^s^"("^params^")"
  | RecGet (r,f) ->
    (parse i r)^"."^f
  | RecSet (r,v) ->
    (parse i r)^" = "^(parse i v)
  | Custom  _ -> assert false
  | Match (s,e,l) -> assert false



and parse_int n = function
  | IntId (s,_) -> s
  | Int i  ->  string_of_int i
  | GInt i  ->  Int32.to_string (i ())
  | IntVecAcc (s,i)  -> (parse n s)^"["^(parse_int n i)^"]"
  | Plus (a,b) as v ->  parse n v
  | Min (a,b) as v ->  parse n v
  | Mul (a,b) as v ->  parse n v
  | Mod (a,b) as v ->  parse n v
  | Div (a,b) as v ->  parse n v
  | App (a,b) as v -> parse n v
  | a -> parse_float n a
(*  | _  -> failwith "error parse_int" *)

and parse_float n = function
  | IntId (s,_) -> s
  | Float f  ->  (string_of_float f)^"f"
  | GFloat f  ->  (string_of_float (f ()))^"f"
  | CastDoubleVar s -> ("(double) spoc_var"^(string_of_int s))
  | Double f  ->  "(double) "^(string_of_float f)
  | IntVecAcc (s,i)  -> (parse n s)^"["^(parse_int n i)^"]"
  | Plusf (a,b) as v ->  parse n v
  | Minf (a,b) as v ->  parse n v
  | Mulf (a,b) as v ->  parse n v
  | Divf (a,b) as v ->  parse n v
  | App (a,b) as v -> parse n v
  | SetV (a,b) as v -> parse n v
  | Intrinsics gv -> parse_intrinsics gv
  | a  -> print_ast a; failwith "cuda error parse_float"

and parse_vect = function
  | IntVect i  -> i
  | _  -> failwith "error parse_vect"

and parse_intrinsics (cudas,opencls) = cudas
