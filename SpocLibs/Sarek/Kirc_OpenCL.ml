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

let return_v = ref ("","")

let rec parse i = function
  | Kern (args,body) -> 
    (let pargs = parse i args in
     let pbody = 
       match body with 
       | Seq (_,_)  -> (parse i body)
       | _  ->  ((parse i body)^"\n"^(space i))
     in
     (pargs ^ pbody)^"}")
  | Local (x,y)  -> (space (i))^""^
                    (parse i x)^";\n"^
                    (space (i))^(parse i y)^
                    "\n"^(space (i))^""
  | VecVar (t,i)  -> 
    (match t with
     | Int _ ->"__global int"
     | Float _ -> "__global float"
     | Double _ -> "__global double"
     | _ -> assert false
    )^("* spoc_var"^(string_of_int (i)))
  | IdName s  ->  s
  | IntVar s -> ("int spoc_var"^(string_of_int s))
  | FloatVar s -> ("float spoc_var"^(string_of_int s))
  | UnitVar v -> assert false
  | CastDoubleVar s -> ("(double) spoc_var"^(string_of_int s))
  | DoubleVar s -> ("double spoc_var"^(string_of_int s))
  | IntArr (s,l) -> ("__shared__ int spoc_var"^
                     (string_of_int s)^"["^
                     (parse i l)^"]")
  | Int32Arr (s,l) -> ("__shared__ int spoc_var"^
                       (string_of_int s)^"["^
                       (parse i l)^"]")
  | Int64Arr (s,l) -> ("__shared__ long spoc_var"^
                       (string_of_int s)^"["^
                       (parse i l)^"]")
  | Float32Arr (s,l) -> ("__shared__ float spoc_var"^
                         (string_of_int s)^"["^
                         (parse i l)^"]")
  | Float64Arr (s,l) -> ("__shared__ double spoc_var"^
                         (string_of_int s)^"["^
                         (parse i l)^"]")
  | Params k -> 
    ("__kernel void spoc_dummy ( "^
     (if (fst !return_v) <> "" then
        (fst !return_v)^", " else "")^(parse i k)^" ) \n{\n")
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
  | Plus  (a,b) -> ((parse_int i a)^" + "^(parse_int i b))
  | Plusf  (a,b) -> ((parse_float i a)^" + "^(parse_float i b))
  | Min  (a,b) -> ((parse_int i a)^" - "^(parse_int i b))
  | Minf  (a,b) -> ((parse_float i a)^" - "^(parse_float i b))
  | Mul  (a,b) -> ((parse_int i a)^" * "^(parse_int i b))
  | Mulf  (a,b) -> ((parse_float i a)^" * "^(parse_float i b))
  | Div  (a,b) -> ((parse_int i a)^" / "^(parse_int i b))
  | Divf  (a,b) -> ((parse_float i a)^" / "^(parse_float i b))

  | Mod  (a,b) -> ((parse_int i a)^" % "^(parse_int i b))
  | Id (s)  -> s
  | Set (var,value) 
  | Acc (var,value) -> 
    ((parse i var)^" = "^
     (parse_int i value))
  | Decl (var) ->
    (parse i var)
  | SetLocalVar (v,gv,k) -> 
    ((parse i v)^" = "^
     (match gv with
      | Intrinsics i -> (parse_intrinsics i)
      | _ -> (parse i gv))^";\n"^(space i)^(parse i k))
  | Return k -> 
    (if (snd !return_v) <> "" then
       snd !return_v
     else
       "")^
    (parse i k)
  | Unit  -> ""
  | IntVecAcc (vec,idx)  -> (parse i vec)^"["^(parse i idx)^"]"
  | SetV (vecacc,value)  -> (
      (parse i vecacc)^" = "^(parse i value)^";") 
  |Int  a  -> string_of_int a
  | Float f -> (string_of_float f)^"f"
  | GInt  a  -> string_of_int (a ())
  | GFloat  a  -> (string_of_float (a ()))^"f"
  | Double f -> string_of_float f
  | IntId (s,_) -> s
  |Intrinsics gv -> parse_intrinsics gv
  | Seq (a,b)  -> (parse i a)^" ;\n"^(space i)^(parse i b)
  | Ife(a,b,c) -> "if ("^(parse i a)^"){\n"^(space (i+2))^(parse (i+2) b)^";}\n"^(space i)^"else{\n"^(space (i+2))^(parse (i+2) c)^";}\n"^(space i)
  | If (a,b) -> "if ("^(parse i a)^")\n"^(space i)^"{\n"^(space (i+2))^(parse (i+2) b)^";\n"^(space i)^"}"^(space i)
  | Or (a,b) -> (parse i a)^" || "^(parse i b)
  | And (a,b) -> (parse i a)^" && "^(parse i b)
  | EqBool (a,b) -> (parse i a)^" == "^(parse i b)
  | LtBool (a,b) -> (parse i a)^" < "^(parse i b)
  | GtBool (a,b) -> (parse i a)^" > "^(parse i b)
  | LtEBool (a,b) -> (parse i a)^" <= "^(parse i b)
  | GtEBool (a,b) -> (parse i a)^" >= "^(parse i b)
  | DoLoop (a,b,c,d) -> 
    let id = parse i a in
    let min = parse i b in
    let max = parse i c in
    let body = parse (i+2) d in
    "for (int "^id^" = "^min^"; "^id^" <= "^max^"; "^id^"++){\n"^(space (i+2))^body^";}"
  | While (a,b) ->
    let cond = parse i a in
    let body = parse (i+2) b in
    "while ("^cond^"){\n"^(space (i+2))^body^";}"
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

and parse_int n = function
  | IntId (s,_) -> s
  | Int i  ->  string_of_int i
  | GInt i  ->  string_of_int (i ())
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
  | Intrinsics gv -> parse_intrinsics gv
  | a  -> print_ast a; failwith "opencl error parse_float"

and parse_vect = function
  | IntVect i  -> i
  | _  -> failwith "error parse_vect"

and parse_intrinsics (cudas,opencls) = opencls
