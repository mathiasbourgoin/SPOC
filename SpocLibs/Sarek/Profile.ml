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
let target_name = "OpenCL"

let global_function = ""
let device_function = ""
let host_function = ""

let global_parameter = ""

let global_variable = ""
let local_variable = ""
let shared_variable = ""

let kern_start  = ""
let kern_end = ""


let parse_intrinsics (cuda,opencl) =
  match opencl with
  | "(get_local_id (0))" -> "thread_idx_x"
  | "(get_local_id (1))" -> "thread_idx_y"
  | "(get_local_id (2))" -> "thread_idx_z"
  | "(get_group_id (0))" -> "block_idx_x"
  | "(get_group_id (1))" -> "block_idx_y"
  | "(get_group_id (2))" -> "block_idx_z"
  | "(get_local_size (0))" -> "block_dim_x"
  | "(get_local_size (1))" -> "block_dim_y"
  | "(get_local_size (2))" -> "block_dim_z"
  | "(get_num_groups (0))" -> "grid_dim_x"
  | "(get_num_groups (1))" -> "grid_dim_y"
  | "(get_num_groups (2))" -> "grid_dim_z"
  | "(get_global_id (0))" -> "global_thread_id"

  | "(float)" -> "float"

  | "return" -> "return"

  | a -> "(* could not translate back to Sarek intrinsic using OpenCL instead*)"^a


let space i =
  String.make i ' '

and  indent i =
  let rec aux acc = function
    | 0 -> acc
    | i -> aux (acc^"  ") (i-1)
  in aux "" i

let default_parser = false
let prof_counter = ref 2
let genereated_functions = Hashtbl.create 10

let gmem_load() =
  "(** ### global_memory loads : "^(Int64.to_string (Spoc.Mem.get !Gen.profile_vect 1))^" **)\n"

let gmem_store() =
    "(** ### global_memory stores : "^(Int64.to_string (Spoc.Mem.get !Gen.profile_vect 0))^" **)\n"

let prof_string()=
  "(**  ### visits : "^(Int64.to_string (Spoc.Mem.get !Gen.profile_vect !prof_counter))^" **)\n"
let prof_time_string()=
  let cycles = Spoc.Mem.get !Gen.profile_vect !prof_counter
  in
  "(** ### total_#_cycles : "^(Int64.to_string cycles)^" **)\n"

let fun_counter = ref 0

let rec parse_fun i a b dev =
  let open Kirc_Ast in
  let rec aux name a =
    let rec aux2 i a =
      match a with
      | Kern (args,body) ->
        (let pargs = aux2 i args  in
         let pbody =
           let a =
           prof_string()^
           (indent (i+2))
           in
           incr prof_counter;
           a^
           match body with
           | Seq (_,_)  -> (aux2 (i+1) body)
           | _  ->  ((aux2 (i+1) body )^"\n")

         in
         (pargs ^ " ->\n"^
          gmem_store()^gmem_load()^"\n" ^indent (i+1) ^ pbody)^indent (i+1)^ "in")
      | Params k -> parse (i+1) k dev
      | a -> parse (i+1) a dev
    in
    aux2 i a

  in
  let name =
    try snd (Hashtbl.find genereated_functions a) with
    | Not_found ->
      incr fun_counter;
      let gen_name =  ("spoc_fun"^string_of_int !fun_counter) in
      let fun_src = aux gen_name a
      in
      Hashtbl.add genereated_functions a (fun_src,gen_name);
      "\n"^indent (i+1) ^"let "^gen_name^" = fun "^fun_src ^"\n" ^indent (i+1)^gen_name
  in
  name





and  parse i a  dev =
  let open Kirc_Ast in
  let rec aux  = function
    | Kern (args, body) -> ("kern "^(parse i args dev)^" ->\n"^
                            gmem_store()^gmem_load()^"\n"^(parse (i) body dev))
    | Local (x,y) -> (*"let mutable " ^ (parse i x)^" in\n"^
                       (indent (i))^(parse i y)^
                          "\n"^(indent (i))*)
      parse i y dev
    | VecVar (t, i, s) ->
       global_parameter^s
    | Params k -> parse i k dev
    | Concat (a,b) ->
      (match b with
      | Empty -> parse i a dev
      | Concat (c,d) -> parse i a dev ^ " " ^parse i b dev
      | _ -> assert false )
    | Seq (a,b) ->
      let a = parse i a dev in
      let b = parse i b dev in
      a^" \n"^(indent i)^b
    | SetV(vecacc, value) -> parse i vecacc dev ^" <- " ^(parse i value dev)^";"
    | IntId(s,_) -> s
    | IntVecAcc (s,v) -> ((parse i s dev)^".[<"^(parse i v dev)^">]")
    | Plus  (a,b) ->
      let a = parse i a dev in
      let b = parse i b dev in
      ("("^ a^" + "^b^")")
    | Plusf  (a,b) ->
      let a = parse i a dev in
      let b = parse i b dev in
      ("("^ a^" +. "^b^")")
    | Min  (a,b) -> ("("^(parse i a dev)^" - "^(parse i b dev)^")")
    | Minf  (a,b) -> ("("^(parse i a dev)^" -. "^(parse i b dev)^")")
    | Mul  (a,b) -> ("("^(parse i a dev)^" * "^(parse i b dev)^")")
    | Mulf  (a,b) -> ("("^(parse i a dev)^" *. "^(parse i b dev)^")")
    | Div  (a,b) -> ("("^(parse i a dev)^" / "^(parse i b dev)^")")
    | Divf  (a,b) -> ("("^(parse i a dev)^" /. "^(parse i b dev)^")")
    | Int i -> (string_of_int i)
    | GInt i  ->  Int32.to_string  (i ())
    | Set (var,value) -> (indent i )^"let mutable "^(parse i var dev)^" = " ^(parse i value dev)^" in"
    | Acc (var, value) -> (parse i var dev)^" := " ^(parse i value dev)^";"
    | Float f -> (string_of_float f)^"f"
    | GFloat  a  -> (string_of_float (a ()))^"f"
    | Double f -> string_of_float f

    | Ife(a,b,c) ->
      let a = parse i a dev in
      let b = parse (i+1) b dev in
      let c = parse (i+1) c dev in
      let iff =
      (indent (i+1))^prof_string()^
                (indent (i+1))^b^"\n"^(indent i)^"\n"^(indent i)
      in
      incr prof_counter;
      let elsee = "else \n"^
      (indent (i+1))^prof_string()^
                  (indent (i+1))^c^";\n"^(indent i)^"\n"^(indent i)
      in
      incr prof_counter;
      "if ("^a^") then \n"^ iff ^elsee

    | If (a,b) ->
      let s =
        "if ("^(parse i a dev )^")"^" then \n"^(indent (i+1))^prof_string()^
        (indent (i+1))^(parse (i+1) b dev)^";\n"^(indent i)^""^(indent i)
      in
      incr prof_counter;
      s
    | Or (a,b) -> (parse i a dev)^" || "^(parse i b dev)
    | And (a,b) -> (parse i a dev)^" && "^(parse i b dev)
    | Not (a) -> "!"^(parse i a dev)
    | EqBool (a,b) -> (parse i a dev)^" = "^(parse i b dev)
    | LtBool (a,b) -> (parse i a dev)^" < "^(parse i b dev)
    | GtBool (a,b) -> (parse i a dev)^" > "^(parse i b dev)
    | LtEBool (a,b) -> (parse i a dev)^" <= "^(parse i b dev)
    | GtEBool (a,b) -> (parse i a dev)^" >= "^(parse i b dev)
    | DoLoop (a,b,c,d) ->
      let id = parse i a dev in
      let min = parse i b dev in
      let max = parse i c dev in
      let body = parse (i+1) d dev in
      "for (int "^id^" = "^min^" to  "^max^"do\n"^(indent (i+1))^body^"done;"

    | While (a,b) ->
      let cond = parse i a dev in
      let body = prof_string()^indent (i+1)^parse (i+1) b dev in
      let s = "while " ^ cond ^" do\n"^
              indent (i+1)^body^
              indent i^"done;"
      in
      incr prof_counter;
      s
    | Unit -> "()"
    | GlobalFun (a,b) ->
      let s = (parse_fun i a b dev) in

      s
    | Intrinsics gv -> parse_intrinsics gv
    | FloatVar (_,s) | IntVar (_,s) -> s
    | Decl (var) ->
      (parse i var dev)
    | App (a,b) ->
      let f = parse i a dev in
      let rec aux = function
        | t::[] -> parse i t dev
        | t::q -> (parse i t dev)^" "^(aux q)
        | [] -> assert false
      in
      (match a with
       | Intrinsics ("return","return") -> f^" "^(aux (Array.to_list b))^" "
       |  _ -> f^" "^(aux (Array.to_list b))^" ")
    | Return k ->
      let s  =
        parse i k dev
      in

      s^
      (match dev.Spoc.Devices.specific_info with
       |  Spoc.Devices.CudaInfo _ ->
            let s =
              prof_time_string() in
            incr prof_counter; s
       | _ -> "")
    | _ -> Kirc_Ast.print_ast a; ""
  in aux  a

let parse i a dev =
  prof_counter := 2;
  Hashtbl.clear genereated_functions;
  let header =
    "(* Profile Kernel *)"
  in
  let footer = ";;"
  in
  (header^"\n" ^ parse 0  a dev ^"\n"^footer^"\n")
