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
let prof_counter = ref 5
    
let generated_functions = Hashtbl.create 10

let gmem_store ()  =
    "(** ### global_memory stores : "^(Int64.to_string (Spoc.Mem.get !Gen.profile_vect 0))^" **)\n"

let gmem_load () =
  "(** ### global_memory loads : "^(Int64.to_string (Spoc.Mem.get !Gen.profile_vect 1))^" **)\n"

let smem_store ()  =""
let smem_load  ()  =""

let prof_string ()=
  "(**  ### visits : "^(Int64.to_string (Spoc.Mem.get !Gen.profile_vect !prof_counter))^" **)\n"

let prof_time_string()=
  let cycles = Spoc.Mem.get !Gen.profile_vect !prof_counter
  in
  "(** ### total_#_cycles : "^(Int64.to_string cycles)^" **)\n"


let branches_string () =
    let nb_branches = Int64.to_string (Spoc.Mem.get !Gen.profile_vect 4)
    and nb_divergents = Int64.to_string (Spoc.Mem.get !Gen.profile_vect 5)
    in
    "(** ### Total branches : "^nb_branches^" **)\n"^
    "(** ### Divergent branches : "^nb_divergents^" **)\n"

let branch_analysis_string c =  
  let numActive = Int64.to_string (Spoc.Mem.get !Gen.profile_vect (c+1)) 
  and numTaken = Int64.to_string (Spoc.Mem.get !Gen.profile_vect (c+2))
  and numNotTaken = Int64.to_string (Spoc.Mem.get !Gen.profile_vect (c+3))
  in
  "(** ### Active threads : "^numActive^" **)\n"^
  "(** ### Branch taken : "^numTaken^" **)\n" ^
  "(** ### Branch not taken : "^numNotTaken^" **)\n"

let memory_div_string () =
  let memDiv = Int64.to_string (Spoc.Mem.get !Gen.profile_vect 6) in
  "(** ### Global Memory divergence : "^memDiv^" **)\n"

let profile_head () =
  gmem_store ()^
  gmem_load ()^
  smem_store ()^
  smem_load ()^
  branches_string ()
    
let fun_counter = ref 0

let rec parse_fun i a b n dev =
  let open Kirc_Ast in
  let rec aux name a =
    let rec aux2 i a =
      match a with
      | Kern (args,body) ->
        (let pargs = aux2 i args  in
         let pbody =
           let s = prof_string() in
           incr prof_counter;
           s^
           
           match body with
           | Seq (_,_)  -> (aux2 (i) body)
           | _  ->  ((aux2 (i) body )^"\n")

         in
         (pargs ^ " ->\n"^
          indent (i+1) ^ pbody)^indent (i+1)^ "in")
      | Params k -> parse (i+1) k dev
      | a -> parse (i+1) a dev
    in
    aux2 i a

  in
  let name =
    try snd (Hashtbl.find generated_functions a) with
    | Not_found ->
      let gen_name =
        if n <> "" then
          n
        else ("spoc_fun"^string_of_int !fun_counter) in
      incr fun_counter;
      let fun_src = aux gen_name a
      in
      Hashtbl.add generated_functions a (fun_src,gen_name);
      "\n"^indent (i+1) ^"let "^gen_name^" = fun "^fun_src ^"\n"^indent (i+1)^gen_name
  in
  name


 (*Profiling counters info                                                            
    pc[0] <- gmem_store                                                                
    pc[1] <- gmem_load                                                                 
    pc[2] <- smem_store                                                                
    pc[3] <- smem_load                                                                 
    pc[4] <- nb_branches                                                               
    pc[5] <- nb_divergent                                                              
 *)
    


and  parse i a  dev =
  let open Kirc_Ast in
  let rec aux  = function
    | Kern (args, body) -> ("kern "^(parse i args dev)^" ->\n"^
                            profile_head ()^
                            (match dev.Spoc.Devices.specific_info with
                             | Spoc.Devices.CudaInfo _ ->
                               memory_div_string()
                             | _ -> "")
                            ^"\n"^(parse (i) body dev))
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
      Printf.printf "Ife %d\n%!" !prof_counter;
      let a = parse i a dev in
      let b_anal = branch_analysis_string !prof_counter in
      prof_counter := !prof_counter + 4; 
      let b = parse (i+1) b dev in
      let c = parse (i+1) c dev in
      (* let iff = *)
      (* (indent (i+1))^prof_string()^ *)
      (*           (indent (i+1))^b^"\n"^(indent i)^"\n"^(indent i) *)
      (* in *)
      (* prof_counter := !prof_counter + 3; *)
      (* let elsee = "else \n"^ *)
      (* (\* (indent (i+1))^prof_string()^ *\) *)
      (* (\*             (indent (i+1))^c^";\n"^(indent i)^"\n"^(indent i) *\) *)
      (* (\* in *\) *)
      b_anal ^
      "if ("^a^") then \n"^ b ^"\n"^(indent i)^"else\n"^indent (i+1)^c
    | If (a,b) ->
      Printf.printf "If %d\n%!" !prof_counter;
      let a = parse i a dev in
      let b_anal = branch_analysis_string !prof_counter in
      prof_counter := !prof_counter + 4; 
      let s =
        "if ("^a^")"^" then \n"^(indent (i+1))^
        (indent (i+1))^(parse (i+1) b dev)^";\n"^(indent i)^""^(indent i)
      in
      b_anal ^ s
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
      Printf.printf "While %d\n%!" !prof_counter;
      let cond = parse i a dev in
      let b_anal = branch_analysis_string !prof_counter in
      prof_counter := !prof_counter + 4; 
      let body = indent (i+1)^parse (i+1) b dev in
      let s = b_anal^"while " ^ cond ^" do\n"^
              body^"\n"^
              indent i^"done;"
      in
      s
    | Unit -> "()"
    | GlobalFun (a,b,n) ->
      let s = (parse_fun i a b n dev) in

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
      let ss = 
        (match dev.Spoc.Devices.specific_info with
         |  Spoc.Devices.CudaInfo _ ->
           let s = indent (i) ^
                   prof_time_string()^ indent (i) in
           incr prof_counter;
           s
         | _ -> "")
      in

      ss^s
    | _ -> Kirc_Ast.print_ast a; ""
  in aux  a

let parse i a dev =
  prof_counter := 6;
  Hashtbl.clear generated_functions;
  let header =
    "(* Profile Kernel *)"
  in
  let footer = ";;"
  in
  (header^"\n" ^ parse 0  a dev ^"\n"^footer^"\n")
