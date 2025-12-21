(******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2015)
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

let profile_vect = ref (Obj.magic None)

let space i = String.make i ' '

and indent i =
  let rec aux acc = function 0 -> acc | i -> aux (acc ^ "  ") (i - 1) in
  aux "" i

let decl_info = function
  | IntVar (i, s, false) -> Some (i, s, "int")
  | FloatVar (i, s, false) -> Some (i, s, "float")
  | DoubleVar (i, s, false) -> Some (i, s, "double")
  | BoolVar (i, s, false) -> Some (i, s, "int")
  | _ -> None

module type CodeGenerator = sig
  (* framework name *)
  val target_name : string

  (* framework dependent qualifiers *)
  val global_function : string

  val device_function : string

  val host_function : string

  val global_parameter : string

  val global_variable : string

  val local_variable : string

  val shared_variable : string

  (* framework dependent code *)
  val kern_start : string

  val kern_end : string

  val parse_intrinsics : intrinsics -> string

  val default_parser : bool

  val parse_fun :
    int -> Kirc_Ast.k_ext -> string -> string -> Spoc.Devices.device -> string

  val parse : int -> Kirc_Ast.k_ext -> Spoc.Devices.device -> string
end

module Generator (M : CodeGenerator) = struct
  let global_funs = Hashtbl.create 0

  let return_v = ref ("", "")

  let global_fun_idx = ref 0

  let protos = ref []

  (*Profiling counters info
    pc[0] <- gmem_store
    pc[1] <- gmem_load
    pc[2] <- smem_store
    pc[3] <- smem_load
    pc[4] <- nb_branches
    pc[5] <- nb_branch_divergent
    pc[6] <- nb_memory_divergent
  *)

  let rec parse_fun ?profile:(prof = false) i a ret_type fname dev =
    begin if M.default_parser then begin
      let aux name a =
        let rec aux2 i a =
          match a with
          | Kern (args, body) ->
              let pargs = aux2 i args in
              let pbody =
                let a =
                  if prof then
                    indent (i + 1)
                    ^ "spoc_atomic_add(prof_cntrs+"
                    ^ string_of_int !profiler_counter
                    ^ ", 1);\n"
                  else ""
                in
                if prof then (
                  Printf.printf "incr in parse_fun \n%!" ;
                  incr profiler_counter) ;
                let b =
                  match M.target_name with
                  | "Cuda" ->
                      if prof then
                        indent (i + 1)
                        ^ ret_type ^ " spoc_res;\n"
                        ^ indent (i + 1)
                        ^ "clock_t start_time = clock();\n"
                      else ""
                  | _ -> ""
                in
                a ^ b
                ^
                match body with
                | Seq (_, _) -> aux2 i body
                | _ -> aux2 i body ^ "\n" ^ indent i
              in
              let s = (pargs ^ pbody) ^ "}" in
              s
          | Params k ->
              let proto =
                M.device_function ^ " " ^ ret_type ^ " " ^ name ^ "  ( "
                ^ (if prof then
                     M.global_parameter
                     ^ (match M.target_name with
                       | "Cuda" -> "unsigned long long int"
                       | _ -> " unsigned long ")
                     ^ " * prof_cntrs, "
                   else "")
                ^ (if fst !return_v <> "" then fst !return_v ^ ", " else "")
                ^ parse ~profile:prof (i + 1) k dev
                ^ " )"
              in
              protos := proto :: !protos ;
              proto ^ "{\n"
          | VecVar (t, _i, s) ->
              M.global_parameter
              ^ (match t with
                | Int _ -> " int"
                | Float _ -> " float"
                | Double _ -> " double"
                | Custom (n, _, _ss) -> " struct " ^ n ^ "_sarek"
                | _ -> assert false)
              ^ "* " ^ s ^ ", int sarek_" ^ s ^ "_length"
          | a -> parse ~profile:prof i a dev
        in
        aux2 i a
      in

      let name =
        try snd (Hashtbl.find global_funs a)
        with Not_found ->
          incr global_fun_idx ;
          let gen_name =
            if fname <> "" then fname
            else "spoc_fun__" ^ string_of_int !global_fun_idx
          in
          let fun_src = aux gen_name a in
          Hashtbl.add global_funs a (fun_src, gen_name) ;
          gen_name
      in
      name
    end
    else M.parse_fun i a ret_type fname dev
    end

  and profiler_counter = ref 5

  and get_profile_counter () =
    Hashtbl.clear global_funs ;
    let a = !profiler_counter in
    profiler_counter := 6 ;
    a

  and parse ?profile:(prof = false) i a dev =
    if M.default_parser then begin
      match a with
      | Kern (args, body) ->
          let pargs = parse ~profile:prof i args dev in
          let pbody =
            match body with
            | Seq (_, _) -> parse ~profile:prof (i + 1) body dev
            | _ -> parse ~profile:prof (i + 1) body dev ^ "\n" ^ indent i
          in
          (pargs ^ "bool spoc_prof_cond;\n" ^ indent (i + 1) ^ pbody)
          ^ M.kern_end
      | Local (x, y) -> (
          match (x, y) with
          | Decl var, Seq (Set (IntId (s, id), value), body) -> (
              match decl_info var with
              | Some (vid, vname, vtype) when vid = id && vname = s ->
                  let value_s = parse ~profile:prof i value dev in
                  let body_s = parse ~profile:prof i body dev in
                  "const " ^ vtype ^ " " ^ vname ^ " = " ^ value_s ^ ";\n"
                  ^ indent i ^ body_s ^ "\n" ^ indent i
              | _ ->
                  parse ~profile:prof i x dev
                  ^ ";\n" ^ indent i
                  ^ parse ~profile:prof i y dev
                  ^ "\n" ^ indent i ^ "")
          | _ ->
              parse ~profile:prof i x dev
              ^ ";\n" ^ indent i
              ^ parse ~profile:prof i y dev
              ^ "\n" ^ indent i ^ "")
      | VecVar (t, _i, s) ->
          M.global_parameter
          ^ (match t with
            | Int _ -> " int"
            | Float _ -> " float"
            | Double _ -> " double"
            | Custom (n, _, _ss) -> " struct " ^ n ^ "_sarek"
            | _ -> assert false)
          ^ "* " ^ s ^ ", int sarek_" ^ s ^ "_length"
      | Block b ->
          indent i ^ "{\n"
          ^ parse ~profile:prof (i + 1) b dev
          ^ "\n" ^ indent i ^ "}"
      | IdName s -> s
      | IntVar (_i, s, _m) -> "int " ^ s
      | FloatVar (_i, s, _m) -> "float " ^ s
      | Custom (n, _s, ss) -> "struct " ^ n ^ "_sarek " ^ ss
      | UnitVar (_v, _s, _m) -> assert false
      | DoubleVar (_i, s, _m) -> "double " ^ s
      | BoolVar (_i, s, _m) -> "int " ^ s
      | Arr (s, l, t, m) ->
          let memspace =
            match m with
            | LocalSpace -> M.local_variable
            | Shared -> M.shared_variable
            | Global -> M.global_variable
          and elttype =
            match t with
            | EInt32 -> "int"
            | EInt64 -> "long"
            | EFloat32 -> "float"
            | EFloat64 -> "double"
          in
          memspace ^ " " ^ elttype ^ " " ^ s ^ "["
          ^ parse ~profile:prof i l dev
          ^ "]"
      | Params k ->
          let rec parse_param k =
            match k with
            | Concat (a, b) -> (
                match b with
                | Empty -> parse_param a
                | Concat (_c, _d) -> parse_param a ^ ", " ^ parse_param b
                | _ -> failwith "parse concat")
            | VecVar (t, _i, s) ->
                M.global_parameter
                ^ (match t with
                  | Int _ -> " int"
                  | Float _ -> " float"
                  | Double _ -> " double"
                  | Custom (n, _, _ss) -> " struct " ^ n ^ "_sarek"
                  | _ -> assert false)
                ^ "* " ^ s ^ ", int sarek_" ^ s ^ "_length"
            | IntVar (_i, s, m) -> (if m then "" else "const ") ^ "int " ^ s
            | FloatVar (_i, s, m) -> (if m then "" else "const ") ^ "float " ^ s
            | DoubleVar (_i, s, m) ->
                (if m then "" else "const ") ^ "double " ^ s
            | BoolVar (_i, s, m) -> (if m then "" else "const ") ^ "int " ^ s
            | Custom (n, _s, ss) -> "struct " ^ n ^ "_sarek " ^ ss
            | Empty -> ""
            | a -> parse ~profile:prof i a dev
          in
          M.kern_start ^ " void spoc_dummy ( "
          ^ (if prof then
               M.global_parameter
               ^ (match M.target_name with
                 | "Cuda" -> "unsigned long long int"
                 | _ -> " unsigned long ")
               ^ " * prof_cntrs, "
             else "")
          ^ (if fst !return_v <> "" then fst !return_v ^ ", " else "")
          ^ parse_param k ^ " ) {\n"
          ^ indent (i + 1)
      | Concat (a, b) -> (
          match b with
          | Empty -> parse ~profile:prof i a dev
          | Concat (_c, _d) ->
              parse ~profile:prof i a dev ^ ", " ^ parse ~profile:prof i b dev
          | _ -> failwith "parse concat")
      | Constr (t, s, l) ->
          "build_" ^ t ^ "_" ^ s ^ "("
          ^ List.fold_left (fun a b -> a ^ parse ~profile:prof i b dev) "" l
          ^ ")"
      | Record (s, l) ->
          let params =
            match l with
            | t :: q ->
                parse ~profile:prof i t dev
                ^ List.fold_left
                    (fun a b -> a ^ ", " ^ parse ~profile:prof i b dev)
                    ""
                    q
            | [] -> assert false
          in
          "build_" ^ s ^ "(" ^ params ^ ")"
      | RecGet (r, f) -> parse ~profile:prof i r dev ^ "." ^ f
      | RecSet (r, v) ->
          parse ~profile:prof i r dev ^ " = " ^ parse ~profile:prof i v dev
      | Plus (a, b) ->
          "("
          ^ parse_int ~profile:prof i a dev
          ^ " + "
          ^ parse_int ~profile:prof i b dev
          ^ ")"
      | Plusf (a, b) ->
          let a = parse_float ~profile:prof i a dev in
          let b = parse_float ~profile:prof i b dev in
          "(" ^ a ^ " + " ^ b ^ ")"
      | Min (a, b) ->
          "("
          ^ parse_int ~profile:prof i a dev
          ^ " - "
          ^ parse_int ~profile:prof i b dev
          ^ ")"
      | Minf (a, b) ->
          "("
          ^ parse_float ~profile:prof i a dev
          ^ " - "
          ^ parse_float ~profile:prof i b dev
          ^ ")"
      | Mul (a, b) ->
          "("
          ^ parse_int ~profile:prof i a dev
          ^ " * "
          ^ parse_int ~profile:prof i b dev
          ^ ")"
      | Mulf (a, b) ->
          "("
          ^ parse_float ~profile:prof i a dev
          ^ " * "
          ^ parse_float ~profile:prof i b dev
          ^ ")"
      | Div (a, b) ->
          "("
          ^ parse_int ~profile:prof i a dev
          ^ " / "
          ^ parse_int ~profile:prof i b dev
          ^ ")"
      | Divf (a, b) ->
          "("
          ^ parse_float ~profile:prof i a dev
          ^ " / "
          ^ parse_float ~profile:prof i b dev
          ^ ")"
      | Mod (a, b) ->
          "("
          ^ parse_int ~profile:prof i a dev
          ^ " % "
          ^ parse_int ~profile:prof i b dev
          ^ ")"
      | Id s -> s
      | Set (var, value) | Acc (var, value) ->
          parse ~profile:prof i var dev
          ^ " = "
          ^ parse ~profile:prof i value dev
      | Decl var -> parse ~profile:prof i var dev
      | SetLocalVar (v, gv, k) ->
          parse ~profile:prof i v dev
          ^ " = "
          ^ (match gv with
            | Intrinsics i -> M.parse_intrinsics i
            | _ -> parse ~profile:prof i gv dev)
          ^ ";\n" ^ indent i
          ^ parse ~profile:prof i k dev
      | Return k -> (
          match k with
          | SetV _ | RecSet _ | Set _ | SetLocalVar _ | IntVecAcc _ | Acc _
          | If _ ->
              parse ~profile:prof i k dev
          | Unit -> ";"
          | _ ->
              if snd !return_v <> "" then snd !return_v
              else
                let s = parse ~profile:prof i k dev in
                if prof then
                  match M.target_name with
                  | "Cuda" ->
                      let ss =
                        indent (i + 1)
                        ^ "spoc_res = " ^ s ^ ";\n"
                        ^ indent (i + 1)
                        ^ "clock_t stop_time = clock();\n"
                        ^ indent (i + 1)
                        ^ "spoc_atomic_add(prof_cntrs+"
                        ^ string_of_int !profiler_counter
                        ^ ", (int)(stop_time - start_time));\n"
                        ^ indent (i + 1)
                        ^ "return spoc_res;\n"
                      in
                      Printf.printf "incr in Return \n%!" ;
                      incr profiler_counter ;
                      ss
                  | _ -> "return " ^ s ^ ";"
                else "return " ^ s ^ ";")
      | IntVecAcc (vec, idx) ->
          if prof then
            "memory_analysis(prof_cntrs, "
            ^ parse ~profile:prof i vec dev
            ^ "+("
            ^ parse ~profile:prof i idx dev
            ^ "), 0, 1)"
          else
            parse ~profile:prof i vec dev
            ^ "["
            ^ parse ~profile:prof i idx dev
            ^ "]"
      | SetV (vecacc, value) -> (
          match vecacc with
          | IntVecAcc (vec, idx) ->
              (if prof then
                 let a =
                   parse ~profile:prof i vec dev
                   ^ "+("
                   ^ parse ~profile:prof i idx dev
                   ^ ")"
                 in
                 indent i ^ "memory_analysis(prof_cntrs, " ^ a ^ ", 1, 0);\n"
               else "")
              ^
              let a = parse ~profile:false 0 vecacc dev in
              indent i ^ a ^ " = " ^ parse ~profile:prof i value dev ^ ";"
          | _ ->
              let a = parse ~profile:false 0 vecacc dev in
              indent i ^ a ^ " = " ^ parse ~profile:prof i value dev ^ ";")
      | Int a -> string_of_int a
      | Float f -> string_of_float f ^ "f"
      | GInt a -> Int32.to_string (a ())
      | GFloat a -> string_of_float (a ()) ^ "f"
      | GFloat64 a -> string_of_float (a ()) ^ "f"
      | Double f -> string_of_float f
      | IntId (s, _) -> s
      | Intrinsics gv -> M.parse_intrinsics gv
      | Seq (a, b) -> (
          match (a, b) with
          | Decl var, Seq (Set (IntId (s, id), value), body) -> (
              match decl_info var with
              | Some (vid, vname, vtype) when vid = id && vname = s ->
                  let value_s = parse ~profile:prof i value dev in
                  let body_s = parse ~profile:prof i body dev in
                  "const " ^ vtype ^ " " ^ vname ^ " = " ^ value_s ^ ";\n"
                  ^ indent i ^ body_s
              | _ ->
                  let a = parse ~profile:prof i a dev in
                  let b = parse ~profile:prof i b dev in
                  a ^ " ;\n" ^ indent i ^ b)
          | _ ->
              let a = parse ~profile:prof i a dev
              and b = parse ~profile:prof i b dev in
              a ^ " ;\n" ^ indent i ^ b)
      | Ife (a, b, c) ->
          let a =
            let a = parse ~profile:prof i a dev in
            let pc = string_of_int !profiler_counter in
            "spoc_prof_cond = (" ^ a ^ ");\n"
            ^ (if prof then (
                 let s =
                   indent i ^ "branch_analysis(prof_cntrs, spoc_prof_cond, "
                   ^ pc ^ ");\n"
                 in
                 Printf.printf "incr in Ife 3 \n%!" ;
                 profiler_counter := !profiler_counter + 4 ;
                 s)
               else "")
            ^ indent i ^ "if ( spoc_prof_cond )"
          in
          let b =
            let b = parse ~profile:prof i b dev in
            b
          in
          let c =
            let c = parse ~profile:prof i c dev in
            c
          in
          let iff =
            "{\n"
            ^ indent (i + 1)
            (* (if prof then *)
            (*    (indent (i))^"spoc_atomic_add(prof_cntrs+"^string_of_int !profiler_counter^", 1); // control if\n" *)
            (*  else "")^ *)
            ^ b
            ^ ";\n" ^ indent i ^ "}\n"
          in
          (*if prof then (
            incr profiler_counter;);*)
          let elsee =
            "{\n"
            ^ indent (i + 1)
            (* (if prof then *)
            (*    (indent (i))^"spoc_atomic_add(prof_cntrs+"^string_of_int !profiler_counter^", 1); // control else\n" *)
            (*  else "")^ *)
            ^ indent i
            ^ c ^ ";\n" ^ indent i ^ "}\n"
          in
          (*if prof then
            (
              Printf.printf "incr in Ife else \n%!";
              profiler_counter := !profiler_counter + 3;
            );*)
          a ^ iff ^ indent i ^ "else" ^ elsee ^ indent i
      | If (a, b) ->
          let a = parse ~profile:prof i a dev in

          let pc = string_of_int !profiler_counter in
          "spoc_prof_cond  = (" ^ a ^ ");\n"
          ^ (if prof then (
               let s =
                 indent i ^ "branch_analysis(prof_cntrs, spoc_prof_cond, " ^ pc
                 ^ ");\n"
               in
               Printf.printf "incr in If 3 \n%!" ;
               profiler_counter := !profiler_counter + 4 ;
               s)
             else (
               profiler_counter := !profiler_counter + 4 ;
               ""))
          ^
          let b = parse ~profile:prof (i + 1) b dev in
          let s =
            indent i ^ "if ( spoc_prof_cond)" ^ "{\n"
            ^ indent (i + 1)
            ^ b ^ ";\n" ^ indent i ^ "}" ^ indent i
          in
          s
      | Or (a, b) ->
          parse ~profile:prof i a dev ^ " || " ^ parse ~profile:prof i b dev
      | And (a, b) ->
          ("(" ^ parse ~profile:prof i a dev)
          ^ ") && ("
          ^ parse ~profile:prof i b dev
          ^ ")"
      | Not a -> "!(" ^ parse ~profile:prof i a dev ^ ")"
      | EqCustom (n, v1, v2) ->
          let v1 = parse ~profile:prof 0 v1 dev
          and v2 = parse ~profile:prof 0 v2 dev in
          (*"switch "^v1^"."^n^"_starek_tag"^*)
          n ^ "(" ^ v1 ^ ", " ^ v2 ^ ")"
      | EqBool (a, b) ->
          parse ~profile:prof i a dev ^ " == " ^ parse ~profile:prof i b dev
      | LtBool (a, b) ->
          parse ~profile:prof i a dev ^ " < " ^ parse ~profile:prof i b dev
      | GtBool (a, b) ->
          parse ~profile:prof i a dev ^ " > " ^ parse ~profile:prof i b dev
      | LtEBool (a, b) ->
          parse ~profile:prof i a dev ^ " <= " ^ parse ~profile:prof i b dev
      | GtEBool (a, b) ->
          parse ~profile:prof i a dev ^ " >= " ^ parse ~profile:prof i b dev
      | DoLoop (a, b, c, d) ->
          let id = parse ~profile:prof i a dev in
          let min = parse ~profile:prof i b dev in
          let max = parse ~profile:prof i c dev in
          let body = parse ~profile:prof (i + 1) d dev in
          "for (int " ^ id ^ " = " ^ min ^ "; " ^ id ^ " <= " ^ max ^ "; " ^ id
          ^ "++){\n"
          ^ indent (i + 1)
          ^ body ^ ";}"
      | While (a, b) ->
          let cond = parse ~profile:prof i a dev in
          let pc = string_of_int !profiler_counter in
          "spoc_prof_cond = (" ^ cond ^ ");\n"
          ^ (if prof then (
               let s =
                 indent i ^ "branch_analysis(prof_cntrs, spoc_prof_cond, " ^ pc
                 ^ ");\n"
               in
               Printf.printf "incr 3 in While  \n%!" ;
               profiler_counter := !profiler_counter + 4 ;
               s)
             else "")
          ^
          let body =
            indent (i + 1)
            ^
            (* (if prof then *)
            (*    (indent (i+1))^"spoc_atomic_add(prof_cntrs+"^string_of_int !profiler_counter^", 1); // control while \n" *)
            (*  else "")^ *)
            parse ~profile:prof (i + 1) b dev
          in
          let s =
            "while ( spoc_prof_cond ){\n"
            (* (indent (i+1)) *)
            (* ^ *)
            (* (if !gmem_load > 0 then *)
            (*    ( *)
            (*      let s = "spoc_atomic_add(prof_cntrs+1,"^(string_of_int !gmem_load)^"); // global mem load\n" in *)
            (*      gmem_load := 0; *)
            (*      indent(i+1)^s *)
            (*    ) *)
            (*  else "")^ *)
            (* (if !gmem_store > 0 then *)
            (*    ( *)
            (*      let s = "spoc_atomic_add(prof_cntrs+0,"^(string_of_int !gmem_load)^"); // global mem store\n" in *)
            (*      gmem_store := 0; *)
            (*      s *)
            (*    ) *)
            (*  else "")^ *)
            ^ body
            ^ ";\n"
            ^ indent (i + 1)
            ^ "spoc_prof_cond = (" ^ cond ^ ");\n"
            ^ (if prof then
                 let s =
                   indent (i + 1)
                   ^ "while_analysis(prof_cntrs, spoc_prof_cond );\n"
                 in
                 s
               else "")
            ^ "\n}"
          in
          (*   ( *)
          (*     Printf.printf "incr in While \n%!"; *)
          (*     incr profiler_counter; *)
          (*   ); *)
          s
      | App (a, b) -> (
          let f = parse ~profile:prof i a dev in
          let rec aux = function
            | t :: [] -> parse ~profile:prof i t dev
            | t :: q -> parse ~profile:prof i t dev ^ ", " ^ aux q
            | [] -> assert false
          in
          match a with
          | Intrinsics ("return", "return") ->
              f ^ " " ^ aux (Array.to_list b) ^ " "
          | Intrinsics (_, _) -> f ^ " (" ^ aux (Array.to_list b) ^ ") "
          | _ ->
              f ^ " ("
              ^ ((if prof then " prof_cntrs, " else "") ^ aux (Array.to_list b))
              ^ ") ")
      | Empty -> ""
      | GlobalFun (a, b, n) ->
          let s = parse_fun ~profile:prof i a b n dev in
          s
      | Match (s, e, l) ->
          let match_e = parse ~profile:prof 0 e dev in
          let switch_content =
            Array.fold_left
              (fun a (j, of_i, b) ->
                let manage_of =
                  match of_i with
                  | None -> ""
                  | Some (typ, cstr, _varn, id) ->
                      indent (i + 1)
                      ^ typ ^ " " ^ id ^ " = " ^ match_e ^ "." ^ s
                      ^ "_sarek_union." ^ s ^ "_sarek_" ^ cstr ^ "." ^ s
                      ^ "_sarek_" ^ cstr ^ "_t" ^ ";\n"
                in
                a ^ "\n\tcase " ^ string_of_int j ^ ":{\n" ^ manage_of
                ^ indent (i + 1)
                ^ parse ~profile:prof (i + 1) b dev
                ^ ";\n"
                ^ indent (i + 1)
                ^ "break;}")
              " "
              l
          in
          "switch (" ^ match_e ^ "." ^ s ^ "_sarek_tag" ^ "){" ^ switch_content
          ^ "}"
      | Native f -> f dev
      | Pragma (opts, k) ->
          "\n" ^ indent i
          ^ List.fold_left (fun a b -> a ^ " " ^ b) "#pragma " opts
          ^ "\n" ^ indent i ^ parse i k dev
      | Unit -> ""
      | Map (f, v1, v2) ->
          indent i ^ "for (int sarek_idx = 0; sarek_idx < SAREK_VEC_LENGTH("
          ^ parse 0 v1 dev ^ "); sarek_idx++){\n"
          ^ indent (i + 1)
          ^ parse (i + 1) v2 dev
          ^ "[sarek_idx] = " ^ parse 0 f dev ^ "( " ^ parse 0 v1 dev
          ^ "[sarek_idx] );}"
      | _ -> assert false
    end
    else M.parse i a dev

  and parse_int ?profile:(prof = false) n a dev =
    match a with
    | IntId (s, _) -> s
    | Int i -> string_of_int i
    | GInt i -> Int32.to_string (i ())
    | IntVecAcc (s, i) ->
        if prof then
          "memory_analysis(prof_cntrs, "
          ^ parse ~profile:prof n s dev
          ^ "+(" ^ parse_int n i dev ^ "), 0, 1)"
        else parse ~profile:prof n s dev ^ "[" ^ parse_int n i dev ^ "]"
    | Plus (_a, _b) as v -> parse ~profile:prof n v dev
    | Min (_a, _b) as v -> parse ~profile:prof n v dev
    | Mul (_a, _b) as v -> parse ~profile:prof n v dev
    | Mod (_a, _b) as v -> parse ~profile:prof n v dev
    | Div (_a, _b) as v -> parse ~profile:prof n v dev
    | App (_a, _b) as v -> parse ~profile:prof n v dev
    | RecGet (_r, _f) as v -> parse ~profile:prof n v dev
    | a -> parse_float ~profile:prof n a dev
  (*  | _  -> assert false; failwith "error parse_int" *)

  and parse_float ?profile:(prof = false) n a dev =
    match a with
    | IntId (s, _) -> s
    | Float f -> string_of_float f ^ "f"
    | GFloat f -> string_of_float (f ()) ^ "f"
    | Double f -> "(double) " ^ string_of_float f
    | IntVecAcc (s, i) ->
        (if prof then "(float) memory_analysis(prof_cntrs, (void*)" else "")
        ^ parse ~profile:prof n s dev
        ^ "[" ^ parse_int n i dev ^ "]"
        ^ if prof then ", 0, 1)" else ""
    | Plusf (_a, _b) as v -> parse ~profile:prof n v dev
    | Minf (_a, _b) as v -> parse ~profile:prof n v dev
    | Mulf (_a, _b) as v -> parse ~profile:prof n v dev
    | Divf (_a, _b) as v -> parse ~profile:prof n v dev
    | App (_a, _b) as v -> parse ~profile:prof n v dev
    | SetV (_a, _b) as v -> parse ~profile:prof n v dev
    | Intrinsics gv -> M.parse_intrinsics gv
    | RecGet (_r, _f) as v -> parse ~profile:prof n v dev
    | Native f -> f dev
    | a ->
        print_ast a ;
        failwith (M.target_name ^ " error parse_float")

  and parse_vect = function
    | IntVect i -> i
    | _ -> failwith (M.target_name ^ " error parse_vect")
end
