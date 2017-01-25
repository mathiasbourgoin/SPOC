open Camlp4.PreCast
open Syntax
open Ast

open Sarek_types
open Typer
open Mparser
open Debug

let get_ff_type p =
    match p with
    | PaId (_loc, IdLid (_loc2, x) ) ->
      (let var = (Hashtbl.find !current_args x) in
       match var.var_type with
       | TInt32  ->
         <:str_item<let $lid:("task_"^x)$ = field task  $str:"task_"^x$ int>>
       | TInt64 ->
         <:str_item<let $lid:("task_"^x)$ = field task  $str:"task_"^x$ int>>
       | TVec k ->
         (match k with
          | TFloat32 ->
            <:str_item<let $lid:("task_"^x)$ = field task  $str:"task_"^x$ (ptr float)>>
          | TInt32 ->
            <:str_item<let $lid:("task_"^x)$ = field task  $str:"task_"^x$ (ptr int)>>
              
          | _ ->  failwith ("gfft : unimplemented yet " ^ ktyp_to_string k)
         )
       | _  -> failwith "error get_ff_type")

  let f p =
    match p with
    | PaId (_loc, IdLid (_loc2, x) ) -> x
    | _ -> assert false


  let rec get_ff_type_str p =
    match p with
    | PaId (_loc, IdLid (_loc2, x) ) ->
      (let var = (Hashtbl.find !current_args x) in
       Printf.sprintf "let task_%s = field task \"task_%s\" %s" x x
         (
           match var.var_type with
           | TFloat32 -> "float"
           | TInt32  ->
             "int"
           | TInt64 ->
             "int"
           | TVec k ->
             (match k with
              | TFloat32 ->
                "(ptr float)"
              | TInt32 -> "(ptr int)"
              | _ ->  failwith "gfft : unimplemented yet"
             )
           | _  -> failwith ("error get_ff_type_str -> "^ (ktyp_to_string var.var_type))))
    | PaTyc (_, x, _ ) -> get_ff_type_str x


  let rec string_of_patt = function
    | PaId(_,x) ->string_of_ident x
    | PaTyc(_,x,_) -> string_of_patt x
    | _ -> assert false


let print_task args nameid _loc =
  let moduleName = (String.capitalize (string_of_ident nameid)) in
  let res = <:expr<
                object (self)
                method offloadTask = $lid:moduleName$.offloadTask
                method noMoreTasks = $lid:moduleName$.accNomoretasks
                method getResult = $lid:moduleName$.getResult
                end
  >> in
                 Printf.printf "open Ctypes\n";
                 Printf.printf "\tmodule %s = struct\n" moduleName;

  Printf.printf "\ttype task\n\t
  let task : task structure typ = structure \"TASK\";;
  %s
  let () = seal task\n"
    (List.fold_left
       (fun a b -> Printf.sprintf "%s;;\n\t%s" a b)
       (get_ff_type_str (List.hd args))
       ((List.map get_ff_type_str (List.tl args))));

  let param_as_tuple_str = (List.fold_left
                              (fun a b -> Printf.sprintf "%s, %s" a b)
                              (string_of_patt (List.hd args))
                              (List.tl (List.map string_of_patt args)))
  in
  Printf.printf "\tlet create_task (%s) =\n"
    param_as_tuple_str;

  Printf.printf "\t\tlet t = allocate_n task 1 in\n%s\t\tt"
    (List.fold_left
       (fun a b -> Printf.sprintf "%s\t\tCtypes.setf !@t task_%s %s;\n" a (string_of_patt b) (string_of_patt b))
       ""
       (args));

  (* TODO : replace fun with functor *)
  Printf.printf"
  let accGetResult () =   let fflib = FastFlow.fflib () in
    Foreign.foreign  ~check_errno:true ~release_runtime_lock:true ~from:fflib \"loadresacc\" (ptr void @-> (ptr (ptr void)) @-> returning void)
  let accOffload () =   let fflib = FastFlow.fflib () in
    Foreign.foreign ~check_errno:true ~release_runtime_lock:true ~from:fflib \"offloadacc\" (ptr void @-> (ptr task) @-> returning void)
  let accNomoretasks () =   let fflib = FastFlow.fflib () in
    Foreign.foreign  ~from:fflib \"nomoretasks\" (ptr void @-> returning void)";

  let rec adapt_string_of_patt = function
    | PaId (_loc, IdLid (_loc2, x) ) ->
      (let var = (Hashtbl.find !current_args x) in
       (
         match var.var_type with
         | TInt32
         | TInt64
         | TFloat32
         | TFloat64  -> x
         | TVec k ->
           let rec t  = function
             | TInt32 -> "(int, Bigarray.int32_elt)"
             | TInt64 -> "(int, Bigarray.int64_elt)"
             | TFloat32 -> "(float, Bigarray.float32_elt)"
             | TFloat64  -> "(float, Bigarray.float64_elt)"
             | TVec k ->
               ((t k)^" Spoc.Vector.vector")
           in
           Printf.sprintf "(Ctypes.bigarray_start array1
                           (Spoc.Vector.to_bigarray_shr (%s : %s)))"
             x (t var.var_type)
         | _  -> failwith "error get_ff_type_str"))
    | PaTyc (_,x,_ ) -> adapt_string_of_patt x
    | _ -> assert false
  in
  let adapted_params_as_tuple =
    (List.fold_left (fun a b -> Printf.sprintf "%s, %s" a b)
       (adapt_string_of_patt (List.hd args))
       (List.tl (List.map adapt_string_of_patt args)))
  in
  Printf.printf "
  let offloadTask acc (%s) =
    let t = create_task (%s) in
    (accOffload ()) acc t"
    param_as_tuple_str
    adapted_params_as_tuple;

  Printf.printf "
  let getResult acc (%s) =
  let t = create_task (%s) in
  let t_ptr = allocate (ptr void) (to_voidp t) in
  (accGetResult ()) acc t_ptr
" param_as_tuple_str adapted_params_as_tuple;
  Printf.printf "
end\n\n";
  res
