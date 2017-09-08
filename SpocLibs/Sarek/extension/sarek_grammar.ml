open Camlp4.PreCast
open Syntax
open Ast

open Sarek_types
open Typer
open Mparser
open Debug

let ano_n = ref 0
let gen_ano_f _loc : ident=
  let s =
    Printf.sprintf "f_anonym_%d" !ano_n in
  incr ano_n;
  IdLid(_loc, s)


let gen_kernel () = ()
    EXTEND Gram
    GLOBAL: str_item expr;
  str_item:
    [
      ["kmodule"; name=UIDENT; "="; "struct"; "begin_const"; mod_consts=LIST0 mod_const_expr; "end_const";
       "begin_fun"; mod_funs = LIST0 mod_funs_expr; "end_fun";"end" ->
       let new_module =
         {
           mod_name = name;
           mod_constants = mod_consts;
           mod_functions = mod_funs;
           mod_modules = Hashtbl.create 0
         } in
       Hashtbl.add modules new_module.mod_name new_module;
       <:str_item<
       module $name$ = struct
       end
       >>
     | "ktype"; name=LIDENT; "="; k = ktype_kind ->
       Hashtbl.add custom_types name k;
       gen_ctypes _loc k name;
      ]
    ];

  ktype_kind :
    [[ "{"; (t,l,m) = klabel_declaration_list; "}" ->
        KRecord (t,l,m)
     | t = fst_constructor -> KSum [t]
     | t = fst_constructor; t2 = kconstructor_list ->
	KSum
	 (gen_constructors _loc t (Some t2));
     ]
    ];
  klabel_declaration_list:
    [ [ t1 = klabel_declaration; ";"; t2 = SELF ->
        gen_labels _loc t1 (Some t2);
        | t1 = klabel_declaration; ";" ->
        gen_labels _loc t1 None;
        | t1 = klabel_declaration ->
        gen_labels _loc t1 None;
      ] ]
  ;
  klabel_declaration:
    [[
      m = OPT "mutable"; s=ident; ":";  t=type_kind ->
      match m with
        None -> (s,t,false)
      |_ -> (s,t,true)
    ]];

  kconstructor_list :
    [[
	"|"; t1 = kconstructor; t2 = SELF ->
	 gen_constructors _loc t1 (Some t2);
      | "|";  t1 = kconstructor  ->
	 gen_constructors _loc t1 (None);
    ]];
  fst_constructor :
    [[
	OPT "|"; t = kconstructor ->  t

    ]];
  kconstructor:
    [[
	c = UIDENT -> (c,None);
      | c = UIDENT; "of" ; t = ctyp (* TODO: real types here! *) -> (c, Some t) ;
    ]];

  mod_const_expr:
    [[name=ident; ":"; typ=ident; "="; cu_value=STRING; cl_value=STRING ->
      let typ =
        match string_of_ident typ with
        | "int32" -> TInt32;
        | "float32" -> TFloat32;
        | "float64" -> TFloat64;
        | _ -> failwith "unimplemented yet"
      in
      (typ, string_of_ident name, cu_value, cl_value)
     ]]
  ;
  mod_funs_expr:
    [[name=ident; ":"; typ=ident; "="; cu_value=STRING; cl_value=STRING ->
      let typ =
        match string_of_ident typ with
        | "int32" -> TInt32;
        | "float32" -> TFloat32;
        | "float64" -> TFloat64;
        | _ -> failwith "unimplemented yet"
      in
      (typ, string_of_ident name, 0, cu_value, cl_value)
     ]]
  ;
  expr:
    [
      ["kern"; args = LIST1 k_patt; "->"; body = kernel_body ->
       (*Printf.printf
         "(*Generated from the sarek syntax extension, \ndo not modify this file*)\n";*)
       (*arg_idx := 0;*)
       return_type := TUnknown;
       arg_list := [];
       extensions := [ ex32 ];
       Hashtbl.clear !current_args;
       Hashtbl.clear !local_fun;
       List.iter new_arg_of_patt args;
       (try
          retype := true;
          while !retype do
            retype := false;
            unknown := 0;
	          Hashtbl.clear !local_fun;
            typer body TUnknown;
            my_eprintf (Printf.sprintf "Unknown : %d \n\n\n%!" !unknown)
          done;
          if !unknown > 0 then
            (
              Hashtbl.iter (fun a b -> if is_unknown b.var_type  then
			       Printf.eprintf "Unknown value type : %s\n" a)  !current_args; 
              failwith "unknown types in this kernel";
              
            )
        with
        | TypeError(expected, given, loc) ->
          (
            Printf.eprintf "%s\n%!" ("\027[31m Type Error \027[00m : expecting : \027[33m"^
                                     (ktyp_to_string expected)^"\027[00m but given : \027[33m"^
                                     (ktyp_to_string given)^"\027[00m in position : "^(Loc.to_string loc)^"");
            exit 1;
          )
        | Unbound_value (value, loc) ->
          (Printf.eprintf "%s\n%!" ("\027[31m Unbound Value \027[00m : \027[33m"^
                                    (value)^"\027[00m in position : "^(Loc.to_string loc)^"");
           exit 2;)
        | Immutable (value, loc) ->
          (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^
                                    (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^"");
           exit 2;)
        | FieldError (record,field,loc) ->
          (Printf.eprintf "%s\n%!" ("\027[31m Type Error \027[00m field : \027[33m"^
                                    (field)^"\027[00m doesn't exist in record type : \027[33m"^record^
                                    "\027[00m in position : "^(Loc.to_string loc)^"");
           exit 2;)

       );
       let new_hash_args = Hashtbl.create (Hashtbl.length !current_args) in
       Hashtbl.iter (Hashtbl.add new_hash_args) !current_args;
       Hashtbl.clear !current_args;
       current_args := new_hash_args;
       let gen_body =
         <:expr<
                 $try Gen_caml.parse_body body with
                 | TypeError(expected, given, loc) ->
                 (
                 Printf.eprintf "%s\n%!" ("\027[31m Type Error \027[00m : expecting : \027[33m"^
                 (ktyp_to_string expected)^"\027[00m but given : \027[33m"^
                 (ktyp_to_string given)^"\027[00m in position : "^(Loc.to_string loc)^"");
                 exit 1;
                 )
                 | Unbound_value (value, loc) ->
                 (Printf.eprintf "%s\n%!" ("\027[31m Unbound Value \027[00m : \027[33m"^
                 (value)^"\027[00m in position : "^(Loc.to_string loc)^"");
                 exit 2;)
                 | Immutable (value, loc) ->
                 (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^
                 (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^"");
                 exit 2;)
                 $>>
       in
       let b_body =
         (try Gen_kir.parse_body2 body true
          with
          | TypeError(expected, given, loc) ->
            (
              Printf.eprintf "%s\n%!" ("\027[31m Type Error \027[00m : expecting : \027[33m"^
                                       (ktyp_to_string expected)^"\027[00m but given : \027[33m"^
                                       (ktyp_to_string given)^"\027[00m in position : "^(Loc.to_string loc)^"");
              exit 1;
            )
          | Unbound_value (value, loc) ->
            (Printf.eprintf "%s\n%!" ("\027[31m Unbound Value \027[00m : \027[33m"^
                                      (value)^"\027[00m in position : "^(Loc.to_string loc)^"");
             exit 2;)
          | Immutable (value, loc) ->
            (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^
                                      (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^"");
             exit 2;))
       in
       let n_body2 = <:expr<params $List.fold_left
                            (fun a b -> <:expr<concat $b$ $a$>>)

<:expr<empty_arg()>>
  ((List.rev_map gen_arg_from_patt2 args))$>> in

let gen_body2 =  <:expr<
                         spoc_gen_kernel
                         $n_body2$
                         $List.fold_left (fun a b ->
                         <:expr<spoc_local_env $b$ $a$>>)
b_body
  !arg_list$>>

in
let gen_args = parse_args args gen_body
in
let ret =
  incr arg_idx;
  match !return_type with
  | TUnknown  -> <:expr<return_unknown (), Dummy>>
  | TInt32 -> <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int32 >>
  | TInt64 ->  <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int64>>
  | TVec TInt32 | TVec TInt64 -> assert false
  | TFloat32 ->  <:expr<return_float $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.float32>>
  | TFloat64 ->  <:expr<return_double $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "" , Vector.float64>>
  | TUnit  -> <:expr<return_unit (), Vector.Unit ((),())>>
  | TBool -> <:expr< return_bool $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int32>>
  | Custom (_, name) ->
    let sarek_name = name^"_sarek" in
    <:expr< Kirc.return_custom1 $str:name$ $str:sarek_name$ "">>

  | t  -> failwith (Printf.sprintf "error ret : %s" (ktyp_to_string t))
in
let fst_ (a,b,c,d,e,f) = a
and snd_ (a,b,c,d,e,f) = b
and thd_ (a,b,c,d,e,f) = c
and fth_ (a,b,c,d,e,f) = d
and ffh_ (a,b,c,d,e,f) = e in

let tup_args, list_args, class_legacy , list_to_args1, list_to_args2=
  let args_fst_list =
    (List.map fst_ (List.map gen_arg_from_patt3 args)) in
  let args_com = paCom_of_list  args_fst_list in


  let args_list =
    let l = (List.map snd_ (List.map gen_arg_from_patt3 args)) in
    exSem_of_list l
  in

  let args_thd_list =
    (List.map thd_ (List.map gen_arg_from_patt3 args)) in
  let args_typ = tySta_of_list args_thd_list in

  let lta1 =
    paSem_of_list
      (List.map fth_ (List.map gen_arg_from_patt3 args)) in

  let args_ffth_list =
    (new_kernel := true;
     List.map ffh_ (List.map gen_arg_from_patt3 args)) in
  let lta2 =
    exCom_of_list args_ffth_list in

  begin
    if List.length args_fst_list = 1 then
      List.hd args_fst_list
    else
      PaTup (_loc, args_com)
  end,

  begin
    ExArr (_loc, args_list)
  end,

  begin
    if List.length args_thd_list = 1 then
      <:ctyp< $List.hd args_thd_list$,(('a,'b) Kernel.kernelArgs) array>>
    else
      <:ctyp< $TyTup (_loc, args_typ)$,(('a,'b) Kernel.kernelArgs) array>>
  end,

  begin
    PaArr (_loc, lta1)
  end,

  begin
    if List.length args_ffth_list = 1 then
      List.hd args_ffth_list
    else
      ExTup (_loc, lta2)
  end
in
let class_name = "kirc_class"^(string_of_int !nb_ker) in
incr nb_ker;
let has_vector = ref false in
List.iter (fun a-> if patt_is_vector a then has_vector := true) args;
let extensions =  match !extensions with
  | [] -> <:expr< [||]>>
  | t::[] -> <:expr< [|$t$|]>>
  | _ -> <:expr<[|$exSem_of_list  !extensions$|]>>
in
let res =
  if !has_vector then
  <:expr< let module M =
          struct
          let exec_fun $tup_args$ = Spoc.Kernel.exec $list_args$;;
          class ['a, 'b] $lid:class_name$ =
          object (self)
          inherit
          [$class_legacy$ ]
          Spoc.Kernel.spoc_kernel "kirc_kernel" "spoc_dummy"
          method exec = exec_fun
          method args_to_list = fun
          $tup_args$ -> $list_args$
          method list_to_args = function
          | $list_to_args1$ -> $list_to_args2$
          | _ -> failwith "spoc_kernel_extension error"
          end
          end
          in
          let open Kirc in
          (new M.$lid:class_name$, {
          ml_kern = $gen_args$;
          body = $gen_body2$;
          ret_val = $ret$;
          extensions = $extensions$;

          }
          )
          >>
else
  <:expr< let module M =
          struct
          let exec_fun $tup_args$ = Spoc.Kernel.exec $list_args$;;
          class ['a, 'b] $lid:class_name$ =
          object (self)
          inherit
          [$class_legacy$ ]
          Spoc.Kernel.spoc_kernel "kirc_kernel" "spoc_dummy"
          method exec = assert false
          method args_to_list = assert false
          method list_to_args = assert false
          end
          end
          in
          let open Kirc in
          (new M.$lid:class_name$, {
          ml_kern = $gen_args$;
          body = $gen_body2$;
          ret_val = $ret$;
          extensions = $extensions$;
          })>>
in
let local =
  Hashtbl.fold (fun key (funv,stri,_) init ->
		<:str_item<
		$stri$ $init$>>) !local_fun
  <:str_item<>>
in
<:expr<
 let module Local_funs = struct
 $local$
end
in let open Local_funs in
$res$
>>
]
];
str_item:
  [
    ["klet"; name = ident; "="; "fun"; args = LIST1 k_patt;
         "->"; body = kernel_body ->
     arg_idx := 0;
     return_type := TUnknown;
     arg_list := [];
     Hashtbl.clear !current_args;
     Hashtbl.clear !local_fun;
     List.iter new_arg_of_patt args;
     my_eprintf ("->->-> global_fun "^(string_of_ident name));
     let cpt = ref 0 in
     retype := true;
     (try
         while !retype  && !cpt < 3 do
           if debug  then
	     incr cpt;
	   retype := false;
           unknown := 0;
           typer body (TUnknown);
           my_eprintf (Printf.sprintf "\nUnknown : %d \n\n\n%!" !unknown)
         done;
	 with
	 | TypeError(expected, given, loc) ->
           (
             failwith ("Type Error : expecting : "^
		       (ktyp_to_string expected)^" but given : "^
		       (ktyp_to_string given)^" in position : "^(Loc.to_string loc)))
         | Immutable (value, loc) ->
           (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^
                                     (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^"");
            exit 2;));

     (* (try  *)
     (*    typer body TUnknown *)
     (*  with *)
     (*  | TypeError(expected, given, loc) ->  *)
     (*    ( *)
     (*      failwith ("Type Error : expecting : "^ *)
     (*                (ktyp_to_string expected)^" but given : "^ *)
     (*                (ktyp_to_string given)^" in position : "^(Loc.to_string loc))) *)
     (*    | Immutable (value, loc) -> *)
     (*      (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^ *)
     (*                                (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^""); *)
     (*       exit 2;));   *)

     let new_hash_args = Hashtbl.create (Hashtbl.length !current_args) in
     Hashtbl.iter (Hashtbl.add new_hash_args) !current_args;
          Hashtbl.clear !current_args;
     current_args := new_hash_args;

     let gen_body =
       <:expr<
               $try Gen_caml.parse_body body
               with
               | TypeError(expected, given, loc) ->
               (
               failwith ("Type Error : expecting : "^
               (ktyp_to_string expected)^" but given : "^
               (ktyp_to_string given)^" in position : "^(Loc.to_string loc)))$>>
     in
     let b_body =
       (try Gen_kir.parse_body2 body true
        with
        | TypeError(expected, given, loc) ->
          failwith ("Type Error : expecting : "^
                    (ktyp_to_string expected)^" but given : "^
                    (ktyp_to_string given)^" in position : "^(Loc.to_string loc))
        | Immutable (value, loc) ->
          (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^
                                    (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^"");
           exit 2;))
     in
     Hashtbl.iter (fun a b -> if b.var_type = TUnknown then
				failwith ("Unknown argument type : "^a))  !current_args ;

	  let n_body2 = <:expr<params $List.fold_left
                          (fun a b -> <:expr<concat $b$ $a$>>)
<:expr<empty_arg()>>
  ((List.rev_map gen_arg_from_patt2 args))$>> in
let gen_body2 =  <:expr<
                         spoc_gen_kernel
                         $n_body2$
                         $List.fold_left (fun a b ->
                         <:expr<spoc_local_env $b$ $a$>>)
b_body
  !arg_list$>>
in
let gen_args = parse_args args gen_body
in
let ret =
  incr arg_idx;
  match !return_type with
  | TUnknown  -> <:expr<return_unknown (), Dummy>>
  | TInt32 -> <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int32 >>
  | TInt64 ->  <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int64>>
  | TVec TInt32 | TVec TInt64 -> assert false
  | TFloat32 ->  <:expr<return_float $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.float32>>
  | TFloat64 ->  <:expr<return_double $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.float64>>
  | TUnit  -> <:expr<return_unit (), Vector.Unit ((),())>>
  | TBool -> <:expr< return_bool $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int32>>
  | Custom (_, name) ->
     let sarek_name = name^"_sarek" in
     let customType = ExId(_loc, (IdLid (_loc,("custom"^(String.capitalize name))))) in
     <:expr< Kirc.return_custom $str:name$ $str:sarek_name$ "", Vector.Custom $customType$>>

  | t  -> failwith (Printf.sprintf "error ret : %s" (ktyp_to_string t))
in
let t =
    List.fold_left (fun seed  p  ->
      match p with
      | (PaId(_,i)) ->
        let value = (Hashtbl.find !current_args (string_of_ident i)) in
        TApp (value.var_type, seed)
      | PaTyc (_,i,t) -> ktyp_of_typ t
      | _ -> assert false) !return_type  (List.rev args)
(*
  Hashtbl.fold (
      fun _ value seed ->
		TApp (seed , value.var_type)) !current_args !return_type in*)
in
my_eprintf ((string_of_ident name)^" ....... "^ktyp_to_string t^"\n");
let task_manager =
  (if Fastflow.fastflow then
     Fastflow.print_task args name _loc 
   else
     <:expr< Obj.magic None >>)
in
Hashtbl.add !global_fun (string_of_ident name)
  {nb_args=0;
   cuda_val="";
   opencl_val=""; typ=t};
<:str_item<
let $id:name$ =
        let open Kirc in
        let a = {
        fun_name = $str:string_of_ident name$;
        ml_fun = $gen_args$;
        funbody = $gen_body2$;
        fun_ret = $ret$;
        fastflow_acc = $task_manager$;
        fun_extensions = [| $match !extensions with
        | [] -> <:expr<>>
| t::[] -> t
| _ -> exSem_of_list  !extensions$|];
}
in
a;;
>>
]
];
k_patt:
  [
    [l = patt -> l
    | "->"; "kern"; l=patt -> l
    ]
  ];
kernel_body:
  [
    [exprs =   sequence -> exprs
    ]
  ];
sequence':
  [
    [
      ->fun e -> e
               | ";" -> fun e -> e
               | ";"; el = sequence ->
		  fun e ->
		  {t=TUnknown; e=Seq(_loc, e, el); loc = _loc}
    ]
  ]
;
sequence:
  [
    [
      e = kexpr; k = sequence' -> k e
    ]
  ];
do_sequence:
  [
    [seq = TRY ["{"; seq = sequence; "}" -> seq] -> seq
    | TRY ["{"; "}"] -> {t=TUnknown; e=Noop; loc=_loc}
    | seq = TRY [seq = sequence; "done" -> seq] -> seq
    | "done" -> {t=TUnknown; e=Noop; loc=_loc}
    ]
  ];
pattern :
  [
    [
      cstr = a_UIDENT; s = OPT ident  ->
      Constr (cstr, s)
    ]
  ];
first_case :
  [
    [ p = pattern; "->"; e = kexpr ->
      (_loc, p, e)
    ]
  ];
match_cases :
  [
    [
      "|"; p = pattern; "->"; e = kexpr ->
      (_loc, p, e)
    ]
  ];
kident :
  [
    "type constraint" LEFTA
      [ "("; x = ident; ":"; t = ctyp; ")" ->
        my_eprintf (Printf.sprintf "adding %s with constraint of type %s \n" (string_of_ident x) (string_of_ctyp t)) ;
        incr arg_idx;
        let arg =  {n= !arg_idx;
                    var_type = ktyp_of_typ t;
                    is_mutable = false;
                    read_only = false;
                    write_only = false;
                    is_global = false;} in
        Hashtbl.add !current_args (string_of_ident x) arg;
      arg,x]
  | "ident"
      [ x = ident ->
      my_eprintf (Printf.sprintf "adding %s\n" (string_of_ident x));
        incr arg_idx;
        let arg =           {n= !arg_idx;
                             var_type = TUnknown;
                             is_mutable = false;
                             read_only = false;
                             write_only = false;
                             is_global = false;} in
        Hashtbl.add !current_args (string_of_ident x) arg;
     arg,x]
  ];

kexpr:
  [
    "let"
      ["let"; opt_mutable = OPT "mutable";  var = kident; "="; y = SELF; "in"; z = sequence  ->
       let arg,var = var
        in   {t=TUnknown;
        e=
          Bind(_loc,
               {t= (
                     (ktyp_to_string arg.var_type);
                   arg.var_type);
                  e= Id (_loc, var);
                  loc = _loc},
                 y, z, (match opt_mutable with
            | None -> false
            | _ -> true)
                );
        loc = _loc};
       |"let"; "open"; i = module_longident; "in"; s = sequence ->
       {
         t = TUnknown;
         e = Open(_loc, i, s);
         loc = _loc
       }]
  | "fun"
      [ "fun"; args = LIST1 k_patt; "->"; body = sequence ->
        (*copy args in local list, (used fo lambda lifting *)
        let args = ref args in
        let lifted = ref [] in
        n_lifted_vals := 0;
        List.iter (fun s  ->
            match s with
            | PaId (_,(IdLid(_,s))) ->
              my_eprintf
                (Printf.sprintf "+*+* %s\n" s)
            | _ -> (); ) !args;
        (*save current kernel environment*)
        let saved_arg_idx = !arg_idx;
        and saved_return_type = !return_type;
        and saved_arg_list = List.map (fun a -> a) !arg_list
        and saved_retype = !retype
        and saved_unknown = !unknown in
        arg_idx := 0;
        return_type := TUnknown;
        arg_list := [];

        let old_args = Hashtbl.create (Hashtbl.length !current_args) in
        Hashtbl.iter (Hashtbl.add old_args) !current_args;

        Hashtbl.clear !current_args;

        List.iter new_arg_of_patt !args;

        retype := true;
        (try
           while !retype do
             retype := false;
             unknown := 0;
             (try
               typer body (TApp (TUnknown, TUnknown));
             with
             | Unbound_value (value, loc) ->
               (
                 (* unbound value in local function, do we need lambda lifitng? *)
                 (try
                    Hashtbl.iter (fun s _ -> my_eprintf (Printf.sprintf "%s\n" s)) old_args;
                    ignore(Hashtbl.find old_args value);

                    (* value found in enclosing kernel/function, needs lambda lifting *)
                    failwith "Lambda lifting not fully implemented yet";
                    args :=  !args @ [(<:patt< $lid:value$ >>)];
                    Printf.eprintf "var : %s needs lambda lifiting\n" value;
                    lifted := value :: !lifted;
                    incr n_lifted_vals;
                    Hashtbl.add !current_args (value)
                      {n= (-1);
                       var_type = TUnknown;
                       is_mutable = false;
                       read_only = false;
                       write_only = false;
                       is_global = false;};
                  with
                  (* not found... *)
                  | Not_found ->
                    (Printf.eprintf "%s\n%!" ("\027[31m Unbound Value \027[00m : \027[33m"^
                                              (value)^"\027[00m in position : "^(Loc.to_string loc)^"");
                     exit 3))
               ));

               my_eprintf (Printf.sprintf "\nUnknown : %d \n\n\n%!" !unknown);
           done;
         with
         | TypeError(expected, given, loc) ->
           (
             failwith ("Type Error : expecting : "^
		       (ktyp_to_string expected)^" but given : "^
		       (ktyp_to_string given)^" in position : "^(Loc.to_string loc)))
         | Immutable (value, loc) ->
           (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^
                                     (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^"");
            exit 2;));

        my_eprintf ("fun_type : "^ktyp_to_string body.t^"\n");

        return_type := body.t;

        let new_hash_args = Hashtbl.create (Hashtbl.length !current_args) in
        Hashtbl.iter (Hashtbl.add new_hash_args) !current_args;
        Hashtbl.clear !current_args;
        current_args := new_hash_args;



        let gen_body =
          <:expr<
                  $try Gen_caml.parse_body body
                  with
                  | TypeError(expected, given, loc) ->
                  (
                  failwith ("Type Error : expecting : "^
                  (ktyp_to_string expected)^" but given : "^
                  (ktyp_to_string given)^" in position : "^(Loc.to_string loc)))$>>
        in
        let b_body =
          (try Gen_kir.parse_body2 body true
           with
           | TypeError(expected, given, loc) ->
             failwith ("Type Error : expecting : "^
                       (ktyp_to_string expected)^" but given : "^
                       (ktyp_to_string given)^" in position : "^(Loc.to_string loc))
           | Immutable (value, loc) ->
             (Printf.eprintf "%s\n%!" ("\027[31m Immutable Value \027[00m : \027[33m"^
                                       (value)^"\027[00m used as mutable in position : "^(Loc.to_string loc)^"");
              exit 2;))
        in
          Hashtbl.iter (fun a b -> if b.var_type = TUnknown then
                           failwith ("Unknown argument type : "^a))  !current_args ;

          let n_body2 = <:expr<params $List.fold_left
                             (fun a b -> <:expr<concat $b$ $a$>>)
<:expr<empty_arg()>>
  ((List.rev_map gen_arg_from_patt2 !args))$>> in
let gen_body2 =  <:expr<
                         spoc_gen_kernel
                         $n_body2$
                         $
                         b_body
                         $>>
in
let gen_args = parse_args !args gen_body
in

let ret =
  incr arg_idx;
  match !return_type with
  | TUnknown  -> <:expr<return_unknown (), Dummy>>
  | TInt32 -> <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int32 >>
  | TInt64 ->  <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int64>>
  | TVec TInt32 | TVec TInt64 -> assert false
  | TFloat32 ->  <:expr<return_float $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.float32>>
  | TFloat64 ->  <:expr<return_double $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.float64>>
  | TUnit  -> <:expr<return_unit (), Vector.Unit ((),())>>
  | TBool -> <:expr< return_bool $ExInt(Loc.ghost, string_of_int (!arg_idx))$ "", Vector.int32>>
  | Custom (_, name) ->
    let sarek_name = name^"_sarek" in
    let customType = ExId(_loc, (IdLid (_loc,("custom"^(String.capitalize name))))) in
    <:expr< Kirc.return_custom $str:name$ $str:sarek_name$ "", Vector.Custom $customType$>>

  | t  -> failwith (Printf.sprintf "error ret : %s" (ktyp_to_string t))
in
let full_typ =
  List.fold_left (fun seed  p  ->
      match p with
      | (PaId(_,i)) ->
        let value = (Hashtbl.find !current_args (string_of_ident i)) in
        TApp (value.var_type, seed)
      | _ -> assert false) !return_type  (List.rev !args)
in
my_eprintf ("/....... "^ktyp_to_string full_typ^"\n");
let funv =  {nb_args=0;
             cuda_val="";
             opencl_val=""; typ=full_typ} in

let local =
  Hashtbl.fold (fun key (funv,stri,_) init ->
      <:str_item<
$stri$ $init$>>) !local_fun
    <:str_item<>>
in
let task_manager =
  (if Fastflow.fastflow then
     Fastflow.print_task !args (gen_ano_f _loc) _loc
   else
     <:expr< Obj.magic None >>)
in
let a = <:expr<
                let open Kirc in
                let local_function  = {
                fun_name="";
                ml_fun = $gen_args$;
	        funbody = $gen_body2$;
                fun_ret = $ret$;
                fastflow_acc = $task_manager$;
                fun_extensions = [| $match !extensions with
		| [] -> <:expr<>>
      | t::[] -> t
      | _ -> exSem_of_list  !extensions$|];
}
in local_function >>  in
let res =
  <:expr<
	  let module Local_funs = struct
	  $local$
	  end
	     in let open Local_funs in
	     $a$>>
in

       (*restore kernel environment*)
       arg_idx := saved_arg_idx;
       return_type := saved_return_type;
       arg_list := List.map (fun a -> a) saved_arg_list;
       retype := saved_retype;
       unknown := saved_unknown;


	      {
    t = full_typ;
    e = Fun (_loc,res,full_typ,funv, !lifted);
    loc = _loc
       }
]
| "native"
    ["$"; code = STRING; "$" ->
     {t = TUnknown; e = Nat (_loc, code); loc = _loc}]
| "if"
    [ "if"; cond=SELF; "then"; cons1=sequence;
      "else"; cons2=sequence ->
		    {t=TUnknown; e= Ife(_loc,cond,cons1,cons2); loc = _loc}
    | "if"; cond=SELF; "then"; cons1 = sequence ->
       {t=TUnknown; e= If(_loc,cond,cons1); loc = _loc}
    ]

  
| "match"
    [
      "match"; x = SELF; "with"; m0 = OPT first_case; m = LIST0 match_cases
        ->
        match m0 with
        | Some m1 ->
          {t=TUnknown; e= Match (_loc, x, m1::m); loc = _loc}
        | None ->
          {t=TUnknown; e= Match (_loc, x, m); loc = _loc}]

  | "mod"  RIGHTA
      [ x = SELF; "mod"; y = SELF -> {t=TInt32; e = Mod(_loc, x,y); loc = _loc}]
  | ":="
      [ x = SELF; ":="; y= SELF  -> {t=(TUnit); e = Acc (_loc, x, y); loc = _loc}
      ]
| "<>-"
      [ x = SELF; "<>-"; y= SELF  ->
        begin
          match x with
          | {t = _; e = ArrGet _ } ->
            {t=(TUnit);
             e = ArrSet (_loc, x, y); loc = _loc}
          | _ -> assert false
        end
      ]  | "<-"
      [ x = SELF; "<-"; y= SELF  ->
        begin
          match x with
          | {t = _; e = VecGet _} ->
            {t=(TUnit);
             e = VecSet (_loc, x, y); loc = _loc}
          | {t = _; e = ArrGet _ } ->
            failwith ("Error in position "^(Loc.to_string _loc)^", 
                       arrays are stored in shared memory and can olny be accessed with '<<-' \n")
          | {t=_; e=RecGet _}->
            {t=TUnit;
             e = RecSet (_loc, x, y); loc = _loc}
          | _ -> assert false
        end
      ]

| "apply" LEFTA
    [ e1 = SELF; e2 = SELF -> {t=(TUnknown); e= App(_loc, e1, [e2]); loc = _loc}
    ]
  | "+" LEFTA
    [ x = SELF; "+!"; y = SELF -> {t=TInt32; e = Plus32(_loc, x,y); loc = _loc};
      | x = SELF; "+!!"; y = SELF -> {t=TInt64; e = Plus64(_loc, x,y); loc = _loc};
      | x = SELF; "+"; y = SELF -> {t=TInt32; e = Plus32(_loc, x,y); loc = _loc};
      | x = SELF; "+."; y = SELF -> {t=TFloat32; e = PlusF32(_loc, x, y); loc = _loc}]
  | "-" LEFTA
    [ x = SELF; "-!"; y = SELF -> {t=TInt32; e = Min32(_loc, x,y); loc = _loc};
      | x = SELF; "-!!"; y = SELF -> {t=TInt64; e = Min64(_loc, x,y); loc = _loc};
      | x = SELF; "-"; y = SELF -> {t=TInt32; e = Min32(_loc, x,y); loc = _loc};
      | x = SELF; "-."; y = SELF -> {t=TFloat32; e = MinF32(_loc, x,y); loc = _loc}]

  | "*" LEFTA
    [ x = SELF; "*!"; y = SELF -> {t=TInt32; e = Mul32(_loc, x,y); loc = _loc};
      | x = SELF; "*!!"; y = SELF -> {t=TInt64; e = Mul64(_loc, x,y); loc = _loc};
      | x = SELF; "*"; y = SELF -> {t=TInt32; e = Mul32(_loc, x,y); loc = _loc};
      | x = SELF; "*."; y = SELF -> {t=TFloat32; e = MulF32(_loc, x,y); loc = _loc}]
  | "/" LEFTA
    [ x = SELF; "/!"; y = SELF -> {t=TInt32; e = Div32(_loc, x,y); loc = _loc};
      | x = SELF; "/!!"; y = SELF -> {t=TInt64; e = Div64(_loc, x,y); loc = _loc};
      | x = SELF; "/"; y = SELF -> {t=TInt32; e = Div32(_loc, x,y); loc = _loc};
      | x = SELF; "/."; y = SELF -> {t=TFloat32; e = DivF32(_loc, x,y); loc = _loc}]
| ":" LEFTA
    [x=SELF; ":"; t=ctyp -> {t=TUnknown; e=TypeConstraint(_loc, x, (ktyp_of_typ t)); loc= _loc};]



| "||"
    [x = SELF; "||"; y = SELF -> {t=TBool;
                                  e = BoolOr (_loc, x, y); loc = _loc} ]

  | "&&"
    [x = SELF; "&&"; y = SELF -> {t=TBool; e = BoolAnd (_loc, x, y); loc = _loc} ]

  | "not"
      ["!"; x = kexpr -> {t=TBool; e = BoolNot (_loc, x); loc = _loc} ]


| "pragma"
    ["pragma"; opt = LIST0 [x = STRING -> x]; y = SELF -> {t=TUnit; e=Pragma(_loc,opt,y); loc=_loc}
    ]
  | "loop"
    [  "for"; x = ident; "="; y=SELF; "to"; z = SELF; "do";  body=do_sequence ->
        {t = TUnknown; e = DoLoop (_loc,
                                {t= TInt32;
                                 e= Id (_loc, x);
                                 loc = _loc}
                               ,y,z,body); loc = _loc};
      | "while"; cond = sequence; "do"; body = do_sequence ->
      {t = TUnknown; e = While (_loc,cond, body); loc = _loc}]

  | "="
    [ x=SELF; "="; y=SELF -> {t=TBool; e= BoolEq(_loc,x,y); loc = _loc};
      | x=SELF; "=!"; y=SELF -> {t=TBool; e= BoolEq32(_loc,x,y); loc = _loc};
      | x=SELF; "=!!"; y=SELF -> {t=TBool; e= BoolEq64(_loc,x,y); loc = _loc};
      | x=SELF; "=."; y=SELF -> {t=TBool; e= BoolEqF32(_loc,x,y); loc = _loc}]
  | "<"
    [ x=SELF; "<"; y=SELF -> {t=TBool; e= BoolLt32(_loc,x,y); loc = _loc};
      | x=SELF; "<!"; y=SELF -> {t=TBool; e= BoolLt32(_loc,x,y); loc = _loc};
      | x=SELF; "<!!"; y=SELF -> {t=TBool; e= BoolLt64(_loc,x,y); loc = _loc};
      | x=SELF; "<."; y=SELF -> {t=TBool; e= BoolLtF32(_loc,x,y); loc = _loc}]

  | "<="
    [ x=SELF; "<="; y=SELF -> {t=TBool; e= BoolLtE32(_loc,x,y); loc = _loc};
      | x=SELF; "<=!"; y=SELF -> {t=TBool; e= BoolLtE32(_loc,x,y); loc = _loc};
      | x=SELF; "<=!!"; y=SELF -> {t=TBool; e= BoolLtE64(_loc,x,y); loc = _loc};
      | x=SELF; "<=."; y=SELF -> {t=TBool; e= BoolLtEF32(_loc,x,y); loc = _loc}]

  |  ">"
      [ x=SELF; ">"; y=SELF -> {t=TBool; e= BoolGt32(_loc,x,y); loc = _loc};
        | x=SELF; ">!"; y=SELF -> {t=TBool; e= BoolGt32(_loc,x,y); loc = _loc};
        | x=SELF; ">!!"; y=SELF -> {t=TBool; e= BoolGt64(_loc,x,y); loc = _loc};
        | x=SELF; ">."; y=SELF -> {t=TBool; e= BoolGtF32(_loc,x,y); loc = _loc}]

  | ">="
      [ x=SELF; ">="; y=SELF -> {t=TBool; e= BoolGtE32(_loc,x,y); loc = _loc};
        | x=SELF; ">=!"; y=SELF -> {t=TBool; e= BoolGtE32(_loc,x,y); loc = _loc};
        | x=SELF; ">=!!"; y=SELF -> {t=TBool; e= BoolGtE64(_loc,x,y); loc = _loc};
        | x=SELF; ">=."; y=SELF -> {t=TBool; e= BoolGtEF32(_loc,x,y); loc = _loc}]


  | "@"
      [ "@"; x = ident ->
        {t=TUnknown;
         e=Ref(_loc,
               {t=TUnknown; e = Id (_loc, x); loc = _loc}
              ); loc = _loc}
      ]

  | "." RIGHTA
      [
        x = SELF; "."; "[<"; y=SELF; ">]"  -> {t=(TUnknown);
					       e = VecGet (_loc, x, y); loc = _loc};
        | x = SELF; "."; "("; y=SELF; ")"  -> {t=(TUnknown);
                                               e = ArrGet (_loc, x, y); loc = _loc};
        |l = UIDENT ; "."; e = SELF -> {t=(TUnknown);
				        e = ModuleAccess (_loc, l, e);
                                        loc = _loc};
        | e1 = kexpr; "."; field = ident -> {t= TUnknown;
                                             e= RecGet (_loc,e1,field);
                                             loc = _loc}
 ]
  | "record"
      [ "{";l = kfields_declaration_list; "}" ->
        (Printf.eprintf "RECORD\n%!";
        {t=TUnknown; e=Record(_loc,l); loc=_loc};)
      ]
  | "simple" NONA
      [ "("; ")" -> {t=TUnknown; e = Noop; loc = _loc};
        |  "(" ;  x= sequence; ")"  ->  x;
        |  "(" ;  x= SELF; ")"  ->  x;
        | "begin" ;  x= sequence; "end"  ->  x;
        | x = FLOAT-> {t=TFloat32; e = Float32 (_loc, x); loc = _loc};
        | x = LIDENT  -> {t=TUnknown; e = Id (_loc, IdLid(_loc,x)); loc = _loc};
        | x = INT32  ->{t=TInt32; e = Int32 (_loc, x); loc = _loc};
        | x = INT  ->{t=TInt32; e = Int32 (_loc, x); loc = _loc}
        | x = a_UIDENT -> {t=TUnknown; e = Id (_loc, IdUid(_loc,x)); loc = _loc};
        | "false" -> {t=TBool; e=False _loc; loc = _loc};
        | "true" -> {t=TBool; e=True _loc; loc = _loc};

      ]


  ];
  kfields_declaration_list:
    [ [ t1 = kfield_declaration; ";"; t2 = SELF -> t1::t2
        | t1 = kfield_declaration; ";" -> [t1]
        | t1 = kfield_declaration ->  [t1]
       ] ]
  ;
  kfield_declaration:
    [[
      s=ident; "=";  t=kexpr -> (_loc,s,t)
    ]];



END
