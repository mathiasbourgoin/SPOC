open Camlp4.PreCast
open Syntax
open Ast

open Typer
open Mparser

let gen_kernel ()= () 

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
     | "ktyp"; name=LIDENT; "="; k = ktype_kind -> 
       gen_ctypes _loc k name;
      ]
    ];
  
  ktype_kind :
    [[ "{"; (t,l) = klabel_declaration_list; "}" ->
       KRecord (t,l)
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
      s=ident; ":";  t=type_kind -> (s,t)
    ]];
  mod_const_expr: 
    [[name=ident; ":"; typ=ident; "="; cu_value=STRING; cl_value=STRING -> 
      let typ =
        match string_of_ident typ with
        | "int" -> TInt;
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
        | "int" -> TInt;
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
       arg_idx := 0;
       return_type := TUnknown;
       arg_list := [];
       extensions := [ ex32 ];
       Hashtbl.clear !current_args;
       List.iter new_arg_of_patt args;
       (try 
          typer body TUnknown
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
       );  
       let new_hash_args = Hashtbl.create (Hashtbl.length !current_args) in
       Hashtbl.iter (Hashtbl.add new_hash_args) !current_args;
       Hashtbl.clear !current_args;
       current_args := new_hash_args;  
       let gen_body = 
         <:expr< 
                 $try parse_body body with
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
         (try parse_body2 body true
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
  (List.rev (List.map gen_arg_from_patt2 args))$>> in 
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
  | TInt | TInt32 -> <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$, Vector.int32 >>
  | TInt64 ->  <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$, Vector.int64>>
  | TVec TInt | TVec TInt32 | TVec TInt64 -> assert false
  | TFloat | TFloat32 ->  <:expr<return_float $ExInt(Loc.ghost, string_of_int (!arg_idx))$, Vector.float32>>
  | TFloat64 ->  <:expr<return_double $ExInt(Loc.ghost, string_of_int (!arg_idx))$, Vector.float64>>
  | TUnit  -> <:expr<return_unit (), Vector.Unit ((),())>>
  | TBool -> <:expr< return_bool $ExInt(Loc.ghost, string_of_int (!arg_idx))$, Vector.Dummy>>
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
          )>>
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
]
];
expr:
  [
    ["kfun"; args = LIST1 k_patt; "->"; body = kernel_body ->
     arg_idx := 0;
     return_type := TUnknown;
     arg_list := [];
     Hashtbl.clear !current_args;
     List.iter new_arg_of_patt args;
     (try 
        typer body TUnknown
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
     let new_hash_args = Hashtbl.create (Hashtbl.length !current_args) in
     Hashtbl.iter (Hashtbl.add new_hash_args) !current_args;
     Hashtbl.clear !current_args;
     current_args := new_hash_args;  
     let gen_body = 
       <:expr< 
               $try parse_body body
               with 
               | TypeError(expected, given, loc) -> 
               (
               failwith ("Type Error : expecting : "^
               (ktyp_to_string expected)^" but given : "^
               (ktyp_to_string given)^" in position : "^(Loc.to_string loc)))$>>
     in
     let b_body = 
       (try parse_body2 body true
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
     let n_body2 = <:expr<params $List.fold_left 
                          (fun a b -> <:expr<concat $b$ $a$>>) 
<:expr<empty_arg()>> 
  (List.rev (List.map gen_arg_from_patt2 args))$>> in 
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
  | TUnknown  -> <:expr<return_unknown ()>>
  | TInt | TInt32 | TInt64 ->  <:expr<return_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$>>
  | TVec TInt | TVec TInt32 | TVec TInt64 -> <:expr<return_vector_int $ExInt(Loc.ghost, string_of_int (!arg_idx))$>>
  | TFloat | TFloat32 ->  <:expr<return_float $ExInt(Loc.ghost, string_of_int (!arg_idx))$>>
  | TFloat64 ->  <:expr<return_double $ExInt(Loc.ghost, string_of_int (!arg_idx))$>>
  | TUnit  -> <:expr<return_unit ()>>
  | TBool -> <:expr< return_bool $ExInt(Loc.ghost, string_of_int (!arg_idx))$>>
  | _  -> failwith "error ret"
in
<:expr< 
        let open Kirc in 
        let a = {
        ml_kern = $gen_args$;
        body = $gen_body2$;
        ret_val = $ret$;
        extensions = [ $match !extensions with 
        | [] -> <:expr<>>
| t::[] -> t 
| _ -> exSem_of_list  !extensions$];
}
in
a>>
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
                               | ";"; el = sequence -> fun e ->
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
kexpr:
  [ 
 "let"
  ["let"; opt_mutable = OPT "mutable";  var = ident; "="; y = SELF; "in"; z = sequence  ->  
     {t=TUnknown; 
      e=  Bind(_loc, 
               {t= TUnknown; 
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
       e = Open (_loc, i, s);
       loc = _loc
     }]
  | "if"
      [ "if"; cond=SELF; "then"; cons1=sequence; 
        "else"; cons2=sequence -> 
        {t=TUnknown; e= Ife(_loc,cond,cons1,cons2); loc = _loc}
              | "if"; cond=SELF; "then"; cons1 = sequence -> 
        {t=TUnknown; e= If(_loc,cond,cons1); loc = _loc}
      ]
  | "mod"  RIGHTA
    [ x = SELF; "mod"; y = SELF -> {t=TInt; e = Mod(_loc, x,y); loc = _loc}]
  | ":=" 
      [ x = SELF; ":="; y= SELF  -> {t=(TUnit); e = Acc (_loc, x, y); loc = _loc}
      ]
  | "apply" LEFTA
    [ e1 = SELF; e2 = SELF -> {t=(TUnknown); e= App(_loc, e1, [e2]); loc = _loc}
    ]	 
  | "<-" 
      [ x = SELF; "<-"; y= SELF  -> 
        begin
          match x with 
          | {t = _; e = VecGet _} ->
            {t=(TUnit);
             e = VecSet (_loc, x, y); loc = _loc}
          | {t = _; e = ArrGet _ } ->
            {t=(TUnit);
             e = ArrSet (_loc, x, y); loc = _loc}
          | _ -> assert false
        end
      ]

  | "+" LEFTA
    [ x = SELF; "+!"; y = SELF -> {t=TInt32; e = Plus32(_loc, x,y); loc = _loc};
      | x = SELF; "+!!"; y = SELF -> {t=TInt64; e = Plus64(_loc, x,y); loc = _loc};
      | x = SELF; "+"; y = SELF -> {t=TInt; e = Plus(_loc, x,y); loc = _loc};
      | x = SELF; "+."; y = SELF -> {t=TFloat32; e = PlusF32(_loc, x,y); loc = _loc}]
  | "-" LEFTA
    [ x = SELF; "-!"; y = SELF -> {t=TInt32; e = Min32(_loc, x,y); loc = _loc};
      | x = SELF; "-!!"; y = SELF -> {t=TInt64; e = Min64(_loc, x,y); loc = _loc};
      | x = SELF; "-"; y = SELF -> {t=TInt; e = Min(_loc, x,y); loc = _loc};
      | x = SELF; "-."; y = SELF -> {t=TFloat32; e = MinF32(_loc, x,y); loc = _loc}]
    
  | "*" LEFTA
    [ x = SELF; "*!"; y = SELF -> {t=TInt32; e = Mul32(_loc, x,y); loc = _loc};
      | x = SELF; "*!!"; y = SELF -> {t=TInt64; e = Mul64(_loc, x,y); loc = _loc};
      | x = SELF; "*"; y = SELF -> {t=TInt; e = Mul(_loc, x,y); loc = _loc};
      | x = SELF; "*."; y = SELF -> {t=TFloat32; e = MulF32(_loc, x,y); loc = _loc}]
  | "/" LEFTA
    [ x = SELF; "/!"; y = SELF -> {t=TInt32; e = Div32(_loc, x,y); loc = _loc};
      | x = SELF; "/!!"; y = SELF -> {t=TInt64; e = Div64(_loc, x,y); loc = _loc};
      | x = SELF; "/"; y = SELF -> {t=TInt; e = Div(_loc, x,y); loc = _loc};
      | x = SELF; "/."; y = SELF -> {t=TFloat32; e = DivF32(_loc, x,y); loc = _loc}]
    
  | "||" LEFTA	
    [x = SELF; "||"; y = SELF -> {t=TBool; 
                                  e = BoolOr (_loc, x, y); loc = _loc} ]
    
  | "&&" RIGHTA	
    [x = SELF; "&&"; y = SELF -> {t=TBool; e = BoolAnd (_loc, x, y); loc = _loc} ]
    


    
  | "loop"
    [ "for"; x = ident; "="; y=SELF; "to"; z = SELF; "do";  body=do_sequence -> 
        {t = TUnknown; e = DoLoop (_loc, 
                                {t= TInt; 
                                 e= Id (_loc, x);
                                 loc = _loc}
                               ,y,z,body); loc = _loc};
      | "while"; cond = sequence; "do"; body = do_sequence -> 
      {t = TUnknown; e = While (_loc,cond, body); loc = _loc}] 
    
  | "="
    [ x=SELF; "="; y=SELF -> {t=TBool; e= BoolEq(_loc,x,y); loc = _loc};
      | x=SELF; "=!"; y=SELF -> {t=TBool; e= BoolEq32(_loc,x,y); loc = _loc};
      | x=SELF; "=!!"; y=SELF -> {t=TBool; e= BoolEq64(_loc,x,y); loc = _loc};
      | x=SELF; "=."; y=SELF -> {t=TBool; e= BoolEqF(_loc,x,y); loc = _loc}]
  | "<" 
    [ x=SELF; "<"; y=SELF -> {t=TBool; e= BoolLt(_loc,x,y); loc = _loc};
      | x=SELF; "<!"; y=SELF -> {t=TBool; e= BoolLt32(_loc,x,y); loc = _loc};
      | x=SELF; "<!!"; y=SELF -> {t=TBool; e= BoolLt64(_loc,x,y); loc = _loc};
      | x=SELF; "<."; y=SELF -> {t=TBool; e= BoolLtF(_loc,x,y); loc = _loc}]
    
  | "<="
    [ x=SELF; "<="; y=SELF -> {t=TBool; e= BoolLtE(_loc,x,y); loc = _loc};
      | x=SELF; "<=!"; y=SELF -> {t=TBool; e= BoolLtE32(_loc,x,y); loc = _loc};
      | x=SELF; "<=!!"; y=SELF -> {t=TBool; e= BoolLtE64(_loc,x,y); loc = _loc};
      | x=SELF; "<=."; y=SELF -> {t=TBool; e= BoolLtEF(_loc,x,y); loc = _loc}]

  |  ">" RIGHTA
      [ x=SELF; ">"; y=SELF -> {t=TBool; e= BoolGt(_loc,x,y); loc = _loc};
        | x=SELF; ">!"; y=SELF -> {t=TBool; e= BoolGt32(_loc,x,y); loc = _loc};
        | x=SELF; ">!!"; y=SELF -> {t=TBool; e= BoolGt64(_loc,x,y); loc = _loc};
        | x=SELF; ">."; y=SELF -> {t=TBool; e= BoolGtF(_loc,x,y); loc = _loc}]

  | ">="
      [ x=SELF; ">="; y=SELF -> {t=TBool; e= BoolGtE(_loc,x,y); loc = _loc};
        | x=SELF; ">=!"; y=SELF -> {t=TBool; e= BoolGtE32(_loc,x,y); loc = _loc};
        | x=SELF; ">=!!"; y=SELF -> {t=TBool; e= BoolGtE64(_loc,x,y); loc = _loc};
        | x=SELF; ">=."; y=SELF -> {t=TBool; e= BoolGtEF(_loc,x,y); loc = _loc}]
		

  | "!"
      [ "!"; x = ident -> 
        {t=TUnknown; 
         e=Ref(_loc,
               {t=TUnknown; e = Id (_loc, x); loc = _loc}
              ); loc = _loc}
      ]
      
  | "." RIGHTA
	[x = SELF; "."; "[<"; y=SELF; ">]"  -> {t=(TUnknown); 
						e = VecGet (_loc, x, y); loc = _loc};
	| x = SELF; "."; "("; y=SELF; ")"  -> {t=(TUnknown); 
                                               e = ArrGet (_loc, x, y); loc = _loc};
	|l = UIDENT ; "."; e = SELF -> {t=(TUnknown); 
					e = ModuleAccess (_loc, l, e); 
                                     loc = _loc}
	]
  | "simple" NONA
      ["(" ;  x= sequence; ")"  ->  x;
       |"begin" ;  x= sequence; "end"  ->  x;
       | "("; ")" -> {t=TUnknown; e = Noop; loc = _loc};
       | x = FLOAT-> {t=TFloat32; e = Float32 (_loc, x); loc = _loc};
       |x = LIDENT  -> {t=TUnknown; e = Id (_loc, IdLid(_loc,x)); loc = _loc};
       |x = INT32  ->{t=TInt32; e = Int32 (_loc, x); loc = _loc};
       |x = INT  ->{t=TInt; e = Int (_loc, x); loc = _loc}] 		


  ];


END
