(******************************************************************************
 * Mathias Bourgoin, Universit Pierre et Marie Curie (2012)
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
open Spoc
open Kernel
open Mem
open Devices
open Vector

open Kirc_Ast


module Kirc_OpenCL = Gen.Generator(Kirc_OpenCL)
module Kirc_Cuda = Gen.Generator(Kirc_Cuda)
module Kirc_Profile = Gen.Generator(Profile)


type extension =
  | ExFloat32
  | ExFloat64


type ('a,'b,'c) kirc_kernel =
  {
    ml_kern : 'a;
    body : Kirc_Ast.k_ext;
    ret_val : Kirc_Ast.k_ext* ('b,'c) Vector.kind;
    extensions : extension array
  }

type ('a,'b,'c,'d) kirc_function =
  {
    fun_name:string;
    ml_fun : 'a;
    funbody : Kirc_Ast.k_ext;
    fun_ret : Kirc_Ast.k_ext* ('b,'c) Vector.kind;
    fastflow_acc : 'd;
    fun_extensions : extension array
  }


type ('a,'b,'c,'d,'e) sarek_kernel =
  ('a,'b) spoc_kernel * ('c,'d,'e) kirc_kernel


let constructors = ref []

let opencl_head = (
  "float spoc_fadd ( float a, float b );\n"^
  "float spoc_fminus ( float a, float b );\n"^
  "float spoc_fmul ( float a, float b );\n"^
  "float spoc_fdiv ( float a, float b );\n"^
  "int logical_and (int, int);\n"^
  "int spoc_powint (int, int);\n"^
  "int spoc_xor (int, int);\n"^


  "float spoc_fadd ( float a, float b ) { return (a + b);}\n"^
  "float spoc_fminus ( float a, float b ) { return (a - b);}\n"^
  "float spoc_fmul ( float a, float b ) { return (a * b);}\n"^
  "float spoc_fdiv ( float a, float b ) { return (a / b);}\n"^
  "int logical_and (int a, int b ) { return (a & b);}\n"^
  "int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"^
  "int spoc_xor (int a, int b ) { return (a^b);}\n"
)

let opencl_float64 = (
  "#ifndef __FLOAT64_EXTENSION__ \n"^
  "#define __FLOAT64_EXTENSION__ \n"^
  "#if defined(cl_khr_fp64)  // Khronos extension available?\n"^
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"^
  "#elif defined(cl_amd_fp64)  // AMD extension available?\n"^
  "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"^
  "#endif\n"^
  "double spoc_dadd ( double a, double b );\n"^
  "double spoc_dminus ( double a, double b );\n"^
  "double spoc_dmul ( double a, double b );\n"^
  "double spoc_ddiv ( double a, double b );\n"^
  "double spoc_dadd ( double a, double b ) { return (a + b);}\n"^
  "double spoc_dminus ( double a, double b ) { return (a - b);}\n"^
  "double spoc_dmul ( double a, double b ) { return (a * b);}\n"^
  "double spoc_ddiv ( double a, double b ) { return (a / b);}\n"^
  "#endif\n"
)

let cuda_float64 =
  (
    "#ifndef __FLOAT64_EXTENSION__ \n"^
    "#define __FLOAT64_EXTENSION__ \n"^
    "__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"^
    "__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"^
    "__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"^
    "__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"^

    "#endif\n"
  )
let cuda_head = (
  "__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"^
  "__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"^
  "__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"^
  "__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"^
  "__device__ int logical_and (int a, int b ) { return (a & b);}\n"^
  "__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"^
  "__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"^
  "__device__ void spoc_atomic_add(unsigned long long int *a, unsigned int b){ atomicAdd(a, b);}"
)

let eint32 = EInt32
let eint64 = EInt32
let efloat32 = EInt32
let efloat64 = EInt32

let global = Global
let local = LocalSpace
let shared = Shared

let new_var i = IdName ("spoc_var"^(string_of_int i))
let new_array i l t m = Arr (i, l, t, m)
let var i s = IntId (s, i) (*("spoc_var"^(string_of_int i)), i)*)
let spoc_gen_kernel args body = Kern (args,body)
let spoc_fun_kernel a b = ()
let global_fun a = GlobalFun (
    a.funbody,
    (match snd a.fun_ret with
    | Vector.Int32 _  -> "int"
    | Vector.Float32 _  -> "float"
    | Vector.Custom _ ->
      (match fst a.fun_ret with
      | CustomVar (s,_,_) -> "struct "^s^"_sarek"
      | _ -> assert false)
    | _ -> "void"),
    a.fun_name)
let seq a b = Seq (a,b)
let app a b = App (a,b)
let spoc_unit () = Unit
let spoc_int a = Int a
let global_int_var a = GInt a
let global_float_var a = GFloat a
let spoc_int32 a = Int (Int32.to_int a)
let spoc_float f = Float f
let spoc_double d = Double d

let spoc_int_id a = Int a(*IntId (a,-1)*)
let spoc_float_id a = Float a
let spoc_plus a b = Plus (a,b)
let spoc_plus_float a b = Plusf (a,b)

let spoc_min a b = Min (a,b)
let spoc_min_float a b = Minf (a,b)

let spoc_mul a b = Mul (a,b)
let spoc_mul_float a b = Mulf (a,b)

let spoc_div a b = Div (a,b)
let spoc_div_float a b = Divf (a,b)

let spoc_mod a b = Mod (a,b)
let spoc_ife a b c = Ife (a,b,c)
let spoc_if a b  = If (a,b)
let spoc_match s e l = Match (s,e,l)
let spoc_case i o e : case = (i,o,e)
let spoc_do a b c d = DoLoop (a,b,c,d)
let spoc_while a b = While (a,b)

let params l = Params l
let spoc_id i = Id ("")
let spoc_constr t c params = Constr (t,c,params)
let spoc_record t params = Record (t,params)
let spoc_return k = Return k
let concat a b = Concat (a,b)
let empty_arg () = Empty
let new_int_var i s = IntVar (i,s)
let new_float_var i s = FloatVar (i,s)
let new_float64_var i s = DoubleVar (i,s)
let new_double_var i s = DoubleVar (i,s)
let new_unit_var i s = UnitVar (i,s)
let new_custom_var n v s = Custom (n,v,s)  (* <--- *)

let new_int_vec_var v s = VecVar (Int 0, v,s)
let new_float_vec_var v s = VecVar (Float 0., v,s)
let new_double_vec_var v s = VecVar (Double 0., v,s)
let new_custom_vec_var n v s = VecVar (Custom (n,0,s), v,s)  (* <--- *)

let int_vect i = IntVect i
let spoc_rec_get r id = RecGet (r,id)
let spoc_rec_set r v = RecSet (r,v)
let set_vect_var vecacc value =
  SetV (vecacc, value)
let set_arr_var arracc value =
  SetV (arracc, value)
let intrinsics a b = Intrinsics (a,b)
let spoc_local_env local_var b = Local(local_var, b)
let spoc_set name value = Set (name, value)
let spoc_declare name = Decl (name)
let spoc_local_var a = a
let spoc_acc a b = Acc (a, b)
let int_var i = i
let int32_var i = i
let float_var f = f
let double_var d = CastDoubleVar (d,"")

let equals a b = EqBool (a,b)
let equals_custom s v1 v2 = EqCustom (s,v1,v2)
let equals32 a b = EqBool (a,b)
let equals64 a b = EqBool (a,b)
let equalsF a b = EqBool (a,b)
let equalsF64 a b = EqBool (a,b)
let b_or a b = Or (a,b)
let b_and a b = And (a,b)
let b_not a  = Not (a)

let lt a b = LtBool (a,b)
let lt32 a b = LtBool (a,b)
let lt64 a b = LtBool (a,b)
let ltF a b = LtBool (a,b)
let ltF64 a b = LtBool (a,b)

let gt a b = GtBool (a,b)
let gt32 a b = GtBool (a,b)
let gt64 a b = GtBool (a,b)
let gtF a b = GtBool (a,b)
let gtF64 a b = GtBool (a,b)

let lte a b = LtEBool (a,b)
let lte32 a b = LtEBool (a,b)
let lte64 a b = LtEBool (a,b)
let lteF a b = LtEBool (a,b)
let lteF64 a b = LtEBool (a,b)

let gte a b = GtEBool (a,b)
let gte32 a b = GtEBool (a,b)
let gte64 a b = GtEBool (a,b)
let gteF a b = GtEBool (a,b)
let gteF64 a b = GtEBool (a,b)

let get_vec a b = IntVecAcc (a,b)
let get_arr a b = IntVecAcc (a,b)
let return_unit () =  Unit

let return_int i s= IntVar (i,s)
let return_float f s= FloatVar (f,s)
let return_double d s= DoubleVar (d,s)
let return_bool b s= BoolVar (b,s)
let return_custom n sn s= CustomVar (n, sn,s)

let spoc_native s = Native s



let print s = Printf.printf "%s}\n" s

let print_ast = Kirc_Ast.print_ast


let debug_print ((ker : ('a, 'b,'c,'d,'e)  sarek_kernel)) =
  let _,k=  ker in
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in
  print_ast k2
;;




let rewrite ker =
  let b = ref false in
  let rec aux kern =
    match kern with
    | Native _ -> kern
    | Block b -> Block (aux b)
    | Kern (k1,k2) ->
      Kern (aux k1, aux k2)
    | Params k ->
      Params (aux k)
    | Plus (k1,k2) ->
      Plus (aux k1, aux k2)
    | Plusf (k1,k2) ->
      (match k1,k2 with
       | Float f1, Float f2 ->
         (b := true; Float (f1 +. f2))
       |  _ -> Plusf (aux k1, aux k2))
    | Min (k1,k2) ->
      Min (aux k1, aux k2)
    | Minf (k1,k2) ->
      (match k1,k2 with
       | Float f1, Float f2 ->
         (b := true; Float (f1 +. f2))
       |  _ -> Minf (aux k1, aux k2))
    | Mul (k1,k2) ->
      Mul (aux k1, aux k2)
    | Mulf (k1,k2) ->
      (match k1,k2 with
       | Float f1, Float f2 ->
         (b := true; Float (f1 +. f2))
       |  _ -> Mulf (aux k1, aux k2))
    | Div (k1,k2) ->
      Div (aux k1, aux k2)
    | Divf (k1,k2) ->
      (match k1,k2 with
       | Float f1, Float f2 ->
         (b := true; Float (f1 +. f2))
       |  _ -> Divf (aux k1, aux k2))
    | Mod (k1,k2) ->
      Mod (aux k1, aux k2)
    | Id _ -> kern
    | IdName _ -> kern
    | IntVar _ -> kern
    | FloatVar _ -> kern
    | UnitVar _ ->
      Seq (kern, kern)
    | CastDoubleVar _ -> kern
    | DoubleVar _ -> kern
    | BoolVar _ -> kern
    | VecVar (k,idx,s ) ->
      VecVar (aux k, idx,s)
    | Concat (k1,k2) ->
      Concat (aux k1, aux k2)
    | Constr (t,c,l) ->
      Constr (t, c, List.map aux l)
    | Record (t,l) ->
      Record (t, List.map aux l)
    | RecGet (r,s) -> RecGet (aux r,s)
    | RecSet (r,v) -> RecSet (aux r, aux v)
    | Empty -> kern
    | Seq (k1, Unit) -> aux k1
    | Seq (k1,k2) ->
      Seq (aux k1, aux k2)
    | Return k ->
      begin
        match k with
        | Return k ->
          (b := true; aux (Return k))
	     | Acc _ | Set _ -> aux k
       | Ife(k1, k2, k3) ->
         (b := true;
          Ife (aux k1, aux (Return k2), aux (Return k3)))
       | If(k1, k2) ->
         (b := true;
          If (aux k1, aux (Return k2)))
       | DoLoop (k1,k2,k3,k4) ->
         (b := true;
          DoLoop (aux k1, aux k2, aux k3, aux (Return k4)))
       | While (k1, k2) ->
         (b:= true;
          While (aux k1, aux (Return k2)))
       | Seq (k1, k2) ->
         ( b:= true;
           Seq ( aux k1, aux (Return k2)))
       | Match (s,a,bb) ->
          (b := true;
           Match (s,aux a,
                  Array.map (fun (i,ofid,e) ->
			     (i,ofid, aux (Return e))) bb))
       | _ ->
         Return (aux k)
      end
    | Acc (k1,k2) ->

       (match k2 with
	| Ife(k1', k2', k3') ->
           (b := true;
            Ife (aux k1',
		 aux (Acc (k1,k2')),
		 aux (Acc (k1,k3'))))
	| If(k1', k2') ->
           (b := true;
            If (aux k1', aux (Acc (k1,k2'))))
	| DoLoop (k1',k2',k3',k4') ->
          (b := true;
           DoLoop (aux k1', aux k2', aux k3', aux (Acc (k1,k4'))))
	| While (k1', k2') ->
           (b:= true;
            While (aux k1', aux (Acc (k1,k2'))))
	| Seq (k1', k2') ->
           ( b:= true;
             Seq ( aux k1', aux (Acc (k1,k2'))))
	| Match (s,a,bb) ->
           (b := true;
            Match (s,aux a,
                   Array.map (fun (i,ofid,e) ->
			      (i,ofid, aux (Acc (k1, e)))) bb))
	| Return _ -> assert false
	|_ ->
	  Acc (aux k1, aux k2))
    | Set (k1,k2) -> aux (Acc (k1,k2))

    | Decl k1 -> aux k1
    | SetV (k1,k2) ->
      (match k2 with
       | Seq (k3, k4) ->
         Seq (k3, SetV (aux k1, aux k4))
       | Ife(k3, k4, k5) ->
         (b := true;
          Ife (aux k3, SetV (aux k1, aux k4), SetV (aux k1, k5)))
       | Match (s,a,bb) ->
         (b := true;
          Match (s,aux a,
                 Array.map (fun (i,ofid,e) ->
                     (i,ofid,SetV(aux k1, aux e))) bb))
       | _ -> SetV (aux k1, aux k2)
      )
    | SetLocalVar (k1,k2,k3) ->
      SetLocalVar (aux k1, aux k2, aux k3)
    | Intrinsics _ -> kern
    | IntId _ -> kern
    | Int _-> kern
    | GInt _ -> kern
    | GFloat _ -> kern
    | Float _ -> kern
    | Double _ -> kern
    | Custom _ -> kern
    | IntVecAcc (k1,k2) ->
      (match k2 with
       |	Seq (k3,k4) ->
         Seq(k3, IntVecAcc (aux k1, aux k4))
       | _ -> IntVecAcc (aux k1, aux k2))
    | Local (k1,k2) ->
      Local (aux k1, aux k2)
    | Ife (k1,k2,k3) ->
      Ife (aux k1, aux k2, aux k3)
    | If (k1,k2) ->
       If (aux k1, aux k2)
    | Not (k) ->
       Not (aux k)
    | Or (k1,k2) ->
      Or (aux k1, aux k2)
    | And (k1,k2) ->
      And (aux k1, aux k2)
    | EqBool (k1,k2) ->
      EqBool (aux k1, aux k2)
    | EqCustom (n,k1,k2) ->
      EqCustom (n,aux k1, aux k2)
    | LtBool (k1,k2) ->
      LtBool (aux k1, aux k2)
    | GtBool (k1,k2) ->
      GtBool (aux k1, aux k2)
    | LtEBool (k1,k2) ->
      LtEBool (aux k1, aux k2)
    | GtEBool (k1,k2) ->
      GtEBool (aux k1, aux k2)
    | DoLoop (k1, k2, k3, k4) ->
      DoLoop (aux k1, aux k2, aux k3, aux k4)
    | Arr (l,t,s,m) -> Arr (l,t,s,m)
    | While (k1, k2) ->
      While (aux k1, aux k2)
    | App (a,b) -> App (aux a, (Array.map aux b))
    | GlobalFun (a,b,n) -> GlobalFun (aux a, b,n)
    | Unit -> kern
    | Match (s,a,b) -> Match (s,aux a,
                              Array.map (fun (i,ofid,e) -> (i,ofid,aux e)) b)
    | CustomVar _ -> kern


  in
  let kern = ref (aux ker) in
  while (!b) do
    b := false;
    kern := aux !kern;
  done;
  !kern
;;



let return_v = ref ("","")

let save file string =
  let channel = open_out file in
  output_string channel string;
  close_out channel;;

let load_file f =
  let ic = open_in f in
  let n = in_channel_length ic in
  let s = String.make  n ' ' in
  really_input ic s 0 n;
  close_in ic;
  (s)



    
let gen_profile ker dev =
  let kir,k = ker in
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in
  return_v := "","";
  let k' = ((Kirc_Profile.parse 0 (fst k3) dev),
            ( match  (fst k3) with
              | IntVar (i,s) | FloatVar (i,s) | DoubleVar (i,s) -> s (*"sspoc_var"^(string_of_int i)^*)^" = "
              | Unit -> ""
              | SetV _ -> ""
              | IntVecAcc _-> ""
              | VecVar _ -> ""
              | _ -> (debug_print
                        (kir,
                         {ml_kern = k1;
                          body = fst k3;
                          ret_val = k3;
                          extensions = k.extensions});  Pervasives.flush stdout; assert false) ))
  in
  Printf.fprintf Spoc.Trac.fileOutput "{\n \"type\":\"profile_kernel\",\n \
                                        \"kernel_id\":%d,\n \
                                        \"source\":\"%s\"\n \
                                       },\n" (!Spoc.Trac.eventId - 1 )(Kirc_Profile.parse 0 (k2) dev)

let gen ?profile:(prof=false) ?return:(r=false) ?only:(o=Devices.Both) ((ker: ('a, 'b, 'c,'d,'e) sarek_kernel)) dev =
  let kir,k = ker in
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in
  return_v := "","";
  let k' = ((Kirc_Cuda.parse ~profile:prof 0 (fst k3) dev),
            ( match  (fst k3) with
              | IntVar (i,s) | FloatVar (i,s) | DoubleVar (i,s) -> s (*"sspoc_var"^(string_of_int i)^*)^" = "
              | Unit -> ""
              | SetV _ -> ""
              | IntVecAcc _-> ""
              | VecVar _ -> ""
              | _ -> (debug_print
                        (kir,
                         {ml_kern = k1;
                          body = fst k3;
                          ret_val = k3;
                          extensions = k.extensions});  Pervasives.flush stdout; assert false) ))
  in

  if r then
    (
      Kirc_Cuda.return_v := k';
      Kirc_OpenCL.return_v := k';
    );

  let gen_cuda () =
    let cuda_head =
      Array.fold_left
        (fun header extension ->
           match extension with
           | ExFloat32 -> header
           | ExFloat64 -> cuda_float64^header) cuda_head k.extensions in
    let src = Kirc_Cuda.parse ~profile:prof 0 (rewrite k2) dev in
    let global_funs = ref "" in
    Hashtbl.iter (fun _ a -> global_funs := !global_funs^(fst a)^"\n") Kirc_Cuda.global_funs;
    let constructors = List.fold_left (fun a b -> "__device__ "^b^a) "\n\n" !constructors in
    save "kirc_kernel.cu" (cuda_head ^ constructors ^  !global_funs ^ src) ;
    ignore(Sys.command ("nvcc --gpu-architecture=sm_30 -m64  -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"));
    let s = (load_file "kirc_kernel.ptx") in
    kir#set_cuda_sources s;
    (*ignore(Sys.command "rm kirc_kernel.cu kirc_kernel.ptx");*)

  and gen_opencl prof =
    let opencl_head =
      Array.fold_left
        (fun header extension ->
           match extension with
           | ExFloat32 -> header
           | ExFloat64 -> opencl_float64^header) opencl_head k.extensions in
    let src = Kirc_OpenCL.parse ~profile:prof 0 (rewrite k2) dev in
    let global_funs = ref "/************* FUNCTION DEFINITIONS ******************/\n"  in
    Hashtbl.iter (fun _ a -> global_funs := !global_funs ^ "\n" ^ (fst a) ^ "\n" ) Kirc_OpenCL.global_funs;
    let constructors =  "/************* CUSTOM TYPES ******************/\n" ^
                        List.fold_left (fun a b -> b^a) "\n\n" !constructors in
    let protos = "/************* FUNCTION PROTOTYPES ******************/\n" ^
                 List.fold_left (fun a b -> b^";\n"^a) "" !Kirc_OpenCL.protos in
    let clkernel = ((if prof then
                       "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
                     else "")^
                    opencl_head ^
                    (if prof then
                      "void spoc_atomic_add(__global ulong *a, ulong b){ atom_add(a, (ulong)b);}\n"
                    else "") ^
                    constructors ^ protos ^ !global_funs ^  src)  in
    save "kirc_kernel.cl" clkernel;
    kir#set_opencl_sources clkernel;

  in
  begin
    match o with
    | Devices.Both ->
      ignore(Kirc_Cuda.get_profile_counter());
      gen_cuda ();
      ignore(Kirc_OpenCL.get_profile_counter());
      gen_opencl prof;
    | Devices.Cuda ->
      ignore(Kirc_Cuda.get_profile_counter());
      gen_cuda ()
    | Devices.OpenCL ->
      ignore(Kirc_OpenCL.get_profile_counter());
      gen_opencl prof
  end;
  kir#reset_binaries ();
  kir,k

let arg_of_vec v  =
  match Vector.kind v with
  | Vector.Int32 _ -> Kernel.VInt32 v
  | Vector.Float32 _ -> Kernel.VFloat32 v
  | Vector.Int64 _ -> Kernel.VInt64 v
  | _ -> assert false


let run ?recompile:(r=false) ((ker: ('a, 'b, 'c,'d,'e) sarek_kernel)) a b q dev =
  let kir,k = ker in
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ ->
     if r then
       ignore(gen ~only:Devices.Cuda (kir,k) dev)
     else
       begin
         match kir#get_cuda_sources () with
         | [] -> ignore(gen  ~only:Devices.Cuda (kir,k) dev)
         | _ -> ()
       end
   | Devices.OpenCLInfo _ ->
     begin
       if r then
         ignore(gen  ~only:Devices.OpenCL (kir,k) dev)
       else
         match kir#get_opencl_sources () with
         | [] -> ignore(gen  ~only:Devices.OpenCL (kir,k) dev)
         | _ -> ()
     end);
  kir#run a b q dev

let profile_run ?recompile:(r=true) ((ker: ('a, 'b, 'c,'d,'e) sarek_kernel)) a b q dev =
  let kir,k = ker in
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ ->
     if r then
       ignore(gen ~profile:true ~only:Devices.Cuda (kir,k) dev)
     else
       begin
         match kir#get_cuda_sources () with
         | [] -> ignore(gen ~profile:true ~only:Devices.Cuda (kir,k) dev)
         | _ -> ()
       end
   | Devices.OpenCLInfo _ ->
     begin
       if r then
         ignore(gen ~profile:true ~only:Devices.OpenCL (kir,k) dev)
       else
         match kir#get_opencl_sources () with
         | [] -> ignore(gen ~profile:true ~only:Devices.OpenCL (kir,k) dev)
         | _ -> ()
     end);
  (*kir#run a b q dev;*)
  let nCounter =
    !(match dev.Devices.specific_info with
     | Devices.CudaInfo _ ->
       Kirc_Cuda.profiler_counter;
     | Devices.OpenCLInfo _ ->
       Kirc_OpenCL.profiler_counter;
     )
  in
  Printf.printf "Number of counters : %d\n\!" nCounter;
  let profiler_counters =  Vector.create Vector.int64 nCounter in
  for i = 0 to nCounter - 1 do
    Mem.set profiler_counters i 0L;
  done;
  (
    let args = kir#args_to_list a in
    let offset = ref 0 in
    kir#compile ~debug:true dev;
    let block,grid = b in 
    let bin = (Hashtbl.find (kir#get_binaries ()) dev) in
    match dev.Devices.specific_info with
    | Devices.CudaInfo cI ->
      let extra = Kernel.Cuda.cuda_create_extra ((Array.length args) + 1) in
      Kernel.Cuda.cuda_load_arg offset extra dev bin 0 (arg_of_vec profiler_counters);
      Array.iteri (Kernel.Cuda.cuda_load_arg offset extra dev bin) args;
      Kernel.Cuda.cuda_launch_grid offset bin grid block extra dev.Devices.general_info 0;

    | Devices.OpenCLInfo _ ->
      Kernel.OpenCL.opencl_load_arg offset dev bin 0 (arg_of_vec profiler_counters);
      Array.iteri (Kernel.OpenCL.opencl_load_arg offset dev bin) args;
      Kernel.OpenCL.opencl_launch_grid bin grid block dev.Devices.general_info 0
  );
  
  Spoc.Tools.iter (fun a -> Printf.printf "%Ld " a) profiler_counters;
  Gen.profile_vect := profiler_counters;
  gen_profile ker dev;
;;
  
  
let compile_kernel_to_files s ((ker: ('a, 'b, 'c,'d,'e) sarek_kernel)) dev =
  let kir,k = ker in
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in
  return_v := "","";
  let k' = ((Kirc_Cuda.parse 0 (fst k3)) dev,
            ( match  (fst k3) with
              | IntVar (i,s) | FloatVar (i,s) | DoubleVar (i,s) -> s^(*"spoc_var"^(string_of_int i)^*)" = "
              | Unit -> ""
              | SetV _ -> ""
              | IntVecAcc _-> ""
              | VecVar _ -> ""
              | _ -> (debug_print
                        (kir,
                         {ml_kern = k1;
                          body = fst k3;
                          ret_val = k3;
                          extensions = k.extensions});  Pervasives.flush stdout; assert false) ))
  in

  Kirc_Cuda.return_v := k';
  Kirc_OpenCL.return_v := k';
  let cuda_head =
    Array.fold_left
      (fun header extension ->
         match extension with
         | ExFloat32 -> header
         | ExFloat64 -> cuda_float64^header) cuda_head k.extensions in
  let opencl_head =
    Array.fold_left
      (fun header extension ->
         match extension with
         | ExFloat32 -> header
         | ExFloat64 -> opencl_float64^header) opencl_head k.extensions in
  save (s^".cu") (cuda_head^(Kirc_Cuda.parse 0 (rewrite k2) dev)) ;
  save (s^".cl") (opencl_head^(Kirc_OpenCL.parse 0 (rewrite k2) dev))




module Std =
struct
  let thread_idx_x = 1l
  let thread_idx_y = 1l
  let thread_idx_z = 1l
  let block_idx_x = 1l
  let block_idx_y = 1l
  let block_idx_z = 1l
  let block_dim_x = 1l
  let block_dim_y = 1l
  let block_dim_z = 1l

  let global_thread_id = 0l
  let return () = ()

  let float64 i = float (Int32.to_int i)
  let float i = float (Int32.to_int i)
  let int_of_float64 f = Int32.of_int (int_of_float f)
  let int_of_float f = Int32.of_int (int_of_float f)

  let block_barrier () = ()

  let make_shared i = Array.make (Int32.to_int i) 0l
  let make_local i = Array.make (Int32.to_int i) 0l
end

module Math =
struct

  let rec pow a  b =
    let rec aux a = function
      | 0 -> 1
      | 1 -> a
      | n ->
        let b = aux a (n / 2) in
        b * b * (if n mod 2 = 0 then 1 else a)
    in
    Int32.of_int (aux (Int32.to_int a) (Int32.to_int b))

  let logical_and = fun a b -> Int32.logand a b
  let xor = fun a b -> Int32.logxor a b

  module Float32 =
  struct
    let add = (+.)
    let minus = (-.)
    let mul = ( *. )
    let div = (/.)

    let pow = ( ** )
    let sqrt = Pervasives.sqrt
    let exp = Pervasives.exp
    let log = Pervasives.log
    let log10 = Pervasives.log10
    let expm1 = Pervasives.expm1
    let log1p = Pervasives.log1p

    let acos = Pervasives.acos
    let cos = Pervasives.cos
    let cosh = Pervasives.cosh
    let asin = Pervasives.asin
    let sin = Pervasives.sin
    let sinh = Pervasives.sinh
    let tan = Pervasives.tan
    let tanh = Pervasives.tanh
    let atan = Pervasives.atan
    let atan2 = Pervasives.atan2
    let hypot = Pervasives.hypot

    let ceil = Pervasives.ceil
    let floor = Pervasives.floor

    let abs_float = Pervasives.abs_float
    let copysign = Pervasives.copysign
    let modf = Pervasives.modf

    let zero = 0.
    let one = 1.

    let make_shared i = Array.make (Int32.to_int i) 0.
    let make_local i = Array.make (Int32.to_int i) 0.
  end

  module Float64 =
  struct
    let add = (+.)
    let minus = (-.)
    let mul = ( *. )
    let div = (/.)

    let pow = ( ** )
    let sqrt = Pervasives.sqrt
    let exp = Pervasives.exp
    let log = Pervasives.log
    let log10 = Pervasives.log10
    let expm1 = Pervasives.expm1
    let log1p = Pervasives.log1p

    let acos = Pervasives.acos
    let cos = Pervasives.cos
    let cosh = Pervasives.cosh
    let asin = Pervasives.asin
    let sin = Pervasives.sin
    let sinh = Pervasives.sinh
    let tan = Pervasives.tan
    let tanh = Pervasives.tanh
    let atan = Pervasives.atan
    let atan2 = Pervasives.atan2
    let hypot = Pervasives.hypot

    let ceil = Pervasives.ceil
    let floor = Pervasives.floor

    let abs_float = Pervasives.abs_float
    let copysign = Pervasives.copysign
    let modf = Pervasives.modf

    let zero = 0.
    let one = 1.
    let of_float32 f = f
    let of_float f = f
    let to_float32 f = f
    let make_shared i = Array.make (Int32.to_int i) 0.
    let make_local i = Array.make (Int32.to_int i) 0.
  end
end
