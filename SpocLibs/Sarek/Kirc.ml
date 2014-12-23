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

type ('a,'b,'c) kirc_function =
  { 
    ml_fun : 'a;
    funbody : Kirc_Ast.k_ext;
    fun_ret : Kirc_Ast.k_ext* ('b,'c) Vector.kind;
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
  "int spoc_xor (int a, int b ) { return (a^b);}\n")

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
  "__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"
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
let var i = IntId (("spoc_var"^(string_of_int i)), i)
let spoc_gen_kernel args body = Kern (args,body)
let spoc_fun_kernel a b = () 
let global_fun a = GlobalFun (a.funbody, match snd a.fun_ret with
					 | Vector.Int32 _  -> "int"
					 | Vector.Float32 _  -> "float"
					 | _ -> "void")
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
let new_int_var i = IntVar i 
let new_float_var i = FloatVar i 
let new_float64_var i = DoubleVar i 
let new_double_var i = DoubleVar i 
let new_unit_var i = UnitVar i
let new_custom_var n v = Custom (n,v)  (* <--- *)

let new_int_vec_var v = VecVar (Int 0, v) 
let new_float_vec_var v = VecVar (Float 0., v) 
let new_double_vec_var v = VecVar (Double 0., v) 
let new_custom_vec_var n v = VecVar (Custom (n,0), v)  (* <--- *)

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
let double_var d = CastDoubleVar d

let equals a b = EqBool (a,b)
let equals_custom s v1 v2 = EqCustom (s,v1,v2)
let equals32 a b = EqBool (a,b)
let equals64 a b = EqBool (a,b)
let equalsF a b = EqBool (a,b)
let equalsF64 a b = EqBool (a,b)
let b_or a b = Or (a,b)
let b_and a b = And (a,b)

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

let return_int i = IntVar i
let return_float f = FloatVar f
let return_double d = DoubleVar d





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
    | VecVar (k,idx ) ->
      VecVar (aux k, idx) 
    | Concat (k1,k2) -> 
      Concat (aux k1, aux k2)
    | Constr (t,c,l) ->
      Constr (t, c, List.map aux l)
    | Record (t,l) ->
      Record (t, List.map aux l)
    | RecGet (r,s) -> RecGet (aux r,s)
    | RecSet (r,v) -> RecSet (aux r, aux v)
    | Empty -> kern
    | Seq (k1,k2) -> 
      Seq (aux k1, aux k2)
    | Return k -> 
      (match k with
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
         Return (aux k))
    | Acc (k1,k2) ->
      Acc (aux k1, aux k2)
    | Set (k1,k2) ->
      Set (aux k1, aux k2)
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
    | GlobalFun (a,b) -> GlobalFun (aux a, b)
    | Unit -> kern
    | Match (s,a,b) -> Match (s,aux a, 
                              Array.map (fun (i,ofid,e) -> (i,ofid,aux e)) b)
    

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

let gen ?return:(r=false) ?only:(o=Devices.Both) ((ker: ('a, 'b, 'c,'d,'e) sarek_kernel)) =
  let kir,k = ker in
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in
  return_v := "","";
  let k' = ((Kirc_Cuda.parse 0 (fst k3)),
            ( match  (fst k3) with
              | IntVar i | FloatVar i | DoubleVar i -> "spoc_var"^(string_of_int i)^" = "
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
    let src = Kirc_Cuda.parse 0 (rewrite k2) in
    let global_funs = ref "" in
    Hashtbl.iter (fun _ a -> global_funs := !global_funs^(fst a)^"\n") Kirc_Cuda.global_funs;
    let constructors = List.fold_left (fun a b -> "__device__ "^b^a) "\n\n" !constructors in
    save "kirc_kernel.cu" (cuda_head ^ constructors ^  !global_funs ^ src) ;
    ignore(Sys.command ("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"));
    let s = (load_file "kirc_kernel.ptx") in
    kir#set_cuda_sources s;
    ignore(Sys.command "rm kirc_kernel.cu kirc_kernel.ptx"); 

  and gen_opencl () =
    let opencl_head = 
      Array.fold_left 
        (fun header extension -> 
           match extension with
           | ExFloat32 -> header
           | ExFloat64 -> opencl_float64^header) opencl_head k.extensions in		
    let src = Kirc_OpenCL.parse 0 (rewrite k2) in
    let global_funs = ref "" in
    Hashtbl.iter (fun _ a -> global_funs := !global_funs ^ "\n" ^ (fst a) ^ "\n" ) Kirc_OpenCL.global_funs;
    let constructors = List.fold_left (fun a b -> b^a) "\n\n" !constructors in
    let clkernel = (opencl_head ^ constructors ^ !global_funs ^  src)  in
    (*save "kirc_kernel.cl" clkernel;*)
    kir#set_opencl_sources clkernel;

  in
  begin
    match o with
    | Devices.Both -> gen_cuda (); gen_opencl();
    | Devices.Cuda -> gen_cuda ()
    | Devices.OpenCL -> gen_opencl ()
  end;
  kir#reset_binaries ();
  kir,k


let run ?recompile:(r=false) ((ker: ('a, 'b, 'c,'d,'e) sarek_kernel)) a b q dev = 
  let kir,k = ker in
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ ->
     if r then
       ignore(gen ~only:Devices.Cuda (kir,k))
     else
       begin
         match kir#get_cuda_sources () with
         | [] -> ignore(gen ~only:Devices.Cuda (kir,k))
         | _ -> ()
       end
   | Devices.OpenCLInfo _ ->
     begin
       if r then
         ignore(gen ~only:Devices.OpenCL (kir,k))
       else
         match kir#get_opencl_sources () with
         | [] -> ignore(gen ~only:Devices.OpenCL (kir,k))
         | _ -> ()
     end);
  kir#run a b q dev


let compile_kernel_to_files s ((ker: ('a, 'b, 'c,'d,'e) sarek_kernel)) =
  let kir,k = ker in 
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in
  return_v := "","";
  let k' = ((Kirc_Cuda.parse 0 (fst k3)),
            ( match  (fst k3) with
              | IntVar i | FloatVar i | DoubleVar i -> "spoc_var"^(string_of_int i)^" = "
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
  save (s^".cu") (cuda_head^(Kirc_Cuda.parse 0 (rewrite k2))) ;
  save (s^".cl") (opencl_head^(Kirc_OpenCL.parse 0 (rewrite k2)))




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
    let to_float32 f = f
    let make_shared i = Array.make (Int32.to_int i) 0.
    let make_local i = Array.make (Int32.to_int i) 0.
  end
end




(************** Composition ****************)

let a_to_vect = function
  | IntVar i  ->  (new_int_vec_var ( i))
  | FloatVar i -> (new_float_vec_var (i))
  | a  -> print_ast a; failwith "a_to_vect"

let a_to_return_vect k1 k2 idx= 
  match k1 with
  | IntVar i  ->  (set_vect_var (get_vec (var i) idx) (k2))
  | FloatVar i  ->  (set_vect_var (get_vec (var i) idx) (k2))
  | _  -> failwith "error a_to_return_vect"

let param_list = ref []


let add_to_param_list a = 
  param_list := a :: !param_list

let rec check_and_transform_to_map a = 
  match a with 
  | Plus (b,c)  -> Plus(check_and_transform_to_map b, check_and_transform_to_map c)
  | Min (b,c)  -> Min(check_and_transform_to_map b, check_and_transform_to_map c)
  | Mul (b,c)  -> Mul(check_and_transform_to_map b, check_and_transform_to_map c)
  | Mod (b,c)  -> Mod(check_and_transform_to_map b, check_and_transform_to_map c)
  | Div (b,c)  -> Div(check_and_transform_to_map b, check_and_transform_to_map c)
  | IntId (v,i)  -> 
    if (List.mem i !param_list) then
      IntVecAcc(IdName ("spoc_var"^(string_of_int i)), 
                Intrinsics ("blockIdx.x*blockDim.x+threadIdx.x","get_global_id (0)"))
      (*(IntId ("spoc_global_id", -1)))*)
    else 
      a
  | _  -> a


let arg_of_vec v  = 
  match Vector.kind v with
  | Int32 _ -> VInt32 v
  | Float32 _ -> VFloat32 v
  | _ -> assert false


let propagate f = function
  | Block b -> Block (f b)
  | Return a  -> Return (f a)
  | Seq (a,b)  -> Seq (f a,  f b)
  | Local (a,b) -> Local (f a, f b)
  | Plus (a,b)  -> Plus (f a, f b)
  | Min (a,b)  -> Min  (f a, f b)
  | Mul (a,b)  -> Mul  (f a, f b)
  | Div (a,b)  -> Div  (f a, f b)
  | Mod (a,b)  -> Mod  (f a, f b)
  | LtBool (a,b)  -> LtBool (f a, f b)
  | GtBool (a,b)  -> GtBool (f a, f b)
  | Ife (a,b,c)  -> Ife (f a, f b, f c)
  | IntId (v,i)  -> IntId (v, i)
  | Kern (a, b) -> Kern (f a, f b)
  | Params a -> Params (f a)
  | Plusf (a, b) -> Plusf (f a, f b)
  | Minf (a, b) -> Minf (f a, f b)
  | Mulf (a, b) -> Mulf (f a, f b)
  | Divf (a, b) -> Divf (f a, f b)
  | Id a -> Id a
  | IdName a -> IdName a
  | IntVar i -> IntVar i
  | FloatVar i -> FloatVar i
  | UnitVar i -> UnitVar i
  | CastDoubleVar i -> assert false
  | DoubleVar i -> DoubleVar i
  | Arr (i, s, t, m) -> Arr (i, f s, t, m)
  | VecVar (a, i) -> VecVar (f a, i)
  | Concat (a, b) -> Concat (f a, f b)
  | Empty -> Empty
  | Set (a, b) -> Set (f a, f b)
  | Decl a -> Decl (f a)
  | SetV (a, b) -> SetV (f a, f b)
  | SetLocalVar (a, b, c) -> SetLocalVar (f a, f b, f c)
  | Intrinsics intr -> Intrinsics intr
  | Int i -> Int i
  | Float f -> Float f
  | Double d -> Double d
  | IntVecAcc (a, b) -> IntVecAcc (f a, f b)
  | Acc (a, b) -> Acc (f a, f b) 
  | If (a, b) -> If (f a, f b)
  | Or (a, b) -> Or (f a, f b)
  | And (a, b) -> And (f a, f b)
  | EqBool (a, b) -> EqBool (f a, f b)
  | EqCustom (n,a, b) -> EqCustom (n,f a, f b)
  | LtEBool (a, b) -> LtEBool (f a, f b)
  | GtEBool (a, b) -> GtEBool (f a, f b)
  | DoLoop (a, b, c, d) -> DoLoop (f a, f b, f c, f d)
  | While (a, b) -> While (f a,f b)
  | App (a, b) -> App ((f a), Array.map f b)
  | GInt foo -> GInt foo
  | GFloat foo -> GFloat foo
  | Unit -> Unit
  | GlobalFun (a,b) -> GlobalFun (f a, b)
  | Constr (a,b,c) -> Constr (a,b,List.map f c)
  | Record (a,c) -> Record (a,List.map f c)
  | RecGet (r,s) -> RecGet (f r, s)
  | RecSet (r,v) -> RecSet (f r, f v)
  | Custom (s,i) -> Custom (s,i)
  | Match (s,a,b) -> Match (s,f a, 
                            Array.map (fun (i,ofid,e) -> (i,ofid,f e)) b)


let map ((ker: ('a, 'b, ('c -> 'd), 'e,'f) sarek_kernel)) ?dev:(device=(Spoc.Devices.init ()).(0)) (vec_in : ('g, 'h) Vector.vector) : ('i, 'j) Vector.vector= 
  let ker2,k = ker in 
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in 
  param_list := [];
  let rec aux = function
    | Kern (args, body)  ->  
      let new_args = 
        match args with
        | Params p ->
          (match p with 
           | Concat (Concat _, _) ->
             failwith "error multiple map args ";  
           | Concat (a, Empty)  ->
             params (concat (a_to_vect a) (concat (a_to_vect (fst k3)) (empty_arg ()))) 
           | _ -> failwith "map type error")
        | _  -> failwith "error map args"
      in let n_body =
        let rec aux curr =
          match curr with
          | Return a  -> a_to_return_vect (fst k3) (aux a) ((intrinsics "blockIdx.x*blockDim.x+threadIdx.x" "get_global_id(0)"))
          | Seq (a,b)  -> seq a (aux b)
          | Local (a,b) -> Local (a, aux b)
          | Plus (a,b)  -> Plus (aux a, aux b)
          | Min (a,b)  -> Min  (aux a, aux b)
          | Mul (a,b)  -> Mul  (aux a, aux b)
          | Div (a,b)  -> Div  (aux a, aux b)
          | Mod (a,b)  -> Mod  (aux a, aux b)
          | LtBool (a,b)  -> LtBool (aux a, aux b)
          | GtBool (a,b)  -> GtBool (aux a, aux b)
          | Ife (a,b,c)  -> Ife (aux a, aux b, aux c)
          | IntId (v,i)  -> 
            if i = 0 then
              IntVecAcc(IdName ("spoc_var"^(string_of_int i)), 
                        Intrinsics ("blockIdx.x*blockDim.x+threadIdx.x","get_global_id (0)"))
            else
              curr 
          | a -> print_ast a; assert false
        in
        aux body
      in
      Kern (new_args, n_body)
    | _ -> failwith "malformed kernel for map"   
  in 
  let res =(ker2,
            { 
              ml_kern = Tools.map (k1) (snd k3);
              body = aux k2;
              ret_val = Unit, Vector.int32;
              extensions = k.extensions;
            })
  in 
  let length = Vector.length vec_in in
  let vec_out =
    (Vector.create (snd k3)  ~dev:device length)
  in
  Mem.to_device vec_in device;
  let spoc_ker, kir_ker = gen res in
  let block = {blockX = 1; blockY = 1; blockZ = 1}
  and grid = {gridX = 1; gridY = 1; gridZ = 1}
  in spoc_ker#compile device;
  begin
    let open Devices in( 
      match device.Devices.specific_info with
      | Devices.CudaInfo cI -> 
        if Vector.length vec_in < 
           (cI.maxThreadsDim.x) then
          (
            grid.gridX <- 1;
            block.blockX <- (Vector.length vec_in)
          )
        else
          (
            block.blockX <- cI.maxThreadsDim.x;
            grid.gridX <- (Vector.length vec_in) / cI.maxThreadsDim.x;
          )
      | Devices.OpenCLInfo oI -> 
        if Vector.length vec_in < oI.Devices.max_work_item_size.Devices.x then
          (
            grid.gridX <- 1;
            block.blockX <- Vector.length vec_in
          )
        else
          (
            block.blockX <- oI.Devices.max_work_item_size.Devices.x;
            grid.gridX <- (Vector.length vec_in) / block.blockX
          )
    )
  end;
  let bin = (Hashtbl.find (spoc_ker#get_binaries ()) device) in
  let offset = ref 0 in
  let extra = Kernel.Cuda.cuda_create_extra 2 in
  (match device.Devices.specific_info with
   | Devices.CudaInfo cI ->
     Kernel.Cuda.cuda_load_arg offset extra device bin 0 (arg_of_vec vec_in);
     Kernel.Cuda.cuda_load_arg offset extra device bin 1 (arg_of_vec vec_out);
     Kernel.Cuda.cuda_launch_grid offset bin grid block extra device.Devices.general_info 0;
   | Devices.OpenCLInfo _ ->
     let clFun = bin in
     let offset = ref 0
     in
     Kernel.OpenCL.opencl_load_arg offset device clFun 0 (arg_of_vec vec_in);
     Kernel.OpenCL.opencl_load_arg offset device clFun 1 (arg_of_vec vec_out);
     Kernel.OpenCL.opencl_launch_grid clFun grid block device.Devices.general_info 0
  );					
  vec_out

let map2 ((ker: ('a, 'b,('c -> 'd -> 'e), 'f,'g) sarek_kernel)) ?dev:(device=(Spoc.Devices.init ()).(0)) (vec_in1 : ('h, 'i) Vector.vector) (vec_in2 : ('j, 'k) Vector.vector) : ('l, 'm) Vector.vector = 
  let ker2,k = ker in
  let (k1,k2,k3) = (k.ml_kern, k.body,k.ret_val) in 
  param_list := [];
  let rec aux = function
    | Kern (args, body)  ->  
      let new_args = 
        match args with
        | Params p ->
          (match p with 
           | Concat (Concat _, Concat _) ->
             failwith "error multiple map2 args ";  
           | Concat (a, Concat (b, Empty)) ->
             params (concat (a_to_vect a) (concat (a_to_vect b) (concat (a_to_vect (fst k3)) (empty_arg ()))))
           | Concat (a, Empty)  ->
             failwith "error too fex map2 args ";  
           | _ -> Printf.printf "+++++> "; print_ast args; failwith "map2 type error")
        | _  -> failwith "error map2 args"
      in let n_body =
        let rec aux curr =
          match curr with
          | Return a  -> a_to_return_vect (fst k3) (aux a) ((intrinsics "blockIdx.x*blockDim.x+threadIdx.x" "get_global_id(0)"))
          | Seq (a,b)  -> seq a (aux b)
          | Local (a,b) -> Local (aux a, aux b)
          | Plus (a,b)  -> Plus (aux a, aux b)
          | Min (a,b)  -> Min  (aux a, aux b)
          | Mul (a,b)  -> Mul  (aux a, aux b)
          | Div (a,b)  -> Div  (aux a, aux b)
          | Mod (a,b)  -> Mod  (aux a, aux b)
          | LtBool (a,b)  -> LtBool (aux a, aux b)
          | GtBool (a,b)  -> GtBool (aux a, aux b)
          | Ife (a,b,c)  -> Ife (aux a, aux b, aux c)
          | IntId (v,i)  -> 
            if i = 0 || i = 1 then
              IntVecAcc(IdName ("spoc_var"^(string_of_int i)), 
                        Intrinsics ("blockIdx.x*blockDim.x+threadIdx.x","get_global_id (0)"))
            else
              curr 
          | a -> print_ast a; propagate aux a
        in
        aux body
      in
      Kern (new_args, n_body)
    | _ -> failwith "malformed kernel for map2"   
  in 
  let res =(ker2,
            { 
              ml_kern = 
                (let map2 = fun f k a b ->
                   let c = Vector.create k (Vector.length a) in
                   for i = 0 to (Vector.length a -1) do
                     Mem.unsafe_set c i (f (Mem.unsafe_get a i) (Mem.unsafe_get b i)) 
                   done;
                   c
                 in	map2 (k1) (snd k3));
              body = aux k2;
              ret_val = Unit, Vector.int32;
              extensions = k.extensions;
            })
  in 
  let length = Vector.length vec_in1 in
  let vec_out =
    (Vector.create (snd k3)  ~dev:device length)
  in
  Mem.to_device vec_in1 device;
  Mem.to_device vec_in2 device;
  let framework = 
    let open Devices in
      match device.Devices.specific_info with
      | Devices.CudaInfo cI -> Devices.Cuda
      | _ -> Devices.OpenCL in

  let spoc_ker, kir_ker = gen ~only:framework res  in
  let block = {blockX = 1; blockY = 1; blockZ = 1}
  and grid = {gridX = 1; gridY = 1; gridZ = 1}
  in spoc_ker#compile  device;
  begin
    let open Devices in( 
      match device.Devices.specific_info with
      | Devices.CudaInfo cI -> 
        if length < 
           (cI.maxThreadsDim.x) then
          (
            grid.gridX <- 1;
            block.blockX <- (length)
          )
        else
          (
            block.blockX <- cI.maxThreadsDim.x;
            grid.gridX <- (length) / cI.maxThreadsDim.x;
          )
      | Devices.OpenCLInfo oI -> 
        if length < oI.Devices.max_work_item_size.Devices.x then
          (
            grid.gridX <- 1;
            block.blockX <- length
          )
        else
          (
            block.blockX <- oI.Devices.max_work_item_size.Devices.x;
            grid.gridX <- (length) / block.blockX
          )
    )
  end;
  let bin = (Hashtbl.find (spoc_ker#get_binaries ()) device) in
  let offset = ref 0 in
  (match device.Devices.specific_info with
   | Devices.CudaInfo cI ->
     let extra = Kernel.Cuda.cuda_create_extra 2 in
     Kernel.Cuda.cuda_load_arg offset extra device bin 0 (arg_of_vec vec_in1);
     Kernel.Cuda.cuda_load_arg offset extra device bin 1 (arg_of_vec vec_in2);
     Kernel.Cuda.cuda_load_arg offset extra device bin 2 (arg_of_vec vec_out);
     Kernel.Cuda.cuda_launch_grid offset bin grid block extra device.Devices.general_info 0;
   | Devices.OpenCLInfo _ ->
     let clFun = bin in
     let offset = ref 0
     in
     Kernel.OpenCL.opencl_load_arg offset device clFun 0 (arg_of_vec vec_in1);
     Kernel.OpenCL.opencl_load_arg offset device clFun 1 (arg_of_vec vec_in2);
     Kernel.OpenCL.opencl_load_arg offset device clFun 2 (arg_of_vec vec_out);
     Kernel.OpenCL.opencl_launch_grid clFun grid block device.Devices.general_info 0
  );                  
  vec_out		



  
