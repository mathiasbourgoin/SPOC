open Spoc
open Kirc
open Kirc_Ast
open Skeletons
(************** Composition ****************)

let a_to_vect = function
  | IntVar (i, s, _m) -> new_int_vec_var i s
  | FloatVar (i, s, _m) -> new_float_vec_var i s
  | a ->
      print_ast a ;
      failwith "a_to_vect"

let a_to_return_vect k1 k2 idx =
  match k1 with
  | IntVar (i, s, _m) -> set_vect_var (get_vec (var i s) idx) k2
  | FloatVar (i, s, _m) -> set_vect_var (get_vec (var i s) idx) k2
  | _ -> failwith "error a_to_return_vect"

let param_list = ref []

let add_to_param_list a = param_list := a :: !param_list

let rec check_and_transform_to_map a =
  match a with
  | Plus (b, c) ->
      Plus (check_and_transform_to_map b, check_and_transform_to_map c)
  | Min (b, c) ->
      Min (check_and_transform_to_map b, check_and_transform_to_map c)
  | Mul (b, c) ->
      Mul (check_and_transform_to_map b, check_and_transform_to_map c)
  | Mod (b, c) ->
      Mod (check_and_transform_to_map b, check_and_transform_to_map c)
  | Div (b, c) ->
      Div (check_and_transform_to_map b, check_and_transform_to_map c)
  | IntId (_v, i) ->
      if List.mem i !param_list then
        IntVecAcc
          ( IdName ("spoc_var" ^ string_of_int i),
            Intrinsics ("blockIdx.x*blockDim.x+threadIdx.x", "get_global_id (0)")
          ) (*(IntId ("spoc_global_id", -1)))*)
      else a
  | _ -> a

let arg_of_vec v =
  match Vector.kind v with
  | Vector.Int32 _ -> Kernel.VInt32 v
  | Vector.Float32 _ -> Kernel.VFloat32 v
  | _ -> assert false

let launch_kernel_with_args bin grid block kernel_arg_tab device =
  let offset = ref 0 in
  match device.Devices.specific_info with
  | Devices.CudaInfo _cI ->
      let extra = Kernel.Cuda.cuda_create_extra (Array.length kernel_arg_tab) in
      Array.iteri
        (fun i arg -> Kernel.Cuda.cuda_load_arg offset extra device bin i arg)
        kernel_arg_tab ;
      Kernel.Cuda.cuda_launch_grid
        offset
        bin
        grid
        block
        extra
        device.Devices.general_info
        0
  | Devices.OpenCLInfo _ ->
      let clFun = bin in
      Array.iteri
        (fun i arg -> Kernel.OpenCL.opencl_load_arg offset device clFun i arg)
        kernel_arg_tab ;
      Kernel.OpenCL.opencl_launch_grid
        clFun
        grid
        block
        device.Devices.general_info
        0

let compute_grid_block_1D device vec_in =
  let open Kernel in
  let block = {blockX = 1; blockY = 1; blockZ = 1}
  and grid = {gridX = 1; gridY = 1; gridZ = 1} in
  let open Devices in
  (match device.Devices.specific_info with
  | Devices.CudaInfo cI ->
      if Vector.length vec_in < cI.maxThreadsDim.x then
        block.blockX <- Vector.length vec_in
      else (
        block.blockX <- cI.maxThreadsDim.x ;
        grid.gridX <- Vector.length vec_in / cI.maxThreadsDim.x)
  | Devices.OpenCLInfo oI ->
      if Vector.length vec_in < oI.Devices.max_work_item_size.Devices.x then
        block.blockX <- Vector.length vec_in
      else (
        block.blockX <- oI.Devices.max_work_item_size.Devices.x ;
        grid.gridX <- Vector.length vec_in / block.blockX)) ;
  (grid, block)

let propagate f expr =
  match expr with
  | Block b -> Block (f b)
  | Return a -> Return (f a)
  | Seq (a, b) -> Seq (f a, f b)
  | Local (a, b) -> Local (f a, f b)
  | Plus (a, b) -> Plus (f a, f b)
  | Min (a, b) -> Min (f a, f b)
  | Mul (a, b) -> Mul (f a, f b)
  | Div (a, b) -> Div (f a, f b)
  | Mod (a, b) -> Mod (f a, f b)
  | LtBool (a, b) -> LtBool (f a, f b)
  | GtBool (a, b) -> GtBool (f a, f b)
  | Ife (a, b, c) -> Ife (f a, f b, f c)
  | IntId (v, i) -> IntId (v, i)
  | Kern (a, b) -> Kern (f a, f b)
  | Params a -> Params (f a)
  | Plusf (a, b) -> Plusf (f a, f b)
  | Minf (a, b) -> Minf (f a, f b)
  | Mulf (a, b) -> Mulf (f a, f b)
  | Divf (a, b) -> Divf (f a, f b)
  | Id a -> Id a
  | IdName _ | IntVar _ | FloatVar _ | DoubleVar _ | UnitVar _ -> expr
  | CastDoubleVar (_i, _s) -> assert false
  | Arr (i, s, t, m) -> Arr (i, f s, t, m)
  | VecVar (a, i, s) -> VecVar (f a, i, s)
  | Concat (a, b) -> Concat (f a, f b)
  | Empty -> Empty
  | Set (a, b) -> Set (f a, f b)
  | Decl a -> Decl (f a)
  | SetV (a, b) -> SetV (f a, f b)
  | SetLocalVar (a, b, c) -> SetLocalVar (f a, f b, f c)
  | Intrinsics intr -> Intrinsics intr
  | IntrinsicRef (path, name) -> IntrinsicRef (path, name)
  | Int i -> Int i
  | Float f -> Float f
  | Double d -> Double d
  | IntVecAcc (a, b) -> IntVecAcc (f a, f b)
  | Acc (a, b) -> Acc (f a, f b)
  | If (a, b) -> If (f a, f b)
  | Or (a, b) -> Or (f a, f b)
  | And (a, b) -> And (f a, f b)
  | EqBool (a, b) -> EqBool (f a, f b)
  | EqCustom (n, a, b) -> EqCustom (n, f a, f b)
  | LtEBool (a, b) -> LtEBool (f a, f b)
  | GtEBool (a, b) -> GtEBool (f a, f b)
  | DoLoop (a, b, c, d) -> DoLoop (f a, f b, f c, f d)
  | While (a, b) -> While (f a, f b)
  | App (a, b) -> App (f a, Array.map f b)
  | GInt foo -> GInt foo
  | GFloat foo -> GFloat foo
  | GIntVar n -> GIntVar n
  | GFloatVar n -> GFloatVar n
  | GFloat64Var n -> GFloat64Var n
  | NativeVar n -> NativeVar n
  | Unit -> Unit
  | GlobalFun (a, b, n) -> GlobalFun (f a, b, n)
  | Constr (a, b, c) -> Constr (a, b, List.map f c)
  | Record (a, c) -> Record (a, List.map f c)
  | RecGet (r, s) -> RecGet (f r, s)
  | RecSet (r, v) -> RecSet (f r, f v)
  | Custom (_s, _i, _ss) -> expr
  | Match (s, a, b) ->
      Match (s, f a, Array.map (fun (i, ofid, e) -> (i, ofid, f e)) b)
  | _ ->
      failwith
        ("Kirk Transform : "
        ^ Kirc_Ast.string_of_ast expr
        ^ " unimplemented yet")

let _map (ker : ('a, 'b, 'c -> 'd, 'e, 'f) sarek_kernel)
    ?dev:(device = (Spoc.Devices.init ()).(0)) (vec_in : ('c, 'h) Vector.vector)
    : ('d, 'j) Vector.vector =
  let ker2, k = ker in
  let k1, k2, k3 = (k.ml_kern, k.body, k.ret_val) in
  param_list := [] ;
  let aux = function
    | Kern (args, body) ->
        let new_args =
          match args with
          | Params p -> (
              match p with
              | Concat (Concat _, _) -> failwith "error multiple map args "
              | Concat (a, Empty) ->
                  params
                    (concat
                       (a_to_vect a)
                       (concat (a_to_vect (fst k3)) (empty_arg ())))
              | _ -> failwith "map type error")
          | _ -> failwith "error map args"
        in
        let n_body =
          let rec aux curr =
            match curr with
            | Return a ->
                a_to_return_vect
                  (fst k3)
                  (aux a)
                  (intrinsics
                     "blockIdx.x*blockDim.x+threadIdx.x"
                     "get_global_id(0)")
            | Seq (a, b) -> seq a (aux b)
            | Local (a, b) -> Local (a, aux b)
            | Plus (a, b) -> Plus (aux a, aux b)
            | Min (a, b) -> Min (aux a, aux b)
            | Mul (a, b) -> Mul (aux a, aux b)
            | Div (a, b) -> Div (aux a, aux b)
            | Mod (a, b) -> Mod (aux a, aux b)
            | LtBool (a, b) -> LtBool (aux a, aux b)
            | GtBool (a, b) -> GtBool (aux a, aux b)
            | Ife (a, b, c) -> Ife (aux a, aux b, aux c)
            | Int a -> Int a
            | IntId (_v, i) ->
                if i = 0 then
                  IntVecAcc
                    ( IdName ("spoc_var" ^ string_of_int i),
                      Intrinsics
                        ( "blockIdx.x*blockDim.x+threadIdx.x",
                          "get_global_id (0)" ) )
                else curr
            | a ->
                print_ast a ;
                assert false
          in
          aux body
        in
        Kern (new_args, n_body)
    | _ -> failwith "malformed kernel for map"
  in
  let res =
    ( ker2,
      {
        ml_kern = Tools.map k1 (snd k3);
        body = aux k2;
        ret_val = (Unit, Vector.int32);
        extensions = k.extensions;
      } )
  in
  let length = Vector.length vec_in in
  let vec_out = Vector.create (snd k3) ~dev:device length in
  Mem.to_device vec_in device ;
  let target =
    match device.Devices.specific_info with
    | Devices.CudaInfo _ -> Devices.Cuda
    | Devices.OpenCLInfo _ -> Devices.OpenCL
  in
  (*spoc_ker, kir_ker =*)
  ignore (gen ~only:target res device) ;
  let spoc_ker, _kir_ker = res in
  let open Kernel in
  let block = {blockX = 1; blockY = 1; blockZ = 1}
  and grid = {gridX = 1; gridY = 1; gridZ = 1} in
  spoc_ker#compile ~debug:true device ;
  begin
    let open Devices in
    match device.Devices.specific_info with
    | Devices.CudaInfo cI ->
        if Vector.length vec_in < cI.maxThreadsDim.x then (
          grid.gridX <- 1 ;
          block.blockX <- Vector.length vec_in)
        else (
          block.blockX <- cI.maxThreadsDim.x ;
          grid.gridX <- Vector.length vec_in / cI.maxThreadsDim.x)
    | Devices.OpenCLInfo oI ->
        if Vector.length vec_in < oI.Devices.max_work_item_size.Devices.x then (
          grid.gridX <- 1 ;
          block.blockX <- Vector.length vec_in)
        else (
          block.blockX <- oI.Devices.max_work_item_size.Devices.x ;
          grid.gridX <- Vector.length vec_in / block.blockX)
  end ;
  let bin = Hashtbl.find (spoc_ker#get_binaries ()) device in
  let offset = ref 0 in
  (match device.Devices.specific_info with
  | Devices.CudaInfo _cI ->
      let extra = Kernel.Cuda.cuda_create_extra 2 in
      Kernel.Cuda.cuda_load_arg offset extra device bin 0 (arg_of_vec vec_in) ;
      Kernel.Cuda.cuda_load_arg offset extra device bin 1 (arg_of_vec vec_out) ;
      Kernel.Cuda.cuda_launch_grid
        offset
        bin
        grid
        block
        extra
        device.Devices.general_info
        0
  | Devices.OpenCLInfo _ ->
      let clFun = bin in
      Kernel.OpenCL.opencl_load_arg offset device clFun 0 (arg_of_vec vec_in) ;
      Kernel.OpenCL.opencl_load_arg offset device clFun 1 (arg_of_vec vec_out) ;
      Kernel.OpenCL.opencl_launch_grid
        clFun
        grid
        block
        device.Devices.general_info
        0) ;
  vec_out

let map2 (ker : ('a, 'b, 'c -> 'd -> 'e, 'f, 'g) sarek_kernel)
    ?dev:(device = (Spoc.Devices.init ()).(0))
    (vec_in1 : ('c, 'i) Vector.vector) (vec_in2 : ('d, 'k) Vector.vector) :
    ('e, 'm) Vector.vector =
  let ker2, k = ker in
  let k1, k2, k3 = (k.ml_kern, k.body, k.ret_val) in
  param_list := [] ;
  let aux = function
    | Kern (args, body) ->
        let new_args =
          match args with
          | Params p -> (
              match p with
              | Concat (Concat _, Concat _) ->
                  failwith "error multiple map2 args "
              | Concat (a, Concat (b, Empty)) ->
                  params
                    (concat
                       (a_to_vect a)
                       (concat
                          (a_to_vect b)
                          (concat (a_to_vect (fst k3)) (empty_arg ()))))
              | Concat (_a, Empty) -> failwith "error too few map2 args "
              | _ ->
                  Printf.printf "+++++> " ;
                  print_ast args ;
                  failwith "map2 type error")
          | _ -> failwith "error map2 args"
        in
        let n_body =
          let rec aux curr =
            match curr with
            | Return a ->
                a_to_return_vect
                  (fst k3)
                  (aux a)
                  (intrinsics
                     "blockIdx.x*blockDim.x+threadIdx.x"
                     "get_global_id(0)")
            | Seq (a, b) -> seq a (aux b)
            | Local (a, b) -> Local (aux a, aux b)
            | Plus (a, b) -> Plus (aux a, aux b)
            | Min (a, b) -> Min (aux a, aux b)
            | Mul (a, b) -> Mul (aux a, aux b)
            | Div (a, b) -> Div (aux a, aux b)
            | Mod (a, b) -> Mod (aux a, aux b)
            | LtBool (a, b) -> LtBool (aux a, aux b)
            | GtBool (a, b) -> GtBool (aux a, aux b)
            | Ife (a, b, c) -> Ife (aux a, aux b, aux c)
            | IntId (_v, i) ->
                if i = 0 || i = 1 then
                  IntVecAcc
                    ( IdName ("spoc_var" ^ string_of_int i),
                      Intrinsics
                        ( "blockIdx.x*blockDim.x+threadIdx.x",
                          "get_global_id (0)" ) )
                else curr
            | a ->
                print_ast a ;
                propagate aux a
          in
          aux body
        in
        Kern (new_args, n_body)
    | _ -> failwith "malformed kernel for map2"
  in
  let res =
    ( ker2,
      {
        ml_kern =
          (let map2 =
            fun f k a b ->
             let c = Vector.create k (Vector.length a) in
             for i = 0 to Vector.length a - 1 do
               Mem.unsafe_set c i (f (Mem.unsafe_get a i) (Mem.unsafe_get b i))
             done ;
             c
           in
           map2 k1 (snd k3));
        body = aux k2;
        ret_val = (Unit, Vector.int32);
        extensions = k.extensions;
      } )
  in
  let length = Vector.length vec_in1 in
  let vec_out = Vector.create (snd k3) ~dev:device length in
  Mem.to_device vec_in1 device ;
  Mem.to_device vec_in2 device ;
  let framework =
    match device.Devices.specific_info with
    | Devices.CudaInfo _cI -> Devices.Cuda
    | _ -> Devices.OpenCL
  in

  let spoc_ker, _kir_ker = gen ~only:framework res device in
  let open Kernel in
  let block = {blockX = 1; blockY = 1; blockZ = 1}
  and grid = {gridX = 1; gridY = 1; gridZ = 1} in
  spoc_ker#compile ~debug:true device ;
  begin
    let open Devices in
    match device.Devices.specific_info with
    | Devices.CudaInfo cI ->
        if length < cI.maxThreadsDim.x then (
          grid.gridX <- 1 ;
          block.blockX <- length)
        else (
          block.blockX <- cI.maxThreadsDim.x ;
          grid.gridX <- length / cI.maxThreadsDim.x)
    | Devices.OpenCLInfo oI ->
        if length < oI.Devices.max_work_item_size.Devices.x then (
          grid.gridX <- 1 ;
          block.blockX <- length)
        else (
          block.blockX <- oI.Devices.max_work_item_size.Devices.x ;
          grid.gridX <- length / block.blockX)
  end ;
  let bin = Hashtbl.find (spoc_ker#get_binaries ()) device in
  let offset = ref 0 in
  (match device.Devices.specific_info with
  | Devices.CudaInfo _cI ->
      let extra = Kernel.Cuda.cuda_create_extra 2 in
      Kernel.Cuda.cuda_load_arg offset extra device bin 0 (arg_of_vec vec_in1) ;
      Kernel.Cuda.cuda_load_arg offset extra device bin 1 (arg_of_vec vec_in2) ;
      Kernel.Cuda.cuda_load_arg offset extra device bin 2 (arg_of_vec vec_out) ;
      Kernel.Cuda.cuda_launch_grid
        offset
        bin
        grid
        block
        extra
        device.Devices.general_info
        0
  | Devices.OpenCLInfo _ ->
      let clFun = bin in
      let offset = ref 0 in
      Kernel.OpenCL.opencl_load_arg offset device clFun 0 (arg_of_vec vec_in1) ;
      Kernel.OpenCL.opencl_load_arg offset device clFun 1 (arg_of_vec vec_in2) ;
      Kernel.OpenCL.opencl_load_arg offset device clFun 2 (arg_of_vec vec_out) ;
      Kernel.OpenCL.opencl_launch_grid
        clFun
        grid
        block
        device.Devices.general_info
        0) ;
  vec_out

let reduce (_ker : ('a, 'b, 'c -> 'c -> 'd, 'e, 'f) sarek_kernel)
    ?dev:(_device = (Spoc.Devices.init ()).(0))
    (_vec_in1 : ('c, 'i) Vector.vector) : 'd =
  Obj.magic ()

let ( ^> ) = fun a b -> a ^ "\n" ^ b

(* (\* External from SPOC *\) *)
(* external opencl_compile : string -> string -> Devices.generalInfo -> Kernel.kernel = *)
(*     "spoc_opencl_compile" *)
(* external cuda_compile : *)
(*   string -> *)
(*   string -> Devices.generalInfo -> Kernel.kernel = *)
(*   "spoc_cuda_compile" *)

let build_new_ker spoc_ker kir_ker ker ml_fun =
  ( spoc_ker,
    {
      ml_kern = ml_fun;
      body = ker;
      ret_val = (Unit, Vector.Unit ((), ()));
      extensions = kir_ker.extensions;
    } )

let map =
 fun (f : ('a, 'b, 'c, 'd, 'e) sarek_kernel)
     ?dev:(device = (Spoc.Devices.init ()).(0))
     (vec_in : ('d, 'h) Vector.vector)
     :
     ('f, 'g) Vector.vector ->
  let spoc_ker, kir_ker = f in
  let ker = map_skeleton kir_ker in
  let vec_out =
    Vector.create (snd kir_ker.ret_val) ~dev:device (Vector.length vec_in)
  in
  Mem.to_device vec_in device ;
  let target =
    match device.Devices.specific_info with
    | Devices.CudaInfo _ -> Devices.Cuda
    | Devices.OpenCLInfo _ -> Devices.OpenCL
  in
  let ((spoc_ker, _kir_ker) as res) =
    build_new_ker
      spoc_ker
      kir_ker
      ker
      (Tools.map kir_ker.ml_kern (snd kir_ker.ret_val))
  in
  ignore (gen ~only:target res device) ;
  spoc_ker#compile ~debug:true device ;
  let grid, block = compute_grid_block_1D device vec_in in

  let bin = Hashtbl.find (spoc_ker#get_binaries ()) device in

  launch_kernel_with_args
    bin
    grid
    block
    [|
      arg_of_vec vec_in; arg_of_vec vec_out; Kernel.Int32 (Vector.length vec_in);
    |]
    device ;
  vec_out

exception Zip of string

let zip =
 fun (f : ('a, 'b, 'c, 'd, 'e) sarek_kernel)
     ?dev:(device = (Spoc.Devices.init ()).(0))
     (vec_in1 : ('f, 'g) Vector.vector)
     (vec_in2 : ('h, 'i) Vector.vector)
     :
     ('j, 'k) Vector.vector ->
  if Vector.length vec_in1 <> Vector.length vec_in2 then
    raise (Zip "incompatible vector sizes") ;

  let spoc_ker, kir_ker = f in
  let ker = zip_skeleton kir_ker in

  let vec_out =
    Vector.create (snd kir_ker.ret_val) ~dev:device (Vector.length vec_in1)
  in
  Mem.to_device vec_in1 device ;
  Mem.to_device vec_in2 device ;
  let target =
    match device.Devices.specific_info with
    | Devices.CudaInfo _ -> Devices.Cuda
    | Devices.OpenCLInfo _ -> Devices.OpenCL
  in
  let ((spoc_ker, _kir_ker) as res) =
    build_new_ker spoc_ker kir_ker ker (fun a b ->
        for i = 0 to Vector.length a - 1 do
          Mem.set vec_out i (kir_ker.ml_kern (Mem.get a i) (Mem.get b i))
        done ;
        vec_out)
  in
  ignore (gen ~only:target res device) ;
  spoc_ker#compile ~debug:true device ;
  let grid, block = compute_grid_block_1D device vec_in1 in

  let bin = Hashtbl.find (spoc_ker#get_binaries ()) device in

  launch_kernel_with_args
    bin
    grid
    block
    [|
      arg_of_vec vec_in1;
      arg_of_vec vec_in2;
      arg_of_vec vec_out;
      Kernel.Int32 (Vector.length vec_in1);
    |]
    device ;
  vec_out
