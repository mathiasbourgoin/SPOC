type kernel
type var = Var of string
and kvect = IntVect of int | Floatvect of int
type intrinsics = string * string
type k_ext =
    Kern of k_ext * k_ext
  | Params of k_ext
  | Plus of k_ext * k_ext
  | Plusf of k_ext * k_ext
  | Min of k_ext * k_ext
  | Minf of k_ext * k_ext
  | Mul of k_ext * k_ext
  | Mulf of k_ext * k_ext
  | Div of k_ext * k_ext
  | Divf of k_ext * k_ext
  | Mod of k_ext * k_ext
  | Id of string
  | IdName of string
  | IntVar of int
  | FloatVar of int
  | UnitVar of int
  | CastDoubleVar of int
  | DoubleVar of int
  | IntArr of int * k_ext
  | Int32Arr of int * k_ext
  | Int64Arr of int * k_ext
  | Float32Arr of int * k_ext
  | Float64Arr of int * k_ext
  | VecVar of k_ext * int
  | Concat of k_ext * k_ext
  | Empty
  | Seq of k_ext * k_ext
  | Return of k_ext
  | Set of k_ext * k_ext
  | Decl of k_ext
  | SetV of k_ext * k_ext
  | SetLocalVar of k_ext * k_ext * k_ext
  | Intrinsics of intrinsics
  | IntId of string * int
  | Int of int
  | Float of float
  | Double of float
  | IntVecAcc of k_ext * k_ext
  | Local of k_ext * k_ext
  | Acc of k_ext * k_ext
  | Ife of k_ext * k_ext * k_ext
  | If of k_ext * k_ext
  | Or of k_ext * k_ext
  | And of k_ext * k_ext
  | EqBool of k_ext * k_ext
  | LtBool of k_ext * k_ext
  | GtBool of k_ext * k_ext
  | LtEBool of k_ext * k_ext
  | GtEBool of k_ext * k_ext
  | DoLoop of k_ext * k_ext * k_ext * k_ext
  | While of k_ext * k_ext
  | App of k_ext * k_ext array
  | GInt of (unit -> int)
  | GFloat of (unit -> float)
  | Unit
type kfun = KernFun of k_ext * k_ext
val print_ast : k_ext -> unit
