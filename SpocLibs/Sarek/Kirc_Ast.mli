type kernel
type var = Var of string
and kvect = IntVect of int | Floatvect of int
type intrinsics = string * string
type elttype = 
  | EInt32
  | EInt64
  | EFloat32
  | EFloat64

type memspace =
  | LocalSpace
  | Global
  | Shared

type k_ext =
    Kern of k_ext * k_ext
  | Block of k_ext
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
  | GlobalFun of k_ext*string
  | IntVar of int
  | FloatVar of int
  | UnitVar of int
  | CastDoubleVar of int
  | DoubleVar of int
  | BoolVar of int
  | Arr of int * k_ext * elttype * memspace
  | VecVar of k_ext * int
  | Concat of k_ext * k_ext
  | Constr of string * string * k_ext list
  | Record of string*k_ext list
  | RecGet of k_ext * string
  | RecSet of k_ext * k_ext
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
  | Custom of string*int
  | CustomVar of string*string
  | IntVecAcc of k_ext * k_ext
  | Local of k_ext * k_ext
  | Acc of k_ext * k_ext
  | Ife of k_ext * k_ext * k_ext
  | If of k_ext * k_ext
  | Match of string*k_ext * case array
  | Or of k_ext * k_ext
  | And of k_ext * k_ext
  | Not of k_ext
  | EqCustom of string * k_ext * k_ext 
  | EqBool of k_ext * k_ext
  | LtBool of k_ext * k_ext
  | GtBool of k_ext * k_ext
  | LtEBool of k_ext * k_ext
  | GtEBool of k_ext * k_ext
  | DoLoop of k_ext * k_ext * k_ext * k_ext
  | While of k_ext * k_ext
  | App of k_ext * k_ext array
  | GInt of (unit -> int32)
  | GFloat of (unit -> float)
  | Unit
and case = int * (string*string*int) option * k_ext

type kfun = KernFun of k_ext * k_ext
val print_ast : k_ext -> unit
