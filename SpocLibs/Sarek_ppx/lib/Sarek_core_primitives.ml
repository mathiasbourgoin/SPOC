(******************************************************************************
 * Sarek PPX - Core Primitives
 *
 * This module defines semantic properties for GPU primitives that are known
 * at compile time by the PPX. This separates semantic analysis (variance,
 * convergence, purity) from device implementations (resolved at JIT time).
 ******************************************************************************)

open Sarek_types

(** Variance levels in the GPU execution model *)
type variance =
  | Uniform  (** Same value for all threads in grid *)
  | BlockVarying  (** Uniform within block, varies between blocks *)
  | ThreadVarying  (** Varies per thread *)

(** Convergence requirements *)
type convergence =
  | NoEffect  (** Does not affect convergence *)
  | ConvergencePoint  (** All workgroup threads must reach together *)

(** Purity classification *)
type purity =
  | Pure  (** No side effects *)
  | Impure  (** Has side effects *)
  | Atomic  (** Atomic memory operation *)

(** A core primitive definition *)
type primitive = {
  name : string;
  typ : typ;
  variance : variance;
  convergence : convergence;
  purity : purity;
  category : string;
}

(* All core primitives *)
let primitives =
  [
    (* === Thread Indices (ThreadVarying) === *)
    {
      name = "thread_idx_x";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "thread_idx_y";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "thread_idx_z";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    {
      name = "global_thread_id";
      typ = t_int32;
      variance = ThreadVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "thread_id";
    };
    (* === Block Indices (BlockVarying) === *)
    {
      name = "block_idx_x";
      typ = t_int32;
      variance = BlockVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "block_id";
    };
    {
      name = "block_idx_y";
      typ = t_int32;
      variance = BlockVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "block_id";
    };
    {
      name = "block_idx_z";
      typ = t_int32;
      variance = BlockVarying;
      convergence = NoEffect;
      purity = Pure;
      category = "block_id";
    };
    (* === Dimensions (Uniform) === *)
    {
      name = "block_dim_x";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "block_dim_y";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "block_dim_z";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "grid_dim_x";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "grid_dim_y";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    {
      name = "grid_dim_z";
      typ = t_int32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "dimension";
    };
    (* === Synchronization === *)
    {
      name = "block_barrier";
      typ = t_fun [t_unit] t_unit;
      variance = Uniform;
      convergence = ConvergencePoint;
      purity = Impure;
      category = "sync";
    };
    (* === Float32 Math (Pure, Uniform inherent variance) === *)
    {
      name = "sin";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "cos";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "tan";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "sqrt";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "rsqrt";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "exp";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "log";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "log2";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "log10";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "fabs";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "floor";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "ceil";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "pow";
      typ = t_fun [t_float32; t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "fma";
      typ = t_fun [t_float32; t_float32; t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "asin";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "acos";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "atan";
      typ = t_fun [t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
    {
      name = "atan2";
      typ = t_fun [t_float32; t_float32] t_float32;
      variance = Uniform;
      convergence = NoEffect;
      purity = Pure;
      category = "math_f32";
    };
  ]

(* Build lookup table for O(1) access *)
let primitive_table : (string, primitive) Hashtbl.t =
  let tbl = Hashtbl.create (List.length primitives) in
  List.iter (fun p -> Hashtbl.add tbl p.name p) primitives ;
  tbl

let find name = Hashtbl.find_opt primitive_table name

let find_exn name = Hashtbl.find primitive_table name

let is_core_primitive name = Hashtbl.mem primitive_table name

let is_thread_varying name =
  match find name with Some p -> p.variance = ThreadVarying | None -> false

let is_convergence_point name =
  match find name with
  | Some p -> p.convergence = ConvergencePoint
  | None -> false

let is_pure name =
  match find name with Some p -> p.purity = Pure | None -> false

let variance_of name = Option.map (fun p -> p.variance) (find name)

(* Variance lattice join (least upper bound) *)
let join_variance v1 v2 =
  match (v1, v2) with
  | Uniform, v | v, Uniform -> v
  | BlockVarying, BlockVarying -> BlockVarying
  | ThreadVarying, _ | _, ThreadVarying -> ThreadVarying

(* Variance ordering: v1 ≤ v2 means v1 is less varying than v2 *)
let variance_leq v1 v2 =
  match (v1, v2) with
  | Uniform, _ -> true
  | BlockVarying, (BlockVarying | ThreadVarying) -> true
  | ThreadVarying, ThreadVarying -> true
  | _ -> false

let primitives_in_category cat =
  List.filter (fun p -> p.category = cat) primitives

(* Pretty printing *)
let pp_variance fmt = function
  | Uniform -> Format.fprintf fmt "Uniform"
  | BlockVarying -> Format.fprintf fmt "BlockVarying"
  | ThreadVarying -> Format.fprintf fmt "ThreadVarying"

let pp_convergence fmt = function
  | NoEffect -> Format.fprintf fmt "NoEffect"
  | ConvergencePoint -> Format.fprintf fmt "ConvergencePoint"

let pp_purity fmt = function
  | Pure -> Format.fprintf fmt "Pure"
  | Impure -> Format.fprintf fmt "Impure"
  | Atomic -> Format.fprintf fmt "Atomic"

let pp_primitive fmt p =
  Format.fprintf
    fmt
    "@[<v 2>%s:@ type: %a@ variance: %a@ convergence: %a@ purity: %a@ \
     category: %s@]"
    p.name
    Sarek_types.pp_typ
    p.typ
    pp_variance
    p.variance
    pp_convergence
    p.convergence
    pp_purity
    p.purity
    p.category
