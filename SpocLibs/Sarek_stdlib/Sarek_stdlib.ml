(******************************************************************************
 * Sarek Standard Library
 *
 * Re-exports all stdlib modules for convenient use.
 *
 * NOTE: Float64 is NOT part of stdlib - it's a separate library (sarek_float64)
 * because not all devices support double precision. Link sarek_float64 only
 * when you need float64 support.
 ******************************************************************************)

module Float32 = Float32
module Int32 = Int32
module Int64 = Int64
module Gpu = Gpu
module Math = Math

(******************************************************************************
 * Force initialization
 *
 * OCaml's lazy module initialization means modules only initialize when
 * something from them is actually used. We force all stdlib modules to
 * initialize at module load time, which registers their intrinsics in the
 * PPX registry.
 *
 * This runs automatically when Sarek_stdlib is linked - no need for consumers
 * to call anything explicitly.
 ******************************************************************************)

let () =
  (* Reference something from each module to force initialization.
     Order doesn't matter for registration since we use qualified names
     (Float32.sqrt vs Float64.sqrt) and open_module handles short names. *)
  ignore Int64.of_float ;
  ignore Float32.sin ;
  ignore Int32.of_float ;
  ignore Gpu.thread_idx_x ;
  ignore Math.xor

(* Keep force_init for explicit calls if needed *)
let force_init () = ()
