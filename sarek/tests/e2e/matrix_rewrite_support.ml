(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Shared kernels and helpers of the matrix-rewrite tests
   (specs/matrix-pattern-rewrite.md). *)

[@@@alert "-sarek_matrix"]

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std

let () =
  Backend_hip.init () ;
  Backend_cuda_c.register () ;
  Test_helpers.Benchmarks.init_backends ()

let failures = ref 0

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK   " ^ s)) fmt

(* ---- Kernels ------------------------------------------------------------- *)

let k_2d =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = thread_idx_y + (block_dim_y * block_idx_y) in
      let col = thread_idx_x + (block_dim_x * block_idx_x) in
      if row < m && col < n then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

let k_1d =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      if row < m then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((row * k) + i) * b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

(* A and B column-major, C row-major. *)
let k_transposed =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0l in
        for i = 0 to k - 1l do
          sum := sum + (a.((i * m) + row) * b.((col * k) + i))
        done ;
        c.((row * n) + col) <- sum
      end]

(* Batched over z, with an epilogue and an initial value read from C. *)
let k_batched =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (c : int32 vector)
        (m : int32)
        (n : int32)
        (k : int32)
        (nb : int32)
        (alpha : int32) ->
      let z = global_idx_z in
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n && z < nb then begin
        let o = (z * m * n) + (row * n) + col in
        let sum = mut c.(o) in
        for i = 0 to k - 1l do
          sum :=
            sum
            + (a.((z * m * k) + (row * k) + i) * b.((z * k * n) + (i * n) + col))
        done ;
        c.(o) <- (alpha * sum) + 3l
      end]

let k_f16 =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            fma
              (float32_of_float16 a.((row * k) + i))
              (float32_of_float16 b.((i * n) + col))
              sum
        done ;
        c.((row * n) + col) <- sum
      end]

let k_f32 =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]

(* ---- Host reference ------------------------------------------------------- *)

type layout = {a_rm : bool; b_rm : bool}

let reference ~layout ~m ~n ~k ~nb ~init ~epi a b c0 =
  let c = Array.copy c0 in
  for z = 0 to nb - 1 do
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let o = (z * m * n) + (i * n) + j in
        let acc = ref (init c0.(o)) in
        for x = 0 to k - 1 do
          let av =
            if layout.a_rm then a.((z * m * k) + (i * k) + x)
            else a.((x * m) + i)
          in
          let bv =
            if layout.b_rm then b.((z * k * n) + (x * n) + j)
            else b.((j * k) + x)
          in
          acc := Int32.add !acc (Int32.mul av bv)
        done ;
        c.(o) <- epi !acc
      done
    done
  done ;
  c

let compare_floats what expected got =
  let bad = ref 0 in
  Array.iteri
    (fun i e ->
      if Int64.bits_of_float e <> Int64.bits_of_float got.(i) then incr bad)
    expected ;
  if !bad = 0 then (
    ok "%s: bit-identical" what ;
    true)
  else (
    fail "%s: %d of %d outputs differ" what !bad (Array.length expected) ;
    false)

(* ---- Runs ------------------------------------------------------------------ *)

let vec arr =
  let v = Vector.create Vector.int32 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

let sentinel = 0x5a5a5a5al

let random_int32 () = Int32.of_int (Random.bits () lxor (Random.bits () lsl 30))

let launch ~rewrite dev ir ~args ~block ~grid =
  if rewrite then Sarek_matrix.run ~device:dev ~ir ~args ~block ~grid ()
  else (
    Execute.run_vectors_plain ~device:dev ~ir ~args ~block ~grid () ;
    Sarek_matrix.Original "(original requested)")

let compare_arrays what expected got =
  let bad = ref 0 and first = ref (-1) in
  Array.iteri
    (fun i e ->
      if e <> got.(i) then (
        incr bad ;
        if !first < 0 then first := i))
    expected ;
  if !bad = 0 then true
  else (
    fail
      "%s: %d of %d outputs differ (first at %d: expected %ld, got %ld)"
      what
      !bad
      (Array.length expected)
      !first
      expected.(!first)
      got.(!first) ;
    false)

type case = {
  name : string;
  kernel : Sarek_ir_types.kernel;
  layout : layout;
  batched : bool;
  geometry :
    m:int ->
    n:int ->
    nb:int ->
    Spoc_framework.Framework_sig.dims * Spoc_framework.Framework_sig.dims;
  full : bool;  (** the geometry covers every output: compare with the host *)
}

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with Some ir -> ir | None -> failwith "no IR"

let cdiv a b = (a + b - 1) / b

let geo_2d ~m ~n ~nb =
  (Execute.dims3d 16 16 1, Execute.dims3d (cdiv n 16) (cdiv m 16) nb)

let geo_1d ~m ~n ~nb:_ = (Execute.dims1d 64, Execute.dims1d (cdiv (m * n) 64))

(* Fewer threads than outputs: the last rows (1-D) or columns (2-D) are left
   alone by the original, and must be by the rewrite. *)
let geo_1d_short ~m ~n ~nb:_ =
  (Execute.dims1d 64, Execute.dims1d (max 1 (m * n / 64)))

let geo_2d_short ~m ~n ~nb =
  (Execute.dims3d 16 16 1, Execute.dims3d (max 1 (n / 16)) (cdiv m 16) nb)

let cases =
  let rm = {a_rm = true; b_rm = true} in
  [
    {
      name = "2d";
      kernel = ir_of k_2d;
      layout = rm;
      batched = false;
      geometry = geo_2d;
      full = true;
    };
    {
      name = "1d quotient";
      kernel = ir_of k_1d;
      layout = rm;
      batched = false;
      geometry = geo_1d;
      full = true;
    };
    {
      name = "1d short launch";
      kernel = ir_of k_1d;
      layout = rm;
      batched = false;
      geometry = geo_1d_short;
      full = false;
    };
    {
      name = "2d short launch";
      kernel = ir_of k_2d;
      layout = rm;
      batched = false;
      geometry = geo_2d_short;
      full = false;
    };
    {
      name = "transposed";
      kernel = ir_of k_transposed;
      layout = {a_rm = false; b_rm = false};
      batched = false;
      geometry = geo_2d;
      full = true;
    };
    {
      name = "batched";
      kernel = ir_of k_batched;
      layout = rm;
      batched = true;
      geometry = geo_2d;
      full = true;
    };
  ]

(* CUDA/C lowers [nvcuda::wmma] 16x16x16 with unsigned 8-bit operands from
   compute capability 7.2 on 32-wide warps (the rule the probe implements,
   restated here independently so that a probe that is missing shows as a
   missing rewrite). *)
let cuda_c_wmma_int dev =
  let caps = dev.Device.capabilities in
  dev.Device.framework = "CUDA/C"
  && compare caps.Spoc_framework.Framework_sig.compute_capability (7, 2) >= 0
  && caps.Spoc_framework.Framework_sig.warp_size = 32

let permits_byte_planes dev =
  Sarek_capability.permits
    (Sarek_coopmat.verdict
       ~support:dev.Device.capabilities.Spoc_framework.Framework_sig.coopmat
       (Sarek_matrix_rewrite.config_of
          Sarek_matrix_rewrite.Int32_byte_planes
          ~operand_elt:Sarek_ir_types.TInt32))

let offers_int dev =
  dev.Device.framework = "Interpreter"
  || (dev.Device.framework = "HIP" || dev.Device.framework = "Vulkan")
     && permits_byte_planes dev
  || cuda_c_wmma_int dev

let vec_f16_of arr =
  let v = Vector.create Vector.float16 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

let vec_f32_zero len =
  let v = Vector.create Vector.float32 len in
  for i = 0 to len - 1 do
    Vector.set v i 0.0
  done ;
  v

(* float16 operands accumulated into C ([c := c + a.b]). *)
let k_f16_accum =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let o = (row * n) + col in
        let sum = mut c.(o) in
        for i = 0 to k - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.(o) <- sum
      end]

(* The float lowerings stay on HIP and the interpreter (specs/matrix-pattern-rewrite.md):
   Vulkan and CUDA/C take the integer lowering only. *)
let offers_float dev =
  dev.Device.framework <> "Vulkan"
  && dev.Device.framework <> "CUDA/C"
  && offers_int dev

(* CUDA/PTX cannot emit the corpus kernels' own intrinsics (global_idx_z), so
   even the original launch raises there; it never rewrites either. *)
let runnable dev = dev.Device.framework <> "CUDA/PTX"
