(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * SCoopmat on Metal (simdgroup_matrix 8x8x8) end to end —
 * specs/coopmat-metal-simdgroup.md.
 *
 * Order matters, because the float opt-in is process-wide and one-way:
 *   1. codegen (any platform): the emitted MSL names the instructions;
 *   2. refusals BEFORE the opt-in, on every Metal device: the f16
 *      configuration is refused for the policy reason, an integer
 *      configuration for the device reason;
 *   3. the opt-in, then a partial-SIMD-group block (refused for the 32-wide
 *      rule: before the opt-in the policy gate fires first), then on Metal
 *      devices: single tiles bit-equal to the
 *      interpreter AND to the closed form measured in
 *      docs/design/f16-relaxed-accuracy.md §2 ("accumulator = C, then add the
 *      products in index order, all in binary32"), every row/column-major
 *      combination, a tile at an offset inside a wider buffer, and a tiled
 *      GEMM (several 8x8 fragments per SIMD-group) over three tilings;
 *   4. positive controls on the same GPU output: a C-last model, a C-dropping
 *      run and a transposed-B run must all be REJECTED by the comparison.
 *
 * The device parts skip (exit 0) when no Metal device is present.
 ******************************************************************************)

open Sarek_ir_types
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module C = Sarek_coopmat
module Gemm = Sarek_gemm.Coopmat
module Fw = Spoc_framework.Framework_sig

let () = Test_helpers.Benchmarks.init_backends ()

let () =
  Sarek_codegen.Sarek_ir_metal.accept_unexecuted_lowering
    ~reason:"test_coopmat_metal_e2e executes the lowering"

let failures = ref 0

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK  " ^ s)) fmt

let contains ~needle s = Test_helpers.string_contains ~needle s

let shape = {C.m = 8; n = 8; k = 8}

let config a b acc =
  {
    C.cfg_shape = shape;
    cfg_a = a;
    cfg_b = b;
    cfg_c = acc;
    cfg_result = acc;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let f16_config = config C.Float16 C.Float16 C.Float32

let int_config = config C.Uint8 C.Uint8 C.Sint32

(** {1 IR: one 8x8x8 tile} *)

let elt = function
  | C.Float16 -> TFloat16
  | C.BFloat16 | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

type layouts = {la : C.layout; lb : C.layout; lc : C.layout; ld : C.layout}

let row =
  {la = C.Row_major; lb = C.Row_major; lc = C.Row_major; ld = C.Row_major}

(** [d = a * b + c] on one tile at [index] with [stride]. [drop_c] never loads C
    (the C-dropping control, built from the same generator as the code it
    controls). *)
let tile_ir ?(drop_c = false) ?(index = 0) ?(stride = 8) ?(lay = row) cfg :
    kernel =
  let fa, fb, fc, fd = C.fragments_of_config cfg in
  let vec id name e =
    ({var_name = name; var_id = id; var_type = TVec e; var_mutable = false}, e)
  in
  let params =
    [
      vec 0 "a" (elt cfg.C.cfg_a);
      vec 1 "b" (elt cfg.C.cfg_b);
      vec 2 "c" (elt cfg.C.cfg_c);
      vec 3 "d" (elt cfg.C.cfg_result);
    ]
  in
  let i = EConst (CInt32 (Int32.of_int index))
  and s = EConst (CInt32 (Int32.of_int stride)) in
  let load dst frag src layout =
    SCoopmat (CM_load {dst; frag; src; index = i; stride = s; layout})
  in
  let body =
    [
      SCoopmat (CM_decl {name = "fa"; frag = fa});
      SCoopmat (CM_decl {name = "fb"; frag = fb});
      SCoopmat (CM_decl {name = "fc"; frag = fc});
      SCoopmat (CM_decl {name = "fd"; frag = fd});
      load "fa" fa "a" lay.la;
      load "fb" fb "b" lay.lb;
    ]
    @ (if drop_c then [] else [load "fc" fc "c" lay.lc])
    @ [
        SCoopmat (CM_muladd {dst = "fd"; a = "fa"; b = "fb"; c = "fc"; cfg});
        SCoopmat
          (CM_store
             {
               src = "fd";
               frag = fd;
               dst = "d";
               index = i;
               stride = s;
               layout = lay.ld;
             });
      ]
  in
  {
    default_kernel with
    kern_name = "coopmat_tile_metal";
    kern_params =
      List.map
        (fun (v, e) ->
          DParam (v, Some {arr_elttype = e; arr_memspace = Global}))
        params;
    kern_body = SSeq body;
  }

(** {1 Inputs and the closed-form model} *)

let rng = Random.State.make [|0x3e7a; 0x51d9|]

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(** [+-(1024 + m)/1024 * 2^e], a binary16 value for [e] in the normal range. *)
let f16_value ~emin ~emax =
  let m = Random.State.int rng 1024 in
  let e = emin + Random.State.int rng (emax - emin + 1) in
  let s = if Random.State.bool rng then 1.0 else -1.0 in
  s *. Float.ldexp (float_of_int (1024 + m) /. 1024.0) e

(** A binary32 value with a full 23-bit mantissa, so that where C enters the sum
    is observable. *)
let f32_value ~emin ~emax =
  let m = Random.State.int rng (1 lsl 23) in
  let e = emin + Random.State.int rng (emax - emin + 1) in
  let s = if Random.State.bool rng then 1.0 else -1.0 in
  s *. Float.ldexp (1.0 +. (float_of_int m /. 8388608.0)) e

let operand n = Array.init n (fun _ -> f16_value ~emin:(-4) ~emax:0)

let addend n = Array.init n (fun _ -> f32_value ~emin:(-1) ~emax:2)

(** The model of docs/design/f16-relaxed-accuracy.md §2 (Metal row): the
    accumulator starts at C, then the [k] products are added in index order,
    each sum rounded to binary32. A product of two binary16 values is exact in
    binary32, and the binary64 sum of a binary32 and such a product is exact for
    the exponent spans generated here, so [round32] of it is the binary32 add.
*)
let model ?(c_last = false) ~a ~b ~c ~m ~n ~k () =
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let prods = List.init k (fun x -> a.((i * k) + x) *. b.((x * n) + j)) in
      if c_last then
        round32
          (List.fold_left (fun s p -> round32 (s +. p)) 0.0 prods +. c.(idx))
      else List.fold_left (fun s p -> round32 (s +. p)) c.(idx) prods)

let vec_f16 arr =
  let v = Vector.create Vector.float16 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

let vec_f32 arr =
  let v = Vector.create Vector.float32 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

let bits_equal x y =
  Array.length x = Array.length y
  &&
  let same = ref true in
  Array.iteri
    (fun i v ->
      if Int32.bits_of_float v <> Int32.bits_of_float y.(i) then same := false)
    x ;
  !same

let count_differing x y =
  let n = ref 0 in
  Array.iteri
    (fun i v ->
      if Int32.bits_of_float v <> Int32.bits_of_float y.(i) then incr n)
    x ;
  !n

(** {1 Runners} *)

let simd = 32

let run_gpu dev ir args ~block ~grid =
  Execute.run_vectors ~device:dev ~ir ~args ~block ~grid () ;
  Transfer.flush dev

let run_interp ir args ~block ~grid =
  Execute.run_interpreter_vectors ~ir ~args ~block ~grid ~parallel:false

let refusal f =
  match f () with () -> None | exception e -> Some (Printexc.to_string e)

(** Run a tile kernel; [len] is the buffer length (the tile may sit inside a
    wider buffer). Returns [d]. *)
let run_tile ~gpu ~len ir ~a ~b ~c =
  let d = vec_f32 (Array.make len 0.0) in
  let args =
    Execute.[Vec (vec_f16 a); Vec (vec_f16 b); Vec (vec_f32 c); Vec d]
  in
  let block = Execute.dims1d simd and grid = Execute.dims1d 1 in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block ~grid
  | None -> run_interp ir args ~block ~grid) ;
  Array.map (fun x -> x) (Vector.to_array d)

(** {1 1. codegen} *)

let check_codegen () =
  let src = Sarek_codegen.Sarek_ir_metal.generate (Gemm.ir ~frag:8 ()) in
  List.iter
    (fun needle ->
      if contains ~needle src then ok "GEMM MSL contains %S" needle
      else fail "GEMM MSL does not contain %S" needle)
    [
      "simdgroup_multiply_accumulate("; "simdgroup_half8x8"; "simdgroup_float8x8";
    ]

(** {1 2. refusals before the opt-in} *)

let gate dev ir () = Execute.check_device_capabilities ~device:dev ir

let check_refusals dev =
  let name = dev.Device.name in
  (match refusal (gate dev (tile_ir f16_config)) with
  | Some msg when contains ~needle:"accept_relaxed_accumulation" msg ->
      ok "%s refuses f16 coopmat for the policy reason" name
  | Some msg -> fail "%s: f16 refusal does not name the opt-in: %s" name msg
  | None -> fail "%s permitted f16 coopmat before the opt-in" name) ;
  (match refusal (gate dev (tile_ir int_config)) with
  | Some _ -> ok "%s refuses u8*u8+s32 (no integer simdgroup_matrix)" name
  | None -> fail "%s permitted an integer coopmat configuration" name) ;
  let sg = dev.Device.capabilities.Fw.warp_size in
  if sg = simd then ok "%s: SIMD width %d" name sg
  else fail "%s: SIMD width %d, expected %d" name sg simd

let check_block_refusal dev =
  let a = Array.make 64 1.0 in
  match
    refusal (fun () ->
        let d = vec_f32 (Array.make 64 0.0) in
        run_gpu
          dev
          (tile_ir f16_config)
          Execute.[Vec (vec_f16 a); Vec (vec_f16 a); Vec (vec_f32 a); Vec d]
          ~block:(Execute.dims1d 16)
          ~grid:(Execute.dims1d 1))
  with
  | None ->
      fail "%s launched a coopmat kernel on a 16-thread block" dev.Device.name
  | Some msg when contains ~needle:"32-wide" msg -> ok "16-thread block refused"
  | Some msg -> fail "16-thread block refused for another reason: %s" msg

(** {1 3. tiles} *)

let check_tile dev =
  let a = operand 64 and b = operand 64 and c = addend 64 in
  let ir = tile_ir f16_config in
  let gpu = run_tile ~gpu:(Some dev) ~len:64 ir ~a ~b ~c in
  let interp = run_tile ~gpu:None ~len:64 ir ~a ~b ~c in
  let m = model ~a ~b ~c ~m:8 ~n:8 ~k:8 () in
  if bits_equal gpu interp then ok "tile: bit-equal to the interpreter (64/64)"
  else
    fail
      "tile: %d of 64 differ from the interpreter"
      (count_differing gpu interp) ;
  if bits_equal gpu m then ok "tile: bit-equal to the C-first binary32 model"
  else fail "tile: %d of 64 differ from the model" (count_differing gpu m) ;
  let c_last = model ~c_last:true ~a ~b ~c ~m:8 ~n:8 ~k:8 () in
  let dropped =
    run_tile ~gpu:None ~len:64 (tile_ir ~drop_c:true f16_config) ~a ~b ~c
  in
  let transposed =
    run_tile
      ~gpu:None
      ~len:64
      (tile_ir ~lay:{row with lb = C.Col_major} f16_config)
      ~a
      ~b
      ~c
  in
  List.iter
    (fun (what, ref_) ->
      let n = count_differing gpu ref_ in
      if n = 0 then fail "control %s was ACCEPTED" what
      else ok "control %s rejected (%d of 64 differ)" what n)
    [
      ("C-last model", c_last);
      ("C-dropping", dropped);
      ("transposed B", transposed);
    ]

let layouts =
  let both = [C.Row_major; C.Col_major] in
  List.concat_map
    (fun la ->
      List.concat_map
        (fun lb ->
          List.concat_map
            (fun lc -> List.map (fun ld -> {la; lb; lc; ld}) both)
            both)
        both)
    both

let layout_name l =
  let c = function C.Row_major -> 'r' | C.Col_major -> 'c' in
  Printf.sprintf "A%c B%c C%c D%c" (c l.la) (c l.lb) (c l.lc) (c l.ld)

let check_layouts dev =
  let a = operand 64 and b = operand 64 and c = addend 64 in
  let bad =
    List.filter
      (fun lay ->
        let ir = tile_ir ~lay f16_config in
        not
          (bits_equal
             (run_tile ~gpu:(Some dev) ~len:64 ir ~a ~b ~c)
             (run_tile ~gpu:None ~len:64 ir ~a ~b ~c)))
      layouts
  in
  match bad with
  | [] ->
      ok
        "all %d layout combinations bit-equal to the interpreter"
        (List.length layouts)
  | l :: _ ->
      fail
        "%d layout combination(s) differ, first: %s"
        (List.length bad)
        (layout_name l)

(** A tile at row 8, column 16 of a 32 x 40 matrix: an implementation that
    ignored the index or assumed a stride of 8 would differ, and the rest of [d]
    must stay zero on both. *)
let check_offset dev =
  let len = 32 * 40 in
  let ir = tile_ir ~index:((8 * 40) + 16) ~stride:40 f16_config in
  let a = operand len and b = operand len and c = addend len in
  let gpu = run_tile ~gpu:(Some dev) ~len ir ~a ~b ~c in
  let interp = run_tile ~gpu:None ~len ir ~a ~b ~c in
  if bits_equal gpu interp then
    ok "offset tile (index 336, stride 40) bit-equal"
  else fail "offset tile: %d of %d differ" (count_differing gpu interp) len

(** {1 4. tiled GEMM, several fragments per SIMD-group} *)

let run_gemm ?(tiling = Gemm.default_tiling) ~gpu ~a ~b ~c ~m ~n ~k () =
  let va = vec_f16 a and vb = vec_f16 b and vc = vec_f32 c in
  let vd = vec_f32 (Array.make (m * n) 0.0) in
  (match gpu with
  | Some dev ->
      Gemm.run ~device:dev ~frag:8 ~tiling ~a:va ~b:vb ~c:vc ~d:vd ~m ~n ~k () ;
      Transfer.flush dev
  | None ->
      Gemm.check_shape ~frag:8 tiling ~m ~n ~k ;
      let block, grid = Gemm.launch_dims ~frag:8 tiling ~m ~n in
      let int x = Execute.Int32 (Int32.of_int x) in
      run_interp
        (Gemm.ir ~tiling ~frag:8 ())
        Execute.[Vec va; Vec vb; Vec vc; Vec vd; int m; int n; int k]
        ~block
        ~grid) ;
  Array.map (fun x -> x) (Vector.to_array vd)

let check_gemm dev ~m ~n ~k ~interp =
  let a = operand (m * k) and b = operand (k * n) and c = addend (m * n) in
  let what = Printf.sprintf "%dx%dx%d" m n k in
  let gpu = run_gemm ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k () in
  let mdl = model ~a ~b ~c ~m ~n ~k () in
  if bits_equal gpu mdl then
    ok "%s GEMM: bit-equal to the C-first binary32 chain" what
  else
    fail
      "%s GEMM: %d of %d differ from the model"
      what
      (count_differing gpu mdl)
      (m * n) ;
  if interp then begin
    let ref_ = run_gemm ~gpu:None ~a ~b ~c ~m ~n ~k () in
    if bits_equal gpu ref_ then ok "%s GEMM: bit-equal to the interpreter" what
    else
      fail
        "%s GEMM: %d differ from the interpreter"
        what
        (count_differing gpu ref_)
  end ;
  let tilings =
    Gemm.
      [
        {frags_m = 1; frags_n = 1; waves = 1};
        {frags_m = 2; frags_n = 1; waves = 2};
      ]
  in
  let moved =
    List.filter
      (fun tiling ->
        not
          (bits_equal
             gpu
             (run_gemm ~tiling ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k ())))
      tilings
  in
  if moved = [] then ok "%s GEMM: three tilings bit-equal" what
  else fail "%s GEMM: %d tiling(s) change the bits" what (List.length moved) ;
  let c_last = model ~c_last:true ~a ~b ~c ~m ~n ~k () in
  let dropped =
    run_gemm ~gpu:(Some dev) ~a ~b ~c:(Array.make (m * n) 0.0) ~m ~n ~k ()
  in
  List.iter
    (fun (w, r) ->
      if bits_equal gpu r then fail "control %s ACCEPTED on %s" w what
      else
        ok
          "control %s rejected on %s (%d differ)"
          w
          what
          (count_differing gpu r))
    [("C-last chain", c_last); ("C-dropping", dropped)]

let () =
  let devices = Array.to_list (Device.init ()) in
  let metal = List.filter (fun d -> d.Device.framework = "Metal") devices in
  List.iter
    (fun d ->
      Printf.printf "device: %s [%s]\n%!" d.Device.name d.Device.framework)
    devices ;
  check_codegen () ;
  List.iter check_refusals metal ;
  C.accept_relaxed_accumulation
    ~reason:
      "test_coopmat_metal_e2e: the Metal closed form is what is under test" ;
  (* After the opt-in: before it the policy gate fires first and the 32-wide
     rule is never reached. *)
  List.iter check_block_refusal metal ;
  (match metal with
  | [] -> print_endline "[SKIP] no Metal device: the device checks did not run"
  | _ ->
      List.iter
        (fun dev ->
          check_tile dev ;
          check_layouts dev ;
          check_offset dev ;
          check_gemm dev ~m:64 ~n:64 ~k:64 ~interp:true ;
          check_gemm dev ~m:128 ~n:128 ~k:256 ~interp:false)
        metal) ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
