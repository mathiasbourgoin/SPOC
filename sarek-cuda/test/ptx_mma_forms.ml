(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The instruction forms [test_ptx_mma_probe] runs, the PTX kernels that
    exercise them, and the host side: operands, fragment packing per the PTX ISA
    layouts, the exact reference and the error bound. Pure (no driver). *)

let warp = 32

(** {1 Instruction forms} *)

type elem = F16 | S8 | U8

type acc = F32 | F16_acc | S32

(** What a form did. The [recorded] field of a form holds ZLUDA v7-preview.3's
    answer on gfx1100 (probe-output.txt, module_0001.log of the 2026-10-10
    investigation); [Wrong] is only ever observed. *)
type status =
  | Exact  (** runs, bit-identical to the CPU reference *)
  | Inexact_within_bound  (** runs, some elements inexact, all within bound *)
  | Not_translated  (** the kernel does not load *)
  | Wrong  (** runs, some element outside the bound / not bit-exact *)

type mma_form = {
  label : string;
  k : int;  (** m16 n8 kK *)
  elem : elem;
  acc : acc;
  satfinite : bool;
  recorded : status;
}

let mma_forms =
  let form ?(satfinite = false) label k elem acc recorded =
    {label; k; elem; acc; satfinite; recorded}
  in
  [
    form "mma m16n8k16 f32.f16.f16.f32" 16 F16 F32 Inexact_within_bound;
    form "mma m16n8k32 s32.s8.s8.s32" 32 S8 S32 Exact;
    form "mma m16n8k32 s32.u8.u8.s32" 32 U8 S32 Not_translated;
    form
      ~satfinite:true
      "mma m16n8k32 satfinite s32.s8.s8.s32"
      32
      S8
      S32
      Not_translated;
    form "mma m16n8k8 f32.f16.f16.f32" 8 F16 F32 Not_translated;
    form "mma m16n8k16 f16.f16.f16.f16" 16 F16 F16_acc Not_translated;
  ]

let elem_bytes = function F16 -> 2 | S8 | U8 -> 1

let elem_ptx = function F16 -> "f16" | S8 -> "s8" | U8 -> "u8"

let acc_ptx = function F32 -> "f32" | F16_acc -> "f16" | S32 -> "s32"

(* 32-bit registers per lane: A holds 16*K elements, B K*8, C and D 16*8. *)
let a_regs f = 16 * f.k * elem_bytes f.elem / 4 / warp

let b_regs f = f.k * 8 * elem_bytes f.elem / 4 / warp

let cd_regs f = match f.acc with F16_acc -> 2 | F32 | S32 -> 4

let reg_list prefix n =
  "{" ^ String.concat "," (List.init n (Printf.sprintf "%%%s%d" prefix)) ^ "}"

let instruction f =
  Printf.sprintf
    "mma.sync.aligned.m16n8k%d.row.col%s.%s.%s.%s.%s %s,%s,%s,%s;"
    f.k
    (if f.satfinite then ".satfinite" else "")
    (acc_ptx f.acc)
    (elem_ptx f.elem)
    (elem_ptx f.elem)
    (acc_ptx f.acc)
    (reg_list "d" (cd_regs f))
    (reg_list "a" (a_regs f))
    (reg_list "b" (b_regs f))
    (reg_list "c" (cd_regs f))

(** The lowest PTX ISA version that admits [.target sm_XY]. [load_from_ptx]
    rewrites [.target] to the device's own SM, so a fixed [.version] would be
    rejected on newer NVIDIA GPUs (sm_86 needs 7.1, sm_90 7.8, sm_120 8.7).
    sm_88 (also what ZLUDA reports), sm_110 and SMs not listed get 9.0, the
    newest version here; ZLUDA parses it. *)
let ptx_isa_version ~major ~minor =
  match (major, minor) with
  | 8, 0 -> "7.0"
  | 8, 6 -> "7.1"
  | 8, 7 -> "7.4"
  | 8, 9 | 9, 0 -> "7.8"
  | 10, 0 | 10, 1 -> "8.6"
  | 12, 0 -> "8.7"
  | 10, 3 | 12, 1 -> "8.8"
  | _ -> if major < 8 then "7.0" else "9.0"

(* Each lane loads its A, B and C registers from its own slice of the input
   buffers, runs [body] (default: the form's instruction) once and stores its
   D registers. *)
let kernel_ptx ~version ?body f =
  let body = match body with Some b -> b | None -> instruction f in
  let lines n fmt = String.concat "" (List.init n fmt) in
  let loads reg ptr n =
    lines n (fun i ->
        Printf.sprintf "ld.global.b32 %%%s%d,[%%%s+%d];\n" reg i ptr (4 * i))
  in
  let offset ptr n =
    Printf.sprintf
      "mul.wide.u32 %%o,%%lane,%d; add.u64 %%%s,%%%s,%%o;\n"
      (4 * n)
      ptr
      ptr
  in
  String.concat
    ""
    [
      ".version " ^ version ^ "\n.target sm_80\n.address_size 64\n";
      ".visible .entry probe(.param .u64 pA,.param .u64 pB,.param .u64 pC,";
      ".param .u64 pD){\n";
      ".reg .b32 %a<8>; .reg .b32 %b<4>; .reg .b32 %c<4>; .reg .b32 %d<4>;\n";
      ".reg .u32 %lane; .reg .u64 %o,%pa,%pb,%pc,%pd;\n";
      "ld.param.u64 %pa,[pA]; ld.param.u64 %pb,[pB];\n";
      "ld.param.u64 %pc,[pC]; ld.param.u64 %pd,[pD];\n";
      "mov.u32 %lane,%tid.x;\n";
      offset "pa" (a_regs f);
      offset "pb" (b_regs f);
      offset "pc" (cd_regs f);
      offset "pd" (cd_regs f);
      loads "a" "pa" (a_regs f);
      loads "b" "pb" (b_regs f);
      loads "c" "pc" (cd_regs f);
      body ^ "\n";
      lines (cd_regs f) (fun i ->
          Printf.sprintf "st.global.b32 [%%pd+%d],%%d%d;\n" (4 * i) i);
      "ret;\n}\n";
    ]

(** {1 Host data: operands, fragment packing, reference} *)

(** binary16 bits of an integer-valued float with |x| <= 2048 (exact). *)
let half_bits x =
  if x = 0.0 then 0
  else
    let u = Int32.to_int (Int32.bits_of_float x) land 0xFFFF_FFFF in
    let sign = (u lsr 16) land 0x8000 in
    let exp = ((u lsr 23) land 0xff) - 127 + 15 in
    sign lor (exp lsl 10) lor ((u lsr 13) land 0x3ff)

let half_value bits =
  let sign = if bits land 0x8000 <> 0 then -1.0 else 1.0 in
  let exp = (bits lsr 10) land 0x1f in
  let man = float_of_int (bits land 0x3ff) in
  if exp = 0 then sign *. man *. ldexp 1.0 (-24)
  else if exp = 31 then nan
  else sign *. ldexp (1.0 +. (man /. 1024.0)) (exp - 15)

(* Random integer operands. Every product and every partial sum is exact in
   the accumulator type, so the exact result is what a correctly rounded
   implementation returns and any inexact element is the implementation's own
   rounding:
   - f16 -> f32: inputs in [-64, 64], |sum| <= 16*64*64 + 2 < 2^24 (exact in
     binary32) but above 2048, so an implementation that accumulated in binary16
     would round and leave the binary32 bound;
   - f16 -> f16: inputs in [-4, 4], |sum| <= 16*16 + 2 <= 2048 (exact in
     binary16), so the result must be bit-exact;
   - 8-bit: row 0 of A and column 0 of B hold the extreme value (-128 or 255),
     as in the C probe; |sum| <= 32*128*128 + 1000. *)
let random_operands f =
  let st = Random.State.make [|f.k; Hashtbl.hash f.label|] in
  let span = match f.acc with F32 -> 64 | F16_acc | S32 -> 4 in
  let pick () =
    match f.elem with
    | F16 -> Random.State.int st ((2 * span) + 1) - span
    | S8 -> Random.State.int st 256 - 128
    | U8 -> Random.State.int st 256
  in
  let a = Array.init 16 (fun _ -> Array.init f.k (fun _ -> pick ())) in
  let b = Array.init f.k (fun _ -> Array.init 8 (fun _ -> pick ())) in
  (match f.elem with
  | F16 -> ()
  | S8 | U8 ->
      let extreme = if f.elem = U8 then 255 else -128 in
      for kk = 0 to f.k - 1 do
        a.(0).(kk) <- extreme ;
        b.(kk).(0) <- extreme
      done) ;
  let c_range = match f.elem with F16 -> 5 | S8 | U8 -> 2001 in
  let c =
    Array.init 16 (fun _ ->
        Array.init 8 (fun _ -> Random.State.int st c_range - (c_range / 2)))
  in
  (a, b, c)

let encode_elem f v =
  match f.elem with F16 -> half_bits (float_of_int v) | S8 | U8 -> v land 0xff

(* Fragment layouts of the PTX ISA (mma m16n8k8/k16 f16, m16n8k32 8-bit) for
   lane 4*g + t and its element i: A (row, col), B (row k, col g), C/D. *)
let a_coord f ~g ~t i =
  match f.elem with
  | F16 -> (g + (8 * (i / 2 mod 2)), (t * 2) + (i mod 2) + (8 * (i / 4)))
  | S8 | U8 -> (g + (8 * (i / 4 mod 2)), (t * 4) + (i mod 4) + (16 * (i / 8)))

let b_row f ~t i =
  match f.elem with
  | F16 -> (t * 2) + (i mod 2) + (8 * (i / 2))
  | S8 | U8 -> (t * 4) + (i mod 4) + (16 * (i / 4))

let cd_coord ~g ~t i = (g + (8 * (i / 2)), (t * 2) + (i mod 2))

let int32_buffer n = Bigarray.(Array1.create int32 c_layout n)

(* [count] elements of [width] bits per lane, packed low bits first into
   [regs] 32-bit registers per lane. *)
let pack ~regs ~count ~width value =
  let buf = int32_buffer (warp * regs) in
  Bigarray.Array1.fill buf 0l ;
  let per_reg = 32 / width in
  for lane = 0 to warp - 1 do
    for i = 0 to count - 1 do
      let r = (lane * regs) + (i / per_reg) in
      let bits = value ~g:(lane / 4) ~t:(lane mod 4) i in
      let shifted = Int32.of_int (bits lsl (width * (i mod per_reg))) in
      buf.{r} <- Int32.logor buf.{r} shifted
    done
  done ;
  buf

let acc_bits f v =
  match f.acc with
  | F32 -> Int32.to_int (Int32.bits_of_float (float_of_int v)) land 0xFFFF_FFFF
  | F16_acc -> half_bits (float_of_int v)
  | S32 -> v land 0xFFFF_FFFF

let pack_inputs f (a, b, c) =
  let width = 8 * elem_bytes f.elem in
  let pa =
    pack
      ~regs:(a_regs f)
      ~count:(16 * f.k / warp)
      ~width
      (fun ~g ~t i ->
        let row, col = a_coord f ~g ~t i in
        encode_elem f a.(row).(col))
  in
  let pb =
    pack
      ~regs:(b_regs f)
      ~count:(f.k * 8 / warp)
      ~width
      (fun ~g ~t i -> encode_elem f b.(b_row f ~t i).(g))
  in
  let c_width = match f.acc with F16_acc -> 16 | F32 | S32 -> 32 in
  let pc =
    pack ~regs:(cd_regs f) ~count:4 ~width:c_width (fun ~g ~t i ->
        let row, col = cd_coord ~g ~t i in
        acc_bits f c.(row).(col))
  in
  (pa, pb, pc)

(* Exact value of D[row][col] and the sum of |terms| (C and the K products). *)
let reference f (a, b, c) row col =
  let sum = ref c.(row).(col) and abs_sum = ref (abs c.(row).(col)) in
  for kk = 0 to f.k - 1 do
    let p = a.(row).(kk) * b.(kk).(col) in
    sum := !sum + p ;
    abs_sum := !abs_sum + abs p
  done ;
  (!sum, !abs_sum)

(* Element i of lane [lane] of the D buffer, as a float (exact for s32). *)
let output_value f d ~lane i =
  match f.acc with
  | F16_acc ->
      let reg = Int32.to_int d.{(lane * 2) + (i / 2)} in
      half_value ((reg lsr (16 * (i mod 2))) land 0xffff)
  | F32 -> Int32.float_of_bits d.{(lane * 4) + i}
  | S32 -> Int32.to_float d.{(lane * 4) + i}

(* gamma_K = K u / (1 - K u) with u = 2^-24 for binary32 accumulation (the
   bound of docs/design/f16-relaxed-accuracy.md §5.2); 0 (bit-exact) for
   integers and for binary16 accumulation, whose operands keep every partial
   sum exact (see [random_operands]; §5.4 admits no f16-accumulate bound). *)
let bound_factor f =
  let gamma u =
    let ku = float_of_int f.k *. u in
    ku /. (1.0 -. ku)
  in
  match f.acc with F32 -> gamma (ldexp 1.0 (-24)) | F16_acc | S32 -> 0.0

(* Legacy wmma, 16x16x16, the forms nvrtc (CUDA 13.0) emits for nvcuda::wmma:
   load A and B (row-major, leading dimension 16), multiply with C = 0, store
   D. All-ones A and B make every D element 16 whatever the fragment layout. *)
let wmma_ptx ~version ~entry ~a_regs ~elem ~mma ~d_reg ~d_type ~zero =
  let regs prefix first n =
    "{"
    ^ String.concat
        ", "
        (List.init n (fun i -> Printf.sprintf "%%%s%d" prefix (first + i)))
    ^ "}"
  in
  let d = regs d_reg 10 8 in
  let c8 = "{" ^ String.concat ", " (List.init 8 (fun _ -> "%z0")) ^ "}" in
  Printf.sprintf
    ".version %s\n\
     .target sm_80\n\
     .address_size 64\n\
     .visible .entry %s(.param .u64 pA, .param .u64 pB, .param .u64 pD)\n\
     {\n\
     .reg .b32 %%r<20>; .reg .f32 %%f<20>; .reg .%s %%z0; .reg .u64 %%pa, \
     %%pb, %%pd;\n\
     .reg .u32 %%ld;\n\
     ld.param.u64 %%pa, [pA]; ld.param.u64 %%pb, [pB]; ld.param.u64 %%pd, [pD];\n\
     mov.u32 %%ld, 16; mov.%s %%z0, %s;\n\
     wmma.load.a.sync.aligned.row.m16n16k16.global.%s %s, [%%pa], %%ld;\n\
     wmma.load.b.sync.aligned.row.m16n16k16.global.%s %s, [%%pb], %%ld;\n\
     wmma.mma.sync.aligned.row.row.m16n16k16.%s %s, %s, %s, %s;\n\
     wmma.store.d.sync.aligned.row.m16n16k16.global.%s [%%pd], %s, %%ld;\n\
     ret;\n\
     }\n"
    version
    entry
    d_type
    d_type
    zero
    elem
    (regs "r" 0 a_regs)
    elem
    (regs "r" a_regs a_regs)
    mma
    d
    (regs "r" 0 a_regs)
    (regs "r" a_regs a_regs)
    c8
    d_type
    d

type wmma_form = {
  wlabel : string;
  entry : string;
  ptx : version:string -> string;
  one_word : int32;  (** a 32-bit word of all-ones A/B elements *)
  words : int;  (** 32-bit words holding the 256 A (or B) elements *)
  d_float : bool;
}

let wmma_forms =
  [
    {
      wlabel = "wmma m16n16k16 f16 -> f32 (load/mma/store)";
      entry = "wf16";
      ptx =
        wmma_ptx
          ~entry:"wf16"
          ~a_regs:8
          ~elem:"f16"
          ~mma:"f32.f32"
          ~d_reg:"f"
          ~d_type:"f32"
          ~zero:"0f00000000";
      one_word = 0x3C003C00l;
      words = 128;
      d_float = true;
    };
    {
      wlabel = "wmma m16n16k16 u8 -> s32 (load/mma/store)";
      entry = "wu8";
      ptx =
        wmma_ptx
          ~entry:"wu8"
          ~a_regs:2
          ~elem:"u8"
          ~mma:"s32.u8.u8.s32"
          ~d_reg:"r"
          ~d_type:"s32"
          ~zero:"0";
      one_word = 0x01010101l;
      words = 64;
      d_float = false;
    };
  ]
