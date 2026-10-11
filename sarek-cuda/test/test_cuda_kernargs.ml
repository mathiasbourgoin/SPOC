(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Kernel-argument marshalling (specs/cheap-repeated-launches.md US-4, C-12 and
    C-13): a launch makes two host allocations (value slots, pointer array)
    whatever the argument count, each value reaches the kernel bit-exact, and
    the blocks outlive a GC that runs between a stream launch and its
    synchronisation.

    The allocation-count and layout cases need no device. The e2e cases launch a
    hand-written PTX kernel and self-skip without a CUDA driver (they run on
    ZLUDA on a non-NVIDIA GPU as well). *)

open Ctypes
open Sarek_cuda
module K = Cuda_api.Kernel

let allocations_of f =
  let before = K.kernarg_blocks_allocated () in
  let r = f () in
  (K.kernarg_blocks_allocated () - before, r)

let ptr_args n = List.init n (fun i -> K.ArgPtr (Nativeint.of_int (16 * i)))

let test_two_allocations () =
  List.iter
    (fun n ->
      let count, _ = allocations_of (fun () -> K.marshal_args (ptr_args n)) in
      Alcotest.(check int)
        (Printf.sprintf "host blocks for %d arguments" n)
        2
        count)
    [0; 1; 3; 10]

let test_layout_and_values () =
  let i32 = -123456789l and i64 = 0x7EDCBA9876543210L in
  let f32 = Int32.float_of_bits 0x7F7FFFFFl and f64 = 1.0e-310 in
  let k =
    K.marshal_args
      [
        K.ArgInt32 i32;
        K.ArgInt64 i64;
        K.ArgFloat32 f32;
        K.ArgFloat64 f64;
        K.ArgPtr 0x1234_5678_9ABCn;
        K.ArgInt32 (-1l);
      ]
  in
  let slot i = K.kernarg_pointer k i in
  let addr i = raw_address_of_ptr (slot i) in
  let base = addr 0 in
  for i = 0 to 5 do
    Alcotest.(check bool)
      (Printf.sprintf "slot %d is 8-byte aligned" i)
      true
      (Nativeint.rem (addr i) 8n = 0n)
  done ;
  Alcotest.(check bool)
    "slots are contiguous 8-byte cells"
    true
    (List.for_all
       (fun i -> addr i = Nativeint.add base (Nativeint.of_int (8 * i)))
       [0; 1; 2; 3; 4; 5]) ;
  Alcotest.(check int32) "int32" i32 !@(from_voidp int32_t (slot 0)) ;
  Alcotest.(check int64) "int64" i64 !@(from_voidp int64_t (slot 1)) ;
  Alcotest.(check int32)
    "float32 bits"
    (Int32.bits_of_float f32)
    (Int32.bits_of_float !@(from_voidp float (slot 2))) ;
  Alcotest.(check int64)
    "float64 bits"
    (Int64.bits_of_float f64)
    (Int64.bits_of_float !@(from_voidp double (slot 3))) ;
  Alcotest.(check bool)
    "pointer"
    true
    (Nativeint.equal 0x1234_5678_9ABCn !@(from_voidp nativeint (slot 4))) ;
  Alcotest.(check int32) "second int32" (-1l) !@(from_voidp int32_t (slot 5))

(* out[0] = (s64)i32, out[1] = bits(f32), out[2] = i64, out[3] = bits(f64). *)
let mixed_ptx =
  {|.version 8.0
.target sm_86
.address_size 64

.entry mixed_args(
    .param .u64 param_out,
    .param .s32 param_i32,
    .param .f32 param_f32,
    .param .s64 param_i64,
    .param .f64 param_f64
)
{
    .reg .s32 %r<2>;
    .reg .b32 %u<2>;
    .reg .s64 %rd<4>;
    .reg .b64 %ud<2>;
    .reg .f32 %f<1>;
    .reg .f64 %d<1>;

    ld.param.u64 %rd0, [param_out];
    ld.param.s32 %r0, [param_i32];
    ld.param.f32 %f0, [param_f32];
    ld.param.s64 %rd1, [param_i64];
    ld.param.f64 %d0, [param_f64];
    cvt.s64.s32 %rd2, %r0;
    st.global.s64 [%rd0], %rd2;
    mov.b32 %u0, %f0;
    cvt.u64.u32 %ud0, %u0;
    st.global.u64 [%rd0+8], %ud0;
    st.global.s64 [%rd0+16], %rd1;
    mov.b64 %ud1, %d0;
    st.global.u64 [%rd0+24], %ud1;
    ret;
}
|}

type mixed = {i32 : int32; f32 : float; i64 : int64; f64 : float}

let sample =
  {
    i32 = -123456789l;
    f32 = Int32.float_of_bits 0x3F9E0652l;
    i64 = 0x7EDCBA9876543210L;
    f64 = 0.1 +. 0.2;
  }

let expected_words m =
  [|
    Int64.of_int32 m.i32;
    Int64.logand (Int64.of_int32 (Int32.bits_of_float m.f32)) 0xFFFFFFFFL;
    m.i64;
    Int64.bits_of_float m.f64;
  |]

(* True iff the four words match [expected_words m]. *)
let words_match out m =
  let want = expected_words m in
  let rec go i =
    i = 4 || (Bigarray.Array1.get out i = want.(i) && go (i + 1))
  in
  go 0

let with_device f =
  if not (Cuda_api.is_driver_available ()) then Alcotest.skip ()
  else begin
    Cuda_api.Device.init () ;
    f (Cuda_api.Device.get 0)
  end

let launch_mixed dev ~stream =
  let out = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 4 in
  Bigarray.Array1.fill out 0L ;
  let d_out = Cuda_api.Memory.alloc dev 4 Bigarray.int64 in
  Cuda_api.Memory.host_to_device ~src:out ~dst:d_out ;
  let kernel = K.load_from_ptx dev ~name:"mixed_args" ~ptx:mixed_ptx in
  K.launch
    kernel
    ~args:
      [
        K.ArgBuffer d_out;
        K.ArgInt32 sample.i32;
        K.ArgFloat32 sample.f32;
        K.ArgInt64 sample.i64;
        K.ArgFloat64 sample.f64;
      ]
    ~grid:(1, 1, 1)
    ~block:(1, 1, 1)
    ~shared_mem:0
    ~stream ;
  (* The launch is asynchronous: only the retained blocks keep the arguments
     alive across this GC. Reusing freed memory makes a freed block read as
     garbage. *)
  Gc.full_major () ;
  Gc.compact () ;
  ignore (Sys.opaque_identity (List.init 10_000 (fun _ -> Bytes.make 16 'x'))) ;
  (match stream with
  | Some s -> Cuda_api.Stream.synchronize s
  | None -> Cuda_api.Device.synchronize dev) ;
  Cuda_api.Memory.device_to_host ~src:d_out ~dst:out ;
  Cuda_api.Memory.free d_out ;
  out

let test_mixed_default () =
  with_device (fun dev ->
      let out = launch_mixed dev ~stream:None in
      Alcotest.(check bool)
        "mixed arguments bit-exact"
        true
        (words_match out sample) ;
      (* Positive control: the comparison goes red on a one-bit difference. *)
      let flipped =
        {
          sample with
          f64 =
            Int64.float_of_bits
              (Int64.logxor (Int64.bits_of_float sample.f64) 1L);
        }
      in
      Alcotest.(check bool)
        "control: a flipped f64 bit is detected"
        false
        (words_match out flipped))

let test_mixed_stream_gc () =
  with_device (fun dev ->
      let s = Cuda_api.Stream.create dev in
      let out = launch_mixed dev ~stream:(Some s) in
      Cuda_api.Stream.destroy s ;
      Alcotest.(check bool)
        "stream launch survives a GC before synchronisation"
        true
        (words_match out sample))

let () =
  Alcotest.run
    "cuda_kernargs"
    [
      ( "marshalling",
        [
          Alcotest.test_case
            "two host blocks per launch"
            `Quick
            test_two_allocations;
          Alcotest.test_case
            "slot layout and values"
            `Quick
            test_layout_and_values;
        ] );
      ( "e2e",
        [
          Alcotest.test_case
            "mixed types, default stream"
            `Quick
            test_mixed_default;
          Alcotest.test_case
            "mixed types, stream + GC"
            `Quick
            test_mixed_stream_gc;
        ] );
    ]
