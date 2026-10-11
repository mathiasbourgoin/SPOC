(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_f16_policy.accept_single_rounding on hardware
 * (specs/f16-narrowing-barrier.md §4).
 *
 * Four dequantisation shapes whose narrowings keep the barrier by default:
 * f16(d * q - dmin * m) (Q4_K), f16(d * q) (exact, but -0 when d < 0 and
 * q = 0), f16(ds * q) and f16(ds * q - dmin * m) with ds = d * dmin, a
 * 22-bit f32 scale whose product with q is not exact, q a 4-bit quant. With
 * the opt-in for this kernel, the generated source has no barrier, and the
 * device may round once where the interpreter rounds twice. The documented
 * bound is checked element by element over every finite binary16 [d] and
 * every byte pattern: |device - interpreter| <= 1 binary16 ulp of the larger
 * plus |p| * 2^-24 for the product term p of the output; a zero of either
 * sign matching a zero. How many elements differ is printed:
 * it shows the opt-in is live on this compiler.
 *
 * Skip-clean off ROCm.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

let k_dequant =
  [%kernel
    fun (out : float16 vector)
        (scales : float16 vector)
        (qs : int32 vector)
        (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then begin
        let d = float32_of_float16 scales.(tid / 64l) in
        let w = qs.(tid) in
        let dmin = float32_of_float16 scales.(tid / 64l lxor 1l) in
        let dm = dmin *. float ((w asr 4l) land 63l) in
        let ds = d *. dmin in
        let o = tid * 4l in
        out.(o) <- float16_of_float32 ((d *. float (w land 15l)) -. dm) ;
        out.(o + 1l) <- float16_of_float32 (d *. float ((w asr 8l) land 15l)) ;
        out.(o + 2l) <- float16_of_float32 (ds *. float ((w asr 16l) land 15l)) ;
        out.(o + 3l) <-
          float16_of_float32 ((ds *. float ((w asr 24l) land 15l)) -. dm)
      end]

let kernel_name = "f16_single_rounding_dequant"

let ir =
  match (snd k_dequant).Sarek.Kirc_types.body_ir with
  | Some ir -> {ir with Sarek_ir_types.kern_name = kernel_name}
  | None -> failwith "no IR"

let f16_value_of_bits b =
  let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
  let exp = (b lsr 10) land 0x1f and man = b land 0x3ff in
  if exp = 31 then None
  else if exp = 0 then Some (sign *. float_of_int man *. ldexp 1.0 (-24))
  else Some (sign *. float_of_int (1024 + man) *. ldexp 1.0 (exp - 25))

let scale_values =
  List.init 0x10000 Fun.id |> List.filter_map f16_value_of_bits |> Array.of_list

let words = Array.length scale_values * 64

(* Byte k of word t holds (t mod 64) * 4 + k. *)
let word t =
  let b k = ((t mod 64 * 4) + k) land 0xff in
  Int32.of_int (b 0 lor (b 1 lsl 8) lor (b 2 lsl 16) lor (b 3 lsl 24))

let field w shift mask = (Int32.to_int w asr shift) land mask

let inputs words =
  let scales = Vector.create Vector.float16 (Array.length scale_values) in
  Array.iteri (fun i x -> Vector.set scales i x) scale_values ;
  let qs = Vector.create Vector.int32 words in
  for t = 0 to words - 1 do
    Vector.set qs t (word t)
  done ;
  let out = Vector.create Vector.float16 (4 * words) in
  (out, Execute.[Vec out; Vec scales; Vec qs; Int words])

let run_device dev =
  let out, args = inputs words in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d 256)
    ~grid:(Execute.dims1d ((words + 255) / 256))
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let run_interp words =
  let out, args = inputs words in
  Execute.run_interpreter_vectors
    ~ir
    ~args
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d ((words + 63) / 64))
    ~parallel:false ;
  Vector.to_array out

(* The interpreter's semantics (checked against the interpreter on a slice
   below): every f32 operation rounds to f32, the narrowing to binary16. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let round16 = Sarek_interp.Sarek_float16.to_float16

(* The product term of output [i]: d * q, or ds * q with ds = d * dmin. *)
let product i =
  let d = scale_values.(i / 256) and k = i mod 4 in
  let scale = if k < 2 then d else f32 (d *. scale_values.(i / 256 lxor 1)) in
  scale *. float_of_int (field (word (i / 4)) (8 * k) 15)

let model i =
  let w = word (i / 4) in
  let dmin = scale_values.(i / 256 lxor 1) in
  let dm = f32 (dmin *. float_of_int (field w 4 63)) in
  let p = f32 (product i) in
  round16 (match i mod 4 with 0 | 3 -> f32 (p -. dm) | _ -> p)

let ulp16 x =
  let ax = Float.abs x in
  if ax < ldexp 1.0 (-14) then ldexp 1.0 (-24)
  else
    let _, e = Float.frexp ax in
    ldexp 1.0 (e - 11)

let within_bound i dev interp =
  if dev = interp then true (* includes +0 against -0 *)
  else if Float.abs dev = Float.infinity || Float.abs interp = Float.infinity
  then
    (* overflow on one side only: the other must be at the binary16 edge *)
    Float.sign_bit dev = Float.sign_bit interp
    && Float.min (Float.abs dev) (Float.abs interp) >= 65504.0
  else
    Float.abs (dev -. interp)
    <= ulp16 (Float.max (Float.abs dev) (Float.abs interp))
       +. (Float.abs (product i) *. ldexp 1.0 (-24))

let source_ok () =
  let src =
    Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir
  in
  let contains needle =
    let nl = String.length needle and hl = String.length src in
    let rec go i =
      i + nl <= hl && (String.sub src i nl = needle || go (i + 1))
    in
    go 0
  in
  let ok =
    (not (contains "__float2half(sarek_f32_barrier("))
    && contains "accept_single_rounding covers this kernel"
  in
  Printf.printf
    "    source: no barrier call, opt-in comment present: %s\n"
    (if ok then "OK" else "FAIL") ;
  ok

let device_ok dev =
  let got = run_device dev in
  let outside = ref 0 and differ = ref 0 in
  Array.iteri
    (fun i g ->
      let e = model i in
      if Int64.bits_of_float g <> Int64.bits_of_float e then incr differ ;
      if not (within_bound i g e) then begin
        if !outside < 5 then
          Printf.printf
            "    outside bound [%d]: device %h interpreter %h\n"
            i
            g
            e ;
        incr outside
      end)
    got ;
  Printf.printf
    "    device vs interpreter: %d/%d differ (single rounding), %d outside the \
     bound\n"
    !differ
    (Array.length got)
    !outside ;
  !outside = 0

(* The model is the interpreter: a slice of 1024 words checked directly. *)
let model_ok () =
  let got = run_interp 1024 in
  let bad = ref 0 in
  Array.iteri
    (fun i g ->
      if Int64.bits_of_float g <> Int64.bits_of_float (model i) then incr bad)
    got ;
  Printf.printf
    "    interpreter vs model (slice): %d/%d differ\n"
    !bad
    (Array.length got) ;
  !bad = 0

let () =
  match
    Device.init () |> Array.to_list
    |> List.filter (fun d -> d.Device.framework = "HIP")
  with
  | [] -> print_endline "[SKIP] no HIP device available"
  | dev :: _ ->
      Sarek_f16_policy.accept_single_rounding
        (Kernel kernel_name)
        ~reason:"test_hip_f16_single_rounding" ;
      Printf.printf "=== f16 single-rounding opt-in: %s ===\n" dev.Device.name ;
      let src = source_ok () in
      let model = model_ok () in
      let device = device_ok dev in
      if src && model && device then print_endline "[PASS]"
      else begin
        print_endline "[FAIL]" ;
        exit 1
      end
