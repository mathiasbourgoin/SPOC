(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Per-shape probe of the warp-level matrix PTX instructions ([mma.sync],
    [wmma.*]) on whatever CUDA driver is present.

    This is a PROBE, not a conformance test of NVIDIA hardware: it records which
    instruction forms the driver translates and whether their results are right.
    It was written against ZLUDA (CUDA-on-ROCm) on an RX 7900 XTX, where the
    answer differs from one form to the next
    (docs/optimization/l15b-mma-probe.md). The statuses recorded below are
    ZLUDA's behaviour, not NVIDIA's.

    One Alcotest case per instruction form, plus a control kernel:
    - the kernel does not load: on ZLUDA (which drops a kernel it cannot parse,
      then [cuModuleGetFunction] returns [CUDA_ERROR_NOT_FOUND]) or below sm_80
      the case SKIPs and prints the error; not translating is a capability
      answer. On another driver at sm_80 or above every form is a native
      instruction, so a load failure means the probe's PTX is wrong: FAIL.
    - the kernel runs: its output is checked against a CPU reference and the
      case FAILS on a wrong result. Integer and f16-accumulate forms must be
      bit-exact (their operands keep every partial sum exact). The
      f32-accumulate forms must stay within [gamma_K * sum |terms|] (C and the K
      products), the bound of docs/design/f16-relaxed-accuracy.md §5.2; their
      operands make binary16 accumulation leave that bound. The §5.4 positive
      controls are not part of this probe. The number of inexact elements is
      printed.
    - the kernel loads but allocation, copy or launch fails: the case FAILS.

    On a ZLUDA device the status recorded for ZLUDA v7-preview.3 (zluda-src
    af8a3b1, gfx1100, 2026-10-10) is printed beside the observed one. A
    difference is reported, never failed, so a ZLUDA upgrade re-runs the probe
    instead of breaking the suite. Skips cleanly with no CUDA driver. *)

open Sarek_cuda
open Ptx_mma_forms

(** {1 Running a kernel} *)

type failure = Not_loaded of string | Run_failed of string

type outcome =
  | Ran of (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
  | Failed of failure

let upload dev h =
  let d = Cuda_api.Memory.alloc dev (Bigarray.Array1.dim h) Bigarray.int32 in
  Cuda_api.Memory.host_to_device ~src:h ~dst:d ;
  d

(* Load [ptx], launch one warp with the [inputs] then an [out_len]-word output
   buffer as arguments, and return the output. *)
let run_kernel dev ~name ~ptx ~inputs ~out_len =
  match Cuda_api.Kernel.load_from_ptx dev ~name ~ptx with
  | exception e -> Failed (Not_loaded (Printexc.to_string e))
  | kernel -> (
      try
        let out = int32_buffer out_len in
        Bigarray.Array1.fill out 0x7fc00000l (* NaN / sentinel *) ;
        let d_out = upload dev out in
        let args =
          List.map (fun h -> Cuda_api.Kernel.ArgBuffer (upload dev h)) inputs
        in
        Cuda_api.Kernel.launch
          kernel
          ~args:(args @ [Cuda_api.Kernel.ArgBuffer d_out])
          ~grid:(1, 1, 1)
          ~block:(warp, 1, 1)
          ~shared_mem:0
          ~stream:None ;
        Cuda_api.Device.synchronize dev ;
        Cuda_api.Memory.device_to_host ~src:d_out ~dst:out ;
        Ran out
      with e -> Failed (Run_failed (Printexc.to_string e)))

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let is_zluda dev = contains dev.Cuda_api.Device.name "ZLUDA"

let version_of dev =
  let major, minor = dev.Cuda_api.Device.compute_capability in
  Sarek_cuda.Cuda_ptx_isa.(to_string (required_version ~major ~minor))

let status_name = function
  | Exact -> "runs, bit-exact"
  | Inexact_within_bound -> "runs, inexact within the bound"
  | Not_translated -> "not translated (kernel does not load)"
  | Wrong -> "runs, WRONG result"

let report dev ~recorded ~observed =
  if is_zluda dev then
    Printf.printf
      "  [INFO] recorded for ZLUDA v7-preview.3 (af8a3b1, gfx1100): %s; \
       observed: %s%s\n\
       %!"
      (status_name recorded)
      (status_name observed)
      (if recorded = observed then "" else "  ** DIFFERS from the record **")

(* Common handling of a kernel that did not produce output. *)
let not_ran dev ~label ~recorded = function
  | Not_loaded msg
    when is_zluda dev || fst dev.Cuda_api.Device.compute_capability < 8 ->
      Printf.printf "  [INFO] %s does not load: %s\n%!" label msg ;
      report dev ~recorded ~observed:Not_translated ;
      Alcotest.skip ()
  | Not_loaded msg ->
      Alcotest.failf
        "%s does not load on a non-ZLUDA sm_80+ device, where it is native: \
         the probe's PTX is suspect: %s"
        label
        msg
  | Run_failed msg ->
      Alcotest.failf
        "%s loaded but did not run (alloc, copy or launch): %s"
        label
        msg

(** {1 The checks} *)

let verify_mma dev f ops d =
  let inexact = ref 0 and outside = ref [] and worst = ref 0.0 in
  for lane = 0 to warp - 1 do
    for i = 0 to 3 do
      let row, col = cd_coord ~g:(lane / 4) ~t:(lane mod 4) i in
      let want, abs_sum = reference f ops row col in
      let got = output_value f d ~lane i in
      let err = Float.abs (got -. float_of_int want) in
      let bound = bound_factor f *. float_of_int abs_sum in
      if not (err = 0.0) then incr inexact ;
      if not (err <= bound) then outside := (row, col, got, want) :: !outside ;
      if bound > 0.0 then worst := Float.max !worst (err /. bound)
    done
  done ;
  Printf.printf
    "  [INFO] %s ran: %d/128 inexact, worst error/bound %.3g\n%!"
    f.label
    !inexact
    !worst ;
  List.iter
    (fun (row, col, got, want) ->
      Printf.printf "  [INFO]   D[%d][%d] = %.9g, want %d\n%!" row col got want)
    (List.filteri (fun i _ -> i < 6) (List.rev !outside)) ;
  let observed =
    if !outside <> [] then Wrong
    else if !inexact > 0 then Inexact_within_bound
    else Exact
  in
  report dev ~recorded:f.recorded ~observed ;
  Alcotest.(check int)
    "elements outside the bound (integers: not bit-exact)"
    0
    (List.length !outside)

let check_mma f dev =
  let ops = random_operands f in
  let pa, pb, pc = pack_inputs f ops in
  match
    run_kernel
      dev
      ~name:"probe"
      ~ptx:(kernel_ptx ~version:(version_of dev) f)
      ~inputs:[pa; pb; pc]
      ~out_len:(warp * cd_regs f)
  with
  | Ran d -> verify_mma dev f ops d
  | Failed why -> not_ran dev ~label:f.label ~recorded:f.recorded why

(* Control: the kernel of the first form with its matrix instruction replaced
   by one mov per D register. It must load and run, or a "does not load"
   verdict above could come from the kernel's structure instead of the
   instruction. *)
let check_control dev =
  let f = List.hd mma_forms in
  let body =
    String.concat
      " "
      (List.init (cd_regs f) (Printf.sprintf "mov.b32 %%d%d,7;"))
  in
  let pa, pb, pc = pack_inputs f (random_operands f) in
  match
    run_kernel
      dev
      ~name:"probe"
      ~ptx:(kernel_ptx ~version:(version_of dev) ~body f)
      ~inputs:[pa; pb; pc]
      ~out_len:(warp * cd_regs f)
  with
  | Ran d ->
      let sevens = List.init (warp * cd_regs f) (fun i -> d.{i} = 7l) in
      Alcotest.(check bool)
        "every lane stored 7"
        true
        (List.for_all Fun.id sevens)
  | Failed (Not_loaded msg | Run_failed msg) ->
      Alcotest.failf "the control kernel (no matrix instruction) failed: %s" msg

let check_wmma w dev =
  let ones () =
    let b = int32_buffer w.words in
    Bigarray.Array1.fill b w.one_word ;
    b
  in
  match
    run_kernel
      dev
      ~name:w.entry
      ~ptx:(w.ptx ~version:(version_of dev))
      ~inputs:[ones (); ones ()]
      ~out_len:256
  with
  | Ran d ->
      let value i =
        if w.d_float then Int32.float_of_bits d.{i} else Int32.to_float d.{i}
      in
      let wrong =
        List.filter (fun i -> value i <> 16.0) (List.init 256 Fun.id)
      in
      Printf.printf
        "  [INFO] %s ran: %d/256 differ from 16\n%!"
        w.wlabel
        (List.length wrong) ;
      report
        dev
        ~recorded:Not_translated
        ~observed:(if wrong = [] then Exact else Wrong) ;
      Alcotest.(check int) "elements different from 16" 0 (List.length wrong)
  | Failed why -> not_ran dev ~label:w.wlabel ~recorded:Not_translated why

(** {1 Runner} *)

let device =
  lazy
    (if not (Cuda_api.is_driver_available ()) then None
     else begin
       Cuda_api.Device.init () ;
       let dev = Cuda_api.Device.get 0 in
       let major, minor = dev.Cuda_api.Device.compute_capability in
       Printf.printf
         "  [INFO] device 0: %s, compute capability %d.%d\n%!"
         dev.Cuda_api.Device.name
         major
         minor ;
       Some dev
     end)

let on_device check () =
  match Lazy.force device with
  | Some dev -> check dev
  | None ->
      Printf.printf "  [SKIP] no CUDA device\n%!" ;
      (* Alcotest captures stdout: a printed skip that returns normally would
         render as a green [OK] under a name that claims the form ran. *)
      Alcotest.skip ()

let () =
  Alcotest.run
    "ptx_mma_probe"
    [
      ( "control",
        [
          Alcotest.test_case
            "same kernel without the matrix instruction runs"
            `Quick
            (on_device check_control);
        ] );
      ( "mma.sync",
        List.map
          (fun f -> Alcotest.test_case f.label `Quick (on_device (check_mma f)))
          mma_forms );
      ( "wmma",
        List.map
          (fun w ->
            Alcotest.test_case w.wlabel `Quick (on_device (check_wmma w)))
          wmma_forms );
    ]
