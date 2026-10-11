(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host-side cost of one kernel launch through [Execute.run_vectors].
 *
 * Launches the same trivial kernel N times back to back WITHOUT synchronising
 * and reports host microseconds per launch (the enqueue cost), then the same
 * loop followed by one [Transfer.flush] (enqueue + drain). Two kernels: 3
 * arguments and 10 arguments, because the per-argument part of the cost is what
 * a many-argument decoder kernel pays most.
 *
 * Information only: prints numbers, asserts nothing beyond the result of one
 * launch being correct. Runs on every enumerated HIP and Native device, or,
 * when SPOC_BENCH_ONLY is set (comma-separated substrings), only on the devices
 * whose "framework: name" contains one of them. On a host with an integrated
 * GPU, restrict it (e.g. HIP_VISIBLE_DEVICES=0 SPOC_BENCH_ONLY="RX 7900,Native").
 *
 *   SPOC_BENCH_LAUNCHES=10000 (default) dune exec .../bench_launch_overhead.exe
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

let () = Sarek_native.Native_plugin.init ()

let k3 =
  [%kernel
    fun (out : float32 vector) (inp : float32 vector) (n : int32) ->
      let tid = global_thread_id in
      if tid < n then out.(tid) <- inp.(tid) +. 1.0]

let k10 =
  [%kernel
    fun (out : float32 vector)
        (inp : float32 vector)
        (idx : int32 vector)
        (w : float32 vector)
        (n : int32)
        (m : int32)
        (k : int32)
        (s : float32)
        (t : float32)
        (u : int32) ->
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          (inp.(tid) *. s) +. t +. w.(tid) +. float (idx.(tid) + m + k + u)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let n = 256

let launches =
  match Sys.getenv_opt "SPOC_BENCH_LAUNCHES" with
  | Some s -> int_of_string s
  | None -> 10_000

let only =
  match Sys.getenv_opt "SPOC_BENCH_ONLY" with
  | None | Some "" -> None
  | Some s -> Some (List.map String.trim (String.split_on_char ',' s))

let contains ~sub s =
  let n = String.length sub and m = String.length s in
  let rec at i = i + n <= m && (String.sub s i n = sub || at (i + 1)) in
  at 0

let selected (dev : Device.t) =
  List.mem dev.framework ["HIP"; "Native"]
  &&
  match only with
  | None -> true
  | Some subs ->
      let label = dev.framework ^ ": " ^ dev.name in
      List.exists (fun sub -> contains ~sub label) subs

let time_us f =
  let t0 = Unix.gettimeofday () in
  f () ;
  (Unix.gettimeofday () -. t0) *. 1e6

let bench dev label ir args =
  let launch () =
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args
      ~block:(Sarek.Execute.dims1d n)
      ~grid:(Sarek.Execute.dims1d 1)
      ()
  in
  (* Warm up: compile, first transfers. *)
  for _ = 1 to 10 do
    launch ()
  done ;
  Transfer.flush dev ;
  let enqueue =
    time_us (fun () ->
        for _ = 1 to launches do
          launch ()
        done)
  in
  Transfer.flush dev ;
  let with_flush =
    time_us (fun () ->
        for _ = 1 to launches do
          launch ()
        done ;
        Transfer.flush dev)
  in
  let per x = x /. float launches in
  Printf.printf
    "  %-12s %-8s %-6s enqueue %8.2f us/launch   enqueue+flush %8.2f us/launch\n\
     %!"
    dev.Device.framework
    label
    (Printf.sprintf "x%d" launches)
    (per enqueue)
    (per with_flush)

let () =
  Printf.printf "bench_launch_overhead (%d launches per measurement)\n" launches ;
  let devices = Device.init () in
  Array.iter
    (fun dev ->
      if selected dev then begin
        Printf.printf "%s: %s\n" dev.Device.framework dev.Device.name ;
        let out = Vector.create Vector.float32 n in
        let inp = Vector.create Vector.float32 n in
        let w = Vector.create Vector.float32 n in
        let idx = Vector.create Vector.int32 n in
        for i = 0 to n - 1 do
          Vector.set inp i (float i) ;
          Vector.set w i 0.5 ;
          Vector.set idx i 1l
        done ;
        bench
          dev
          "3-arg"
          (ir_of k3)
          Sarek.Execute.[Vec out; Vec inp; Int32 256l] ;
        (* 1 launch of k3 has run last: out = inp + 1. *)
        let ok = ref true in
        for i = 0 to n - 1 do
          if Vector.get out i <> float i +. 1.0 then ok := false
        done ;
        if not !ok then begin
          print_endline "  RESULT MISMATCH (3-arg)" ;
          exit 1
        end ;
        bench
          dev
          "10-arg"
          (ir_of k10)
          Sarek.Execute.
            [
              Vec out;
              Vec inp;
              Vec idx;
              Vec w;
              Int32 256l;
              Int32 1l;
              Int32 2l;
              Float32 2.0;
              Float32 0.25;
              Int32 3l;
            ] ;
        (* out = inp*2 + 0.25 + 0.5 + (1+1+2+3) *)
        for i = 0 to n - 1 do
          if Vector.get out i <> (float i *. 2.0) +. 0.25 +. 0.5 +. 7.0 then
            ok := false
        done ;
        if not !ok then begin
          print_endline "  RESULT MISMATCH (10-arg)" ;
          exit 1
        end
      end)
    devices
