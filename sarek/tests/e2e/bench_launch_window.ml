(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Launch window benchmark (specs/launch-window.md, CHECK-4).
 *
 * A decode-like chain: per "layer", 18 launches shaped like one linear-
 * attention + dense-FFN layer of a ~0.8B transformer (row norm, four
 * independent matvecs on the same activation, element-wise gates, two
 * independent row norms, output matvec, residual add, norm, gate/up matvecs,
 * SwiGLU, down matvec, residual add). f32 weights, shared by all layers.
 * A "token" is [layers] layers followed by [Transfer.flush].
 *
 * For window off and on, interleaved over [reps] repetitions, it prints:
 *   - backend launches per token (window statistics),
 *   - wall microseconds per token (host + device, synchronised per token),
 *   - host microseconds per submitted launch (enqueue only, no flush).
 * Results of the last token are compared bit for bit between the two modes.
 *
 * It runs on every enumerated non-CPU device (HIP, CUDA, Metal, OpenCL,
 * Vulkan), or on those whose framework is listed in SPOC_BENCH_FRAMEWORK
 * (comma separated, e.g. "Metal" or "CUDA/PTX"). "Off" is the pre-window
 * behaviour (every launch issued at once); "on" is the backend's default.
 *
 *   SPOC_BENCH_TOKENS=50 SPOC_BENCH_REPS=5 SPOC_BENCH_LAYERS=24 \
 *     dune exec sarek/tests/e2e/bench_launch_window.exe
 ******************************************************************************)

[@@@warning "-32-33-34-69"]

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
open Sarek

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let env name default =
  match Sys.getenv_opt name with Some s -> int_of_string s | None -> default

let tokens = env "SPOC_BENCH_TOKENS" 50

let reps = env "SPOC_BENCH_REPS" 5

let layers = env "SPOC_BENCH_LAYERS" 24

let wanted_frameworks =
  match Sys.getenv_opt "SPOC_BENCH_FRAMEWORK" with
  | Some s -> Some (String.split_on_char ',' s)
  | None -> None

let selected (dev : Device.t) =
  (not dev.capabilities.is_cpu)
  &&
  match wanted_frameworks with
  | None -> true
  | Some l -> List.mem dev.framework l

(* y[row] = sum_k w[row*n + k] * x[k], 8 rows per 256-thread work-group. *)
let matvec =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (rows : int32)
        (n : int32) ->
      let%shared (red : float32) = 256l in
      let tid = thread_idx_x in
      let lane = tid land 31l in
      let row = (block_idx_x * 8l) + (tid / 32l) in
      let acc = mut 0.0 in
      let%superstep partial =
        if row < rows then begin
          let k = mut lane in
          while k < n do
            acc := acc +. (w.((row * n) + k) *. x.(k)) ;
            k := k + 32l
          done
        end ;
        red.(tid) <- acc
      in
      let%superstep r16 =
        if lane < 16l then red.(tid) <- red.(tid) +. red.(tid + 16l)
      in
      let%superstep r8 =
        if lane < 8l then red.(tid) <- red.(tid) +. red.(tid + 8l)
      in
      let%superstep r4 =
        if lane < 4l then red.(tid) <- red.(tid) +. red.(tid + 4l)
      in
      let%superstep r2 =
        if lane < 2l then red.(tid) <- red.(tid) +. red.(tid + 2l)
      in
      if lane = 0l && row < rows then y.(row) <- red.(tid) +. red.(tid + 1l)]

(* out[r*n + i] = x[r*n + i] / sqrt(mean(x[r]^2) + 1e-6) * w[i] *)
let rmsnorm =
  [%kernel
    fun (x : float32 vector)
        (w : float32 vector)
        (out : float32 vector)
        (n : int32) ->
      let%shared (red : float32) = 256l in
      let tid = thread_idx_x in
      let base = block_idx_x * n in
      let acc = mut 0.0 in
      let%superstep partial =
        let i = mut tid in
        while i < n do
          let v = x.(base + i) in
          acc := acc +. (v *. v) ;
          i := i + 256l
        done ;
        red.(tid) <- acc
      in
      let%superstep r128 =
        if tid < 128l then red.(tid) <- red.(tid) +. red.(tid + 128l)
      in
      let%superstep r64 =
        if tid < 64l then red.(tid) <- red.(tid) +. red.(tid + 64l)
      in
      let%superstep r32 =
        if tid < 32l then red.(tid) <- red.(tid) +. red.(tid + 32l)
      in
      let%superstep r16 =
        if tid < 16l then red.(tid) <- red.(tid) +. red.(tid + 16l)
      in
      let%superstep r8 =
        if tid < 8l then red.(tid) <- red.(tid) +. red.(tid + 8l)
      in
      let%superstep r4 =
        if tid < 4l then red.(tid) <- red.(tid) +. red.(tid + 4l)
      in
      let%superstep r2 =
        if tid < 2l then red.(tid) <- red.(tid) +. red.(tid + 2l)
      in
      let%superstep r1 =
        if tid < 1l then red.(tid) <- red.(tid) +. red.(tid + 1l)
      in
      let scale = 1.0 /. sqrt ((red.(0l) /. float n) +. 0.000001) in
      let i = mut tid in
      while i < n do
        out.(base + i) <- x.(base + i) *. scale *. w.(i) ;
        i := i + 256l
      done]

let gates =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (g : float32 vector)
        (beta : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then begin
        g.(i) <- exp (0.0 -. a.(i)) ;
        beta.(i) <- 1.0 /. (1.0 +. exp (0.0 -. b.(i)))
      end]

let silu_mul =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (out : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then begin
        let av = a.(i) in
        out.(i) <- av /. (1.0 +. exp (0.0 -. av)) *. b.(i)
      end]

let add_inplace =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (n : int32) ->
      let i = global_thread_id in
      if i < n then x.(i) <- x.(i) +. y.(i)]

let copy =
  [%kernel
    fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
      let i = global_thread_id in
      if i < n then dst.(i) <- src.(i) *. 0.5]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* Shapes modelled on a ~0.8B model: d = 1024, ffn = 3584, mixer 6144/2048. *)
let d = 1024

let ffn = 3584

let qkv = 6144

let inner = 2048

let heads = 16

let head_dim = 128

let fill v seed =
  let n = Vector.length v in
  for i = 0 to n - 1 do
    Vector.set
      v
      i
      (Float.of_int ((((i * 7919) + seed) mod 2003) - 1001) /. 8192.0)
  done

let vec ?(seed = 0) n =
  let v = Vector.create Vector.float32 n in
  fill v seed ;
  v

type model = {
  w_qkv : (float, Bigarray.float32_elt) Vector.t;
  w_z : (float, Bigarray.float32_elt) Vector.t;
  w_a : (float, Bigarray.float32_elt) Vector.t;
  w_b : (float, Bigarray.float32_elt) Vector.t;
  w_o : (float, Bigarray.float32_elt) Vector.t;
  w_g : (float, Bigarray.float32_elt) Vector.t;
  w_u : (float, Bigarray.float32_elt) Vector.t;
  w_d : (float, Bigarray.float32_elt) Vector.t;
  norm : (float, Bigarray.float32_elt) Vector.t;
  hnorm : (float, Bigarray.float32_elt) Vector.t;
}

let model () =
  {
    w_qkv = vec ~seed:1 (qkv * d);
    w_z = vec ~seed:2 (inner * d);
    w_a = vec ~seed:3 (heads * d);
    w_b = vec ~seed:4 (heads * d);
    w_o = vec ~seed:5 (d * inner);
    w_g = vec ~seed:6 (ffn * d);
    w_u = vec ~seed:7 (ffn * d);
    w_d = vec ~seed:8 (d * ffn);
    norm = vec ~seed:9 d;
    hnorm = vec ~seed:10 head_dim;
  }

type state = {
  x : (float, Bigarray.float32_elt) Vector.t;
  h : (float, Bigarray.float32_elt) Vector.t;
  t1 : (float, Bigarray.float32_elt) Vector.t;
  t2 : (float, Bigarray.float32_elt) Vector.t;
  t3 : (float, Bigarray.float32_elt) Vector.t;
  qn : (float, Bigarray.float32_elt) Vector.t;
  kn : (float, Bigarray.float32_elt) Vector.t;
  a : (float, Bigarray.float32_elt) Vector.t;
  b : (float, Bigarray.float32_elt) Vector.t;
  g : (float, Bigarray.float32_elt) Vector.t;
  beta : (float, Bigarray.float32_elt) Vector.t;
  f1 : (float, Bigarray.float32_elt) Vector.t;
  f2 : (float, Bigarray.float32_elt) Vector.t;
  f3 : (float, Bigarray.float32_elt) Vector.t;
}

let state () =
  {
    x = vec ~seed:11 d;
    h = vec d;
    t1 = vec qkv;
    t2 = vec inner;
    t3 = vec inner;
    qn = vec inner;
    kn = vec inner;
    a = vec heads;
    b = vec heads;
    g = vec heads;
    beta = vec heads;
    f1 = vec ffn;
    f2 = vec ffn;
    f3 = vec ffn;
  }

let layer dev m s =
  let open Execute in
  let go k args ~groups ~threads =
    run_vectors
      ~device:dev
      ~ir:(ir_of k)
      ~args
      ~block:(dims1d threads)
      ~grid:(dims1d groups)
      ()
  in
  let mv w x y rows n =
    go
      matvec
      [Vec w; Vec x; Vec y; Int rows; Int n]
      ~groups:((rows + 7) / 8)
      ~threads:256
  in
  let ew k args n =
    go k (args @ [Int n]) ~groups:((n + 255) / 256) ~threads:256
  in
  go rmsnorm [Vec s.x; Vec m.norm; Vec s.h; Int d] ~groups:1 ~threads:256 ;
  mv m.w_qkv s.h s.t1 qkv d ;
  mv m.w_z s.h s.t2 inner d ;
  mv m.w_a s.h s.a heads d ;
  mv m.w_b s.h s.b heads d ;
  ew gates [Vec s.a; Vec s.b; Vec s.g; Vec s.beta] heads ;
  ew copy [Vec s.t1; Vec s.t3] inner ;
  go
    rmsnorm
    [Vec s.t3; Vec m.hnorm; Vec s.qn; Int head_dim]
    ~groups:heads
    ~threads:256 ;
  go
    rmsnorm
    [Vec s.t2; Vec m.hnorm; Vec s.kn; Int head_dim]
    ~groups:heads
    ~threads:256 ;
  ew silu_mul [Vec s.qn; Vec s.kn; Vec s.t3] inner ;
  mv m.w_o s.t3 s.h d inner ;
  ew add_inplace [Vec s.x; Vec s.h] d ;
  go rmsnorm [Vec s.x; Vec m.norm; Vec s.h; Int d] ~groups:1 ~threads:256 ;
  mv m.w_g s.h s.f1 ffn d ;
  mv m.w_u s.h s.f2 ffn d ;
  ew silu_mul [Vec s.f1; Vec s.f2; Vec s.f3] ffn ;
  mv m.w_d s.f3 s.h d ffn ;
  ew add_inplace [Vec s.x; Vec s.h] d

let token dev m s =
  for _ = 1 to layers do
    layer dev m s
  done ;
  Transfer.flush dev

(* Runs [tokens] tokens; returns (µs per token, launches per token, result). *)
let measure dev m ~window =
  Execute.Window.set_enabled window ;
  let s = state () in
  token dev m s ;
  (* warm-up: uploads, compilation *)
  Execute.Window.reset_stats () ;
  let t0 = Unix.gettimeofday () in
  for _ = 1 to tokens do
    token dev m s
  done ;
  let dt = Unix.gettimeofday () -. t0 in
  let st = Execute.Window.stats () in
  let bits = Array.init d (fun i -> Int32.bits_of_float (Vector.get s.x i)) in
  (dt *. 1e6 /. float tokens, float st.issued /. float tokens, bits)

(* Host µs per submitted launch, enqueue only (window full of pending work is
   flushed by the bound, not by a sync). *)
let enqueue_cost dev m ~window =
  Execute.Window.set_enabled window ;
  let s = state () in
  token dev m s ;
  let t0 = Unix.gettimeofday () in
  for _ = 1 to layers do
    layer dev m s
  done ;
  let dt = Unix.gettimeofday () -. t0 in
  Transfer.flush dev ;
  dt *. 1e6 /. float (layers * 18)

let spread xs =
  let lo = List.fold_left min infinity xs
  and hi = List.fold_left max neg_infinity xs in
  let sorted = List.sort compare xs in
  (lo, List.nth sorted (List.length sorted / 2), hi)

let () =
  let devs = Device.init () in
  Array.iter
    (fun (dev : Device.t) ->
      if selected dev then begin
        Printf.printf
          "== %s [%s]: %d layers x 18 launches, %d tokens, %d reps ==\n%!"
          dev.name
          dev.framework
          layers
          tokens
          reps ;
        (* "On" must defer on this device: a backend outside the default set
           (Metal) is opted in for the run and restored afterwards. *)
        Execute.Window.set_enabled true ;
        let opted_in = not (Execute.Window.defers dev.framework) in
        if opted_in then
          Execute.Window.set_backend_enabled ~backend:dev.framework true ;
        Printf.printf
          "  window on = %s\n%!"
          (if opted_in then "opted in (backend is not in the default set)"
           else "the backend's default") ;
        let m = model () in
        let runs =
          List.init reps (fun _ ->
              let off = measure dev m ~window:false in
              let on = measure dev m ~window:true in
              (off, on))
        in
        let col f = List.map f runs in
        let us (u, _, _) = u and per (_, l, _) = l in
        let show label xs =
          let lo, med, hi = spread xs in
          Printf.printf
            "  %-34s min %8.1f  median %8.1f  max %8.1f\n%!"
            label
            lo
            med
            hi
        in
        show "window off: us/token" (col (fun (o, _) -> us o)) ;
        show "window on:  us/token" (col (fun (_, n) -> us n)) ;
        show "window off: launches/token" (col (fun (o, _) -> per o)) ;
        show "window on:  launches/token" (col (fun (_, n) -> per n)) ;
        let same = List.for_all (fun ((_, _, a), (_, _, b)) -> a = b) runs in
        let (_, _, ref_off), (_, _, ref_on) = List.hd runs in
        let off_det = List.for_all (fun ((_, _, a), _) -> a = ref_off) runs in
        (* A device whose window-off runs already disagree with each other is
           nondeterministic without the window; on/off equality says nothing
           there. *)
        Printf.printf
          "  off runs agree with each other: %b; on runs: %b\n%!"
          off_det
          (List.for_all (fun (_, (_, _, b)) -> b = ref_on) runs) ;
        Printf.printf "  results bit-identical on/off: %b\n%!" same ;
        let eo = List.init reps (fun _ -> enqueue_cost dev m ~window:false) in
        let en = List.init reps (fun _ -> enqueue_cost dev m ~window:true) in
        show "window off: host us/launch (enqueue)" eo ;
        show "window on:  host us/launch (enqueue)" en ;
        Execute.Window.set_enabled true ;
        if opted_in then
          Execute.Window.set_backend_enabled ~backend:dev.framework false ;
        if reps >= 2 && off_det && not same then exit 1
      end)
    devs
