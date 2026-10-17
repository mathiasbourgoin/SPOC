(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sub-group probe of an OpenCL device
    (docs/plans/warp-shuffle-intake-2026-10-17.md, section 6.5).

    OpenCL does not tie the sub-group lane to the linear local id. The warp
    shuffles are specified over the lane ([lane = linear thread id mod W]), so a
    device may run them only if that equality is MEASURED. The probe compiles
    one tiny kernel and runs it for two block shapes, 64x1 and 8x8; each
    work-item reports [get_sub_group_size()] and [get_sub_group_local_id()]. The
    device is accepted when every work-item reports the same size [W] and its
    lane equals its linear id modulo [W] in both shapes. [W] also replaces the
    constant 32 the plugin used to report as [warp_size].

    The probe is made once per device and cached by the caller. It uses its own
    context and queue, released before returning, so it never perturbs the
    plugin's per-device state. Any failure is "no sub-group support": the
    feature is not claimed. *)

type t = {
  width : int;  (** the sub-group size reported by the device *)
  lane_mapping_ok : bool;
      (** [lane = linear thread id mod width] in every probed shape *)
  failing_shape : (int * int) option;
      (** the first block shape (x, y) that broke the mapping *)
}

let probe_source =
  {|#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void sarek_warp_probe(__global int *out) {
  size_t t = get_local_id(0)
           + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));
  out[2 * t] = (int)get_sub_group_size();
  out[2 * t + 1] = (int)get_sub_group_local_id();
}
|}

(* The shapes of section 6.5 that fit a 64-thread block: 1-D and 2-D. *)
let shapes = [(64, 1); (8, 8)]

let shape_fits (d : Opencl_api.Device.t) (x, y) =
  x * y <= d.Opencl_api.Device.max_work_group_size
  && (d.max_work_item_dims >= if y > 1 then 2 else 1)
  && d.max_work_item_sizes.(0) >= x
  && (y = 1 || d.max_work_item_sizes.(1) >= y)

(* Run the probe kernel for one shape; returns (sizes, lanes) per work-item. *)
let run_shape queue kernel out (x, y) =
  let n = x * y in
  Opencl_api.Kernel.launch queue kernel ~global:(x, y, 1) ~local:(x, y, 1) ;
  Opencl_api.CommandQueue.finish queue ;
  let host = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout (2 * n) in
  Opencl_api.Memory.device_to_host queue ~src:out ~dst:host ;
  Array.init n (fun t ->
      ( Int32.to_int (Bigarray.Array1.get host (2 * t)),
        Int32.to_int (Bigarray.Array1.get host ((2 * t) + 1)) ))

let evaluate_shape width results =
  let ok = ref (64 mod width = 0) in
  Array.iteri
    (fun t (size, lane) ->
      if size <> width || lane <> t mod width then ok := false)
    results ;
  !ok

let measure (d : Opencl_api.Device.t) : t =
  let context = Opencl_api.Context.create d in
  let queue = Opencl_api.CommandQueue.create context () in
  Fun.protect
    ~finally:(fun () ->
      (try Opencl_api.CommandQueue.release queue with _ -> ()) ;
      try Opencl_api.Context.release context with _ -> ())
    (fun () ->
      let program =
        Opencl_api.Program.create_from_source context probe_source
      in
      Opencl_api.Program.build
        program
        ~options:(Opencl_api.Device.subgroup_build_options d)
        () ;
      let kernel = Opencl_api.Kernel.create program "sarek_warp_probe" in
      let out = Opencl_api.Memory.alloc context 128 Bigarray.int32 in
      Opencl_api.Kernel.set_arg_buffer kernel 0 out ;
      let usable = List.filter (shape_fits d) shapes in
      Fun.protect ~finally:(fun () ->
          (try Opencl_api.Memory.free out with _ -> ()) ;
          (try Opencl_api.Kernel.release kernel with _ -> ()) ;
          try Opencl_api.Program.release program with _ -> ())
      @@ fun () ->
      match usable with
      | [] -> {width = 0; lane_mapping_ok = false; failing_shape = None}
      | first :: _ ->
          let width =
            let sizes = run_shape queue kernel out first in
            fst sizes.(0)
          in
          let failing =
            List.find_opt
              (fun shape ->
                let results = run_shape queue kernel out shape in
                not (evaluate_shape width results))
              usable
          in
          {
            width;
            lane_mapping_ok = width > 0 && failing = None;
            failing_shape = failing;
          })

(** [probe d] measures [d], or [None] when the device cannot run the probe (no
    sub-group support, a compile or launch failure). Never raises. *)
let probe (d : Opencl_api.Device.t) : t option =
  try Some (measure d) with _ -> None

(** The capability a probe result earns: [Warp_shuffle] only when the lane
    mapping was measured, in every probed shape, on a device that reported a
    sub-group size. The extension strings alone never grant it. *)
let features_of_probe (probe : t option) : Sarek_ir_analysis.feature list =
  match probe with
  | Some {width; lane_mapping_ok = true; _} when width > 0 ->
      [Sarek_ir_analysis.Warp_shuffle]
  | _ -> []

(** The [warp_size] a device reports: the measured sub-group size, or 1 when the
    device has no sub-group support (no warp structure is known). It used to be
    the constant 32. *)
let warp_size_of_probe (probe : t option) : int =
  match probe with Some {width; _} when width > 0 -> width | _ -> 1
