open Spoc
open Kernel
open Vector

type point = {
  mutable x: float;
  mutable y: float;
}

external getsizeofpoint: unit -> int = "custom_getsizeofpoint"
external extget: customarray -> int -> point = "custom_extget"
external extset: customarray -> int -> point -> unit = "custom_extset"

let customPoint = {
  size = getsizeofpoint();
  get = extget;
  set = extset;
}

let nbPoint = 2_000_000;;
let ray = 10.0;;


kernel gpuPi : point Spoc.Vector.vcustom -> Spoc.Vector.vfloat32 -> int -> float -> unit = "kernels/CustomPi" "pi"
 


let cpuPI field size =
  let resVec =
    Array.map (fun pt -> ((pt.x *. pt.x) +. (pt.y *. pt.y)) <= (ray *. ray))
      field in
  let res = Array.fold_left (fun i res -> if res then i + 1 else i) 0 resVec
  in res

let gpuPI gpuField vbool dev size =
  let threadsPerBlock =  match dev.Devices.specific_info with
          | Devices.OpenCLInfo clI -> 
            (match clI.Devices.device_type with
              | Devices.CL_DEVICE_TYPE_CPU -> 1
              | _  ->   256)
          | _  -> 256 in
  let blocksPerGrid =
    (((size / 32) + threadsPerBlock) - 1) / threadsPerBlock in
  let block =
    {
      Spoc.Kernel.blockX = threadsPerBlock;
      Spoc.Kernel.blockY = 1;
      Spoc.Kernel.blockZ = 1;
    }
  and grid =
    {
      Spoc.Kernel.gridX = blocksPerGrid;
      Spoc.Kernel.gridY = 1;
      Spoc.Kernel.gridZ = 1;
    }
  in
  (Spoc.Kernel.run dev (block, grid) gpuPi
      (gpuField, vbool, nbPoint, (ray *. ray));
    let pio4 =
      int_of_float
        (Tools.fold_left (fun a b -> a +. b) 0. vbool)
    in pio4)

let multiGpuPI gpuField1 gpuField2 vbool1 vbool2 dev1 dev2 size =
  let threadsPerBlock = 512 in
  let blocksPerGrid =
    ((((size / 32) / 2) + threadsPerBlock) - 1) / threadsPerBlock in
  let block =
    {
      Spoc.Kernel.blockX = threadsPerBlock;
      Spoc.Kernel.blockY = 1;
      Spoc.Kernel.blockZ = 1;
    }
  and grid =
    {
      Spoc.Kernel.gridX = blocksPerGrid;
      Spoc.Kernel.gridY = 1;
      Spoc.Kernel.gridZ = 1;
    }
  in
  (Spoc.Kernel.run dev1 (block, grid) gpuPi
      (gpuField1, vbool1, (size / 2), (ray *. ray));
    Spoc.Kernel.run dev2 (block, grid) gpuPi
      (gpuField2, vbool2, (size / 2), (ray *. ray));
    let pio4_1 =
      int_of_float
        (Tools.fold_left (fun a b -> a +. b) 0. vbool1)
    and pio4_2 =
      int_of_float
        (Tools.fold_left (fun a b -> a +. b) 0. vbool2)
    in
    let pio4 = pio4_1 + pio4_2 in pio4)

let _ =
  (Random.self_init ();
      let devices = Spoc.Devices.init () in
      let dev = ref devices.(0) in
      let size = ref nbPoint in
      let arg1 = ("-device" , Arg.Int (fun i  -> dev := devices.(i)), "number of the device [0]")
      and arg2 = ("-size" , Arg.Int (fun i  -> size := i), "width of the image to compute [2_000_000]")
      in
      Arg.parse ([arg1;arg2;]) (fun s -> ()) "";

    let pio4 = ref 0 in (*CPU COMPUTATION*)
    let t0 = Unix.gettimeofday () in
    let field =
      Array.init !size
        (fun _ -> { x = Random.float ray; y = Random.float ray; }) in
    (pio4 := cpuPI field !size;
     Printf.printf "CPU Computation : PI = %d/%d = %.10G\n" !pio4 !size
       ((4. *. (float !pio4)) /. (float !size));
     
      (* GPU COMPUTATION *)
    
      (
        Printf.printf "Will use device : %s\n" !dev.Spoc.Devices.general_info.Spoc.Devices.name;
        Spoc.Kernel.compile !dev gpuPi;
        Spoc.Mem.auto_transfers false;
        let vbool = Spoc.Vector.create float32 !size in
        let gpuField =
          Spoc.Tools.map
            (fun _ -> { x = Random.float ray; y = Random.float ray; })
            (Custom customPoint) vbool
        in
        (Spoc.Mem.to_device gpuField ~queue_id: 1 !dev;
          Spoc.Mem.to_device vbool ~queue_id: 0 !dev ;
          pio4 := gpuPI gpuField vbool !dev !size;
          Printf.printf "GPU Computation : PI = %d/%d = %.10G\n" !pio4 !size
            ((4. *. (float !pio4)) /. (float !size));
          Pervasives.flush stdout;
        ))))

