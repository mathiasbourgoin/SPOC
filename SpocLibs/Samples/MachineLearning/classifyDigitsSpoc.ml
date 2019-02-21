
open Spoc
open Kirc
    
let comp_ml = kern trainingSet data res setSize dataSize ->
  let open Std in
  let tid = thread_idx_x + block_dim_x * block_idx_x in
  if tid < setSize then
    (
      let mutable diff = 0 in
      let mutable toAdd = 0 in
      let mutable i = 0 in
      while(i < dataSize) do
        toAdd := data.[<i>] - trainingSet.[<tid*dataSize + i>];
        diff := diff + (toAdd * toAdd);
        i := i + 1;
      done;
      res.[<tid>] <- diff)
  else
    return ()
(*kernel comp_ml : Spoc.Vector.vint32 -> Spoc.Vector.vint32 -> Spoc.Vector.vint32 -> int -> int -> unit = "kernels/Ml_kernel" "kernel_compute"*)
 
type labelPixels = { label: int; pixels: int array }

let devices = Spoc.Devices.init ()


let dev = ref devices.(try int_of_string Sys.argv.(1) with | _ -> 0)
let auto_transfers = ref true
let compute_gpu = ref true
let trainSet_size = ref 5000 
let data_size = ref 784
    
let trainSet = Spoc.Vector.create Spoc.Vector.int32 (!trainSet_size * !data_size)
let results = Spoc.Vector.create Spoc.Vector.int32 !trainSet_size
let vect_validation = Spoc.Vector.create Spoc.Vector.int32 !data_size
let block = { Spoc.Kernel.blockX = 1024; Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1; }
let grid = { Spoc.Kernel.gridX = 5; Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1; }

let parse_args () =
  let arg1 = ("-device" , Arg.Int (fun i -> dev := devices.(i)), "number of the device [0]")
  and arg2 = ("-auto" , Arg.Bool (fun b -> auto_transfers := b), "let Spoc handles transfers automatically [false]")
  and arg3 = ("-compute" , Arg.Bool (fun b -> compute_gpu := b), "computation should be done on the gpu [true]") in
  Arg.parse ([arg1; arg2; arg3]) (fun s -> ()) ""

let load_data_to_GPU data =
  Printf.printf "Will use device : %s\n" (!dev).Spoc.Devices.general_info.Spoc.Devices.name;
  Printf.printf "Loading a vector with the training data\n%!";
  let f = (fun index elem ->
      let pix_list = elem.pixels in
      for i = 0 to !data_size-1 do
        Spoc.Mem.set trainSet (index * !data_size + i) (Int32.of_int pix_list.(i));
      done) in
  Array.iteri f data;
  if (not !auto_transfers) then begin
    Printf.printf "Transfering training data to GPU\n%!";
    Spoc.Mem.to_device trainSet !dev;
    Spoc.Mem.to_device results !dev
  end;
  (*Kirc.gen comp_ml  !dev;*)
  Printf.printf "Done\n%!"
  
let read_lines name : string list =
  Printf.printf "Reading data from file : %s\n%!" name;
  let ic = open_in name in
  let try_read () =
    try Some (input_line ic) with End_of_file -> None in
  let rec loop acc = match try_read () with
    | Some s -> loop (s :: acc)
    | None -> close_in ic; List.rev acc in
  loop []

let slurp_file file =
  List.tl (read_lines file)
  |> List.map (fun line -> Str.split (Str.regexp ",") line )
  |> List.map (fun numline -> List.map (fun (x:string) -> int_of_string x) numline)
  |> List.map (fun line ->
    { label= List.hd line;
      pixels= Array.of_list @@ List.tl line })
  |> Array.of_list
       

let array_fold_left2 f acc a1 a2 =
  let open Array in
  let len = length a1 in
  let rec iter acc i =
    if i = len then acc
    else
      let v1 = unsafe_get a1 i in
      let v2 = unsafe_get a2 i in
      iter (f acc v1 v2) (i+1)
  in
  iter acc 0

let distance p1 p2 =
  sqrt
  @@ float_of_int
  @@ array_fold_left2 (fun acc a b -> let d = a - b in acc + d * d) 0 p1 p2

let classify (pixels: int array) trainingset =
  fst (
    Array.fold_left (fun ((min_label, min_dist) as min) (x : labelPixels) ->
      let dist = distance pixels x.pixels in
      if dist < min_dist then (x.label, dist) else min)
      (max_int, max_float) (* a tiny hack *)
      trainingset)

let r = ref true

let classify_gpu (pixels: int array) trainingset =
  Array.iteri (fun i x -> Spoc.Mem.set vect_validation i (Int32.of_int x)) pixels;
  if (not !auto_transfers) then Spoc.Mem.to_device vect_validation !dev;
  if !r then
    (
      r := not !r;
      Kirc.profile_run comp_ml (trainSet, vect_validation, results, !trainSet_size, !data_size) (block, grid)  0 !dev;
      ignore(Kirc.gen comp_ml !dev); 
    )
  else
      Kirc.run comp_ml (trainSet, vect_validation, results, !trainSet_size, !data_size) (block, grid)  0 !dev;
  if (not !auto_transfers) then Spoc.Mem.to_cpu results ();
  Spoc.Devices.flush !dev ();
  let rec loop min id res =
    if id = !trainSet_size then res
    else begin
      let nmin = sqrt (Int32.to_float (Spoc.Mem.get results id)) in
      if min > nmin then let nres = id in (loop nmin (id+1) nres)
      else (loop min (id+1) res)
    end in
  trainingset.(loop max_float 0 0).label

let num_correct trainingset validationsample =
  load_data_to_GPU trainingset;
  if !compute_gpu then Array.fold_left (fun sum p -> sum + if classify_gpu p.pixels trainingset = p.label then 1 else 0) 0 validationsample
  else Array.fold_left (fun sum p -> sum + if classify p.pixels trainingset = p.label then 1 else 0) 0 validationsample
      
let _ =
  parse_args ();
  let validationsample = slurp_file "./validationsample.csv" in
  let trainingset = slurp_file "./trainingsample.csv" in

  Printf.printf "\nResults with training set of %i vectors of length %i\n%!" !trainSet_size !data_size;
  Printf.printf "Percentage correct:%f\n" ((float_of_int(num_correct trainingset validationsample)/. (float_of_int(Array.length validationsample)))*.100.0)

