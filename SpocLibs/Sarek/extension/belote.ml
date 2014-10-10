open Spoc

ktype couleur = Pique | Coeur | Carreau | Trefle ;;

ktype coulval = {c:couleur; v : int32} ;;

ktype carte = 
  As of couleur
  | Roi of couleur
  | Dame of couleur
  | Valet of couleur 
  | Autre of coulval ;;

let  calcul_valeur  = kern cartes atout valeurs  ->
  let valeur = fun a atout->
    match a with 
    | As c -> 11
    | Roi c -> 4
    | Dame c -> 3
    | Valet c -> if c = atout then 20 else 2
    | Autre cv ->
      if cv.v = 10 then 10 else if (cv.c = atout) && (cv.v = 9) then 14 else 0 
  in
  let open Std in
  let i = thread_idx_x + block_dim_x * block_idx_x in
  valeurs.[<i>] <- valeur  cartes.[<i>] atout.[<0>]
      




let _ = 
  let devs = Spoc.Devices.init ~only:Devices.OpenCL() in
  let dev = devs.(0) in
  ignore(Kirc.gen ~only:Devices.OpenCL calcul_valeur);
  Printf.printf "%s\n%!" (List.hd ((fst calcul_valeur)#get_opencl_sources ()));
  (fst calcul_valeur)#compile ~debug:true dev;
