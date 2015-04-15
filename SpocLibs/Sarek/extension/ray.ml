open Spoc
let cpt = ref 0

let tot_time = ref 0.

let measure_time f s =
  Printf.printf "Using : %s\n%!" s;
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s : time %d : %Fs\n%!" s !cpt (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;

ktype point3 = {x : float; y : float; z : float;}
ktype color = {r : float; g : float; b : float;}
ktype sphere = {
    spos : point3;
    radius : float;
    scol : color;
    sshine : float;
  }
ktype plane ={
  ppos : point3;
  norm : point3;
  pcol : color;
  pshine : float;
}
ktype obj =
  | Sphere of sphere
  | Plane of plane
ktype light = {loc : point3; lcol : color;}
ktype hit = {dist : float; obj : obj;}
ktype matchres =
  | MatchHit of hit
  | MatchMiss
ktype ray = {orig : point3; dir: point3}


let raytrace objects nb_objs width height rays eye lights ambiant bounce =
  let point_diff  = fun x y ->
    {x = x.x -. y.x;
     y = x.y -. y.y;
     z = x.z -. y.z;
    }
  in
  let point_add  = fun x y ->
    {x = x.x +. y.x;
     y = x.y +. y.y;
     z = x.z +. y.z;
    }
  in
  let dotprod  = fun a b ->
    a.x *. b.x +. a.y *. b.y +. a.z *. b.z

  in
  let point_mag = fun p ->
    sqrt (dotprod p p)
  in
  let point_div = fun p s  ->
    {x = p.x /. s;
     y = p.y /. s;
     z = p.z /.s;
    }
  in
  let point_mul = fun p s  ->
    {x = p.x *. s;
     y = p.y *. s;
     z = p.z *.s;
    }
  in
  let unit_length  = fun p ->
    point_div p (point_mag p)
  in
  let distance_to obj origin direction =
    match obj with
    | Sphere s ->  
      let p = point_add origin
        (point_mul direction (dotprod (point_diff s.spos origin) direction))
      in 
      let d_cp = point_mag (point_diff p s.spos) in
      let sep = point_diff p origin 
      in
      Printf.printf "distance_to_sphere %g %g - %g %g\n" d_cp s.radius (dotprod sep direction) 0.;
      if d_cp >= s.radius || (dotprod sep direction) <= 0. then
        MatchMiss
      else
        MatchHit {dist = (point_mag sep) -. sqrt (s.radius *. s.radius -. d_cp *. d_cp);
                  obj = obj;}         
    | Plane p ->
      let theta = dotprod direction p.norm in
      if theta >= 0. then
        MatchMiss
      else
        MatchHit {dist = (dotprod (point_diff p.ppos origin) p.norm) /. theta;
                  obj = obj}



  in
  let castray = fun origin dir ->
    let r = ref MatchMiss in
    for i = 0 to nb_objs - 1 do
      let dist = distance_to (Mem.get objects i) origin dir in
      match dist with
      | MatchMiss  -> (); (*do nothing *)
      | MatchHit m -> (* keep only the nearest hit *)
        match !r with
        | MatchMiss -> r := dist
        | MatchHit old_d ->
          if old_d.dist > m.dist then
            r := dist;
    done;
    !r
   
      
  in 
  let add_color c1 c2 =
    {r = c1.r +. c2.r;
     g = c1.g +. c2.g;
     b = c1.b +. c2.b}
  in
  let scale_colour c1 s = 
    {r = c1.r *. s;
     g = c1.g *. s;
     b = c1.b *. s}
  in
  let mul_colour c1 c2 =
  {r = c1.r *. c2.r;
   g = c1.g *. c2.g;
   b = c1.b *. c2.b}
  
  in
  let applyLights = fun pos normal ->
    let color = ref {r = 0.; g = 0.; b = 0.} in
    for i = 0 to Vector.length lights - 1  do
      let light = Mem.get lights i in
      let lp_p = point_diff light.loc pos in
      let dist = point_mag lp_p in
      let dir = point_mul lp_p (1.0 /. dist) in
      let mag = (dotprod normal dir) /. (dist *. dist) in
      color := add_color !color {r= light.lcol.r *. mag; g= light.lcol.g *. mag; b = light.lcol.b *. mag};
      (* checks in haskell *)
    done;
    !color
  in
  let img = Array.init (width*height) (fun _ -> {r= 0.; g= 0.; b= 0.}) in
  let bce = ref bounce in
  while !bce > 0 do  
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let r = (Mem.get rays (y*width+x)) in
        let dir = r.dir in
        let orig = r.orig  in
        let hit = castray orig dir  in
        (match hit with
         | MatchMiss -> ()(*Printf.printf "MISS\n";*)
         | MatchHit m -> 
           let point = point_add  orig (point_mul dir m.dist) in
           let normal,shine,col = 
           (match m.obj with
             | Sphere s ->
               unit_length (point_diff point s.spos), s.sshine,s.scol
             | Plane p ->    
               p.norm,p.pshine, p.pcol)
           in
           let colour = img.(y*width+x) in
           let newdir =  point_diff dir  (point_mul normal (2.0 *. (dotprod normal dir))) in
           Mem.set rays (y*width+x) {orig = point; dir =  newdir};
           let direct = applyLights  point normal in
           let lightning = add_color direct  (Mem.get ambiant 0) in
           let light_in =
             if !bce = bounce then
               scale_colour lightning (1.0 -. shine)                
               else
             scale_colour lightning (shine)                      
           in
           img.(y*width+x) <- add_color col  (mul_colour light_in col) (* maybe clamp *);
           let r,g,b = 
             let c= img.(y*width+x) in
             c.r,c.g, c.b
           in
           Printf.printf "(%d, %d) r : %g, g : %g, b : %g\n" x y r g b;      
           
        );
      done
    done;
    bce := !bce - 1;
  done;
  img




let devid = 1


let _ =
  let devs = Devices.init ~only:Devices.OpenCL () in
  let dev = devs.(devid) in

  let width = 800 in
  let height = 600 in
  let fov = 100 in
  let zoom = 1 in
  let bounces = 4 in 
  let framerate = 30 in


  let light = {
    loc = {x=300.; y=(-300.); z=(-100.)};
    lcol = {r=150_000.; g=150_000.; b=150_000.};
  }
  in
  let ambiant_light =
    {r=0.3;
     g=0.3;
     b = 0.3;
    }
  in
  let lights = Vector.create (Vector.Custom customLight) 1 in
  let ambiant = Vector.create (Vector.Custom customColor) 1 in
  Mem.set lights 0 light;
  Mem.set ambiant 0 ambiant_light;
  
  let scene = Vector.create (Vector.Custom customObj) 5 in

  Mem.set scene 0
    (Sphere {
        spos = {x=0.; y=80.; z=0.};
        radius = 20.;
        scol = {r = 1.; g=0.3; b=1.};
        sshine = 0.4;
      });

  Mem.set scene 1
    (Sphere {
        spos = {x=0.; y=(-40.); z=200.};
        radius = 100.;
        scol = {r = 0.4; g=0.4; b=1.};
        sshine = 0.8;
      });

  Mem.set scene 2
    (Sphere {
        spos = {x=(0.); y=(-40.); z=(-200.)};
        radius = 100.;
        scol = {r = 0.4; g=0.4; b=1.};
        sshine = 0.5;
      });

  Mem.set scene 3
    (Sphere {
        spos = {x=0.; y=(-150.); z=(-100.)};
        radius = 50.;
        scol = {r = 1.; g=1.; b=1.};
        sshine = 0.8;
      });

  Mem.set scene 4
    (Plane {
        ppos = {x=0.; y=100.; z=0.};
        norm = {x=0.; y=(-0.9805807); z=(-0.19611613)};
        pcol = {r=1.; g=1.; b=1.};
        pshine = 0.2;
      });

 let eye = {x=(0.); y=(-100.); z=(-700.)}
  in
  let cast_view_ray = kern width height fov eye rays ->
    let point_diff  = fun x y ->
      {x = x.x -. y.x;
       y = x.y -. y.y;
       z = x.z -. y.z;
      }
    in
    let unit_length  = fun p ->
      let point_mag = fun p ->
        let dotprod  = fun a b ->
          a.x *. b.x +. a.y *. b.y +. a.z *. b.z
        in
        Math.Float32.sqrt (dotprod p p)
      in
      let point_div = fun p s  ->
        {x = p.x /. s;
         y = p.y /. s;
         z = p.z /.s;
        }
      in
      point_div p (point_mag p)
    in
    let open Std in
    let y = thread_idx_y + block_dim_y * block_idx_y in
    let x = thread_idx_x + block_dim_x * block_idx_x in
    if x < width && y < height then
      let aspect = (float width) /. (float height) in
      let fovX = (float fov) *. aspect in
      let fovY = float fov in
      let dir = 
        unit_length (
          point_diff
            {x = (float x) *. fovX;
             y = fovY *. (float (0 - y));
             z = 0.}
            eye.[<0>]) in
      rays.[<y*width+x>] <- {orig = eye.[<0>];
                             dir = dir}
                             
  in
  let rays = Vector.create (Vector.Custom customRay) (width*height) in
  let eyev = Vector.create (Vector.Custom customPoint3) 1 in
  Mem.set eyev 0 eye;

  let threadsPerBlock = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 16)
    | _ -> 16
  in
  let blocksPerGridX = (width + threadsPerBlock -1) / threadsPerBlock in
  let blocksPerGridY = (height + threadsPerBlock -1) / threadsPerBlock in

  let block = {Spoc.Kernel.blockX = threadsPerBlock;
               Spoc.Kernel.blockY = threadsPerBlock;
               Spoc.Kernel.blockZ = 1;} in
  let grid = {Spoc.Kernel.gridX = blocksPerGridX;
              Spoc.Kernel.gridY = blocksPerGridY;
              Spoc.Kernel.gridZ = 1;} in
  
  let name = dev.Spoc.Devices.general_info.Spoc.Devices.name in
  measure_time (fun () ->
      Kirc.run  cast_view_ray
        (width, height, fov, eyev, rays) 
        (block,grid) 0 dev;
      Mem.to_cpu rays ();
      Devices.flush dev ();) ("GPU "^name);


  let rawimg = raytrace scene (Vector.length scene) width height rays eye lights ambiant bounces in
  let colors = Array.init height 
      (fun y -> Array.init width 
          (fun x -> 
             let r,g,b = 
               let c= rawimg.(y*width+x) in
               c.r,c.g, c.b
             in
             Printf.printf "r : %g, g : %g, b : %g\n" r g b;
             Graphics.rgb (int_of_float (r *.255.) ) (int_of_float (g *. 255.)) (int_of_float (b *. 255.)))) in
  Graphics.open_graph "";
  Graphics.resize_window width height;
  Graphics.draw_image (Graphics.make_image colors) 0 0 ;
  Graphics.synchronize ();
  ignore (read_int ());

  
