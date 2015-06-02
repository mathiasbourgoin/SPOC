open Spoc

let tot_time = ref 0.

let measure_time f s =
  Printf.printf "Using : %s --> %!" s;
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "   %Fs\n%!"  (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
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
ktype color_shine = {color : color; shine : float}

let raytrace objects nb_objs width height rays eye lights ambiant bounce imgs shine_colors =
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
      (*Printf.printf "distance_to_sphere %g %g - %g %g\n" d_cp s.radius (dotprod sep direction) 0.;*)
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
  let checkray = fun orig dir dist ->
    let r = ref true in
    let hit = ref false in
    let i = ref 0 in
    while not !hit && (!i < nb_objs - 1 ) do
      let hit2 = distance_to (Mem.get objects !i) orig dir in
      match hit2 with
      | MatchMiss -> 
        (r := false; incr i; )
      | MatchHit m ->  
        (r := (m.dist < dist);
         hit := true;)
    done;
    !r
  in
  let applyLights = fun pos normal dist_ ->
    let color = ref {r = 0.; g = 0.; b = 0.} in
    for i = 0 to Vector.length lights - 1  do
      let light = Mem.get lights i in
      let lp_p = point_diff light.loc pos in
      let dist = point_mag lp_p in
      let dir = point_mul lp_p (1.0 /. dist) in
      let mag = (dotprod normal dir) /. (dist *. dist) in
      let refl = {r= light.lcol.r *. mag; 
                  g= light.lcol.g *. mag; 
                  b = light.lcol.b *. mag} in
      if not (checkray pos dir dist) then
        color := add_color !color refl
    done;
    !color
  in
  let clamp_colour c =
    let clamp x =
      if x < 0. then 0.
      else if x > 1. then 1.
      else x in
    {r = clamp c.r;
     g = clamp c.g;
     b = clamp c.b;}
  in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let bce = ref bounce in
      while !bce > 0 do
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
		p.norm,p.pshine,
                let x_,z_ = point.x, point.z in
                let v1 = (int_of_float (x_ /. 100.) + 1) mod 2 in
                let v2 = (int_of_float (z_ /. 100.) + 1) mod 2 in
                let v3 = (if x_ < 0.000 then 1 else 0) in
                let v4 = (if z_ < 0.000 then 1 else 0) in
                if v1 lxor v2 lxor v3 lxor v4 = 1 then
                  p.pcol
                else
                  {r = 0.4; g=0.4; b=0.4}

             )
           in
           let newdir =  point_diff dir  (point_mul normal (2.0 *. (dotprod normal dir))) in
           Mem.set rays (y*width+x) {orig = point; dir =  newdir};
	   let idx = (bounce - !bce)*width*height in
           let colour = imgs.(idx + y*width+x) in
           let direct = applyLights point normal m.dist in
           let lighting = add_color direct  (Mem.get ambiant 0) in
	   imgs.(idx + y*width+x) <- lighting;
           shine_colors.(idx + y*width+x) <- {color = col; shine = shine;});
	  bce := !bce - 1;
      done;
      bce := bounce - 1;
      (* recursion would be way more sexy *)
      while !bce >= 0 do
	let idx = (!bce)*width*height in
	let pred = (!bce+1)*width*height in
	let lighting = imgs.(idx + y*width+x) in
	let shine = shine_colors.(idx + y*width+x).shine in
	let col = shine_colors.(idx + y*width+x).color in

	let refl = 
	  if !bce = bounce - 1 then
	    {r = 0.; g = 0.; b = 0.}
	  else
	    imgs.(pred + y*width+x) in
	let light_in =
	  add_color
	    (scale_colour refl shine)
	    (scale_colour lighting (1.0 -. shine))
	in 
	let light_out = (mul_colour light_in col) in
        imgs.(idx + y*width+x) <- clamp_colour light_out;
      	bce := !bce -1;
      done;
    done;
  done

	 
	 


let devid = 0


let _ =
  let devs = Devices.init ~only:Devices.OpenCL () in
  let dev = devs.(devid) in

  let width = 800 in
  let height = 600 in
  let fov = 1. in
  let zoom = 1 in
  let bounces = 4 in 
  let framerate = 30 in


  let light = {
    loc = {x=(300.); y=(-300.); z=(-100.)};
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

  Mem.set scene 1
    (Sphere {
        spos = {x=40.; y=80.; z=0.};
        radius = 20.;
        scol = {r = 1.; g=0.3; b=1.};
        sshine = 0.4;
      });

  Mem.set scene 2
    (Sphere {
        spos = {x=(-200.); y=(0.); z=200.};
        radius = 100.;
        scol = {r = 0.4; g=0.4; b=1.};
        sshine = 0.8;
      });

  Mem.set scene 3
    (Sphere {
        spos = {x=(200.); y=(0.); z=(-200.)};
        radius = 100.;
        scol = {r = 0.4; g=0.4; b=1.};
        sshine = 0.5;
      });

  Mem.set scene 4
    (Sphere {
        spos = {x=(0.); y=(-150.); z=(-100.)};
        radius = 50.;
        scol = {r = 1.; g=1.; b=1.5};
        sshine = 0.8;
      });

  Mem.set scene 0
    (Plane {
        ppos = {x=0.; y=100.; z=0.};
        norm = {x=0.; y=(-0.9805807); z=(-0.19611613)};
        pcol = {r=1.; g=1.; b=1.};
        pshine = 0.2;
      });
  let eye = {x=(50.); y=(-200.); z=(-700.)}
  in
  let rays = Vector.create (Vector.Custom customRay) (width*height) in
  let eyev = Vector.create (Vector.Custom customPoint3) 1 in
  Mem.set eyev 0 eye;
  
  let cast_view_ray = kern width height fov eye rays ->
    let point_diff  = fun a b ->
      {x = a.x -. b.x;
       y = a.y -. b.y;
       z = a.z -. b.z;
      }
    in
    let normalise  = fun p ->
      let mag = fun p ->
        Math.Float32.sqrt (p.x *. p.x +. p.y *. p.y +. p.z *. p.z)
      in
      let mul = fun s p  ->
        {x = p.x *. s;
         y = p.y *. s;
         z = p.z *. s;
        }
      in
      mul (1.0 /. (mag p)) p
    in
    let open Std in
    let y_ = thread_idx_y + block_dim_y * block_idx_y in
    let x_ = thread_idx_x + block_dim_x * block_idx_x in
    if x_ < width && y_ < height then
      (
        let aspect = (float width) /. (float height) in
        let fovX = (fov) *. aspect in
        let fovY = fov in
        
        (* let midx = (width / 2) in *)
        (* let fsizex2 = (Std.float width) /. 2. in *)
        (* let x = (Std.float (x_ - midx)) /. fsizex2 in *)

        (* let midy = (height / 2) in *)
        (* let fsizey2 = (Std.float height) /. 2. in *)
        (* let y = (Std.float (y_ - midy)) /. fsizey2 in *)
        let x = float ((width / 2) - x_) in
        let y = float ((height / 2) - y_)in
        
        
        let dir = 
          normalise (
              point_diff
		{x = (x) *. fovX;
		 y = fovY *.  ( 0. -. y);
		 z = 0.}
		eye.[<0>]) in
        rays.[<y_*width+x_>] <- {orig = eye.[<0>];
                                 dir = dir}
      )
  in

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

  let shine_colors = 
    Array.init (width*height*bounces)
	       (fun _ ->
		{color = {r= 0.; g= 0.; b= 0.};
		 shine = 0.}) in
  let imgs =
    Array.init (width*height*bounces)
	       (fun _ ->
		{r= 0.; g= 0.; b= 0.}) in
  
  measure_time 
    (fun () -> 
     raytrace scene (Vector.length scene)
	      width height rays eye lights ambiant bounces imgs shine_colors)
    "CPU to raytrace";
  
  (*let rawimg = imgs.(0) in*)
  
  let colors =
    measure_time
      (fun () -> Array.init height 
      (fun y -> Array.init width 
          (fun x -> 
             let r,g,b = 
               let c= imgs.(y*width+x) in
               c.r,c.g, c.b
             in
             (*Printf.printf "r : %g, g : %g, b : %g\n" r g b;*)
             Graphics.rgb (int_of_float (r *.255.) ) (int_of_float (g *. 255.)) (int_of_float (b *. 255.))))
      ) "CPU to convert into colors"
  in
  Graphics.open_graph "";
  Graphics.resize_window width height;
  Graphics.draw_image (Graphics.make_image colors) 0 0 ;
  Graphics.synchronize ();
  ignore(read_line ());
  let image = Rgb24.make width height
			 {Color.Rgb.r = 0; g=0; b=0; } in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let r,g,b =
	let c= imgs.(y*width+x) in
	let i x = int_of_float (x *. 255.) in
	i c.r, i c.g, i c.b
      in
      Rgb24.set image x y {Color.Rgb.r = r; g=g; b=b;}
    done
  done;
  Png.save "ray.png" [] (Images.Rgb24 image)

  
