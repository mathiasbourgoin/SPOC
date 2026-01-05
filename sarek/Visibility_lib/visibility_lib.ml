type float32 = float

let[@sarek.module] public_add (x : float32) (y : float32) : float32 = x +. y

let[@sarek.module_private] private_scale (x : float32) : float32 = x *. 2.0
