open Spoc

external%kernel mandelbrot : Spoc.Vector.vint32 -> int -> int -> int -> unit = "kernels/Mandelbrot" "mandelbrot"


(*external%kernel test: int -> int -> int vector = "test_file" "test_fun"*)

let _ = Printf.printf "ok"
