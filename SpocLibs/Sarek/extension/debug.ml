let retype = ref false
let unknown = ref 0
let debug = false

let my_eprintf s =
  if debug then
    (output_string stderr s;
     flush stderr;)
  else ()
