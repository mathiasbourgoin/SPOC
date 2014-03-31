open Dom_html

module Printf = struct
  let printf f = 
    Printf.ksprintf 
      (fun s -> document##write(Js.string (s^"<BR>")); ) f
end
