open Dom_html

let append_text e s = Dom.appendChild e (document##createTextNode (Js.string s))

let newLine _ = Dom_html.createBr document

module Printf = struct

  let printf f = 
    let body = 
      Js.Opt.get (document##getElementById (Js.string "section1")) 
    (fun () -> assert false)
    in
    Printf.ksprintf 
      (fun s -> 
       Dom.appendChild body (newLine ());
       append_text body s;
       Dom.appendChild body (newLine ())) f
  let sprintf = Printf.sprintf
end
