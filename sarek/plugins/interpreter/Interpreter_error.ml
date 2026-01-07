(** Error handling for Interpreter backend plugin *)

include Spoc_framework.Backend_error.Make (struct
  let name = "Interpreter"
end)
