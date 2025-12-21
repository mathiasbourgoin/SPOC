(******************************************************************************
 * Cross-module Sarek type definitions
 ******************************************************************************)

type float32 = float

type vec2 = {x : float32; y : float32} [@@sarek.type]

let[@sarek.module] add_vec (p : vec2) : float32 = p.x +. p.y
