type comparison = | Eq | Lt | Gt

val compOpp : comparison -> comparison

type positive = | XI of positive | XO of positive | XH

type z = | Z0 | Zpos of positive | Zneg of positive

module Pos : sig val succ : positive -> positive

val add : positive -> positive -> positive

val add_carry : positive -> positive -> positive

val pred_double : positive -> positive

val mul : positive -> positive -> positive

val iter : ('a1 -> 'a1) -> 'a1 -> positive -> 'a1

val div2 : positive -> positive

val div2_up : positive -> positive

val compare_cont : comparison -> positive -> positive -> comparison

val compare : positive -> positive -> comparison

val of_succ_nat : int -> positive end

module Z : sig val double : z -> z

val succ_double : z -> z

val pred_double : z -> z

val pos_sub : positive -> positive -> z

val add : z -> z -> z

val opp : z -> z

val sub : z -> z -> z

val mul : z -> z -> z

val pow_pos : z -> positive -> z

val pow : z -> z -> z

val compare : z -> z -> comparison

val leb : z -> z -> bool

val ltb : z -> z -> bool

val min : z -> z -> z

val of_nat : int -> z

val pos_div_eucl : positive -> z -> z * z

val div_eucl : z -> z -> z * z

val modulo : z -> z -> z

val div2 : z -> z

val shiftl : z -> z -> z

val shiftr : z -> z -> z end

val word_modulus : int -> z

val to_unsigned : int -> z -> z

val to_signed : int -> z -> z

val shift_amount : int -> z -> z

val ashr_w : int -> z -> z -> z

val lshr_w : int -> z -> z -> z

type shr_op = | OpShr | OpLshr32 | OpLshr64

type shr_flavour = | FlavS | FlavB

val shr_operand_ok : shr_op -> bool -> bool

val shr_flavour_of : shr_op -> shr_flavour

val shr_width_of : shr_op -> bool -> int

val shr_insn : shr_op -> bool -> shr_flavour * int

val shr_eval : shr_op -> bool -> z -> z -> z
