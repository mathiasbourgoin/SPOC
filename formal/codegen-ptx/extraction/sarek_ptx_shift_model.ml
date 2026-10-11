type comparison = | Eq | Lt | Gt

(** val compOpp : comparison -> comparison **)

let compOpp = function | Eq -> Eq | Lt -> Gt | Gt -> Lt

type positive = | XI of positive | XO of positive | XH

type z = | Z0 | Zpos of positive | Zneg of positive

module Pos = struct (** val succ : positive -> positive **)

let rec succ = function | XI p -> XO (succ p) | XO p -> XI p | XH -> XO XH

(** val add : positive -> positive -> positive **)

let rec add x y = match x with | XI p -> (match y with | XI q -> XO (add_carry
  p q) | XO q -> XI (add p q) | XH -> XO (succ p)) | XO p -> (match y with |
  XI q -> XI (add p q) | XO q -> XO (add p q) | XH -> XI p) | XH -> (match y
  with | XI q -> XO (succ q) | XO q -> XI q | XH -> XO XH)

(** val add_carry : positive -> positive -> positive **)

and add_carry x y = match x with | XI p -> (match y with | XI q -> XI
  (add_carry p q) | XO q -> XO (add_carry p q) | XH -> XI (succ p)) | XO p ->
  (match y with | XI q -> XO (add_carry p q) | XO q -> XI (add p q) | XH -> XO
  (succ p)) | XH -> (match y with | XI q -> XI (succ q) | XO q -> XO (succ q)
  | XH -> XI XH)

(** val pred_double : positive -> positive **)

let rec pred_double = function | XI p -> XI (XO p) | XO p -> XI (pred_double
  p) | XH -> XH

(** val mul : positive -> positive -> positive **)

let rec mul x y = match x with | XI p -> add y (XO (mul p y)) | XO p -> XO
  (mul p y) | XH -> y

(** val iter : ('a1 -> 'a1) -> 'a1 -> positive -> 'a1 **)

let rec iter f x = function | XI n' -> f (iter f (iter f x n') n') | XO n' ->
  iter f (iter f x n') n' | XH -> f x

(** val div2 : positive -> positive **)

let div2 = function | XI p0 -> p0 | XO p0 -> p0 | XH -> XH

(** val div2_up : positive -> positive **)

let div2_up = function | XI p0 -> succ p0 | XO p0 -> p0 | XH -> XH

(** val compare_cont : comparison -> positive -> positive -> comparison **)

let rec compare_cont r x y = match x with | XI p -> (match y with | XI q ->
  compare_cont r p q | XO q -> compare_cont Gt p q | XH -> Gt) | XO p ->
  (match y with | XI q -> compare_cont Lt p q | XO q -> compare_cont r p q |
  XH -> Gt) | XH -> (match y with | XH -> r | _ -> Lt)

(** val compare : positive -> positive -> comparison **)

let compare = compare_cont Eq

(** val of_succ_nat : int -> positive **)

let rec of_succ_nat n = (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun
  _ -> XH) (fun x -> succ (of_succ_nat x)) n end

module Z = struct (** val double : z -> z **)

let double = function | Z0 -> Z0 | Zpos p -> Zpos (XO p) | Zneg p -> Zneg (XO
  p)

(** val succ_double : z -> z **)

let succ_double = function | Z0 -> Zpos XH | Zpos p -> Zpos (XI p) | Zneg p ->
  Zneg (Pos.pred_double p)

(** val pred_double : z -> z **)

let pred_double = function | Z0 -> Zneg XH | Zpos p -> Zpos (Pos.pred_double
  p) | Zneg p -> Zneg (XI p)

(** val pos_sub : positive -> positive -> z **)

let rec pos_sub x y = match x with | XI p -> (match y with | XI q -> double
  (pos_sub p q) | XO q -> succ_double (pos_sub p q) | XH -> Zpos (XO p)) | XO
  p -> (match y with | XI q -> pred_double (pos_sub p q) | XO q -> double
  (pos_sub p q) | XH -> Zpos (Pos.pred_double p)) | XH -> (match y with | XI q
  -> Zneg (XO q) | XO q -> Zneg (Pos.pred_double q) | XH -> Z0)

(** val add : z -> z -> z **)

let add x y = match x with | Z0 -> y | Zpos x' -> (match y with | Z0 -> x |
  Zpos y' -> Zpos (Pos.add x' y') | Zneg y' -> pos_sub x' y') | Zneg x' ->
  (match y with | Z0 -> x | Zpos y' -> pos_sub y' x' | Zneg y' -> Zneg
  (Pos.add x' y'))

(** val opp : z -> z **)

let opp = function | Z0 -> Z0 | Zpos x0 -> Zneg x0 | Zneg x0 -> Zpos x0

(** val sub : z -> z -> z **)

let sub m n = add m (opp n)

(** val mul : z -> z -> z **)

let mul x y = match x with | Z0 -> Z0 | Zpos x' -> (match y with | Z0 -> Z0 |
  Zpos y' -> Zpos (Pos.mul x' y') | Zneg y' -> Zneg (Pos.mul x' y')) | Zneg x'
  -> (match y with | Z0 -> Z0 | Zpos y' -> Zneg (Pos.mul x' y') | Zneg y' ->
  Zpos (Pos.mul x' y'))

(** val pow_pos : z -> positive -> z **)

let pow_pos z0 = Pos.iter (mul z0) (Zpos XH)

(** val pow : z -> z -> z **)

let pow x = function | Z0 -> Zpos XH | Zpos p -> pow_pos x p | Zneg _ -> Z0

(** val compare : z -> z -> comparison **)

let compare x y = match x with | Z0 -> (match y with | Z0 -> Eq | Zpos _ -> Lt
  | Zneg _ -> Gt) | Zpos x' -> (match y with | Zpos y' -> Pos.compare x' y' |
  _ -> Gt) | Zneg x' -> (match y with | Zneg y' -> compOpp (Pos.compare x' y')
  | _ -> Lt)

(** val leb : z -> z -> bool **)

let leb x y = match compare x y with | Gt -> false | _ -> true

(** val ltb : z -> z -> bool **)

let ltb x y = match compare x y with | Lt -> true | _ -> false

(** val min : z -> z -> z **)

let min n m = match compare n m with | Gt -> m | _ -> n

(** val of_nat : int -> z **)

let of_nat n = (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun _ -> Z0)
  (fun n0 -> Zpos (Pos.of_succ_nat n0)) n

(** val pos_div_eucl : positive -> z -> z * z **)

let rec pos_div_eucl a b = match a with | XI a' -> let (q, r) = pos_div_eucl
  a' b in let r' = add (mul (Zpos (XO XH)) r) (Zpos XH) in if ltb r' b then
  ((mul (Zpos (XO XH)) q), r') else ((add (mul (Zpos (XO XH)) q) (Zpos XH)),
  (sub r' b)) | XO a' -> let (q, r) = pos_div_eucl a' b in let r' = mul (Zpos
  (XO XH)) r in if ltb r' b then ((mul (Zpos (XO XH)) q), r') else ((add (mul
  (Zpos (XO XH)) q) (Zpos XH)), (sub r' b)) | XH -> if leb (Zpos (XO XH)) b
  then (Z0, (Zpos XH)) else ((Zpos XH), Z0)

(** val div_eucl : z -> z -> z * z **)

let div_eucl a b = match a with | Z0 -> (Z0, Z0) | Zpos a' -> (match b with |
  Z0 -> (Z0, a) | Zpos _ -> pos_div_eucl a' b | Zneg b' -> let (q, r) =
  pos_div_eucl a' (Zpos b') in (match r with | Z0 -> ((opp q), Z0) | _ ->
  ((opp (add q (Zpos XH))), (add b r)))) | Zneg a' -> (match b with | Z0 ->
  (Z0, a) | Zpos _ -> let (q, r) = pos_div_eucl a' b in (match r with | Z0 ->
  ((opp q), Z0) | _ -> ((opp (add q (Zpos XH))), (sub b r))) | Zneg b' -> let
  (q, r) = pos_div_eucl a' (Zpos b') in (q, (opp r)))

(** val modulo : z -> z -> z **)

let modulo a b = let (_, r) = div_eucl a b in r

(** val div2 : z -> z **)

let div2 = function | Z0 -> Z0 | Zpos p -> (match p with | XH -> Z0 | _ ->
  Zpos (Pos.div2 p)) | Zneg p -> Zneg (Pos.div2_up p)

(** val shiftl : z -> z -> z **)

let shiftl a = function | Z0 -> a | Zpos p -> Pos.iter (mul (Zpos (XO XH))) a
  p | Zneg p -> Pos.iter div2 a p

(** val shiftr : z -> z -> z **)

let shiftr a n = shiftl a (opp n) end

(** val word_modulus : int -> z **)

let word_modulus w = Z.pow (Zpos (XO XH)) (Z.of_nat w)

(** val to_unsigned : int -> z -> z **)

let to_unsigned w z0 = Z.modulo z0 (word_modulus w)

(** val to_signed : int -> z -> z **)

let to_signed w z0 = let u = to_unsigned w z0 in if Z.ltb u (Z.pow (Zpos (XO
  XH)) (Z.sub (Z.of_nat w) (Zpos XH))) then u else Z.sub u (word_modulus w)

(** val shift_amount : int -> z -> z **)

let shift_amount w k = Z.min (to_unsigned (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ 0)))))))))))))))))))))))))))))))) k)
  (Z.of_nat w)

(** val ashr_w : int -> z -> z -> z **)

let ashr_w w a k = to_unsigned w (Z.shiftr (to_signed w a) (shift_amount w k))

(** val lshr_w : int -> z -> z -> z **)

let lshr_w w a k = Z.shiftr (to_unsigned w a) (shift_amount w k)

type shr_op = | OpShr | OpLshr32 | OpLshr64

type shr_flavour = | FlavS | FlavB

(** val shr_operand_ok : shr_op -> bool -> bool **)

let shr_operand_ok op wide = match op with | OpShr -> true | OpLshr32 -> if
  wide then false else true | OpLshr64 -> wide

(** val shr_flavour_of : shr_op -> shr_flavour **)

let shr_flavour_of = function | OpShr -> FlavS | _ -> FlavB

(** val shr_width_of : shr_op -> bool -> int **)

let shr_width_of op wide = match op with | OpShr -> if wide then
  Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) else
  Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  0))))))))))))))))))))))))))))))) | OpLshr32 -> Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  0))))))))))))))))))))))))))))))) | OpLshr64 -> Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

(** val shr_insn : shr_op -> bool -> shr_flavour * int **)

let shr_insn op wide = ((shr_flavour_of op), (shr_width_of op wide))

(** val shr_eval : shr_op -> bool -> z -> z -> z **)

let shr_eval op wide a k = let w = shr_width_of op wide in (match
  shr_flavour_of op with | FlavS -> ashr_w w a k | FlavB -> lshr_w w a k)
