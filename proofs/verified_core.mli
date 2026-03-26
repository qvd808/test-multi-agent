
type __ = Obj.t

type bool =
| True
| False

type nat =
| O
| S of nat

type ('a, 'b) sum =
| Inl of 'a
| Inr of 'b

type ('a, 'b) prod =
| Pair of 'a * 'b

val snd : ('a1, 'a2) prod -> 'a2

type comparison =
| Eq
| Lt
| Gt

val compOpp : comparison -> comparison

val id : 'a1 -> 'a1

type 'a sig0 = 'a
  (* singleton inductive, whose constructor was exist *)

type sumbool =
| Left
| Right

type 'a sumor =
| Inleft of 'a
| Inright

val add : nat -> nat -> nat

val mul : nat -> nat -> nat

type positive =
| XI of positive
| XO of positive
| XH

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Pos :
 sig
  type mask =
  | IsNul
  | IsPos of positive
  | IsNeg
 end

module Coq_Pos :
 sig
  val succ : positive -> positive

  val add : positive -> positive -> positive

  val add_carry : positive -> positive -> positive

  val pred_double : positive -> positive

  type mask = Pos.mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  val succ_double_mask : mask -> mask

  val double_mask : mask -> mask

  val double_pred_mask : positive -> mask

  val sub_mask : positive -> positive -> mask

  val sub_mask_carry : positive -> positive -> mask

  val sub : positive -> positive -> positive

  val mul : positive -> positive -> positive

  val size_nat : positive -> nat

  val compare_cont : comparison -> positive -> positive -> comparison

  val compare : positive -> positive -> comparison

  val max : positive -> positive -> positive

  val ggcdn :
    nat -> positive -> positive -> (positive, (positive, positive) prod) prod

  val ggcd :
    positive -> positive -> (positive, (positive, positive) prod) prod

  val iter_op : ('a1 -> 'a1 -> 'a1) -> positive -> 'a1 -> 'a1

  val to_nat : positive -> nat

  val of_nat : nat -> positive

  val of_succ_nat : nat -> positive
 end

module Z :
 sig
  val double : z -> z

  val succ_double : z -> z

  val pred_double : z -> z

  val pos_sub : positive -> positive -> z

  val add : z -> z -> z

  val opp : z -> z

  val sub : z -> z -> z

  val mul : z -> z -> z

  val compare : z -> z -> comparison

  val sgn : z -> z

  val leb : z -> z -> bool

  val abs : z -> z

  val of_nat : nat -> z

  val to_pos : z -> positive

  val ggcd : z -> z -> (z, (z, z) prod) prod
 end

val z_lt_dec : z -> z -> sumbool

val z_lt_ge_dec : z -> z -> sumbool

val z_lt_le_dec : z -> z -> sumbool

type q = { qnum : z; qden : positive }

val qplus : q -> q -> q

val qmult : q -> q -> q

val qopp : q -> q

val qminus : q -> q -> q

val qlt_le_dec : q -> q -> sumbool

val qarchimedean : q -> positive

val qred : q -> q

type cReal = (positive -> q)

type cRealLt = positive

val cRealLt_lpo_dec :
  cReal -> cReal -> (__ -> (nat -> sumbool) -> nat sumor) -> (cRealLt, __) sum

val inject_Q : q -> cReal

val cReal_plus : cReal -> cReal -> cReal

val cReal_opp : cReal -> cReal

val qCauchySeq_bound : (positive -> q) -> (positive -> positive) -> positive

val cReal_mult : cReal -> cReal -> cReal

val sig_forall_dec : (nat -> sumbool) -> nat sumor

val lowerCutBelow : (q -> bool) -> q

val lowerCutAbove : (q -> bool) -> q

type dReal = (q -> bool)

val dRealQlim_rec : (q -> bool) -> nat -> nat -> q

val dRealQlim : dReal -> nat -> q

val dRealAbstr : cReal -> dReal

val dRealRepr : dReal -> cReal

module type RbaseSymbolsSig =
 sig
  type coq_R

  val coq_Rabst : cReal -> coq_R

  val coq_Rrepr : coq_R -> cReal

  val coq_R0 : coq_R

  val coq_R1 : coq_R

  val coq_Rplus : coq_R -> coq_R -> coq_R

  val coq_Rmult : coq_R -> coq_R -> coq_R

  val coq_Ropp : coq_R -> coq_R
 end

module RbaseSymbolsImpl :
 RbaseSymbolsSig

val rminus :
  RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R

val iPR_2 : positive -> RbaseSymbolsImpl.coq_R

val iPR : positive -> RbaseSymbolsImpl.coq_R

val iZR : z -> RbaseSymbolsImpl.coq_R

val total_order_T :
  RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R -> sumbool sumor

val rle_dec : RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R -> sumbool

type envParams = { commission_rate : RbaseSymbolsImpl.coq_R;
                   max_position : z; share_size : z }

type agentState = { cash : RbaseSymbolsImpl.coq_R; shares_held : z }

type action =
| Hold
| Buy
| Sell

val r_of_Z : z -> RbaseSymbolsImpl.coq_R

val step :
  envParams -> agentState -> RbaseSymbolsImpl.coq_R -> action -> agentState
