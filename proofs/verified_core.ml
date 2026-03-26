
type __ = Obj.t
let __ = let rec f _ = Obj.repr f in Obj.repr f

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

(** val snd : ('a1, 'a2) prod -> 'a2 **)

let snd = function
| Pair (_, y) -> y

type comparison =
| Eq
| Lt
| Gt

(** val compOpp : comparison -> comparison **)

let compOpp = function
| Eq -> Eq
| Lt -> Gt
| Gt -> Lt

(** val id : 'a1 -> 'a1 **)

let id x =
  x

type 'a sig0 = 'a
  (* singleton inductive, whose constructor was exist *)

type sumbool =
| Left
| Right

type 'a sumor =
| Inleft of 'a
| Inright

module Coq__1 = struct
 (** val add : nat -> nat -> nat **)
 let rec add n m =
   match n with
   | O -> m
   | S p -> S (add p m)
end
include Coq__1

(** val mul : nat -> nat -> nat **)

let rec mul n m =
  match n with
  | O -> O
  | S p -> add m (mul p m)

type positive =
| XI of positive
| XO of positive
| XH

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Pos =
 struct
  type mask =
  | IsNul
  | IsPos of positive
  | IsNeg
 end

module Coq_Pos =
 struct
  (** val succ : positive -> positive **)

  let rec succ = function
  | XI p -> XO (succ p)
  | XO p -> XI p
  | XH -> XO XH

  (** val add : positive -> positive -> positive **)

  let rec add x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XI (add p q0)
       | XO q0 -> XO (add p q0)
       | XH -> XI p)
    | XH -> (match y with
             | XI q0 -> XO (succ q0)
             | XO q0 -> XI q0
             | XH -> XO XH)

  (** val add_carry : positive -> positive -> positive **)

  and add_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XI (add_carry p q0)
       | XO q0 -> XO (add_carry p q0)
       | XH -> XI (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XH ->
      (match y with
       | XI q0 -> XI (succ q0)
       | XO q0 -> XO (succ q0)
       | XH -> XI XH)

  (** val pred_double : positive -> positive **)

  let rec pred_double = function
  | XI p -> XI (XO p)
  | XO p -> XI (pred_double p)
  | XH -> XH

  type mask = Pos.mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  (** val succ_double_mask : mask -> mask **)

  let succ_double_mask = function
  | IsNul -> IsPos XH
  | IsPos p -> IsPos (XI p)
  | IsNeg -> IsNeg

  (** val double_mask : mask -> mask **)

  let double_mask = function
  | IsPos p -> IsPos (XO p)
  | x0 -> x0

  (** val double_pred_mask : positive -> mask **)

  let double_pred_mask = function
  | XI p -> IsPos (XO (XO p))
  | XO p -> IsPos (XO (pred_double p))
  | XH -> IsNul

  (** val sub_mask : positive -> positive -> mask **)

  let rec sub_mask x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> double_mask (sub_mask p q0)
       | XO q0 -> succ_double_mask (sub_mask p q0)
       | XH -> IsPos (XO p))
    | XO p ->
      (match y with
       | XI q0 -> succ_double_mask (sub_mask_carry p q0)
       | XO q0 -> double_mask (sub_mask p q0)
       | XH -> IsPos (pred_double p))
    | XH -> (match y with
             | XH -> IsNul
             | _ -> IsNeg)

  (** val sub_mask_carry : positive -> positive -> mask **)

  and sub_mask_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> succ_double_mask (sub_mask_carry p q0)
       | XO q0 -> double_mask (sub_mask p q0)
       | XH -> IsPos (pred_double p))
    | XO p ->
      (match y with
       | XI q0 -> double_mask (sub_mask_carry p q0)
       | XO q0 -> succ_double_mask (sub_mask_carry p q0)
       | XH -> double_pred_mask p)
    | XH -> IsNeg

  (** val sub : positive -> positive -> positive **)

  let sub x y =
    match sub_mask x y with
    | IsPos z0 -> z0
    | _ -> XH

  (** val mul : positive -> positive -> positive **)

  let rec mul x y =
    match x with
    | XI p -> add y (XO (mul p y))
    | XO p -> XO (mul p y)
    | XH -> y

  (** val size_nat : positive -> nat **)

  let rec size_nat = function
  | XI p0 -> S (size_nat p0)
  | XO p0 -> S (size_nat p0)
  | XH -> S O

  (** val compare_cont : comparison -> positive -> positive -> comparison **)

  let rec compare_cont r x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> compare_cont r p q0
       | XO q0 -> compare_cont Gt p q0
       | XH -> Gt)
    | XO p ->
      (match y with
       | XI q0 -> compare_cont Lt p q0
       | XO q0 -> compare_cont r p q0
       | XH -> Gt)
    | XH -> (match y with
             | XH -> r
             | _ -> Lt)

  (** val compare : positive -> positive -> comparison **)

  let compare =
    compare_cont Eq

  (** val max : positive -> positive -> positive **)

  let max p p' =
    match compare p p' with
    | Gt -> p
    | _ -> p'

  (** val ggcdn :
      nat -> positive -> positive -> (positive, (positive, positive) prod)
      prod **)

  let rec ggcdn n a b =
    match n with
    | O -> Pair (XH, (Pair (a, b)))
    | S n0 ->
      (match a with
       | XI a' ->
         (match b with
          | XI b' ->
            (match compare a' b' with
             | Eq -> Pair (a, (Pair (XH, XH)))
             | Lt ->
               let Pair (g, p) = ggcdn n0 (sub b' a') a in
               let Pair (ba, aa) = p in
               Pair (g, (Pair (aa, (add aa (XO ba)))))
             | Gt ->
               let Pair (g, p) = ggcdn n0 (sub a' b') b in
               let Pair (ab, bb) = p in
               Pair (g, (Pair ((add bb (XO ab)), bb))))
          | XO b0 ->
            let Pair (g, p) = ggcdn n0 a b0 in
            let Pair (aa, bb) = p in Pair (g, (Pair (aa, (XO bb))))
          | XH -> Pair (XH, (Pair (a, XH))))
       | XO a0 ->
         (match b with
          | XI _ ->
            let Pair (g, p) = ggcdn n0 a0 b in
            let Pair (aa, bb) = p in Pair (g, (Pair ((XO aa), bb)))
          | XO b0 -> let Pair (g, p) = ggcdn n0 a0 b0 in Pair ((XO g), p)
          | XH -> Pair (XH, (Pair (a, XH))))
       | XH -> Pair (XH, (Pair (XH, b))))

  (** val ggcd :
      positive -> positive -> (positive, (positive, positive) prod) prod **)

  let ggcd a b =
    ggcdn (Coq__1.add (size_nat a) (size_nat b)) a b

  (** val iter_op : ('a1 -> 'a1 -> 'a1) -> positive -> 'a1 -> 'a1 **)

  let rec iter_op op p a =
    match p with
    | XI p0 -> op a (iter_op op p0 (op a a))
    | XO p0 -> iter_op op p0 (op a a)
    | XH -> a

  (** val to_nat : positive -> nat **)

  let to_nat x =
    iter_op Coq__1.add x (S O)

  (** val of_nat : nat -> positive **)

  let rec of_nat = function
  | O -> XH
  | S x -> (match x with
            | O -> XH
            | S _ -> succ (of_nat x))

  (** val of_succ_nat : nat -> positive **)

  let rec of_succ_nat = function
  | O -> XH
  | S x -> succ (of_succ_nat x)
 end

module Z =
 struct
  (** val double : z -> z **)

  let double = function
  | Z0 -> Z0
  | Zpos p -> Zpos (XO p)
  | Zneg p -> Zneg (XO p)

  (** val succ_double : z -> z **)

  let succ_double = function
  | Z0 -> Zpos XH
  | Zpos p -> Zpos (XI p)
  | Zneg p -> Zneg (Coq_Pos.pred_double p)

  (** val pred_double : z -> z **)

  let pred_double = function
  | Z0 -> Zneg XH
  | Zpos p -> Zpos (Coq_Pos.pred_double p)
  | Zneg p -> Zneg (XI p)

  (** val pos_sub : positive -> positive -> z **)

  let rec pos_sub x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> double (pos_sub p q0)
       | XO q0 -> succ_double (pos_sub p q0)
       | XH -> Zpos (XO p))
    | XO p ->
      (match y with
       | XI q0 -> pred_double (pos_sub p q0)
       | XO q0 -> double (pos_sub p q0)
       | XH -> Zpos (Coq_Pos.pred_double p))
    | XH ->
      (match y with
       | XI q0 -> Zneg (XO q0)
       | XO q0 -> Zneg (Coq_Pos.pred_double q0)
       | XH -> Z0)

  (** val add : z -> z -> z **)

  let add x y =
    match x with
    | Z0 -> y
    | Zpos x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> Zpos (Coq_Pos.add x' y')
       | Zneg y' -> pos_sub x' y')
    | Zneg x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> pos_sub y' x'
       | Zneg y' -> Zneg (Coq_Pos.add x' y'))

  (** val opp : z -> z **)

  let opp = function
  | Z0 -> Z0
  | Zpos x0 -> Zneg x0
  | Zneg x0 -> Zpos x0

  (** val sub : z -> z -> z **)

  let sub m n =
    add m (opp n)

  (** val mul : z -> z -> z **)

  let mul x y =
    match x with
    | Z0 -> Z0
    | Zpos x' ->
      (match y with
       | Z0 -> Z0
       | Zpos y' -> Zpos (Coq_Pos.mul x' y')
       | Zneg y' -> Zneg (Coq_Pos.mul x' y'))
    | Zneg x' ->
      (match y with
       | Z0 -> Z0
       | Zpos y' -> Zneg (Coq_Pos.mul x' y')
       | Zneg y' -> Zpos (Coq_Pos.mul x' y'))

  (** val compare : z -> z -> comparison **)

  let compare x y =
    match x with
    | Z0 -> (match y with
             | Z0 -> Eq
             | Zpos _ -> Lt
             | Zneg _ -> Gt)
    | Zpos x' -> (match y with
                  | Zpos y' -> Coq_Pos.compare x' y'
                  | _ -> Gt)
    | Zneg x' ->
      (match y with
       | Zneg y' -> compOpp (Coq_Pos.compare x' y')
       | _ -> Lt)

  (** val sgn : z -> z **)

  let sgn = function
  | Z0 -> Z0
  | Zpos _ -> Zpos XH
  | Zneg _ -> Zneg XH

  (** val leb : z -> z -> bool **)

  let leb x y =
    match compare x y with
    | Gt -> False
    | _ -> True

  (** val abs : z -> z **)

  let abs = function
  | Zneg p -> Zpos p
  | x -> x

  (** val of_nat : nat -> z **)

  let of_nat = function
  | O -> Z0
  | S n0 -> Zpos (Coq_Pos.of_succ_nat n0)

  (** val to_pos : z -> positive **)

  let to_pos = function
  | Zpos p -> p
  | _ -> XH

  (** val ggcd : z -> z -> (z, (z, z) prod) prod **)

  let ggcd a b =
    match a with
    | Z0 -> Pair ((abs b), (Pair (Z0, (sgn b))))
    | Zpos a0 ->
      (match b with
       | Z0 -> Pair ((abs a), (Pair ((sgn a), Z0)))
       | Zpos b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zpos aa), (Zpos bb))))
       | Zneg b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zpos aa), (Zneg bb)))))
    | Zneg a0 ->
      (match b with
       | Z0 -> Pair ((abs a), (Pair ((sgn a), Z0)))
       | Zpos b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zneg aa), (Zpos bb))))
       | Zneg b0 ->
         let Pair (g, p) = Coq_Pos.ggcd a0 b0 in
         let Pair (aa, bb) = p in
         Pair ((Zpos g), (Pair ((Zneg aa), (Zneg bb)))))
 end

(** val z_lt_dec : z -> z -> sumbool **)

let z_lt_dec x y =
  match Z.compare x y with
  | Lt -> Left
  | _ -> Right

(** val z_lt_ge_dec : z -> z -> sumbool **)

let z_lt_ge_dec =
  z_lt_dec

(** val z_lt_le_dec : z -> z -> sumbool **)

let z_lt_le_dec =
  z_lt_ge_dec

type q = { qnum : z; qden : positive }

(** val qplus : q -> q -> q **)

let qplus x y =
  { qnum = (Z.add (Z.mul x.qnum (Zpos y.qden)) (Z.mul y.qnum (Zpos x.qden)));
    qden = (Coq_Pos.mul x.qden y.qden) }

(** val qmult : q -> q -> q **)

let qmult x y =
  { qnum = (Z.mul x.qnum y.qnum); qden = (Coq_Pos.mul x.qden y.qden) }

(** val qopp : q -> q **)

let qopp x =
  { qnum = (Z.opp x.qnum); qden = x.qden }

(** val qminus : q -> q -> q **)

let qminus x y =
  qplus x (qopp y)

(** val qlt_le_dec : q -> q -> sumbool **)

let qlt_le_dec x y =
  z_lt_le_dec (Z.mul x.qnum (Zpos y.qden)) (Z.mul y.qnum (Zpos x.qden))

(** val qarchimedean : q -> positive **)

let qarchimedean q0 =
  let { qnum = a; qden = _ } = q0 in
  (match a with
   | Zpos p -> Coq_Pos.add p XH
   | _ -> XH)

(** val qred : q -> q **)

let qred q0 =
  let { qnum = q1; qden = q2 } = q0 in
  let Pair (r1, r2) = snd (Z.ggcd q1 (Zpos q2)) in
  { qnum = r1; qden = (Z.to_pos r2) }

type cReal = (positive -> q)

type cRealLt = positive

(** val cRealLt_lpo_dec :
    cReal -> cReal -> (__ -> (nat -> sumbool) -> nat sumor) -> (cRealLt, __)
    sum **)

let cRealLt_lpo_dec x y lpo =
  let s =
    lpo __ (fun n ->
      let s =
        qlt_le_dec { qnum = (Zpos (XO XH)); qden = (Coq_Pos.of_nat (S n)) }
          (qminus (y (Coq_Pos.of_nat (S n))) (x (Coq_Pos.of_nat (S n))))
      in
      (match s with
       | Left -> Right
       | Right -> Left))
  in
  (match s with
   | Inleft s0 -> Inl (Coq_Pos.of_nat (S s0))
   | Inright -> Inr __)

(** val inject_Q : q -> cReal **)

let inject_Q q0 _ =
  q0

(** val cReal_plus : cReal -> cReal -> cReal **)

let cReal_plus x y n =
  qred (qplus (x (Coq_Pos.mul (XO XH) n)) (y (Coq_Pos.mul (XO XH) n)))

(** val cReal_opp : cReal -> cReal **)

let cReal_opp x n =
  qopp (x n)

(** val qCauchySeq_bound :
    (positive -> q) -> (positive -> positive) -> positive **)

let qCauchySeq_bound qn cvmod =
  match (qn (cvmod XH)).qnum with
  | Z0 -> XH
  | Zpos p -> Coq_Pos.add p XH
  | Zneg p -> Coq_Pos.add p XH

(** val cReal_mult : cReal -> cReal -> cReal **)

let cReal_mult x y n =
  qmult
    (x
      (Coq_Pos.mul
        (Coq_Pos.mul (XO XH)
          (Coq_Pos.max (qCauchySeq_bound x id) (qCauchySeq_bound y id))) n))
    (y
      (Coq_Pos.mul
        (Coq_Pos.mul (XO XH)
          (Coq_Pos.max (qCauchySeq_bound x id) (qCauchySeq_bound y id))) n))

(** val sig_forall_dec : (nat -> sumbool) -> nat sumor **)

let sig_forall_dec =
  failwith "AXIOM TO BE REALIZED"

(** val lowerCutBelow : (q -> bool) -> q **)

let lowerCutBelow f =
  let s =
    sig_forall_dec (fun n ->
      let b = f (qopp { qnum = (Z.of_nat n); qden = XH }) in
      (match b with
       | True -> Right
       | False -> Left))
  in
  (match s with
   | Inleft s0 -> qopp { qnum = (Z.of_nat s0); qden = XH }
   | Inright -> assert false (* absurd case *))

(** val lowerCutAbove : (q -> bool) -> q **)

let lowerCutAbove f =
  let s =
    sig_forall_dec (fun n ->
      let b = f { qnum = (Z.of_nat n); qden = XH } in
      (match b with
       | True -> Left
       | False -> Right))
  in
  (match s with
   | Inleft s0 -> { qnum = (Z.of_nat s0); qden = XH }
   | Inright -> assert false (* absurd case *))

type dReal = (q -> bool)

(** val dRealQlim_rec : (q -> bool) -> nat -> nat -> q **)

let rec dRealQlim_rec f n = function
| O -> assert false (* absurd case *)
| S p0 ->
  let b =
    f
      (qplus (lowerCutBelow f) { qnum = (Z.of_nat p0); qden =
        (Coq_Pos.of_nat (S n)) })
  in
  (match b with
   | True ->
     qplus (lowerCutBelow f) { qnum = (Z.of_nat p0); qden =
       (Coq_Pos.of_nat (S n)) }
   | False -> dRealQlim_rec f n p0)

(** val dRealQlim : dReal -> nat -> q **)

let dRealQlim x n =
  let s = lowerCutAbove x in
  let s0 = qarchimedean (qminus s (lowerCutBelow x)) in
  dRealQlim_rec x n (mul (S n) (Coq_Pos.to_nat s0))

(** val dRealAbstr : cReal -> dReal **)

let dRealAbstr x =
  let h = fun q0 n ->
    let s =
      qlt_le_dec
        (qplus q0 { qnum = (Zpos XH); qden = (Coq_Pos.of_nat (S n)) })
        (x (Coq_Pos.of_nat (S n)))
    in
    (match s with
     | Left -> Right
     | Right -> Left)
  in
  (fun q0 ->
  match sig_forall_dec (h q0) with
  | Inleft _ -> True
  | Inright -> False)

(** val dRealRepr : dReal -> cReal **)

let dRealRepr x n =
  dRealQlim x (Coq_Pos.to_nat n)

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

module RbaseSymbolsImpl =
 struct
  type coq_R = dReal

  (** val coq_Rabst : cReal -> dReal **)

  let coq_Rabst =
    dRealAbstr

  (** val coq_Rrepr : dReal -> cReal **)

  let coq_Rrepr =
    dRealRepr

  (** val coq_Rquot1 : __ **)

  let coq_Rquot1 =
    __

  (** val coq_Rquot2 : __ **)

  let coq_Rquot2 =
    __

  (** val coq_R0 : coq_R **)

  let coq_R0 =
    coq_Rabst (inject_Q { qnum = Z0; qden = XH })

  (** val coq_R1 : coq_R **)

  let coq_R1 =
    coq_Rabst (inject_Q { qnum = (Zpos XH); qden = XH })

  (** val coq_Rplus : coq_R -> coq_R -> coq_R **)

  let coq_Rplus x y =
    coq_Rabst (cReal_plus (coq_Rrepr x) (coq_Rrepr y))

  (** val coq_Rmult : coq_R -> coq_R -> coq_R **)

  let coq_Rmult x y =
    coq_Rabst (cReal_mult (coq_Rrepr x) (coq_Rrepr y))

  (** val coq_Ropp : coq_R -> coq_R **)

  let coq_Ropp x =
    coq_Rabst (cReal_opp (coq_Rrepr x))

  type coq_Rlt = __

  (** val coq_R0_def : __ **)

  let coq_R0_def =
    __

  (** val coq_R1_def : __ **)

  let coq_R1_def =
    __

  (** val coq_Rplus_def : __ **)

  let coq_Rplus_def =
    __

  (** val coq_Rmult_def : __ **)

  let coq_Rmult_def =
    __

  (** val coq_Ropp_def : __ **)

  let coq_Ropp_def =
    __

  (** val coq_Rlt_def : __ **)

  let coq_Rlt_def =
    __
 end

(** val rminus :
    RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R **)

let rminus r1 r2 =
  RbaseSymbolsImpl.coq_Rplus r1 (RbaseSymbolsImpl.coq_Ropp r2)

(** val iPR_2 : positive -> RbaseSymbolsImpl.coq_R **)

let rec iPR_2 = function
| XI p0 ->
  RbaseSymbolsImpl.coq_Rmult
    (RbaseSymbolsImpl.coq_Rplus RbaseSymbolsImpl.coq_R1
      RbaseSymbolsImpl.coq_R1)
    (RbaseSymbolsImpl.coq_Rplus RbaseSymbolsImpl.coq_R1 (iPR_2 p0))
| XO p0 ->
  RbaseSymbolsImpl.coq_Rmult
    (RbaseSymbolsImpl.coq_Rplus RbaseSymbolsImpl.coq_R1
      RbaseSymbolsImpl.coq_R1) (iPR_2 p0)
| XH ->
  RbaseSymbolsImpl.coq_Rplus RbaseSymbolsImpl.coq_R1 RbaseSymbolsImpl.coq_R1

(** val iPR : positive -> RbaseSymbolsImpl.coq_R **)

let iPR = function
| XI p0 -> RbaseSymbolsImpl.coq_Rplus RbaseSymbolsImpl.coq_R1 (iPR_2 p0)
| XO p0 -> iPR_2 p0
| XH -> RbaseSymbolsImpl.coq_R1

(** val iZR : z -> RbaseSymbolsImpl.coq_R **)

let iZR = function
| Z0 -> RbaseSymbolsImpl.coq_R0
| Zpos n -> iPR n
| Zneg n -> RbaseSymbolsImpl.coq_Ropp (iPR n)

(** val total_order_T :
    RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R -> sumbool sumor **)

let total_order_T r1 r2 =
  let s =
    cRealLt_lpo_dec (RbaseSymbolsImpl.coq_Rrepr r1)
      (RbaseSymbolsImpl.coq_Rrepr r2) (fun _ -> sig_forall_dec)
  in
  (match s with
   | Inl _ -> Inleft Left
   | Inr _ ->
     let s0 =
       cRealLt_lpo_dec (RbaseSymbolsImpl.coq_Rrepr r2)
         (RbaseSymbolsImpl.coq_Rrepr r1) (fun _ -> sig_forall_dec)
     in
     (match s0 with
      | Inl _ -> Inright
      | Inr _ -> Inleft Right))

(** val rle_dec :
    RbaseSymbolsImpl.coq_R -> RbaseSymbolsImpl.coq_R -> sumbool **)

let rle_dec r1 r2 =
  let h = total_order_T r1 r2 in
  (match h with
   | Inleft _ -> Left
   | Inright -> Right)

type envParams = { commission_rate : RbaseSymbolsImpl.coq_R;
                   max_position : z; share_size : z }

type agentState = { cash : RbaseSymbolsImpl.coq_R; shares_held : z }

type action =
| Hold
| Buy
| Sell

(** val r_of_Z : z -> RbaseSymbolsImpl.coq_R **)

let r_of_Z =
  iZR

(** val step :
    envParams -> agentState -> RbaseSymbolsImpl.coq_R -> action -> agentState **)

let step params s current_price action0 =
  let comm_rate = params.commission_rate in
  let max_pos = params.max_position in
  let size = params.share_size in
  (match action0 with
   | Hold -> s
   | Buy ->
     let cost_per_share =
       RbaseSymbolsImpl.coq_Rmult current_price
         (RbaseSymbolsImpl.coq_Rplus (iZR (Zpos XH)) comm_rate)
     in
     let total_cost = RbaseSymbolsImpl.coq_Rmult (r_of_Z size) cost_per_share
     in
     (match rle_dec total_cost s.cash with
      | Left ->
        (match Z.leb (Z.add s.shares_held size) max_pos with
         | True ->
           { cash = (rminus s.cash total_cost); shares_held =
             (Z.add s.shares_held size) }
         | False -> s)
      | Right -> s)
   | Sell ->
     (match Z.leb size s.shares_held with
      | True ->
        let revenue_per_share =
          RbaseSymbolsImpl.coq_Rmult current_price
            (rminus (iZR (Zpos XH)) comm_rate)
        in
        let total_revenue =
          RbaseSymbolsImpl.coq_Rmult (r_of_Z size) revenue_per_share
        in
        { cash = (RbaseSymbolsImpl.coq_Rplus s.cash total_revenue);
        shares_held = (Z.sub s.shares_held size) }
      | False -> s))
