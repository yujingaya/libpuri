/// Trait for associative binary operations with an identity element.
///
/// The trait requires that the following property holds:
/// ```ignore
/// // The operation * is associative.
/// (a * b) * c == a * (b * c)
/// 
/// // There exists an identity element.
/// a * id == id * a == a
/// ```
/// This property cannot be checked by the compiler so the implementer should verify it by themself.
pub trait Monoid: Clone + Eq {
    // TODO(yujingaya) remove identity requirement with non-full binary tree?
    // Could be a semigroup
    // reference: https://codeforces.com/blog/entry/18051
    const ID: Self;
    fn op(&self, rhs: &Self) -> Self;
}

/// Trait for the relationship between two monoids in the [`LazySegTree`](crate::LazySegTree)
///
/// A lazy segment tree requires two monoids **M** and **A**, which represent _the
/// property of interval_ and _the lazy action on the range_, respectively.
/// 
/// **M** only need to be monoid while **A** need to satisfy both monoid and lazy action properties.
/// By implementing `LazyAct` you are confirming that the action of **A** on **M** satisfies
/// following conditions:
///
/// ```ignore
/// // The identity of `A` should map m from `M` to itself.
/// id(m) == m
/// 
/// // The mapping corresponds to an element of `A` should be homomorphic.
/// f(m * n) == f(m) * f(n)
/// ```
pub trait LazyAct<M: Monoid>: Monoid {
    fn act(&self, m: &M) -> M;
}
