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
    /// An identity requirement could be lifted if we implement the tree with complete tree instead
    /// of perfect tree, making this trait name a semigroup. But for the sake of simplicity, we
    /// leave it this way for now.
    const ID: Self;
    /// An associative binary operator on monoid elements.
    fn op(&self, rhs: &Self) -> Self;
}

/// Trait for the relationship between two monoids in the [`LazySegTree`](crate::LazySegTree)
///
/// A lazy segment tree requires two monoids **M** and **A**, which represent _the
/// property of interval_ and _the lazy action on the range_, respectively.
/// 
/// **M** only need to be a monoid while **A** need to satisfy both monoid and lazy action properties.
/// By implementing `LazyAct` you are confirming that the action of **A** on **M** satisfies
/// following conditions:
///
/// ```ignore
/// // Composite of two maps is the product of two corresponding monoids.
/// f(g(m)) == (f * g)(m)
/// 
/// // The identity of `A` should map m from `M` to itself.
/// id(m) == m
/// 
/// // The corresponding map of an element of `A` should be homomorphic.
/// f(m * n) == f(m) * f(n)
/// ```
///
/// Here the first two conditions are for a monoid action
/// and the third is only for the lazy segment
/// tree.
pub trait LazyAct<M: Monoid>: Monoid {
    // TODO(yujingaya) Also could be a semigroup.
    // We can just use Option::None instead of the identity.
    /// An [action](https://en.wikipedia.org/wiki/Semigroup_action) of a lazy monoid to an interval
    /// monoid.
    fn act(&self, m: &M) -> M;
}
