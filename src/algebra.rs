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

/// Trait that defines how a range update should be applied to a range property
/// in the [`LazySegTree`](crate::LazySegTree)
///
/// A lazy segment tree requires two monoids and one action:
/// - `M`: A monoid that represents the property of a range.
/// - `A`: A monoid that reprenents the update on a range.
/// - `L`: An action that represents the application of an update to a property.
///
/// Note that `L: A * M -> M` and we can [curry](https://en.wikipedia.org/wiki/Currying) it to be
/// `L: A -> M -> M`. So an element of `A` corresponds to a map `M -> M`.
/// For the sake of example we'll denote an element of `A` and the map `M -> M` corresponds to the
/// element with same symbols like `f, g` and elements of `M` as `m, n`.
///
/// To be a monoid action, first and second condition should be met.
/// The third condition is required for the lazy segment tree to work.
///
/// ```ignore
/// // The identity of `A` should map an element from `M` to itself.
/// id(m) == m
///
/// // For any f, g in `A`, the map f * g is same as the composition map f ∘ g.
/// (f * g)(m) == f(g(m))
///
/// // The corresponding map of an element of `A` should be homomorphic.
/// f(m * n) == f(m) * f(n)
/// ```
pub trait LazyAct<M: Monoid>: Monoid {
    // TODO(yujingaya) Also could be a semigroup.
    // We can just use Option::None instead of the identity.
    /// Apply a range update `self` to a range property `m`.
    fn act(&self, m: &M) -> M;
}
