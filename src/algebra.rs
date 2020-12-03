/// Trait for associative binary operations with an identity element.
///
/// Monoid requires that the following property holds:
/// ```ignore
/// // The operation * is associative.
/// (a * b) * c == a * (b * c)
///
/// // There exists an identity element.
/// a * id == id * a == a
/// ```
/// This property cannot be checked by the compiler so the implementer should verify it by themself.
pub trait Monoid {
    /// The identity element.
    const ID: Self;
    /// An associative binary operator on monoid elements.
    fn op(&self, rhs: &Self) -> Self;
}

/// Trait for an action of an algebraic structure on a set `M`
///
/// Action requires that the following properties holds when we implement `Act<M>` for `A`:
///
/// ```ignore
/// // For any f, g in `A`, the map f * g is same as the composition map f ∘ g.
/// (f * g)(m) == f(g(m))
/// 
/// // If A has an identity, the identity should map an element from `M` to itself.
/// id(m) == m
/// ```
/// 
/// This property cannot be checked by the compiler so the implementer should verify it by themself.
pub trait Act<M> {
    /// An action of `self` on an element of `M`
    fn act(&self, m: &M) -> M;
}
