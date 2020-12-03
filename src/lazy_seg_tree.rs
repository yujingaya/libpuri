use crate::algebra::{LazyAct, Monoid};
use crate::util::IntoIndex;
use std::fmt::{self, Debug};
use std::iter::FromIterator;
use std::ptr;
// TODO(yujingaya) Rewrite tests/doctests to reuse MinMax, Add structs when cfg(doctest) is stable.
// reference: https://github.com/rust-lang/rust/issues/67295

/// A segment tree that supports range query and range update.
///
/// A lazy segment requires a [`Monoid`] that represents a property of an interval and another
/// monoid that represents a range operation. The latter should also conform to a [`LazyAct`] which
/// defines how the range operation should be applied to a range. Check out the documentation of
/// both traits for further details.
///
/// # Examples
/// Following example supports two operations:
///
/// - Query minimum and maximum numbers within an interval.
/// - Add a number to each element within an interval.
///
/// ```
/// use libpuri::{Monoid, LazyAct, LazySegTree};
///
/// #[derive(Clone, Debug, PartialEq, Eq)]
/// struct MinMax(i64, i64);
/// impl Monoid for MinMax {
///     const ID: Self = MinMax(i64::MAX, i64::MIN);
///     fn op(&self, rhs: &Self) -> Self {
///         MinMax(self.0.min(rhs.0), self.1.max(rhs.1))
///     }
/// }
///
/// #[derive(Clone, Debug, PartialEq, Eq)]
/// struct Add(i64);
/// impl Monoid for Add {
///     const ID: Self = Add(0);
///     fn op(&self, rhs: &Self) -> Self {
///         Add(self.0.saturating_add(rhs.0))
///     }
/// }
///
/// impl LazyAct<MinMax> for Add {
///     fn act(&self, m: &MinMax) -> MinMax {
///         if m == &MinMax::ID {
///             MinMax::ID
///         } else {
///             MinMax(m.0 + self.0, m.1 + self.0)
///         }
///     }
/// }
///
/// // Initialize with [0, 0, 0, 0, 0, 0]
/// let mut lazy_tree: LazySegTree<MinMax, Add> = (0..6).map(|_| MinMax(0, 0)).collect();
/// assert_eq!(lazy_tree.get(..), MinMax(0, 0));
///
/// // Range update [5, 5, 5, 5, 0, 0]
/// lazy_tree.act(0..4, Add(5));
///
/// // Another range update [5, 5, 47, 47, 42, 42]
/// lazy_tree.act(2..6, Add(42));
///
/// assert_eq!(lazy_tree.get(1..3), MinMax(5,  47));
/// assert_eq!(lazy_tree.get(3..5), MinMax(42, 47));
///
/// // Set index 3 to 0 [5, 5, 47, 0, 42, 42]
/// lazy_tree.set(3, MinMax(0, 0));
///
/// assert_eq!(lazy_tree.get(..), MinMax(0,  47));
/// assert_eq!(lazy_tree.get(3..5), MinMax(0,  42));
/// assert_eq!(lazy_tree.get(0), MinMax(5, 5));
/// ```
///
// LazyAct of each node represents an action not yet commited to its children but already commited
// to the node itself.
// Conld be refactored into `Vec<(M, Option<A>)>`
pub struct LazySegTree<M: Monoid, A: LazyAct<M>>(Vec<(M, A)>);

impl<M: Monoid, A: LazyAct<M>> LazySegTree<M, A> {
    fn size(&self) -> usize {
        self.0.len() / 2
    }

    fn height(&self) -> u32 {
        self.0.len().trailing_zeros()
    }

    fn act_lazy(node: &mut (M, A), a: &A) {
        *node = (a.act(&node.0), a.op(&node.1));
    }

    fn propagate_to_children(&mut self, i: usize) {
        let a = self.0[i].1.clone();
        Self::act_lazy(&mut self.0[i * 2], &a);
        Self::act_lazy(&mut self.0[i * 2 + 1], &a);

        self.0[i].1 = A::ID;
    }

    fn propagate(&mut self, start: usize, end: usize) {
        for i in (1..self.height()).rev() {
            if (start >> i) << i != start {
                self.propagate_to_children(start >> i);
            }
            if (end >> i) << i != end {
                self.propagate_to_children((end - 1) >> i);
            }
        }
    }

    fn update(&mut self, i: usize) {
        self.0[i].0 = self.0[i * 2].0.op(&self.0[i * 2 + 1].0);
    }
}

impl<M: Monoid, A: LazyAct<M>> LazySegTree<M, A> {
    /// Constructs a new lazy segment tree with at least given number of intervals can be stored.
    ///
    /// The segment tree will be initialized with the identity elements.
    ///
    /// # Complexity
    /// O(n).
    ///
    /// If you know the initial elements in advance,
    /// [`from_iter_sized()`](LazySegTree::from_iter_sized) should be preferred over `new()`.
    ///
    /// Initializing with the identity elements and updating n elements will tax you O(nlog(n)),
    /// whereas `from_iter_sized()` is O(n) by computing the interval properties only once.
    ///
    /// # Examples
    /// ```
    /// # use libpuri::{LazySegTree, Monoid, LazyAct};
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct MinMax(i64, i64);
    /// # impl Monoid for MinMax {
    /// #     const ID: Self = MinMax(i64::MAX, i64::MIN);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         MinMax(self.0.min(rhs.0), self.1.max(rhs.1))
    /// #     }
    /// # }
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct Add(i64);
    /// # impl Monoid for Add {
    /// #     const ID: Self = Add(0);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         Add(self.0.saturating_add(rhs.0))
    /// #     }
    /// # }
    /// #
    /// # impl LazyAct<MinMax> for Add {
    /// #     fn act(&self, m: &MinMax) -> MinMax {
    /// #         if m == &MinMax::ID {
    /// #             MinMax::ID
    /// #         } else {
    /// #             MinMax(m.0 + self.0, m.1 + self.0)
    /// #         }
    /// #     }
    /// # }
    /// let mut lazy_tree: LazySegTree<MinMax, Add> = LazySegTree::new(5);
    ///
    /// // Initialized with [id, id, id, id, id]
    /// assert_eq!(lazy_tree.get(..), MinMax::ID);
    /// ```
    pub fn new(size: usize) -> Self {
        LazySegTree(vec![(M::ID, A::ID); size.next_power_of_two() * 2])
    }

    /// Constructs a new lazy segment tree with given intervals properties.
    ///
    /// # Complexity
    /// O(n).
    ///
    /// # Examples
    /// ```
    /// # use libpuri::{LazySegTree, Monoid, LazyAct};
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct MinMax(i64, i64);
    /// # impl Monoid for MinMax {
    /// #     const ID: Self = MinMax(i64::MAX, i64::MIN);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         MinMax(self.0.min(rhs.0), self.1.max(rhs.1))
    /// #     }
    /// # }
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct Add(i64);
    /// # impl Monoid for Add {
    /// #     const ID: Self = Add(0);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         Add(self.0.saturating_add(rhs.0))
    /// #     }
    /// # }
    /// #
    /// # impl LazyAct<MinMax> for Add {
    /// #     fn act(&self, m: &MinMax) -> MinMax {
    /// #         if m == &MinMax::ID {
    /// #             MinMax::ID
    /// #         } else {
    /// #             MinMax(m.0 + self.0, m.1 + self.0)
    /// #         }
    /// #     }
    /// # }
    /// let v = [0, 42, 17, 6, -11].iter().map(|&i| MinMax(i, i));
    /// let mut lazy_tree: LazySegTree<MinMax, Add> = LazySegTree::from_iter_sized(v, 5);
    ///
    /// // Initialized with [0, 42, 17, 6, -11]
    /// assert_eq!(lazy_tree.get(..), MinMax(-11, 42));
    /// ```
    pub fn from_iter_sized<I: IntoIterator<Item = M>>(iter: I, size: usize) -> Self {
        let mut iter = iter.into_iter();
        let size = size.next_power_of_two();
        let mut v = Vec::with_capacity(size * 2);

        let v_ptr: *mut (M, A) = v.as_mut_ptr();

        unsafe {
            v.set_len(size * 2);

            for i in 0..size {
                ptr::write(
                    v_ptr.add(size + i),
                    if let Some(m) = iter.next() {
                        (m, A::ID)
                    } else {
                        (M::ID, A::ID)
                    },
                );
            }

            for i in (1..size).rev() {
                ptr::write(v_ptr.add(i), (v[i * 2].0.op(&v[i * 2 + 1].0), A::ID));
            }
        }

        LazySegTree(v)
    }

    /// Queries on the given interval.
    ///
    /// Note that any [`RangeBounds`](std::ops::RangeBounds) can be used including
    /// `..`, `a..`, `..b`, `..=c`, `d..e`, or `f..=g`.
    /// You can just `seg_tree.get(..)` to get the interval property of the entire elements and
    /// `lazy_tree.get(a)` to get a specific element.
    /// # Examples
    /// ```
    /// # use libpuri::{LazySegTree, Monoid, LazyAct};
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct MinMax(i64, i64);
    /// # impl Monoid for MinMax {
    /// #     const ID: Self = MinMax(i64::MAX, i64::MIN);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         MinMax(self.0.min(rhs.0), self.1.max(rhs.1))
    /// #     }
    /// # }
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct Add(i64);
    /// # impl Monoid for Add {
    /// #     const ID: Self = Add(0);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         Add(self.0.saturating_add(rhs.0))
    /// #     }
    /// # }
    /// #
    /// # impl LazyAct<MinMax> for Add {
    /// #     fn act(&self, m: &MinMax) -> MinMax {
    /// #         if m == &MinMax::ID {
    /// #             MinMax::ID
    /// #         } else {
    /// #             MinMax(m.0 + self.0, m.1 + self.0)
    /// #         }
    /// #     }
    /// # }
    /// // [0, 42, 6, 7, 2]
    /// let mut lazy_tree: LazySegTree<MinMax, Add> = [0, 42, 6, 7, 2].iter()
    ///     .map(|&n| MinMax(n, n))
    ///     .collect();
    ///
    /// assert_eq!(lazy_tree.get(..), MinMax(0, 42));
    ///
    /// // [5, 47, 11, 7, 2]
    /// lazy_tree.act(0..3, Add(5));
    ///
    /// // [5, 47, 4, 0, -5]
    /// lazy_tree.act(2..5, Add(-7));
    ///
    /// assert_eq!(lazy_tree.get(..), MinMax(-5, 47));
    /// assert_eq!(lazy_tree.get(..4), MinMax(0, 47));
    /// assert_eq!(lazy_tree.get(2), MinMax(4, 4));
    /// ```
    pub fn get<R>(&mut self, range: R) -> M
    where
        R: IntoIndex,
    {
        let (mut start, mut end) = range.into_index(self.size());
        start += self.size();
        end += self.size();

        self.propagate(start, end);

        let mut m = M::ID;

        while start < end {
            if start % 2 == 1 {
                m = self.0[start].0.op(&m);
                start += 1;
            }

            if end % 2 == 1 {
                end -= 1;
                m = self.0[end].0.op(&m);
            }

            start /= 2;
            end /= 2;
        }

        m
    }

    /// Sets an element with given index to the value. It propagates its update to its ancestors.
    ///
    /// It takes O(log(n)) to propagate the update as the height of the tree is log(n).
    ///
    /// # Examples
    /// ```
    /// # use libpuri::{LazySegTree, Monoid, LazyAct};
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct MinMax(i64, i64);
    /// # impl Monoid for MinMax {
    /// #     const ID: Self = MinMax(i64::MAX, i64::MIN);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         MinMax(self.0.min(rhs.0), self.1.max(rhs.1))
    /// #     }
    /// # }
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct Add(i64);
    /// # impl Monoid for Add {
    /// #     const ID: Self = Add(0);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         Add(self.0.saturating_add(rhs.0))
    /// #     }
    /// # }
    /// #
    /// # impl LazyAct<MinMax> for Add {
    /// #     fn act(&self, m: &MinMax) -> MinMax {
    /// #         if m == &MinMax::ID {
    /// #             MinMax::ID
    /// #         } else {
    /// #             MinMax(m.0 + self.0, m.1 + self.0)
    /// #         }
    /// #     }
    /// # }
    /// // [0, 42, 6, 7, 2]
    /// let mut lazy_tree: LazySegTree<MinMax, Add> = [0, 42, 6, 7, 2].iter()
    ///     .map(|&n| MinMax(n, n))
    ///     .collect();
    ///
    /// assert_eq!(lazy_tree.get(..), MinMax(0, 42));
    ///
    /// // [0, 1, 6, 7, 2]
    /// lazy_tree.set(1, MinMax(1, 1));
    ///
    /// assert_eq!(lazy_tree.get(1), MinMax(1, 1));
    /// assert_eq!(lazy_tree.get(..), MinMax(0, 7));
    /// assert_eq!(lazy_tree.get(2..), MinMax(2, 7));
    /// ```
    pub fn set(&mut self, i: usize, m: M) {
        let i = i + self.size();

        for h in (1..=self.height()).rev() {
            self.propagate_to_children(i >> h);
        }

        self.0[i] = (m, A::ID);

        for h in 1..=self.height() {
            self.update(i >> h);
        }
    }

    /// Apply an action to elements within given range.
    ///
    /// It takes O(log(n)).
    ///
    /// # Examples
    /// ```
    /// # use libpuri::{LazySegTree, Monoid, LazyAct};
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct MinMax(i64, i64);
    /// # impl Monoid for MinMax {
    /// #     const ID: Self = MinMax(i64::MAX, i64::MIN);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         MinMax(self.0.min(rhs.0), self.1.max(rhs.1))
    /// #     }
    /// # }
    /// #
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct Add(i64);
    /// # impl Monoid for Add {
    /// #     const ID: Self = Add(0);
    /// #     fn op(&self, rhs: &Self) -> Self {
    /// #         Add(self.0.saturating_add(rhs.0))
    /// #     }
    /// # }
    /// #
    /// # impl LazyAct<MinMax> for Add {
    /// #     fn act(&self, m: &MinMax) -> MinMax {
    /// #         if m == &MinMax::ID {
    /// #             MinMax::ID
    /// #         } else {
    /// #             MinMax(m.0 + self.0, m.1 + self.0)
    /// #         }
    /// #     }
    /// # }
    /// // [0, 42, 6, 7, 2]
    /// let mut lazy_tree: LazySegTree<MinMax, Add> = [0, 42, 6, 7, 2].iter()
    ///     .map(|&n| MinMax(n, n))
    ///     .collect();
    ///
    /// assert_eq!(lazy_tree.get(..), MinMax(0, 42));
    ///
    /// // [0, 30, -6, 7, 2]
    /// lazy_tree.act(1..3, Add(-12));
    ///
    /// assert_eq!(lazy_tree.get(1), MinMax(30, 30));
    /// assert_eq!(lazy_tree.get(..), MinMax(-6, 30));
    /// assert_eq!(lazy_tree.get(2..), MinMax(-6, 7));
    /// ```
    pub fn act<R>(&mut self, range: R, a: A)
    where
        R: IntoIndex,
    {
        let (mut start, mut end) = range.into_index(self.size());
        start += self.size();
        end += self.size();

        self.propagate(start, end);

        {
            let mut start = start;
            let mut end = end;

            while start < end {
                if start % 2 == 1 {
                    Self::act_lazy(&mut self.0[start], &a);
                    start += 1;
                }
                if end % 2 == 1 {
                    end -= 1;
                    Self::act_lazy(&mut self.0[end], &a);
                }

                start /= 2;
                end /= 2;
            }
        }

        for i in 1..=self.height() {
            if (start >> i) << i != start {
                self.update(start >> i);
            }
            if (end >> i) << i != end {
                self.update((end - 1) >> i);
            }
        }
    }
}

impl<M, A> Debug for LazySegTree<M, A>
where
    M: Debug + Monoid,
    A: Debug + LazyAct<M>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tree = "LazySegTree\n".to_owned();
        for h in 0..self.height() {
            for i in 1 << h..1 << (h + 1) {
                tree.push_str(&if self.0[i].0 == M::ID {
                    "(id, ".to_owned()
                } else {
                    format!("({:?}, ", self.0[i].0)
                });
                tree.push_str(&if self.0[i].1 == A::ID {
                    "id) ".to_owned()
                } else {
                    format!("{:?}) ", self.0[i].1)
                });
            }
            tree.pop();
            tree.push('\n');
        }

        f.write_str(&tree)
    }
}

/// You can `collect` into a lazy segment tree.
impl<M, A> FromIterator<M> for LazySegTree<M, A>
where
    M: Monoid,
    A: LazyAct<M>,
{
    fn from_iter<I: IntoIterator<Item = M>>(iter: I) -> Self {
        let v: Vec<M> = iter.into_iter().collect();
        let len = v.len();

        LazySegTree::from_iter_sized(v, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct MinMax(i64, i64);
    impl Monoid for MinMax {
        const ID: Self = MinMax(i64::MAX, i64::MIN);
        fn op(&self, rhs: &Self) -> Self {
            MinMax(self.0.min(rhs.0), self.1.max(rhs.1))
        }
    }
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Add(i64);
    impl Monoid for Add {
        const ID: Self = Add(0);
        fn op(&self, rhs: &Self) -> Self {
            Add(self.0.saturating_add(rhs.0))
        }
    }
    impl LazyAct<MinMax> for Add {
        fn act(&self, m: &MinMax) -> MinMax {
            if m == &MinMax::ID {
                MinMax::ID
            } else {
                MinMax(m.0 + self.0, m.1 + self.0)
            }
        }
    }

    #[test]
    fn min_max_and_range_add() {
        // [id, id, id, id, id, id, id, id]
        let mut t: LazySegTree<MinMax, Add> = LazySegTree::new(8);
        assert_eq!(t.get(..), MinMax::ID);

        // [0,  0,  0,  0,  0,  0, id, id]
        for i in 0..6 {
            t.set(i, MinMax(0, 0));
        }

        // [5,  5,  5,  5,  0,  0, id, id]
        t.act(0..=3, Add(5));

        // [5,  5, 47, 47, 42, 42, id, id]
        t.act(2..=5, Add(42));

        assert_eq!(t.get(0..=1), MinMax(5, 5));
        assert_eq!(t.get(1..=2), MinMax(5, 47));
        assert_eq!(t.get(2..=3), MinMax(47, 47));
        assert_eq!(t.get(3..=4), MinMax(42, 47));
        assert_eq!(t.get(4..=5), MinMax(42, 42));
        assert_eq!(t.get(5..=6), MinMax(42, 42));
        assert_eq!(t.get(0..=5), MinMax(5, 47));
        assert_eq!(t.get(6..=7), MinMax::ID);
        assert_eq!(t.get(5), MinMax(42, 42));
    }

    #[test]
    fn many_intervals() {
        let mut t: LazySegTree<MinMax, Add> = LazySegTree::new(88);

        for i in 0..88 {
            t.set(i, MinMax(0, 0));
        }

        t.act(0..20, Add(5));
        t.act(20..40, Add(42));
        t.act(40..60, Add(-5));
        t.act(60..88, Add(17));
        t.act(10..70, Add(1));

        assert_eq!(t.get(..), MinMax(-4, 43));
        assert_eq!(t.get(0..20), MinMax(5, 6));
        assert_eq!(t.get(70..88), MinMax(17, 17));
        assert_eq!(t.get(40), MinMax(-4, -4));
    }
}
