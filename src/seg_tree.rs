use super::algebra::Monoid;
use super::util::IntoIndex;
use std::iter::FromIterator;
use std::ptr;

/// A segment tree that supports range query.
///
/// A segment tree is used when you want to query on properties of interval. For instance, you have
/// a list of numbers and you want to get a minimum value of certain interval. If you compute it on
/// the fly, it would require (m - 1) comparisons for the length m interval. We can use a segment tree
/// to efficiently compute the minimum element.
///
/// Each node of a segment tree represents a union of intervals of their child nodes and each leaf
/// node means an interval containing only one element. Following is an example segment tree of
/// elements [1, 42, 16, 3, 5].
/// <pre>
///                          +-----+
///                          |  1  |
///                          +-----+
///                          [0, 8)
///                     /               \
///                    /                 \
///            +---+                         +---+
///            | 1 |                         | 5 |
///            +---+                         +---+  
///            [0, 4)                        [4, 8)  
///           /     \                       /     \
///          /       \                     /       \
///      +---+       +---+             +---+       +----+  
///      | 1 |       | 3 |             | 5 |       | id |   
///      +---+       +---+             +---+       +----+   
///      [0, 2)      [2, 4)            [4, 6)      [6, 8)  
///     /    |       |    \           /    |       |    \
///    /     |       |     \         /     |       |     \
/// +---+  +----+  +----+  +---+  +---+  +----+  +----+  +----+
/// | 1 |  | 42 |  | 16 |  | 3 |  | 5 |  | id |  | id |  | id |
/// +---+  +----+  +----+  +---+  +---+  +----+  +----+  +----+
/// [0, 1) [1, 2)  [2, 3)  [3, 4) [4, 5) [5, 6)  [6, 7)  [7, 8)
/// </pre>
///
/// When you update an element, it propagates from the leaf to the top. If we update 16 to 2, then
/// the parent node would be updated to 3 -> 2, but the [0, 4) and root node won't be updated as 1
/// is less than 2.
///
/// When querying, it visits non-overlapping nodes within the inteval and computes the minimum among
/// the nodes. For example if we query minimum element in an interval [1, 4), it first visits [1, 2)
/// node and then it visits [2, 4). Then it computes min(42, 3) which is 3. Note that it only visits
/// two nodes at each height of the tree hence the time complexity O(log(n)).
///
/// # Use a segment tree when:
/// - You only update one element at a time.
/// - You want to efficiently query on a property of interval.
/// - The interval property is computed by [associative operations](https://en.wikipedia.org/wiki/Associative_property) on the elements in the interval.
///
/// To be precise, a segment tree requires the elements to implement the [`Monoid`] trait.
///
/// # Performance
/// Given n elements, it computes the interval property in O(log(n)) at the expense of O(log(n))
/// update time.
///
/// If we were to store the elements with `Vec`, it would take O(m) for length m interval query
/// and O(1) to update.
///
/// # Examples
/// ```
/// use libpuri::{Monoid, SegTree};
///
/// // We'll use the segment tree to compute interval sum.
/// #[derive(Clone, Debug, PartialEq, Eq)]
/// struct Sum(i64);
///
/// impl Monoid for Sum {
///     const ID: Self = Sum(0);
///     fn op(&self, rhs: &Self) -> Self { Sum(self.0 + rhs.0) }
/// }
///
/// // Segment tree can be initialized from an iterator of the monoid
/// let mut seg_tree: SegTree<Sum> = [1, 2, 3, 4, 5].iter().map(|&n| Sum(n)).collect();
///
/// // Add elements within range 0..=3
/// assert_eq!(seg_tree.get(0..=3), Sum(10));
///
/// // Update element at 2 to 42
/// seg_tree.set(2, Sum(42));
/// assert_eq!(seg_tree.get(0..=3), Sum(49));
/// ```
#[derive(Debug)]
pub struct SegTree<M: Monoid>(Vec<M>);

impl<M: Monoid> SegTree<M> {
    fn size(&self) -> usize {
        self.0.len() / 2
    }
}

impl<M: Monoid> SegTree<M> {
    /// Constructs a new segment tree with at least given number of interval propertiess can be stored.
    ///
    /// The segment tree will be initialized with the identity elements.
    /// 
    /// # Complexity
    /// O(n).
    /// 
    /// If you know the initial elements in advance, [`from_iter_sized()`](SegTree::from_iter_sized) should be preferred over
    /// `new()`.
    ///
    /// Initializing with the identity elements and updating n elements will tax you O(nlog(n)),
    /// whereas `from_iter_sized()` is O(n) by computing the interval properties only once.
    ///
    /// # Examples
    /// ```
    /// use libpuri::{Monoid, SegTree};
    ///
    /// #[derive(Clone, Debug, PartialEq, Eq)]
    /// struct MinMax(i64, i64);
    ///
    /// impl Monoid for MinMax {
    ///     const ID: Self = Self(i64::MAX, i64::MIN);
    ///     fn op(&self, rhs: &Self) -> Self { Self(self.0.min(rhs.0), self.1.max(rhs.1)) }
    /// }
    ///
    /// // ⚠️ Use `from_iter_sized` whenever possible.
    /// // See the Complexity paragraph above for how it differs.
    /// let mut seg_tree: SegTree<MinMax> = SegTree::new(4);
    ///
    /// seg_tree.set(0, MinMax(1, 1));
    /// assert_eq!(seg_tree.get(0..4), MinMax(1, 1));
    ///
    /// seg_tree.set(3, MinMax(4, 4));
    /// assert_eq!(seg_tree.get(0..2), MinMax(1, 1));
    /// assert_eq!(seg_tree.get(2..4), MinMax(4, 4));
    ///
    /// seg_tree.set(2, MinMax(3, 3));
    /// assert_eq!(seg_tree.get(0..3), MinMax(1, 3));
    /// assert_eq!(seg_tree.get(0..4), MinMax(1, 4));
    /// ```
    pub fn new(size: usize) -> Self {
        SegTree(vec![M::ID; size.next_power_of_two() * 2])
    }

    /// Constructs a new segment tree with given interval properties.
    ///
    /// # Complexity
    /// O(n).
    ///
    /// # Examples
    /// ```
    /// use libpuri::{Monoid, SegTree};
    ///
    /// #[derive(Clone, Debug, PartialEq, Eq)]
    /// struct Sum(i64);
    ///
    /// impl Monoid for Sum {
    ///     const ID: Self = Self(0);
    ///     fn op(&self, rhs: &Self) -> Self { Self(self.0 + rhs.0) }
    /// }
    ///
    /// let mut seg_tree: SegTree<Sum> = SegTree::from_iter_sized(
    ///     [1, 2, 3, 42].iter().map(|&i| Sum(i)),
    ///     4
    /// );
    ///
    /// assert_eq!(seg_tree.get(0..4), Sum(48));
    ///
    /// // [1, 2, 3, 0]
    /// seg_tree.set(3, Sum(0));
    /// assert_eq!(seg_tree.get(0..2), Sum(3));
    /// assert_eq!(seg_tree.get(2..4), Sum(3));
    ///
    /// // [1, 2, 2, 0]
    /// seg_tree.set(2, Sum(2));
    /// assert_eq!(seg_tree.get(0..3), Sum(5));
    /// assert_eq!(seg_tree.get(0..4), Sum(5));
    /// ```
    pub fn from_iter_sized<I: IntoIterator<Item = M>>(iter: I, size: usize) -> Self {
        let mut iter = iter.into_iter();
        let size = size.next_power_of_two();
        let mut v = Vec::with_capacity(size * 2);

        let v_ptr: *mut M = v.as_mut_ptr();

        unsafe {
            v.set_len(size * 2);

            for i in 0..size {
                ptr::write(
                    v_ptr.add(size + i),
                    if let Some(m) = iter.next() { m } else { M::ID },
                );
            }

            for i in (1..size).rev() {
                ptr::write(v_ptr.add(i), v[i * 2].op(&v[i * 2 + 1]));
            }
        }

        SegTree(v)
    }

    /// Queries on the given interval.
    ///
    /// Note that any [`RangeBounds`](std::ops::RangeBounds) can be used including
    /// `..`, `a..`, `..b`, `..=c`, `d..e`, or `f..=g`.
    /// You can just `seg_tree.get(..)` to get the interval property of the entire elements and
    /// `seg_tree.get(a)` to get a specific element.
    /// # Examples
    /// ```
    /// # use libpuri::{Monoid, SegTree};
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct MinMax(i64, i64);
    /// # impl Monoid for MinMax {
    /// #     const ID: Self = Self(i64::MAX, i64::MIN);
    /// #     fn op(&self, rhs: &Self) -> Self { Self(self.0.min(rhs.0), self.1.max(rhs.1)) }
    /// # }
    /// let mut seg_tree: SegTree<MinMax> = SegTree::new(4);
    ///
    /// assert_eq!(seg_tree.get(..), MinMax::ID);
    ///
    /// seg_tree.set(0, MinMax(42, 42));
    ///
    /// assert_eq!(seg_tree.get(..), MinMax(42, 42));
    /// assert_eq!(seg_tree.get(1..), MinMax::ID);
    /// assert_eq!(seg_tree.get(0), MinMax(42, 42));
    /// ```
    pub fn get<R>(&self, range: R) -> M
    where
        R: IntoIndex,
    {
        let (mut start, mut end) = range.into_index(self.size());
        start += self.size();
        end += self.size();

        let mut m = M::ID;
        while start < end {
            if start % 2 == 1 {
                m = self.0[start].op(&m);
                start += 1;
            }

            if end % 2 == 1 {
                end -= 1;
                m = self.0[end].op(&m);
            }

            start /= 2;
            end /= 2;
        }

        m
    }

    /// Sets an element with index i to a value m. It propagates its update to its parent.
    ///
    /// It takes O(log(n)) to propagate the update as the height of the tree is log(n).
    ///
    /// # Examples
    /// ```
    /// # use libpuri::{Monoid, SegTree};
    /// # #[derive(Clone, Debug, PartialEq, Eq)]
    /// # struct MinMax(i64, i64);
    /// # impl Monoid for MinMax {
    /// #     const ID: Self = Self(i64::MAX, i64::MIN);
    /// #     fn op(&self, rhs: &Self) -> Self { Self(self.0.min(rhs.0), self.1.max(rhs.1)) }
    /// # }
    /// let mut seg_tree: SegTree<MinMax> = SegTree::new(4);
    ///
    /// seg_tree.set(0, MinMax(4, 4));
    /// ```
    pub fn set(&mut self, mut i: usize, m: M) {
        i += self.size();
        self.0[i] = m;

        while i > 1 {
            i /= 2;
            self.0[i] = self.0[i * 2].op(&self.0[i * 2 + 1]);
        }
    }
}

/// You can `collect` into a segment tree.
impl<M: Monoid> FromIterator<M> for SegTree<M> {
    fn from_iter<I: IntoIterator<Item = M>>(iter: I) -> Self {
        let v: Vec<M> = iter.into_iter().collect();
        let len = v.len();

        SegTree::from_iter_sized(v, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_sum() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct Sum(i64);
        impl Monoid for Sum {
            const ID: Self = Sum(0);
            fn op(&self, rhs: &Self) -> Self {
                Sum(self.0 + rhs.0)
            }
        }

        let mut seg_tree: SegTree<Sum> = (1..=5).map(|n| Sum(n)).collect();

        for &(update, i, j) in [(true, 2, 6), (false, 1, 4), (true, 4, 2), (false, 2, 4)].iter() {
            if update {
                seg_tree.set(i, Sum(j as i64));
            } else {
                assert_eq!(seg_tree.get(i..=j).0, if i == 1 { 17 } else { 12 });
            }
        }
    }
}
