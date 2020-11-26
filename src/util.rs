use std::ops::{
    Bound::*, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

/// Into inclusive start and exclusive end indices.
pub trait IntoIndex {
    fn into_index(self, size: usize) -> (usize, usize);
}

macro_rules! into_index_impl {
    ($($range:ty)*) => {
        $(
            impl IntoIndex for $range {
                fn into_index(self, size: usize) -> (usize, usize) {
                    into_index(self, size)
                }
            }
        )*
    };
}

into_index_impl! { Range<usize> RangeFrom<usize> RangeInclusive<usize> RangeToInclusive<usize>
RangeTo<usize> RangeFull }

impl IntoIndex for usize {
    fn into_index(self, _: usize) -> (usize, usize) {
        (self, self + 1)
    }
}

// TODO(yujingaya) Be defensive with some assertion
pub fn into_index<R: RangeBounds<usize>>(range: R, size: usize) -> (usize, usize) {
    (
        match range.start_bound() {
            Included(&start) => start,
            Excluded(&start) => start + 1,
            Unbounded => 0,
        },
        match range.end_bound() {
            Included(&end) => end + 1,
            Excluded(&end) => end,
            Unbounded => size,
        },
    )
}
