//! Idiomatic rust competitive programming library.
mod seg_tree;
pub use self::seg_tree::*;

mod lazy_seg_tree;
pub use self::lazy_seg_tree::*;

mod algebra;
pub use self::algebra::*;

mod util;
pub use self::util::IntoIndex;
