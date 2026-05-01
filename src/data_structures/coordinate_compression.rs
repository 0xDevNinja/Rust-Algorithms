//! Coordinate compression: maps a sparse set of comparable values onto a dense
//! range of indices `0..k` (where `k` is the number of distinct values) while
//! preserving order. Runs in O(n log n) time, O(n) extra space.
//!
//! Useful when an algorithm's complexity depends on the value range (Fenwick
//! trees, segment trees, offline range queries) but the actual values are
//! sparse, large, or negative. Compressing first turns a value-indexed
//! structure of size `max(values)` into one of size `k`.

/// Compresses a slice of comparable values into dense ranks.
///
/// Returns `(ranks, sorted_unique)` where:
/// - `ranks[i]` is the 0-based rank of `values[i]` in the sorted, deduplicated
///   set of input values, so `sorted_unique[ranks[i]] == values[i]`.
/// - `sorted_unique` is the sorted vector of distinct input values.
///
/// Empty input returns `(vec![], vec![])`. Runs in O(n log n).
pub fn coordinate_compress<T: Ord + Clone>(values: &[T]) -> (Vec<usize>, Vec<T>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut sorted_unique: Vec<T> = values.to_vec();
    sorted_unique.sort();
    sorted_unique.dedup();
    let ranks: Vec<usize> = values
        .iter()
        .map(|v| sorted_unique.binary_search(v).expect("value present"))
        .collect();
    (ranks, sorted_unique)
}

/// Query-only coordinate compressor: build once over a value set, then look up
/// the rank of arbitrary values in O(log k).
pub struct Compressor<T> {
    sorted: Vec<T>,
}

impl<T: Ord + Clone> Compressor<T> {
    /// Builds a compressor over the distinct values in `values`. O(n log n).
    pub fn new(values: &[T]) -> Self {
        let mut sorted: Vec<T> = values.to_vec();
        sorted.sort();
        sorted.dedup();
        Self { sorted }
    }

    /// Returns the dense rank of `value` if it is present, else `None`.
    /// O(log k).
    pub fn rank(&self, value: &T) -> Option<usize> {
        self.sorted.binary_search(value).ok()
    }

    /// Number of distinct values in the compressed set.
    pub const fn len(&self) -> usize {
        self.sorted.len()
    }

    /// True if the compressed set is empty.
    pub const fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// Borrows the sorted, deduplicated value set.
    pub fn sorted_unique(&self) -> &[T] {
        &self.sorted
    }
}

#[cfg(test)]
mod tests {
    use super::{coordinate_compress, Compressor};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let (ranks, sorted): (Vec<usize>, Vec<i32>) = coordinate_compress(&[]);
        assert!(ranks.is_empty());
        assert!(sorted.is_empty());
    }

    #[test]
    fn single() {
        let (ranks, sorted) = coordinate_compress(&[42_i32]);
        assert_eq!(ranks, vec![0]);
        assert_eq!(sorted, vec![42]);
    }

    #[test]
    fn duplicates_collapse() {
        let (ranks, sorted) = coordinate_compress(&[5, 5, 5, 5]);
        assert_eq!(ranks, vec![0, 0, 0, 0]);
        assert_eq!(sorted, vec![5]);
    }

    #[test]
    fn descending_input() {
        let (ranks, sorted) = coordinate_compress(&[100, 50, 25, 10, 1]);
        assert_eq!(ranks, vec![4, 3, 2, 1, 0]);
        assert_eq!(sorted, vec![1, 10, 25, 50, 100]);
    }

    #[test]
    fn mixed_signs_i64() {
        let input: Vec<i64> = vec![-1_000_000_000_000, 0, 1_000_000_000_000, -1, 1, 0];
        let (ranks, sorted) = coordinate_compress(&input);
        assert_eq!(
            sorted,
            vec![-1_000_000_000_000, -1, 0, 1, 1_000_000_000_000]
        );
        assert_eq!(ranks, vec![0, 2, 4, 1, 3, 2]);
    }

    #[test]
    fn duplicates_with_distinct_values() {
        let (ranks, sorted) = coordinate_compress(&[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]);
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 9]);
        assert_eq!(ranks, vec![2, 0, 3, 0, 4, 6, 1, 5, 4, 2, 4]);
    }

    #[test]
    fn compressor_rank_present() {
        let c = Compressor::new(&[10, 20, 30, 20, 10]);
        assert_eq!(c.rank(&10), Some(0));
        assert_eq!(c.rank(&20), Some(1));
        assert_eq!(c.rank(&30), Some(2));
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        assert_eq!(c.sorted_unique(), &[10, 20, 30]);
    }

    #[test]
    fn compressor_rank_absent_returns_none() {
        let c = Compressor::new(&[10, 20, 30]);
        assert_eq!(c.rank(&15), None);
        assert_eq!(c.rank(&0), None);
        assert_eq!(c.rank(&100), None);
    }

    #[test]
    fn compressor_empty() {
        let c: Compressor<i32> = Compressor::new(&[]);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.rank(&0), None);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn roundtrip(values: Vec<i32>) -> bool {
        let values: Vec<i32> = values.into_iter().take(50).collect();
        let (ranks, sorted) = coordinate_compress(&values);
        if ranks.len() != values.len() {
            return false;
        }
        for (i, v) in values.iter().enumerate() {
            if &sorted[ranks[i]] != v {
                return false;
            }
        }
        // ranks must lie in [0, sorted.len())
        ranks.iter().all(|&r| r < sorted.len() || values.is_empty())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn ranks_preserve_order(values: Vec<i32>) -> bool {
        let values: Vec<i32> = values.into_iter().take(50).collect();
        let (ranks, _) = coordinate_compress(&values);
        for i in 0..values.len() {
            for j in 0..values.len() {
                let want = values[i].cmp(&values[j]);
                let got = ranks[i].cmp(&ranks[j]);
                if want != got {
                    return false;
                }
            }
        }
        true
    }
}
