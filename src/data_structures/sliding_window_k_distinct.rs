//! Sliding-window enumeration of maximal intervals with exactly `k` distinct
//! values.
//!
//! Given a sequence `seq` and a target `k`, this module enumerates every
//! *maximal* contiguous interval `[l..=r]` whose elements form exactly `k`
//! distinct values. An interval is **maximal** when:
//!
//! 1. It cannot be extended to the right: either `r == seq.len() - 1`, or
//!    `seq[r + 1]` is a value not already present in `seq[l..=r]` (so
//!    extending would push the distinct count above `k`).
//! 2. It cannot be extended to the left: either `l == 0`, or `seq[l - 1]` is
//!    a value not already present in `seq[l..=r]` (so extending would push
//!    the distinct count above `k`).
//!
//! Equivalently, neither endpoint can move outward without exceeding `k`
//! distinct values. When `k == 0` the only interval with exactly zero
//! distinct values is the trivially-empty one, so the enumeration is empty.
//!
//! The implementation uses a two-pointer sweep paired with a
//! `HashMap<&T, usize>` of value counts. Each index is visited a constant
//! number of times by either pointer, giving an overall complexity of
//! `O(n)` in time and `O(k)` in extra memory (beyond the output vector).

use std::collections::HashMap;
use std::hash::Hash;

/// Enumerates every maximal contiguous interval `[l..=r]` of `seq` containing
/// exactly `k` distinct values, returned as inclusive `(l, r)` pairs in
/// left-to-right order of `r`.
///
/// See the [module-level documentation](self) for the precise definition of
/// *maximal*. In short, a returned interval cannot be extended in either
/// direction without exceeding `k` distinct values.
///
/// - Time: `O(n)` where `n = seq.len()`.
/// - Space: `O(k)` auxiliary, plus the returned vector.
///
/// Returns an empty vector when `k == 0`, when `seq` is empty, or when `seq`
/// contains fewer than `k` distinct values.
pub fn enumerate_k_distinct<T: Eq + Hash + Clone>(seq: &[T], k: usize) -> Vec<(usize, usize)> {
    let n = seq.len();
    if k == 0 || n == 0 {
        return Vec::new();
    }

    // Precompute, for each r, the smallest l such that seq[l..=r] has at most
    // k distinct values. This sequence is non-decreasing in r, so a single
    // two-pointer sweep computes it in O(n).
    let mut l_min: Vec<usize> = Vec::with_capacity(n);
    let mut counts: HashMap<&T, usize> = HashMap::new();
    let mut l: usize = 0;
    for r in 0..n {
        *counts.entry(&seq[r]).or_insert(0) += 1;
        while counts.len() > k {
            let cnt = counts
                .get_mut(&seq[l])
                .expect("left pointer value must be tracked");
            *cnt -= 1;
            if *cnt == 0 {
                counts.remove(&seq[l]);
            }
            l += 1;
        }
        l_min.push(l);
    }

    // An interval is maximal exactly when it equals (l_min[r], r) for some r,
    // the window has exactly k distinct values, and the right boundary cannot
    // be extended without introducing a new distinct value (either r is the
    // last index, or l_min strictly increases at r + 1, meaning seq[r + 1]
    // was a fresh distinct value that forced the left pointer forward).
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut window: HashMap<&T, usize> = HashMap::new();
    let mut cur_l: usize = 0;
    for r in 0..n {
        // Advance the auxiliary window to match [l_min[r] .. r].
        *window.entry(&seq[r]).or_insert(0) += 1;
        while cur_l < l_min[r] {
            let cnt = window
                .get_mut(&seq[cur_l])
                .expect("left pointer value must be tracked");
            *cnt -= 1;
            if *cnt == 0 {
                window.remove(&seq[cur_l]);
            }
            cur_l += 1;
        }

        if window.len() != k {
            continue;
        }
        let right_maximal = r + 1 == n || l_min[r + 1] > l_min[r];
        if right_maximal {
            out.push((l_min[r], r));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::enumerate_k_distinct;

    #[test]
    fn empty_sequence_returns_empty() {
        let seq: [i32; 0] = [];
        assert_eq!(enumerate_k_distinct(&seq, 1), Vec::<(usize, usize)>::new());
        assert_eq!(enumerate_k_distinct(&seq, 0), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn k_zero_is_always_empty() {
        assert_eq!(
            enumerate_k_distinct(&[1, 2, 3], 0),
            Vec::<(usize, usize)>::new()
        );
        assert_eq!(
            enumerate_k_distinct::<i32>(&[], 0),
            Vec::<(usize, usize)>::new()
        );
    }

    #[test]
    fn k_one_runs_of_equal_values() {
        // Maximal runs of a single distinct value.
        let seq = [1, 1, 2, 2, 2, 3, 1];
        assert_eq!(
            enumerate_k_distinct(&seq, 1),
            vec![(0, 1), (2, 4), (5, 5), (6, 6)]
        );
    }

    #[test]
    fn k_exceeds_distinct_count_returns_empty() {
        let seq = [1, 1, 2, 2];
        assert_eq!(enumerate_k_distinct(&seq, 3), Vec::<(usize, usize)>::new());
        assert_eq!(enumerate_k_distinct(&seq, 99), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn classic_example_1_2_1_3_2() {
        let seq = [1, 2, 1, 3, 2];
        // [0,2] = {1,2}; [2,3] = {1,3}; [3,4] = {3,2}.
        assert_eq!(enumerate_k_distinct(&seq, 2), vec![(0, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn all_same_with_k_one_is_single_interval() {
        let seq = [7, 7, 7, 7];
        assert_eq!(enumerate_k_distinct(&seq, 1), vec![(0, 3)]);
    }

    #[test]
    fn k_equals_distinct_count_one_global_interval() {
        let seq = [1, 2, 3, 1, 2];
        assert_eq!(enumerate_k_distinct(&seq, 3), vec![(0, 4)]);
    }

    #[test]
    fn strings_supported_via_generics() {
        let seq: Vec<String> = ["a", "b", "a", "c"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        assert_eq!(enumerate_k_distinct(&seq, 2), vec![(0, 2), (2, 3)]);
    }

    #[test]
    fn returned_intervals_are_truly_maximal() {
        let seq = [4, 5, 4, 6, 7, 5, 5, 8];
        let k = 2;
        let out = enumerate_k_distinct(&seq, k);
        for &(l, r) in &out {
            // Exactly k distinct in [l..=r].
            let mut s = std::collections::HashSet::new();
            for v in &seq[l..=r] {
                s.insert(v);
            }
            assert_eq!(s.len(), k, "[{l},{r}] should have {k} distinct");
            // Cannot extend right.
            if r + 1 < seq.len() {
                assert!(!s.contains(&&seq[r + 1]), "[{l},{r}] not right-maximal");
            }
            // Cannot extend left.
            if l > 0 {
                assert!(!s.contains(&&seq[l - 1]), "[{l},{r}] not left-maximal");
            }
        }
    }
}
