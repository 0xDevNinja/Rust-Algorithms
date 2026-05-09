//! Bin packing via the First-Fit-Decreasing (FFD) heuristic.
//!
//! Given a list of item sizes and a uniform bin `capacity`, distributes the
//! items into a minimal number of bins so that the sum of sizes in each bin
//! does not exceed `capacity`. FFD sorts the items by size in *descending*
//! order and, for each item in turn, places it into the first already-open
//! bin that still has enough remaining capacity. If no open bin fits the
//! current item, a new bin is opened.
//!
//! Time complexity: `O(n^2)` in the worst case — sorting costs `O(n log n)`,
//! and each of the `n` items scans through up to `n` open bins.
//! Space complexity: `O(n)` for the indirection / output bins.
//!
//! Approximation guarantee: classic bin packing is NP-hard, but FFD achieves
//! a worst-case bound of `FFD(I) <= 11/9 · OPT(I) + 1` (Dosa, 2007 — tight
//! constant; Johnson's original 1973 analysis gave `11/9·OPT + 4`). In other
//! words, FFD never uses more than about 22 % more bins than the unknown
//! optimum, plus a small additive slack. This is provably the best constant
//! achievable by *any* sort-then-first-fit family and is the standard
//! reference baseline for bin packing heuristics.
//!
//! Edge cases / panics:
//! - An item strictly greater than `capacity` has no feasible bin: the
//!   function panics. Callers that wish to *skip* over-large items should
//!   filter them beforehand.
//! - A `capacity` of `0` together with an empty `items` slice yields an empty
//!   `Vec`. A `capacity` of `0` with any non-empty input panics (every item
//!   exceeds capacity).
//! - Item size `0` always fits and is placed into the first bin (a new one is
//!   opened on the very first such item if none yet exists).

/// Packs `items` into bins of size `capacity` using First-Fit-Decreasing.
///
/// Returns the bins as a `Vec<Vec<u64>>` of item *sizes*; each inner `Vec` is
/// guaranteed to sum to at most `capacity`. The relative order of items
/// within a bin reflects the order in which FFD placed them (i.e. descending
/// by size, with ties broken by stable original-index order).
///
/// # Panics
/// Panics if any element of `items` is strictly greater than `capacity` —
/// such items have no valid bin and the input is rejected as ill-formed.
///
/// Time: `O(n^2)`. Space: `O(n)`.
///
/// See the module-level documentation for the FFD approximation guarantee
/// (`FFD <= 11/9·OPT + 1`).
#[must_use]
pub fn first_fit_decreasing(items: &[u64], capacity: u64) -> Vec<Vec<u64>> {
    let bins_by_index = first_fit_decreasing_indices(items, capacity);
    bins_by_index
        .into_iter()
        .map(|bin| bin.into_iter().map(|i| items[i]).collect())
        .collect()
}

/// Same as [`first_fit_decreasing`], but each inner `Vec` contains the
/// *original indices* of the items rather than their sizes — convenient when
/// callers need to map placements back to the input slice.
///
/// # Panics
/// Panics if any element of `items` is strictly greater than `capacity`.
///
/// Time: `O(n^2)`. Space: `O(n)`.
#[must_use]
pub fn first_fit_decreasing_indices(items: &[u64], capacity: u64) -> Vec<Vec<usize>> {
    if items.is_empty() {
        return Vec::new();
    }

    // Reject inputs we cannot pack at all rather than silently dropping the
    // offending item or blowing past `capacity`.
    if let Some(bad) = items.iter().position(|&s| s > capacity) {
        panic!(
            "bin_packing_ffd: item at index {bad} has size {} > capacity {capacity}",
            items[bad]
        );
    }

    // Sort *indices* in descending order by item size. Stable so that ties
    // preserve the caller's original ordering, which keeps the output
    // deterministic and matches the expected canonical FFD layout.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&i, &j| items[j].cmp(&items[i]));

    // `bins` holds the indices placed in each bin; `remaining` mirrors it
    // with the leftover capacity, so we can avoid summing on every probe.
    let mut bins: Vec<Vec<usize>> = Vec::new();
    let mut remaining: Vec<u64> = Vec::new();

    for idx in order {
        let size = items[idx];
        // First-fit scan over the open bins.
        let mut placed = false;
        for (b, rem) in remaining.iter_mut().enumerate() {
            if *rem >= size {
                bins[b].push(idx);
                *rem -= size;
                placed = true;
                break;
            }
        }
        if !placed {
            // No existing bin fits — open a new one.
            bins.push(vec![idx]);
            remaining.push(capacity - size);
        }
    }

    bins
}

#[cfg(test)]
mod tests {
    use super::{first_fit_decreasing, first_fit_decreasing_indices};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_items_returns_empty_bins() {
        let bins = first_fit_decreasing(&[], 10);
        assert!(bins.is_empty());
        let bins_idx = first_fit_decreasing_indices(&[], 10);
        assert!(bins_idx.is_empty());
    }

    #[test]
    fn single_item_under_capacity_one_bin() {
        let bins = first_fit_decreasing(&[5], 10);
        assert_eq!(bins, vec![vec![5]]);
    }

    #[test]
    fn single_item_at_capacity_one_bin() {
        let bins = first_fit_decreasing(&[10], 10);
        assert_eq!(bins, vec![vec![10]]);
    }

    #[test]
    fn canonical_ffd_packing() {
        // Items [4,8,1,4,2,1] sorted desc -> [8,4,4,2,1,1], capacity 10.
        // Place 8 -> bin0 (rem 2). Place 4 -> 4 > 2, open bin1 (rem 6).
        // Place 4 -> fits bin1 (rem 2). Place 2 -> fits bin0 (rem 0).
        // Place 1 -> 1 > 0, fits bin1 (rem 1). Place 1 -> fits bin1 (rem 0).
        // Result: [[8,2], [4,4,1,1]] (2 bins).
        let bins = first_fit_decreasing(&[4, 8, 1, 4, 2, 1], 10);
        assert_eq!(bins, vec![vec![8, 2], vec![4, 4, 1, 1]]);
        // Each bin within capacity.
        for bin in &bins {
            assert!(bin.iter().sum::<u64>() <= 10);
        }
    }

    #[test]
    fn all_items_equal_to_capacity_one_per_bin() {
        let bins = first_fit_decreasing(&[7, 7, 7, 7], 7);
        assert_eq!(bins.len(), 4);
        for bin in &bins {
            assert_eq!(bin, &vec![7]);
        }
    }

    #[test]
    fn tight_pairs_pack_two_per_bin() {
        // Items pair up exactly: (9+1), (8+2), (7+3), capacity 10.
        // Sorted desc: [9,8,7,3,2,1]. Each large item opens a new bin and the
        // small ones fill the residual capacity in first-fit order.
        let bins = first_fit_decreasing(&[9, 1, 8, 2, 7, 3], 10);
        assert_eq!(bins.len(), 3);
        for bin in &bins {
            assert_eq!(bin.iter().sum::<u64>(), 10);
        }
    }

    #[test]
    fn zero_size_items_share_a_bin() {
        // Zero-size items always fit; FFD packs them into the first bin.
        let bins = first_fit_decreasing(&[0, 0, 0], 5);
        assert_eq!(bins, vec![vec![0, 0, 0]]);
    }

    #[test]
    fn indices_variant_returns_original_positions() {
        // Items [4,8,1,4,2,1] with capacity 10.
        // Sorted desc by size with stable tie-break (preserve original order):
        //   [8 (idx1), 4 (idx0), 4 (idx3), 2 (idx4), 1 (idx2), 1 (idx5)].
        // Placement mirrors the canonical_ffd_packing test, but in indices:
        //   bin0: [1, 4] (sizes 8, 2). bin1: [0, 3, 2, 5] (sizes 4, 4, 1, 1).
        let bins = first_fit_decreasing_indices(&[4, 8, 1, 4, 2, 1], 10);
        assert_eq!(bins, vec![vec![1, 4], vec![0, 3, 2, 5]]);
    }

    #[test]
    #[should_panic(expected = "bin_packing_ffd")]
    fn item_larger_than_capacity_panics() {
        let _ = first_fit_decreasing(&[3, 11, 4], 10);
    }

    #[test]
    #[should_panic(expected = "bin_packing_ffd")]
    fn zero_capacity_with_positive_item_panics() {
        let _ = first_fit_decreasing(&[1], 0);
    }

    #[test]
    fn zero_capacity_empty_items_is_ok() {
        let bins = first_fit_decreasing(&[], 0);
        assert!(bins.is_empty());
    }

    /// Conservative upper bound on the number of bins FFD may emit. The
    /// classical theoretical guarantee is `11/9·OPT + 1`; since `OPT` itself
    /// is at most `n` (one bin per item), `2n + 1` is a generous slack.
    fn ffd_bin_upper_bound(n: usize, sum: u64, capacity: u64) -> usize {
        let by_capacity = sum.div_ceil(capacity.max(1)) as usize;
        // 11/9 ≈ 1.223; allow a small additive constant on top of ceil(sum/cap)
        // — FFD's worst case is well within `2 * by_capacity + 2`.
        2 * by_capacity + 2 + n / 4
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn property_each_item_placed_once_and_within_capacity(raw: Vec<u8>) -> bool {
        // Cap n at 16 to keep the test fast; map u8 sizes into a sensible range.
        let capacity: u64 = 16;
        let items: Vec<u64> = raw
            .into_iter()
            .take(16)
            // Use modulo to guarantee `size <= capacity` (no panic path).
            .map(|x| u64::from(x) % (capacity + 1))
            .collect();

        let bins = first_fit_decreasing_indices(&items, capacity);

        // Every item appears exactly once.
        let mut seen = vec![false; items.len()];
        for bin in &bins {
            for &i in bin {
                if seen[i] {
                    return false;
                }
                seen[i] = true;
            }
        }
        if seen.iter().any(|&s| !s) {
            return false;
        }

        // Every bin fits inside `capacity`.
        for bin in &bins {
            let total: u64 = bin.iter().map(|&i| items[i]).sum();
            if total > capacity {
                return false;
            }
        }

        // Bin count is bounded above by the conservative slack.
        let sum: u64 = items.iter().sum();
        bins.len() <= ffd_bin_upper_bound(items.len(), sum, capacity)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn property_sizes_match_indices(raw: Vec<u8>) -> bool {
        let capacity: u64 = 16;
        let items: Vec<u64> = raw
            .into_iter()
            .take(16)
            .map(|x| u64::from(x) % (capacity + 1))
            .collect();

        let by_size = first_fit_decreasing(&items, capacity);
        let by_index = first_fit_decreasing_indices(&items, capacity);

        if by_size.len() != by_index.len() {
            return false;
        }
        for (sbin, ibin) in by_size.iter().zip(by_index.iter()) {
            if sbin.len() != ibin.len() {
                return false;
            }
            for (s, &i) in sbin.iter().zip(ibin.iter()) {
                if *s != items[i] {
                    return false;
                }
            }
        }
        true
    }
}
