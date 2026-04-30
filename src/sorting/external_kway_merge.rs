//! External k-way merge sort.
//!
//! Merges `k` pre-sorted runs into a single sorted sequence using a binary
//! min-heap, achieving O(N log k) time where N is the total number of elements
//! across all runs. This is the merge phase of an external sort: when the input
//! is too large to fit in memory, it is split into sorted runs (each small
//! enough to sort in RAM) that are written to disk, then streamed back through
//! a k-way merge that only keeps `k` items resident at once.
//!
//! In a real external sort each run would be backed by a buffered file reader;
//! here we expose the in-memory shape `Vec<Vec<T>>` so the algorithm can be
//! tested and reused without an I/O layer. Swapping the input for an iterator
//! of file-backed readers is the only change needed to make it external.
//!
//! Stability is not preserved across runs: ties are broken by whichever run
//! the heap pops first.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Merges `runs`, each assumed to be sorted in non-decreasing order, into a
/// single sorted `Vec<T>`.
///
/// Uses a min-heap of `(value, run_index)` so each pop costs O(log k) and the
/// total work is O(N log k) for `N` elements across `k` runs.
///
/// Empty input returns an empty `Vec`. A single run is cloned out unchanged.
/// Empty runs in the input are skipped.
///
/// # Examples
///
/// ```
/// use rust_algorithms::sorting::external_kway_merge::k_way_merge;
///
/// let runs = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];
/// assert_eq!(k_way_merge(runs), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
pub fn k_way_merge<T: Ord + Clone>(runs: Vec<Vec<T>>) -> Vec<T> {
    if runs.is_empty() {
        return Vec::new();
    }
    if runs.len() == 1 {
        return runs.into_iter().next().unwrap_or_default();
    }

    let total: usize = runs.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);

    // Cursor into each run.
    let mut cursors = vec![0usize; runs.len()];
    let mut heap: BinaryHeap<Reverse<(T, usize)>> = BinaryHeap::with_capacity(runs.len());

    // Seed the heap with the first element of every non-empty run.
    for (idx, run) in runs.iter().enumerate() {
        if let Some(first) = run.first() {
            heap.push(Reverse((first.clone(), idx)));
            cursors[idx] = 1;
        }
    }

    while let Some(Reverse((value, idx))) = heap.pop() {
        out.push(value);
        let cursor = cursors[idx];
        if cursor < runs[idx].len() {
            heap.push(Reverse((runs[idx][cursor].clone(), idx)));
            cursors[idx] = cursor + 1;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::k_way_merge;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_input() {
        let runs: Vec<Vec<i32>> = vec![];
        assert!(k_way_merge(runs).is_empty());
    }

    #[test]
    fn all_empty_runs() {
        let runs: Vec<Vec<i32>> = vec![vec![], vec![], vec![]];
        assert!(k_way_merge(runs).is_empty());
    }

    #[test]
    fn single_run() {
        let runs = vec![vec![1, 2, 3, 4, 5]];
        assert_eq!(k_way_merge(runs), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn single_empty_run() {
        let runs: Vec<Vec<i32>> = vec![vec![]];
        assert!(k_way_merge(runs).is_empty());
    }

    #[test]
    fn two_runs() {
        let runs = vec![vec![1, 3, 5, 7], vec![2, 4, 6, 8]];
        assert_eq!(k_way_merge(runs), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn many_runs_k_ten() {
        let runs: Vec<Vec<i32>> = (0..10).map(|i| vec![i, i + 10, i + 20]).collect();
        let mut expected: Vec<i32> = (0..30).collect();
        expected.sort();
        assert_eq!(k_way_merge(runs), expected);
    }

    #[test]
    fn different_sized_runs() {
        let runs = vec![
            vec![1],
            vec![2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![],
            vec![0, 11],
        ];
        assert_eq!(
            k_way_merge(runs),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        );
    }

    #[test]
    fn with_duplicates() {
        let runs = vec![vec![1, 1, 2, 3], vec![1, 2, 2, 4], vec![3, 3, 5]];
        assert_eq!(k_way_merge(runs), vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5]);
    }

    #[test]
    fn negative_and_mixed() {
        let runs = vec![vec![-5, -1, 0], vec![-3, 2, 7], vec![-10, 4]];
        assert_eq!(k_way_merge(runs), vec![-10, -5, -3, -1, 0, 2, 4, 7]);
    }

    #[quickcheck]
    fn matches_sorted_flatten(mut runs: Vec<Vec<i32>>) -> bool {
        // Bound the input: k <= 10 runs, total <= 100 elements.
        runs.truncate(10);
        let mut total = 0;
        for run in &mut runs {
            let remaining = 100usize.saturating_sub(total);
            if run.len() > remaining {
                run.truncate(remaining);
            }
            total += run.len();
            run.sort();
        }

        let mut expected: Vec<i32> = runs.iter().flatten().copied().collect();
        expected.sort();

        k_way_merge(runs) == expected
    }
}
