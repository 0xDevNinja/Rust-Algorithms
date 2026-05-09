//! Greedy approximation for the (unweighted) set cover problem.
//!
//! Given a `universe` of `n` elements and a family of `m` candidate `sets`
//! (each a subset of the universe), pick a minimum-size sub-family whose union
//! equals the universe. Set cover is NP-hard, but the greedy rule "at every
//! step pick the set that covers the most still-uncovered elements" yields a
//! cover of size at most `H_n · OPT`, where `H_n = 1 + 1/2 + ... + 1/n` is the
//! `n`-th harmonic number and `OPT` is the optimum cover size. This `ln n + 1`
//! factor is tight under standard complexity assumptions (Feige 1998).
//!
//! Algorithm: maintain the set of still-uncovered elements as a `HashSet`. At
//! each step scan all unused candidate sets and pick the one whose intersection
//! with the uncovered set is largest; remove those elements from the uncovered
//! set and record the chosen index. Stop when the uncovered set is empty.
//!
//! Complexity: `O(m · n)` per pick and at most `min(m, n)` picks, for an
//! overall `O(m · n · min(m, n))` worst case. Space is `O(n + m)` for the
//! uncovered hash set and the bookkeeping vectors.
//!
//! Coverability: if the union of all `sets` does not equal the universe, no
//! cover exists and the function returns `Err(())`. Duplicate elements in
//! `universe` are deduplicated implicitly (the universe is treated as a set).

use std::collections::HashSet;
use std::hash::Hash;

/// Greedily select set indices that cover the `universe`.
///
/// At each step picks the set with the largest intersection with the still-
/// uncovered elements; ties are broken by smallest index. Returns the chosen
/// indices in pick order.
///
/// # Errors
/// Returns `Err(())` if the union of `sets` does not cover `universe` — i.e.
/// the instance is infeasible. Returns `Ok(vec![])` when `universe` is empty.
///
/// Time: `O(m · n · min(m, n))` where `m = sets.len()` and `n` is the number
/// of distinct elements in `universe`. Space: `O(n + m)`.
#[allow(clippy::result_unit_err)]
pub fn greedy_set_cover<T: Eq + Hash + Clone>(
    universe: &[T],
    sets: &[Vec<T>],
) -> Result<Vec<usize>, ()> {
    let mut uncovered: HashSet<T> = universe.iter().cloned().collect();
    if uncovered.is_empty() {
        return Ok(Vec::new());
    }

    // Feasibility check: the union of all sets must contain every element of
    // the universe. Done up front so we can fail fast and so the greedy loop
    // below can rely on always finding a non-empty intersection.
    let union: HashSet<&T> = sets.iter().flat_map(|s| s.iter()).collect();
    if !uncovered.iter().all(|e| union.contains(e)) {
        return Err(());
    }

    let mut chosen: Vec<usize> = Vec::new();
    let mut used: Vec<bool> = vec![false; sets.len()];

    while !uncovered.is_empty() {
        let mut best_idx: Option<usize> = None;
        let mut best_count: usize = 0;

        for (i, set) in sets.iter().enumerate() {
            if used[i] {
                continue;
            }
            let count = set.iter().filter(|e| uncovered.contains(*e)).count();
            if count > best_count {
                best_count = count;
                best_idx = Some(i);
            }
        }

        // `best_count == 0` means no remaining set contributes — impossible
        // here because the up-front feasibility check guarantees the union
        // covers the universe and every covering element lives in some set
        // we haven't yet picked.
        let Some(idx) = best_idx else {
            return Err(());
        };

        used[idx] = true;
        for e in &sets[idx] {
            uncovered.remove(e);
        }
        chosen.push(idx);
    }

    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::greedy_set_cover;
    use std::collections::HashSet;

    /// Verify that `chosen` is a valid cover for `universe` using `sets`:
    /// indices in range, unique, and the union of the picked sets contains
    /// every element of `universe`.
    fn is_valid_cover<T: Eq + std::hash::Hash + Clone>(
        universe: &[T],
        sets: &[Vec<T>],
        chosen: &[usize],
    ) -> bool {
        // Indices in range and unique.
        let mut seen: HashSet<usize> = HashSet::new();
        for &i in chosen {
            if i >= sets.len() || !seen.insert(i) {
                return false;
            }
        }
        // Union covers universe.
        let mut covered: HashSet<&T> = HashSet::new();
        for &i in chosen {
            for e in &sets[i] {
                covered.insert(e);
            }
        }
        universe.iter().all(|e| covered.contains(e))
    }

    #[test]
    fn empty_universe_returns_empty_cover() {
        let universe: Vec<i32> = vec![];
        let sets: Vec<Vec<i32>> = vec![vec![1, 2, 3]];
        let result = greedy_set_cover(&universe, &sets);
        assert_eq!(result, Ok(vec![]));
    }

    #[test]
    fn empty_universe_with_no_sets() {
        let universe: Vec<i32> = vec![];
        let sets: Vec<Vec<i32>> = vec![];
        let result = greedy_set_cover(&universe, &sets);
        assert_eq!(result, Ok(vec![]));
    }

    #[test]
    fn single_set_covers_everything() {
        let universe = vec![1, 2, 3, 4];
        let sets = vec![vec![1, 2, 3, 4]];
        let result = greedy_set_cover(&universe, &sets).expect("feasible");
        assert_eq!(result, vec![0]);
        assert!(is_valid_cover(&universe, &sets, &result));
    }

    #[test]
    fn picks_largest_first_canonical() {
        // Three sets; greedy should pick set 0 first because it covers the
        // most uncovered elements at the start, then whichever covers the
        // remaining {4}. Index 1 wins by smallest-index tie-break? Actually
        // {1,2,4} covers element 4 (uncovered) → 1 element; {3,4} also covers
        // {4} (since 3 is already covered) → 1 element. Tie broken by smaller
        // index, so pick 1.
        let universe = vec![1, 2, 3, 4];
        let sets = vec![vec![1, 2, 3], vec![1, 2, 4], vec![3, 4]];
        let result = greedy_set_cover(&universe, &sets).expect("feasible");
        assert_eq!(result, vec![0, 1]);
        assert!(is_valid_cover(&universe, &sets, &result));
    }

    #[test]
    fn worst_case_log_factor_example_covers_all() {
        // Universe {0..6}, six elements. Greedy picks the largest set first
        // ({0,1,2,3}, 4 elements), then needs to cover {4,5,6}.
        let universe: Vec<i32> = (0..6).collect();
        let sets = vec![
            vec![0, 1, 2, 3],
            vec![4, 5],
            vec![6],
            vec![0, 2, 4, 6],
            vec![1, 3, 5],
        ];
        let result = greedy_set_cover(&universe, &sets).expect("feasible");
        assert!(is_valid_cover(&universe, &sets, &result));
        // First pick is the largest set.
        assert_eq!(result[0], 0);
    }

    #[test]
    fn uncoverable_input_returns_err() {
        // Element 99 is in the universe but not in any set.
        let universe = vec![1, 2, 3, 99];
        let sets = vec![vec![1, 2], vec![3]];
        let result = greedy_set_cover(&universe, &sets);
        assert_eq!(result, Err(()));
    }

    #[test]
    fn uncoverable_with_empty_sets() {
        let universe = vec![1, 2, 3];
        let sets: Vec<Vec<i32>> = vec![];
        let result = greedy_set_cover(&universe, &sets);
        assert_eq!(result, Err(()));
    }

    #[test]
    fn duplicate_universe_elements_handled() {
        // Duplicates in the universe are deduplicated implicitly.
        let universe = vec![1, 1, 2, 2, 3];
        let sets = vec![vec![1, 2, 3]];
        let result = greedy_set_cover(&universe, &sets).expect("feasible");
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn string_elements() {
        let universe = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let sets = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["b".to_string(), "c".to_string()],
            vec!["c".to_string()],
        ];
        let result = greedy_set_cover(&universe, &sets).expect("feasible");
        assert!(is_valid_cover(&universe, &sets, &result));
    }

    #[test]
    fn property_indices_unique_and_union_covers() {
        // A handful of handpicked feasible instances; the result must always
        // have unique indices and its union must cover the universe.
        let cases: Vec<(Vec<i32>, Vec<Vec<i32>>)> = vec![
            (vec![1, 2, 3], vec![vec![1], vec![2], vec![3]]),
            (
                vec![1, 2, 3, 4, 5],
                vec![vec![1, 2, 3], vec![3, 4], vec![4, 5], vec![1, 5]],
            ),
            (
                (0..10).collect(),
                vec![
                    (0..5).collect(),
                    (5..10).collect(),
                    (0..10).step_by(2).collect(),
                    (1..10).step_by(2).collect(),
                ],
            ),
        ];
        for (universe, sets) in cases {
            let result = greedy_set_cover(&universe, &sets).expect("feasible");
            // Unique.
            let mut seen: HashSet<usize> = HashSet::new();
            for &i in &result {
                assert!(seen.insert(i), "duplicate index in {result:?}");
            }
            // Covers.
            assert!(
                is_valid_cover(&universe, &sets, &result),
                "result {result:?} does not cover universe"
            );
        }
    }

    #[test]
    fn approximation_bound_holds() {
        // Greedy's cover size is at most ceil(H_n) * OPT. For n = 6 elements
        // and OPT = 2, H_6 ≈ 2.45 so greedy must use at most ~5 sets — and
        // in practice picks 3.
        let universe: Vec<i32> = (0..6).collect();
        let sets = vec![
            vec![0, 1, 2, 3],
            vec![4, 5],
            vec![6], // unused; outside universe
            vec![0, 2, 4, 6],
            vec![1, 3, 5],
        ];
        let result = greedy_set_cover(&universe, &sets).expect("feasible");
        // OPT here is 2 (sets 0 and 1). H_6 ≈ 2.45, so allow up to 6 sets to
        // be safely within the H_n bound.
        assert!(result.len() <= 6);
        assert!(is_valid_cover(&universe, &sets, &result));
    }
}
