//! Permutations and k-combinations of a slice via backtracking.
//!
//! # Algorithms
//! - **Permutations**: Heap-style recursive swap-in-place. A working buffer
//!   is mutated by swapping the element at index `i` with each `j >= i`,
//!   recursing on `i + 1`, then swapping back. At `i == n`, the current
//!   buffer state is cloned into the output.
//! - **Combinations**: Choose-or-skip recursion. At each index `i`, the
//!   algorithm either includes `items[i]` in the running selection and
//!   recurses on `i + 1`, or skips it and recurses on `i + 1`. When the
//!   selection has length `k`, it is cloned into the output.
//!
//! # Complexity
//! - `permutations`: **Time** `O(n · n!)` — there are `n!` outputs, each of
//!   length `n` to materialise. **Space** `O(n)` auxiliary (recursion +
//!   working buffer), excluding the `O(n · n!)` output itself.
//! - `combinations`: **Time** `O(k · C(n, k))` — there are `C(n, k)` outputs,
//!   each of length `k` to materialise. **Space** `O(k)` auxiliary
//!   (recursion + selection buffer), excluding the `O(k · C(n, k))` output.
//!
//! # Stability and duplicates
//! Neither generator de-duplicates. If the input contains equal elements,
//! `permutations` still returns exactly `n!` outputs (with indistinguishable
//! repeats), and `combinations` still returns exactly `C(n, k)` outputs.
//! Combinations preserve the original index order of the input within each
//! output vector.

/// Returns all `n!` permutations of `items`, where `n = items.len()`.
///
/// Empty input returns `vec![vec![]]` (the single empty permutation).
/// Equal elements are not de-duplicated: an input of length `n` always
/// yields exactly `n!` output vectors.
pub fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let mut buf: Vec<T> = items.to_vec();
    permute(&mut buf, 0, &mut out);
    out
}

/// Recursive helper for `permutations`. Swaps element `start` with each
/// element at index `j >= start`, recurses, then swaps back to restore the
/// buffer for the next branch.
fn permute<T: Clone>(buf: &mut [T], start: usize, out: &mut Vec<Vec<T>>) {
    if start == buf.len() {
        out.push(buf.to_vec());
        return;
    }
    for j in start..buf.len() {
        buf.swap(start, j);
        permute(buf, start + 1, out);
        buf.swap(start, j);
    }
}

/// Returns all `C(n, k)` `k`-combinations of `items`, where `n = items.len()`.
///
/// Each output preserves the original index order of `items`. Returns
/// `vec![vec![]]` when `k == 0` and `vec![]` when `k > n`. Equal elements
/// are not de-duplicated.
pub fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    if k > items.len() {
        return out;
    }
    let mut buf: Vec<T> = Vec::with_capacity(k);
    combine(items, 0, k, &mut buf, &mut out);
    out
}

/// Recursive helper for `combinations`. At index `i`, either include
/// `items[i]` and recurse on `i + 1`, or skip and recurse on `i + 1`.
fn combine<T: Clone>(items: &[T], i: usize, k: usize, buf: &mut Vec<T>, out: &mut Vec<Vec<T>>) {
    if buf.len() == k {
        out.push(buf.clone());
        return;
    }
    // Not enough remaining elements to reach length k.
    if items.len() - i < k - buf.len() {
        return;
    }
    // Choose items[i].
    buf.push(items[i].clone());
    combine(items, i + 1, k, buf, out);
    buf.pop();
    // Skip items[i].
    combine(items, i + 1, k, buf, out);
}

#[cfg(test)]
mod tests {
    use super::{combinations, permutations};
    use quickcheck_macros::quickcheck;
    use std::collections::HashSet;

    fn factorial(n: usize) -> usize {
        (1..=n).product::<usize>().max(1)
    }

    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut result: usize = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    // ---------- permutations ----------

    #[test]
    fn permutations_empty() {
        let out: Vec<Vec<i32>> = permutations::<i32>(&[]);
        assert_eq!(out, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn permutations_single_element() {
        let out = permutations(&[42]);
        assert_eq!(out, vec![vec![42]]);
    }

    #[test]
    fn permutations_three_elements_returns_six_distinct() {
        let input = [1, 2, 3];
        let out = permutations(&input);
        assert_eq!(out.len(), 6);
        // Every output is a permutation of the input (same multiset).
        let mut sorted_input = input.to_vec();
        sorted_input.sort_unstable();
        for perm in &out {
            assert_eq!(perm.len(), input.len());
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, sorted_input);
        }
        // All 6 outputs are pairwise distinct (input has no duplicates).
        let unique: HashSet<Vec<i32>> = out.into_iter().collect();
        assert_eq!(unique.len(), 6);
    }

    // Documented behaviour: no de-duplication. Input with duplicates still
    // produces exactly n! outputs, including indistinguishable repeats.
    #[test]
    fn permutations_with_duplicates_returns_n_factorial() {
        let input = [1, 1, 2];
        let out = permutations(&input);
        assert_eq!(out.len(), factorial(3));
        // Distinct *values* are fewer than n! when duplicates exist.
        let unique: HashSet<Vec<i32>> = out.into_iter().collect();
        assert_eq!(unique.len(), 3); // {1,1,2}, {1,2,1}, {2,1,1}
    }

    // ---------- combinations ----------

    #[test]
    fn combinations_k_zero() {
        let out = combinations(&[1, 2, 3], 0);
        assert_eq!(out, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn combinations_k_zero_empty_input() {
        let out: Vec<Vec<i32>> = combinations::<i32>(&[], 0);
        assert_eq!(out, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn combinations_k_greater_than_n() {
        let out: Vec<Vec<i32>> = combinations(&[1, 2], 5);
        assert!(out.is_empty());
    }

    #[test]
    fn combinations_k_equals_n() {
        let out = combinations(&[1, 2, 3], 3);
        assert_eq!(out, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn combinations_5_choose_2() {
        let input = [1, 2, 3, 4, 5];
        let out = combinations(&input, 2);
        assert_eq!(out.len(), 10);
        // Each output is unique and a 2-subset preserving original order.
        let unique: HashSet<Vec<i32>> = out.iter().cloned().collect();
        assert_eq!(unique.len(), 10);
        for combo in &out {
            assert_eq!(combo.len(), 2);
            assert!(combo[0] < combo[1], "expected sorted order: {combo:?}");
            for value in combo {
                assert!(input.contains(value));
            }
        }
    }

    // ---------- quickcheck property tests ----------

    // For any input slice of length n ≤ 6, permutations must:
    //   1. produce exactly n! outputs;
    //   2. each output is a permutation of the input (same multiset).
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_permutations_count_and_multiset(items: Vec<u8>) -> bool {
        let items: Vec<u8> = items.into_iter().take(6).collect();
        let n = items.len();
        let out = permutations(&items);
        if out.len() != factorial(n) {
            return false;
        }
        let mut input_sorted = items;
        input_sorted.sort_unstable();
        out.into_iter().all(|p| {
            let mut p_sorted = p;
            p_sorted.sort_unstable();
            p_sorted == input_sorted
        })
    }

    // For any input slice of length n ≤ 6 and every k in 0..=n, combinations
    // must:
    //   1. produce exactly C(n, k) outputs;
    //   2. each output has length k;
    //   3. each output's elements appear in the same relative order as in
    //      the input (i.e. it is a sub-sequence by index).
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_combinations_count_and_subsequence(items: Vec<u8>) -> bool {
        let items: Vec<u8> = items.into_iter().take(6).collect();
        let n = items.len();
        for k in 0..=n {
            let out = combinations(&items, k);
            if out.len() != binomial(n, k) {
                return false;
            }
            for combo in &out {
                if combo.len() != k {
                    return false;
                }
                // Verify combo is a sub-sequence of items by index walk.
                let mut idx = 0;
                for value in combo {
                    while idx < items.len() && items[idx] != *value {
                        idx += 1;
                    }
                    if idx == items.len() {
                        return false;
                    }
                    idx += 1;
                }
            }
        }
        true
    }
}
