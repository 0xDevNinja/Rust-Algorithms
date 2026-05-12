//! Inversion count via merge sort.
//!
//! An inversion is a pair of indices `(i, j)` with `i < j` and `arr[i] > arr[j]`.
//! Counting them naively is `O(n^2)`; piggy-backing the count onto a merge sort
//! brings it down to `O(n log n)` time with `O(n)` auxiliary space.
//!
//! Key insight: during the merge step, when we pick an element from the right
//! half because it is strictly smaller than the current left-half element,
//! every remaining element in the left half forms an inversion with it. So we
//! add `left.len() - left_idx` to the running counter.
//!
//! Equal elements are not counted as inversions.
//!
//! Complexity: `O(n log n)` time, `O(n)` extra space.
//! Returns `u64` so counts up to `n*(n-1)/2` fit comfortably for `n` up to a
//! few billion.

/// Counts the number of inversions in `arr` — pairs `(i, j)` with `i < j` and
/// `arr[i] > arr[j]`.
///
/// Runs a non-mutating merge sort over a cloned working buffer and tallies
/// inversions during each merge.
pub fn inversion_count<T: Ord + Clone>(arr: &[T]) -> u64 {
    if arr.len() < 2 {
        return 0;
    }
    let mut buf: Vec<T> = arr.to_vec();
    let mut scratch: Vec<T> = arr.to_vec();
    sort_count(&mut buf, &mut scratch)
}

fn sort_count<T: Ord + Clone>(slice: &mut [T], scratch: &mut [T]) -> u64 {
    let n = slice.len();
    if n < 2 {
        return 0;
    }
    let mid = n / 2;
    let (left, right) = slice.split_at_mut(mid);
    let (sl, sr) = scratch.split_at_mut(mid);
    let mut count = sort_count(left, sl);
    count += sort_count(right, sr);
    count += merge_count(left, right, scratch);
    slice.clone_from_slice(&scratch[..n]);
    count
}

fn merge_count<T: Ord + Clone>(left: &[T], right: &[T], out: &mut [T]) -> u64 {
    let (mut i, mut j, mut k) = (0, 0, 0);
    let mut inv: u64 = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            out[k] = left[i].clone();
            i += 1;
        } else {
            out[k] = right[j].clone();
            // every remaining element in `left` is greater than `right[j]`.
            inv += (left.len() - i) as u64;
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        out[k] = left[i].clone();
        i += 1;
        k += 1;
    }
    while j < right.len() {
        out[k] = right[j].clone();
        j += 1;
        k += 1;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::inversion_count;

    fn brute_force<T: Ord>(arr: &[T]) -> u64 {
        let mut count: u64 = 0;
        for i in 0..arr.len() {
            for j in (i + 1)..arr.len() {
                if arr[i] > arr[j] {
                    count += 1;
                }
            }
        }
        count
    }

    #[test]
    fn empty() {
        let a: Vec<i32> = vec![];
        assert_eq!(inversion_count(&a), 0);
    }

    #[test]
    fn single() {
        assert_eq!(inversion_count(&[42]), 0);
    }

    #[test]
    fn sorted() {
        assert_eq!(inversion_count(&[1, 2, 3, 4, 5, 6]), 0);
    }

    #[test]
    fn reverse_sorted() {
        for n in 0..=10usize {
            let v: Vec<i32> = (0..n as i32).rev().collect();
            let expected = (n as u64) * (n.saturating_sub(1) as u64) / 2;
            assert_eq!(inversion_count(&v), expected);
        }
    }

    #[test]
    fn small_known() {
        assert_eq!(inversion_count(&[2, 4, 1, 3, 5]), 3);
    }

    #[test]
    fn duplicates_only() {
        assert_eq!(inversion_count(&[1, 1, 1]), 0);
        assert_eq!(inversion_count(&[7, 7, 7, 7, 7]), 0);
    }

    #[test]
    fn duplicates_mixed() {
        // (3,1),(3,2),(3,2) -> 3 inversions
        assert_eq!(inversion_count(&[3, 1, 2, 2]), 3);
    }

    #[test]
    fn strings() {
        let v = vec!["pear", "apple", "banana"];
        assert_eq!(inversion_count(&v), 2);
    }

    #[test]
    fn matches_brute_force_small() {
        // Deterministic LCG for reproducibility — no rand dependency.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..40 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let len = ((state >> 32) as usize) % 12;
            let mut v: Vec<i32> = Vec::with_capacity(len);
            for _ in 0..len {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                v.push(((state >> 40) as i32) % 7);
            }
            assert_eq!(inversion_count(&v), brute_force(&v), "mismatch on {v:?}");
        }
    }
}
