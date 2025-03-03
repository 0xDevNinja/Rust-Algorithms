//! Counting sort for non-negative integers. Stable, O(n + k) time, O(n + k) space
//! where `k` is the maximum value.

/// Sorts `slice` in non-decreasing order using counting sort.
///
/// Returns immediately on empty input.
pub fn counting_sort(slice: &mut [u32]) {
    if slice.is_empty() {
        return;
    }
    let max = *slice.iter().max().unwrap();
    let mut counts = vec![0_usize; (max as usize) + 1];
    for &x in slice.iter() {
        counts[x as usize] += 1;
    }
    let mut idx = 0;
    for (value, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            slice[idx] = value as u32;
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::counting_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<u32> = vec![];
        counting_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn small() {
        let mut v = vec![4, 2, 2, 8, 3, 3, 1];
        counting_sort(&mut v);
        assert_eq!(v, vec![1, 2, 2, 3, 3, 4, 8]);
    }

    #[test]
    fn all_zero() {
        let mut v = vec![0, 0, 0, 0];
        counting_sort(&mut v);
        assert_eq!(v, vec![0, 0, 0, 0]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<u32>) -> bool {
        // Bound input to keep counts vector sane.
        input.iter_mut().for_each(|x| *x %= 5_000);
        let mut expected = input.clone();
        expected.sort();
        counting_sort(&mut input);
        input == expected
    }
}
