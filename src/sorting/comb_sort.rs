//! Comb sort. In-place, not stable, O(n²) worst, ~O(n log n) typical.
//!
//! Bubble-sort variant with a shrinking gap (Knuth shrink factor 1.3).
//! Eliminates "turtles" — small values near the end that slow down bubble
//! sort — by comparing distant elements first.

const SHRINK: f64 = 1.3;

/// Sorts `slice` in non-decreasing order using comb sort.
pub fn comb_sort<T: Ord>(slice: &mut [T]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    let mut gap = n;
    let mut sorted = false;
    while !sorted {
        gap = ((gap as f64) / SHRINK) as usize;
        if gap <= 1 {
            gap = 1;
            sorted = true;
        }
        for i in 0..(n - gap) {
            if slice[i] > slice[i + gap] {
                slice.swap(i, i + gap);
                sorted = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::comb_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        comb_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn single() {
        let mut v = vec![42];
        comb_sort(&mut v);
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn random() {
        let mut v = vec![8, 4, 1, 56, 3, -44, 23, -6, 28, 0];
        comb_sort(&mut v);
        assert_eq!(v, vec![-44, -6, 0, 1, 3, 4, 8, 23, 28, 56]);
    }

    #[test]
    fn reverse_eliminates_turtles() {
        let mut v: Vec<i32> = (0..50).rev().collect();
        comb_sort(&mut v);
        assert_eq!(v, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn all_equal() {
        let mut v = vec![3; 20];
        comb_sort(&mut v);
        assert_eq!(v, vec![3; 20]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        comb_sort(&mut input);
        input == expected
    }
}
