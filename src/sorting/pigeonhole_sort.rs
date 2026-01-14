//! Pigeonhole sort. O(n + range) time and space, where `range = max - min + 1`.
//! Stable. Practical only when the key range is comparable to the element count.

/// Sorts `slice` of `i64` values in non-decreasing order using pigeonhole sort.
///
/// Allocates one bucket per distinct possible key in `[min, max]`. Will panic
/// if the range overflows `usize`.
pub fn pigeonhole_sort(slice: &mut [i64]) {
    if slice.len() < 2 {
        return;
    }
    let min = *slice.iter().min().unwrap();
    let max = *slice.iter().max().unwrap();
    let range = (max - min + 1) as usize;
    let mut holes = vec![0_usize; range];
    for &x in slice.iter() {
        holes[(x - min) as usize] += 1;
    }
    let mut idx = 0;
    for (offset, &count) in holes.iter().enumerate() {
        for _ in 0..count {
            slice[idx] = min + offset as i64;
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pigeonhole_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<i64> = vec![];
        pigeonhole_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn single() {
        let mut v = vec![17_i64];
        pigeonhole_sort(&mut v);
        assert_eq!(v, vec![17]);
    }

    #[test]
    fn negatives_and_positives() {
        let mut v = vec![3_i64, -1, 4, -1, 5, -9, 2, 6];
        pigeonhole_sort(&mut v);
        assert_eq!(v, vec![-9, -1, -1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn dense_range() {
        let mut v: Vec<i64> = (0..50).rev().collect();
        pigeonhole_sort(&mut v);
        assert_eq!(v, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn all_equal() {
        let mut v = vec![5_i64; 30];
        pigeonhole_sort(&mut v);
        assert_eq!(v, vec![5_i64; 30]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i16>) -> bool {
        // Use i16 to bound range and keep memory manageable.
        let mut as_i64: Vec<i64> = input.iter().map(|&x| x as i64).collect();
        pigeonhole_sort(&mut as_i64);
        input.sort();
        as_i64
            .iter()
            .zip(input.iter())
            .all(|(a, b)| *a == *b as i64)
    }
}
