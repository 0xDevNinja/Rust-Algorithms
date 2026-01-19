//! Fibonacci search on a sorted slice. O(log n) comparisons; each split is
//! computed using only addition and subtraction (no division), historically
//! useful when division is expensive (early hardware, magnetic-tape access).

/// Returns the index of an element equal to `target` in the sorted `slice`,
/// or `None`. The slice MUST be sorted in non-decreasing order.
pub fn fibonacci_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let n = slice.len();
    if n == 0 {
        return None;
    }
    // Build the smallest Fibonacci number >= n.
    let (mut fk_minus_2, mut fk_minus_1) = (0_usize, 1_usize);
    let mut fk = fk_minus_1 + fk_minus_2;
    while fk < n {
        fk_minus_2 = fk_minus_1;
        fk_minus_1 = fk;
        fk = fk_minus_1 + fk_minus_2;
    }
    let mut offset: isize = -1;
    while fk > 1 {
        let i = (offset + fk_minus_2 as isize).min(n as isize - 1) as usize;
        match slice[i].cmp(target) {
            std::cmp::Ordering::Equal => return Some(i),
            std::cmp::Ordering::Less => {
                fk = fk_minus_1;
                fk_minus_1 = fk_minus_2;
                fk_minus_2 = fk - fk_minus_1;
                offset = i as isize;
            }
            std::cmp::Ordering::Greater => {
                fk = fk_minus_2;
                fk_minus_1 -= fk_minus_2;
                fk_minus_2 = fk - fk_minus_1;
            }
        }
    }
    if fk_minus_1 == 1 && ((offset + 1) as usize) < n && &slice[(offset + 1) as usize] == target {
        return Some((offset + 1) as usize);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::fibonacci_search;

    #[test]
    fn empty() {
        let v: [i32; 0] = [];
        assert_eq!(fibonacci_search(&v, &0), None);
    }

    #[test]
    fn single_present() {
        let v = [42];
        assert_eq!(fibonacci_search(&v, &42), Some(0));
    }

    #[test]
    fn single_absent() {
        let v = [42];
        assert_eq!(fibonacci_search(&v, &7), None);
    }

    #[test]
    fn finds_each_element() {
        let v: Vec<i32> = (1..=50).collect();
        for (i, x) in v.iter().enumerate() {
            assert_eq!(fibonacci_search(&v, x), Some(i), "value {x}");
        }
    }

    #[test]
    fn missing_in_middle() {
        let v = [1, 3, 5, 7, 9, 11, 13];
        assert_eq!(fibonacci_search(&v, &4), None);
        assert_eq!(fibonacci_search(&v, &14), None);
    }

    #[test]
    fn boundary_values() {
        let v: Vec<i32> = (0..1024).collect();
        assert_eq!(fibonacci_search(&v, &0), Some(0));
        assert_eq!(fibonacci_search(&v, &1023), Some(1023));
    }

    #[test]
    fn duplicates_returns_some_match() {
        let v = vec![1, 2, 2, 2, 3];
        let result = fibonacci_search(&v, &2);
        assert!(result.is_some());
        assert_eq!(v[result.unwrap()], 2);
    }
}
