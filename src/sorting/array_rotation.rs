//! In-place left rotation of a slice using Bentley's classical algorithms
//! from *Programming Pearls*.
//!
//! - Three-reversal: reverse(0, k), reverse(k, n), reverse(0, n).
//!   O(n) time, O(1) extra space, no `Clone` bound required.
//! - Juggling: gcd(n, k) cycles, each shifting elements by `k` modulo `n`.
//!   O(n) time, O(1) extra space, requires `Clone` for the temp slot.

/// Greatest common divisor (Euclid). Used by the juggling rotation.
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Rotates `arr` left by `k` positions using the three-reversal trick.
///
/// `k` is reduced modulo `arr.len()`. Empty and single-element slices are
/// no-ops. Runs in O(n) time with O(1) extra space and does not require
/// `T: Clone`.
pub fn rotate_three_reversal<T>(arr: &mut [T], k: usize) {
    let n = arr.len();
    if n < 2 {
        return;
    }
    let k = k % n;
    if k == 0 {
        return;
    }
    arr[..k].reverse();
    arr[k..].reverse();
    arr.reverse();
}

/// Rotates `arr` left by `k` positions using the juggling algorithm.
///
/// Splits the slice into `gcd(n, k)` independent cycles and shifts each
/// cycle by `k` positions. `k` is reduced modulo `arr.len()`. Empty and
/// single-element slices are no-ops. Runs in O(n) time with O(1) extra
/// space; requires `T: Clone` for the rotating temporary.
pub fn rotate_juggling<T: Clone>(arr: &mut [T], k: usize) {
    let n = arr.len();
    if n < 2 {
        return;
    }
    let k = k % n;
    if k == 0 {
        return;
    }
    let cycles = gcd(n, k);
    for start in 0..cycles {
        let temp = arr[start].clone();
        let mut i = start;
        loop {
            let j = (i + k) % n;
            if j == start {
                break;
            }
            arr[i] = arr[j].clone();
            i = j;
        }
        arr[i] = temp;
    }
}

#[cfg(test)]
mod tests {
    use super::{rotate_juggling, rotate_three_reversal};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_is_noop() {
        let mut a: Vec<i32> = vec![];
        rotate_three_reversal(&mut a, 3);
        assert!(a.is_empty());
        let mut b: Vec<i32> = vec![];
        rotate_juggling(&mut b, 3);
        assert!(b.is_empty());
    }

    #[test]
    fn single_element_is_noop() {
        let mut a = vec![42];
        rotate_three_reversal(&mut a, 7);
        assert_eq!(a, vec![42]);
        let mut b = vec![42];
        rotate_juggling(&mut b, 7);
        assert_eq!(b, vec![42]);
    }

    #[test]
    fn k_zero_is_noop() {
        let mut a = vec![1, 2, 3, 4];
        rotate_three_reversal(&mut a, 0);
        assert_eq!(a, vec![1, 2, 3, 4]);
        let mut b = vec![1, 2, 3, 4];
        rotate_juggling(&mut b, 0);
        assert_eq!(b, vec![1, 2, 3, 4]);
    }

    #[test]
    fn k_equals_len_is_identity() {
        let mut a = vec![1, 2, 3, 4, 5];
        rotate_three_reversal(&mut a, 5);
        assert_eq!(a, vec![1, 2, 3, 4, 5]);
        let mut b = vec![1, 2, 3, 4, 5];
        rotate_juggling(&mut b, 5);
        assert_eq!(b, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn k_larger_than_len_uses_modulo() {
        // n=5, k=12 -> effective k=2.
        let mut a = vec![1, 2, 3, 4, 5];
        rotate_three_reversal(&mut a, 12);
        assert_eq!(a, vec![3, 4, 5, 1, 2]);
        let mut b = vec![1, 2, 3, 4, 5];
        rotate_juggling(&mut b, 12);
        assert_eq!(b, vec![3, 4, 5, 1, 2]);
    }

    #[test]
    fn canonical_rotate_by_three() {
        let mut a = vec![1, 2, 3, 4, 5, 6, 7];
        rotate_three_reversal(&mut a, 3);
        assert_eq!(a, vec![4, 5, 6, 7, 1, 2, 3]);
        let mut b = vec![1, 2, 3, 4, 5, 6, 7];
        rotate_juggling(&mut b, 3);
        assert_eq!(b, vec![4, 5, 6, 7, 1, 2, 3]);
    }

    #[test]
    fn juggling_multiple_cycles_k2_n6() {
        // gcd(6, 2) = 2 — exercises the multi-cycle path.
        let mut v = vec![1, 2, 3, 4, 5, 6];
        rotate_juggling(&mut v, 2);
        assert_eq!(v, vec![3, 4, 5, 6, 1, 2]);
    }

    #[test]
    fn juggling_multiple_cycles_k4_n6() {
        // gcd(6, 4) = 2.
        let mut v = vec![1, 2, 3, 4, 5, 6];
        rotate_juggling(&mut v, 4);
        assert_eq!(v, vec![5, 6, 1, 2, 3, 4]);
    }

    #[test]
    fn juggling_single_cycle_when_coprime() {
        // gcd(7, 3) = 1 — one cycle covers everything.
        let mut v = vec![1, 2, 3, 4, 5, 6, 7];
        rotate_juggling(&mut v, 3);
        assert_eq!(v, vec![4, 5, 6, 7, 1, 2, 3]);
    }

    #[test]
    fn three_reversal_no_clone_required() {
        // Box<i32> is not Copy; this confirms the reversal variant
        // works on non-Clone-friendly element types.
        let mut v: Vec<Box<i32>> = (1..=5).map(Box::new).collect();
        rotate_three_reversal(&mut v, 2);
        let got: Vec<i32> = v.into_iter().map(|b| *b).collect();
        assert_eq!(got, vec![3, 4, 5, 1, 2]);
    }

    #[quickcheck]
    fn both_methods_agree(input: Vec<i32>, k: usize) -> bool {
        let mut a = input.clone();
        let mut b = input;
        rotate_three_reversal(&mut a, k);
        rotate_juggling(&mut b, k);
        a == b
    }

    #[quickcheck]
    fn rotation_is_a_permutation(input: Vec<i32>, k: usize) -> bool {
        let mut a = input.clone();
        rotate_three_reversal(&mut a, k);
        let mut sorted_in = input;
        sorted_in.sort();
        let mut sorted_out = a;
        sorted_out.sort();
        sorted_in == sorted_out
    }
}
