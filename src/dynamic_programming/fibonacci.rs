//! Fibonacci number computation: bottom-up DP with O(1) extra space.

/// Returns the `n`-th Fibonacci number. Saturates at `u128::MAX` rather than
/// panicking on overflow.
pub fn fibonacci(n: u32) -> u128 {
    if n == 0 {
        return 0;
    }
    let (mut a, mut b): (u128, u128) = (0, 1);
    for _ in 1..n {
        let next = a.saturating_add(b);
        a = b;
        b = next;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::fibonacci;

    #[test]
    fn first_terms() {
        let expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        for (n, &v) in expected.iter().enumerate() {
            assert_eq!(fibonacci(n as u32), v as u128);
        }
    }

    #[test]
    fn larger() {
        assert_eq!(fibonacci(50), 12_586_269_025);
    }

    #[test]
    fn saturates_on_overflow() {
        assert_eq!(fibonacci(u32::MAX), u128::MAX);
    }
}
