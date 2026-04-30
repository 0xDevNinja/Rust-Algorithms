//! Square-matrix fast exponentiation by repeated squaring, modulo a `u64`.
//!
//! Given a `k × k` matrix `M`, computes `M^n mod p` in O(k³ log n) time. The
//! main use case is closing linear recurrences: any sequence defined by a
//! constant-coefficient recurrence of order `k` can be advanced `n` steps by
//! raising its companion matrix to the `n`-th power, giving an O(k³ log n)
//! evaluation of `a_n` instead of the naive O(n). Classic example: Fibonacci
//! via `[[1,1],[1,0]]^n`. All intermediate products use `u128` to avoid
//! overflow when multiplying two values already reduced modulo a `u64`.

/// Modular multiplication of two compatible matrices.
///
/// Computes `(A · B) mod modulus` where `A` is `r × k` and `B` is `k × c`.
/// Intermediate products are accumulated in `u128` so any inputs already
/// reduced modulo a `u64` are safe regardless of `k`.
///
/// # Panics
/// Panics if the inner dimensions do not match (i.e. `A`'s column count
/// differs from `B`'s row count) or if either matrix has zero rows.
#[must_use]
pub fn mat_mul_mod(a: &[Vec<u64>], b: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
    assert!(!a.is_empty() && !b.is_empty(), "matrices must be non-empty");
    let r = a.len();
    let k = a[0].len();
    let c = b[0].len();
    assert!(
        a.iter().all(|row| row.len() == k),
        "left matrix has ragged rows"
    );
    assert!(
        b.iter().all(|row| row.len() == c),
        "right matrix has ragged rows"
    );
    assert_eq!(b.len(), k, "inner dimensions must match for multiplication");
    let m = u128::from(modulus);
    let mut out = vec![vec![0_u64; c]; r];
    for i in 0..r {
        for t in 0..k {
            let a_it = u128::from(a[i][t]);
            if a_it == 0 {
                continue;
            }
            for j in 0..c {
                let prod = (a_it * u128::from(b[t][j])) % m;
                let sum = (u128::from(out[i][j]) + prod) % m;
                out[i][j] = sum as u64;
            }
        }
    }
    out
}

/// Fast exponentiation of a square matrix modulo `modulus`.
///
/// Returns `matrix^exp mod modulus` using binary exponentiation in
/// O(k³ log exp) for a `k × k` matrix. `matrix^0` returns the `k × k`
/// identity (each entry reduced modulo `modulus`).
///
/// # Panics
/// Panics if `matrix` is empty or non-square.
#[must_use]
pub fn mat_pow_mod(matrix: &[Vec<u64>], exp: u64, modulus: u64) -> Vec<Vec<u64>> {
    assert!(!matrix.is_empty(), "matrix must be non-empty");
    let k = matrix.len();
    assert!(
        matrix.iter().all(|row| row.len() == k),
        "matrix must be square"
    );
    let one = u64::from(modulus != 1);
    let mut result: Vec<Vec<u64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| if i == j { one } else { 0 })
                .collect::<Vec<u64>>()
        })
        .collect();
    // Reduce the base modulo `modulus` once up-front.
    let mut base: Vec<Vec<u64>> = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|&v| if modulus == 0 { v } else { v % modulus })
                .collect()
        })
        .collect();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = mat_mul_mod(&result, &base, modulus);
        }
        e >>= 1;
        if e > 0 {
            base = mat_mul_mod(&base, &base, modulus);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{mat_mul_mod, mat_pow_mod};
    use quickcheck_macros::quickcheck;

    const P: u64 = 1_000_000_007;

    fn fib(n: u64) -> u64 {
        let (mut a, mut b) = (0_u128, 1_u128);
        for _ in 0..n {
            let next = (a + b) % u128::from(P);
            a = b;
            b = next;
        }
        a as u64
    }

    fn fib_via_matrix(n: u64) -> u64 {
        let m = vec![vec![1_u64, 1], vec![1, 0]];
        let r = mat_pow_mod(&m, n, P);
        // [[F(n+1), F(n)], [F(n), F(n-1)]]
        r[0][1]
    }

    #[test]
    fn one_by_one() {
        let m = vec![vec![3_u64]];
        // 3^10 mod P
        let r = mat_pow_mod(&m, 10, P);
        let expected = (0..10).fold(1_u128, |acc, _| (acc * 3) % u128::from(P)) as u64;
        assert_eq!(r, vec![vec![expected]]);
    }

    #[test]
    fn one_by_one_zero_exp() {
        let m = vec![vec![7_u64]];
        assert_eq!(mat_pow_mod(&m, 0, P), vec![vec![1]]);
    }

    #[test]
    fn identity_powers_identity() {
        let id = vec![vec![1_u64, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
        for n in [0_u64, 1, 2, 5, 100, 1_000_000] {
            assert_eq!(mat_pow_mod(&id, n, P), id);
        }
    }

    #[test]
    fn zero_exponent_returns_identity() {
        let m = vec![vec![2_u64, 3], vec![5, 7]];
        let r = mat_pow_mod(&m, 0, P);
        assert_eq!(r, vec![vec![1, 0], vec![0, 1]]);
    }

    #[test]
    fn first_power_returns_self_mod() {
        let m = vec![vec![2_u64, 3], vec![5, 7]];
        let r = mat_pow_mod(&m, 1, P);
        assert_eq!(r, m);
    }

    #[test]
    fn fibonacci_recurrence_small() {
        // Spot-check the first several Fibonacci numbers.
        let expected: [u64; 11] = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        for (n, &v) in expected.iter().enumerate() {
            assert_eq!(fib_via_matrix(n as u64), v);
        }
    }

    #[test]
    fn fibonacci_recurrence_large() {
        for n in [50_u64, 100, 1_000, 10_000, 100_000] {
            assert_eq!(fib_via_matrix(n), fib(n));
        }
    }

    #[test]
    fn modular_consistency_pre_reduce() {
        // Reducing the base modulo p before exponentiating must give the same
        // result as exponentiating then reducing.
        let m = vec![vec![P + 2, P + 3], vec![2 * P + 5, P + 7]];
        let m_reduced: Vec<Vec<u64>> = m
            .iter()
            .map(|r| r.iter().map(|v| v % P).collect())
            .collect();
        for n in [0_u64, 1, 2, 5, 17, 64] {
            assert_eq!(mat_pow_mod(&m, n, P), mat_pow_mod(&m_reduced, n, P));
        }
    }

    #[test]
    fn mul_dimension_mismatch_panics() {
        let a = vec![vec![1_u64, 2, 3]];
        let b = vec![vec![1_u64], vec![2]];
        let result = std::panic::catch_unwind(|| mat_mul_mod(&a, &b, P));
        assert!(result.is_err());
    }

    #[test]
    fn pow_non_square_panics() {
        let a = vec![vec![1_u64, 2, 3], vec![4, 5, 6]];
        let result = std::panic::catch_unwind(|| mat_pow_mod(&a, 3, P));
        assert!(result.is_err());
    }

    fn naive_pow(matrix: &[Vec<u64>], exp: u64, modulus: u64) -> Vec<Vec<u64>> {
        let k = matrix.len();
        let mut acc: Vec<Vec<u64>> = (0..k)
            .map(|i| (0..k).map(|j| u64::from(i == j)).collect())
            .collect();
        for _ in 0..exp {
            acc = mat_mul_mod(&acc, matrix, modulus);
        }
        acc
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_naive_iterative_product(seed: Vec<u64>, k_pick: u8, exp_pick: u8) -> bool {
        let k = (k_pick % 3) as usize + 1; // k ∈ {1, 2, 3}
        let exp = u64::from(exp_pick % 51); // exp ∈ [0, 50]
        let mut vals: Vec<u64> = seed.into_iter().take(k * k).collect();
        while vals.len() < k * k {
            vals.push(0);
        }
        let m: Vec<Vec<u64>> = (0..k)
            .map(|i| (0..k).map(|j| vals[i * k + j] % P).collect())
            .collect();
        mat_pow_mod(&m, exp, P) == naive_pow(&m, exp, P)
    }
}
