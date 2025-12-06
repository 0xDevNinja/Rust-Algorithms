//! Matrix-chain multiplication: minimum scalar multiplications. O(n³).

/// Given a chain of matrices where matrix `i` has dimensions
/// `dims[i] × dims[i+1]`, returns the minimum number of scalar
/// multiplications required to compute their product.
pub fn min_matrix_chain(dims: &[u64]) -> u64 {
    let n = dims.len();
    if n < 3 {
        return 0;
    }
    let m = n - 1; // number of matrices
    let mut dp = vec![vec![0_u64; m]; m];
    for len in 2..=m {
        for i in 0..=m - len {
            let j = i + len - 1;
            let mut best = u64::MAX;
            for k in i..j {
                let cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1];
                if cost < best {
                    best = cost;
                }
            }
            dp[i][j] = best;
        }
    }
    dp[0][m - 1]
}

#[cfg(test)]
mod tests {
    use super::min_matrix_chain;

    #[test]
    fn classic_clrs() {
        // Matrices A1(30x35) A2(35x15) A3(15x5) A4(5x10) A5(10x20) A6(20x25)
        let dims = [30, 35, 15, 5, 10, 20, 25];
        assert_eq!(min_matrix_chain(&dims), 15_125);
    }

    #[test]
    fn two_matrices() {
        // 10x20 · 20x30 = 6000.
        assert_eq!(min_matrix_chain(&[10, 20, 30]), 6_000);
    }

    #[test]
    fn single_matrix() {
        assert_eq!(min_matrix_chain(&[5, 10]), 0);
    }

    #[test]
    fn empty() {
        assert_eq!(min_matrix_chain(&[]), 0);
    }
}
