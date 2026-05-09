//! Count walks of length `k` between every pair of vertices via matrix
//! exponentiation of the adjacency matrix.
//!
//! For an `n × n` non-negative integer adjacency matrix `A` (with `A[i][j]`
//! interpreted as the multiplicity of the edge `i -> j`), the entry
//! `(A^k)[i][j]` counts the number of length-`k` walks from `i` to `j`.
//!
//! Complexity: `O(n^3 log k)` time, `O(n^2)` space. Multiplications are
//! performed in `u128` to avoid overflow when `modulus` is close to
//! `u64::MAX`.

/// Multiply two `n × n` matrices over `Z/modulus`. Assumes both matrices are
/// square and of the same dimension.
fn mat_mul(a: &[Vec<u64>], b: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
    let n = a.len();
    let m = modulus as u128;
    let mut out = vec![vec![0u64; n]; n];
    for i in 0..n {
        for l in 0..n {
            let ail = a[i][l] as u128 % m;
            if ail == 0 {
                continue;
            }
            for j in 0..n {
                let blj = b[l][j] as u128 % m;
                let acc = out[i][j] as u128;
                out[i][j] = ((acc + ail * blj) % m) as u64;
            }
        }
    }
    out
}

/// Returns the identity `n × n` matrix modulo `modulus`. If `modulus == 1`
/// every entry collapses to 0.
fn identity(n: usize, modulus: u64) -> Vec<Vec<u64>> {
    let mut id = vec![vec![0u64; n]; n];
    let one = u64::from(modulus != 1);
    for (i, row) in id.iter_mut().enumerate().take(n) {
        row[i] = one;
    }
    id
}

/// Reduce every entry of `m` modulo `modulus`.
fn reduce(m: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
    let mu = modulus as u128;
    m.iter()
        .map(|row| row.iter().map(|&x| (x as u128 % mu) as u64).collect())
        .collect()
}

/// Returns `adj^k mod modulus`. The `(i, j)` entry of the result is the
/// number of length-`k` walks from vertex `i` to vertex `j`, taken modulo
/// `modulus`.
///
/// `adj` must be square. Entries may be any non-negative `u64` and are
/// interpreted as edge multiplicities. `k = 0` returns the identity matrix
/// (only zero-length walks `i -> i`). When `n == 0`, an empty `Vec` is
/// returned regardless of `k`.
///
/// # Panics
///
/// Panics if `modulus == 0` or if `adj` is not a square matrix.
pub fn count_walks_mod(adj: &[Vec<u64>], k: u64, modulus: u64) -> Vec<Vec<u64>> {
    assert!(modulus != 0, "modulus must be non-zero");
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    for row in adj {
        assert_eq!(row.len(), n, "adjacency matrix must be square");
    }

    if k == 0 {
        return identity(n, modulus);
    }
    if k == 1 {
        return reduce(adj, modulus);
    }

    let mut base = reduce(adj, modulus);
    let mut result = identity(n, modulus);
    let mut exp = k;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mat_mul(&result, &base, modulus);
        }
        exp >>= 1;
        if exp > 0 {
            base = mat_mul(&base, &base, modulus);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::count_walks_mod;

    fn naive_pow(adj: &[Vec<u64>], k: u64, modulus: u64) -> Vec<Vec<u64>> {
        let n = adj.len();
        let m = modulus as u128;
        // Start with identity.
        let mut result: Vec<Vec<u64>> = (0..n)
            .map(|i| (0..n).map(|j| u64::from(i == j)).collect())
            .collect();
        for _ in 0..k {
            let mut next = vec![vec![0u64; n]; n];
            for i in 0..n {
                for l in 0..n {
                    let v = result[i][l] as u128;
                    if v == 0 {
                        continue;
                    }
                    for j in 0..n {
                        let acc = next[i][j] as u128;
                        next[i][j] = ((acc + v * (adj[l][j] as u128)) % m) as u64;
                    }
                }
            }
            result = next;
        }
        result
    }

    /// Brute-force walk counter via DFS enumeration. Only viable for tiny
    /// graphs; counts walks of length exactly `k` from `s` to `t`.
    fn dfs_count(adj: &[Vec<u64>], s: usize, t: usize, k: u64) -> u128 {
        let n = adj.len();
        if k == 0 {
            return u128::from(s == t);
        }
        let mut total: u128 = 0;
        for v in 0..n {
            let m = adj[s][v] as u128;
            if m == 0 {
                continue;
            }
            total += m * dfs_count(adj, v, t, k - 1);
        }
        total
    }

    #[test]
    fn k_zero_returns_identity() {
        let adj = vec![vec![0, 1, 0], vec![1, 0, 1], vec![0, 1, 0]];
        let r = count_walks_mod(&adj, 0, 1_000_000_007);
        assert_eq!(r, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    }

    #[test]
    fn k_one_returns_adj() {
        let adj = vec![vec![0, 2, 1], vec![3, 0, 0], vec![1, 4, 0]];
        let r = count_walks_mod(&adj, 1, 1_000_000_007);
        assert_eq!(r, adj);
    }

    #[test]
    fn empty_graph() {
        let adj: Vec<Vec<u64>> = Vec::new();
        let r = count_walks_mod(&adj, 5, 7);
        assert!(r.is_empty());
    }

    #[test]
    fn path_graph_length_two() {
        // 0 - 1 - 2 path, undirected => symmetric adjacency.
        let adj = vec![vec![0, 1, 0], vec![1, 0, 1], vec![0, 1, 0]];
        let r = count_walks_mod(&adj, 2, 1_000_000_007);
        // Closed length-2 walks from 0: only 0-1-0, count = 1.
        assert_eq!(r[0][0], 1);
        // From 1: 1-0-1 and 1-2-1, count = 2.
        assert_eq!(r[1][1], 2);
        // 0 to 2 via length 2: 0-1-2, count = 1.
        assert_eq!(r[0][2], 1);
    }

    #[test]
    fn triangle_k3_closed_walks_length_two() {
        // K3: each vertex has 2 neighbors.
        let adj = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
        let r = count_walks_mod(&adj, 2, 1_000_000_007);
        for i in 0..3 {
            assert_eq!(r[i][i], 2, "closed length-2 walks from {i} should be 2");
        }
    }

    #[test]
    fn matches_naive_product_for_small_k() {
        // Mixed directed graph with multiplicities.
        let adj = vec![
            vec![0, 2, 0, 1],
            vec![1, 0, 3, 0],
            vec![0, 1, 0, 2],
            vec![2, 0, 1, 0],
        ];
        let modulus = 1_000_000_007;
        for k in 0..=6 {
            let fast = count_walks_mod(&adj, k, modulus);
            let slow = naive_pow(&adj, k, modulus);
            assert_eq!(fast, slow, "mismatch at k={k}");
        }
    }

    #[test]
    fn modulus_is_applied() {
        // K3 grows fast: (A^k)[i][i] for k large is ~ 2^k / 3 + ...
        let adj = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
        let modulus = 7u64;
        let r = count_walks_mod(&adj, 20, modulus);
        let slow = naive_pow(&adj, 20, modulus);
        assert_eq!(r, slow);
        for row in &r {
            for &x in row {
                assert!(x < modulus);
            }
        }
    }

    #[test]
    fn modulus_one_yields_zero_matrix() {
        let adj = vec![vec![0, 1], vec![1, 0]];
        let r = count_walks_mod(&adj, 5, 1);
        for row in r {
            for x in row {
                assert_eq!(x, 0);
            }
        }
    }

    #[test]
    fn large_modulus_no_overflow() {
        // modulus close to 2^63 forces u128 multiplication.
        let modulus: u64 = (1u64 << 62) - 57;
        let adj = vec![
            vec![modulus - 1, modulus - 2, 0],
            vec![3, modulus - 5, 7],
            vec![1, 9, modulus - 11],
        ];
        let fast = count_walks_mod(&adj, 5, modulus);
        let slow = naive_pow(&adj, 5, modulus);
        assert_eq!(fast, slow);
        for row in &fast {
            for &x in row {
                assert!(x < modulus);
            }
        }
    }

    #[test]
    fn matches_dfs_enumeration_small_graphs() {
        // A handful of tiny graphs (n ≤ 4, k ≤ 5) compared against DFS
        // enumeration. Small simple LCG keeps the test deterministic.
        let mut state: u64 = 0x00C0_FFEE;
        let mut next = || -> u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        let modulus: u64 = 1_000_003;
        for _ in 0..30 {
            let n = ((next() % 4) as usize) + 1; // 1..=4
            let k = next() % 6; // 0..=5
            let adj: Vec<Vec<u64>> = (0..n)
                .map(|_| (0..n).map(|_| next() % 3).collect())
                .collect();
            let fast = count_walks_mod(&adj, k, modulus);
            for s in 0..n {
                for t in 0..n {
                    let want = (super::tests::dfs_count(&adj, s, t, k) % modulus as u128) as u64;
                    assert_eq!(
                        fast[s][t], want,
                        "mismatch n={n} k={k} s={s} t={t} adj={adj:?}"
                    );
                }
            }
        }
    }
}
