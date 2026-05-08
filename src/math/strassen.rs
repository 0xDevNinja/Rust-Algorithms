//! Strassen's matrix multiplication for square real matrices.
//!
//! Multiplies two `n x n` matrices using the Strassen recurrence, which
//! computes the product with seven recursive sub-multiplications instead of
//! the eight required by the schoolbook algorithm. The recursion halves the
//! dimension at each step, so it requires the size to be a power of two; we
//! pad the inputs with zeros to the next power of two and trim the result at
//! the end.
//!
//! Below a small `BASE_THRESHOLD` we fall back to the schoolbook algorithm
//! since the constant factor of Strassen dominates on tiny matrices.
//!
//! # Complexity
//! Time: `O(n^log2 7) ≈ O(n^2.807)`. Space: `O(n^2)`.
//!
//! Reference: Strassen, "Gaussian Elimination is not Optimal" (1969).

/// Crossover dimension below which schoolbook multiplication is used.
const BASE_THRESHOLD: usize = 64;

/// A square dense matrix represented as a vector of row vectors.
type Matrix = Vec<Vec<f64>>;

/// The four `k x k` quadrants of a `2k x 2k` matrix, in row-major order
/// (top-left, top-right, bottom-left, bottom-right).
type Quadrants = (Matrix, Matrix, Matrix, Matrix);

/// Multiplies two square `f64` matrices `a` and `b` using Strassen's
/// algorithm, padding to the next power of two when necessary.
///
/// # Panics
/// Panics if `a` and `b` are not square matrices of identical dimension.
pub fn strassen(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    assert_eq!(b.len(), n, "matrices must have equal dimension");
    assert!(a.iter().all(|row| row.len() == n), "a must be square");
    assert!(b.iter().all(|row| row.len() == n), "b must be square");

    if n == 0 {
        return Vec::new();
    }

    // Pad to next power of two if needed.
    let m = next_power_of_two(n);
    if m == n && n.is_power_of_two() {
        let padded = strassen_recursive(a, b);
        return padded;
    }

    let a_pad = pad(a, m);
    let b_pad = pad(b, m);
    let c_pad = strassen_recursive(&a_pad, &b_pad);
    trim(&c_pad, n)
}

/// Recursive Strassen multiply on a power-of-two square matrix.
fn strassen_recursive(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n <= BASE_THRESHOLD {
        return schoolbook(a, b);
    }

    let k = n / 2;

    // Split a and b into four k×k blocks.
    let (a11, a12, a21, a22) = split(a, k);
    let (b11, b12, b21, b22) = split(b, k);

    // Seven Strassen products.
    let m1 = strassen_recursive(&add(&a11, &a22), &add(&b11, &b22));
    let m2 = strassen_recursive(&add(&a21, &a22), &b11);
    let m3 = strassen_recursive(&a11, &sub(&b12, &b22));
    let m4 = strassen_recursive(&a22, &sub(&b21, &b11));
    let m5 = strassen_recursive(&add(&a11, &a12), &b22);
    let m6 = strassen_recursive(&sub(&a21, &a11), &add(&b11, &b12));
    let m7 = strassen_recursive(&sub(&a12, &a22), &add(&b21, &b22));

    // Assemble the four output blocks.
    let c11 = add(&sub(&add(&m1, &m4), &m5), &m7);
    let c12 = add(&m3, &m5);
    let c21 = add(&m2, &m4);
    let c22 = add(&sub(&add(&m1, &m3), &m2), &m6);

    join(&c11, &c12, &c21, &c22)
}

/// Standard `O(n^3)` triple-loop multiplication. Used as the recursion base
/// case and as a reference inside tests.
fn schoolbook(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            for j in 0..n {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// Element-wise sum of two equally-sized square matrices.
fn add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// Element-wise difference `a - b` for equally-sized square matrices.
fn sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// Splits a `2k x 2k` matrix into its four `k x k` quadrants.
fn split(a: &[Vec<f64>], k: usize) -> Quadrants {
    let mut a11 = vec![vec![0.0_f64; k]; k];
    let mut a12 = vec![vec![0.0_f64; k]; k];
    let mut a21 = vec![vec![0.0_f64; k]; k];
    let mut a22 = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            a11[i][j] = a[i][j];
            a12[i][j] = a[i][j + k];
            a21[i][j] = a[i + k][j];
            a22[i][j] = a[i + k][j + k];
        }
    }
    (a11, a12, a21, a22)
}

/// Joins four `k x k` quadrants into a single `2k x 2k` matrix.
fn join(c11: &[Vec<f64>], c12: &[Vec<f64>], c21: &[Vec<f64>], c22: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = c11.len();
    let n = 2 * k;
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..k {
        for j in 0..k {
            c[i][j] = c11[i][j];
            c[i][j + k] = c12[i][j];
            c[i + k][j] = c21[i][j];
            c[i + k][j + k] = c22[i][j];
        }
    }
    c
}

/// Pads a matrix to size `m x m` (with `m >= n`) by appending zero rows and
/// zero columns. Original entries are preserved at the top-left.
fn pad(a: &[Vec<f64>], m: usize) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut p = vec![vec![0.0_f64; m]; m];
    for i in 0..n {
        for j in 0..n {
            p[i][j] = a[i][j];
        }
    }
    p
}

/// Returns the top-left `n x n` submatrix.
fn trim(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            t[i][j] = a[i][j];
        }
    }
    t
}

/// Smallest power of two `>= n`. Returns 1 for `n == 0`.
const fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::{schoolbook, strassen};

    fn approx_eq(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (ra, rb) in a.iter().zip(b.iter()) {
            if ra.len() != rb.len() {
                return false;
            }
            for (x, y) in ra.iter().zip(rb.iter()) {
                if (x - y).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        m
    }

    /// Tiny xorshift PRNG so tests are deterministic without extra deps.
    struct Xs64(u64);
    impl Xs64 {
        const fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f64(&mut self) -> f64 {
            // Map to [-1, 1).
            let u = self.next_u64() >> 11; // 53 bits
            let f = (u as f64) / ((1_u64 << 53) as f64);
            f.mul_add(2.0, -1.0)
        }
    }

    fn random_matrix(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Xs64::new(seed);
        (0..n)
            .map(|_| (0..n).map(|_| rng.next_f64()).collect())
            .collect()
    }

    #[test]
    fn identity_times_a_equals_a() {
        let a = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ];
        let i = identity(4);
        let c = strassen(&i, &a);
        assert!(approx_eq(&c, &a, 1e-12));
    }

    #[test]
    fn two_by_two_closed_form() {
        // [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let expected = vec![vec![19.0, 22.0], vec![43.0, 50.0]];
        let c = strassen(&a, &b);
        assert!(approx_eq(&c, &expected, 1e-12));
    }

    #[test]
    fn eight_by_eight_random_vs_schoolbook() {
        let a = random_matrix(8, 0x00C0_FFEE);
        let b = random_matrix(8, 0x0BAD_BEEF);
        let want = schoolbook(&a, &b);
        let got = strassen(&a, &b);
        assert!(approx_eq(&got, &want, 1e-9));
    }

    #[test]
    fn three_by_three_non_power_of_two() {
        // 3x3 path forces padding to 4x4 internally.
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let b = vec![
            vec![9.0, 8.0, 7.0],
            vec![6.0, 5.0, 4.0],
            vec![3.0, 2.0, 1.0],
        ];
        let want = schoolbook(&a, &b);
        let got = strassen(&a, &b);
        assert_eq!(got.len(), 3);
        assert!(approx_eq(&got, &want, 1e-12));
    }

    #[test]
    fn five_by_five_non_power_of_two() {
        let a = random_matrix(5, 42);
        let b = random_matrix(5, 1337);
        let want = schoolbook(&a, &b);
        let got = strassen(&a, &b);
        assert_eq!(got.len(), 5);
        assert!(approx_eq(&got, &want, 1e-9));
    }

    #[test]
    fn one_by_one() {
        let a = vec![vec![3.0]];
        let b = vec![vec![4.0]];
        let c = strassen(&a, &b);
        let expected = vec![vec![12.0_f64]];
        assert!(approx_eq(&c, &expected, 1e-12));
    }

    #[test]
    fn empty_matrix() {
        let a: Vec<Vec<f64>> = Vec::new();
        let b: Vec<Vec<f64>> = Vec::new();
        let c = strassen(&a, &b);
        assert!(c.is_empty());
    }

    #[cfg(test)]
    mod property {
        use super::{approx_eq, random_matrix, schoolbook};
        use crate::math::strassen::strassen;
        use quickcheck_macros::quickcheck;

        /// Random small square matrices: `strassen` must agree with
        /// `schoolbook` within an `f64` tolerance.
        #[quickcheck]
        fn matches_schoolbook(size_seed: u8, a_seed: u64, b_seed: u64) -> bool {
            let n = (size_seed as usize % 10) + 1; // 1..=10
            let a = random_matrix(n, a_seed.wrapping_add(1));
            let b = random_matrix(n, b_seed.wrapping_add(1));
            let want = schoolbook(&a, &b);
            let got = strassen(&a, &b);
            approx_eq(&got, &want, 1e-9)
        }
    }
}
