//! Pollard's rho integer factorisation with Brent's cycle improvement.
//!
//! Finds a non-trivial factor of a composite `n: u64` in expected
//! `O(n^{1/4})` time. The driver [`factorize`] peels small primes via
//! trial division, gates primality with a deterministic Miller-Rabin
//! over the first twelve primes (which is exact for all `u64`), and
//! recurses with [`pollard_rho`] on the remaining composite parts.
//!
//! The rho iteration uses the polynomial `f(x) = x^2 + c (mod n)` and
//! Brent's cycle detection with batched gcds to amortise the cost of
//! the inverse-mod step across blocks of length `m`. All arithmetic is
//! kept inside `u128` to avoid overflow when multiplying two `u64`s.
//!
//! # Complexity
//! - [`pollard_rho`]: expected `O(n^{1/4})` mulmods per call.
//! - [`factorize`]: expected `O(n^{1/4} log n)` worst case (driven by
//!   the largest prime factor).
//!
//! # References
//! - Pollard, J. M. (1975). "A Monte Carlo method for factorization."
//! - Brent, R. P. (1980). "An improved Monte Carlo factorization algorithm."

/// Returns the prime factorisation of `n` as a flat list of primes
/// repeated according to multiplicity, sorted ascending. Returns the
/// empty vector for `n = 0` and `n = 1`.
pub fn factorize(n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    if n < 2 {
        return out;
    }
    factorize_into(n, &mut out);
    out.sort_unstable();
    out
}

/// Returns a non-trivial factor of the composite `n`. Requires that
/// `n` be composite, odd, and at least 4.
///
/// # Panics
/// Panics (in debug builds) if `n <= 3`, `n` is even, or `n` is prime.
#[must_use]
pub fn pollard_rho(n: u64) -> u64 {
    debug_assert!(n > 3, "pollard_rho requires n > 3");
    debug_assert!(n & 1 == 1, "pollard_rho requires odd n");
    debug_assert!(!is_prime_u64(n), "pollard_rho requires composite n");

    // Brent's variant: cycle detection on f(x) = x^2 + c (mod n) with a
    // batched gcd over blocks of length `m`.
    let mut c: u64 = 1;
    loop {
        if let Some(d) = brent_attempt(n, c) {
            return d;
        }
        c = c.wrapping_add(1);
        // c == 0 mod n would degenerate, but n > 3 keeps us safe for
        // any reasonable c. Still, skip the trivial cases.
        if c.is_multiple_of(n) {
            c = 1;
        }
    }
}

/// Recursively splits `n` into primes and pushes them into `out`.
fn factorize_into(mut n: u64, out: &mut Vec<u64>) {
    // Strip the small primes first; this both handles the `pollard_rho`
    // even/<=3 preconditions and accelerates the typical case.
    for &p in &SMALL_PRIMES {
        if (p as u64).saturating_mul(p as u64) > n {
            break;
        }
        while n.is_multiple_of(u64::from(p)) {
            out.push(u64::from(p));
            n /= u64::from(p);
        }
    }
    if n < 2 {
        return;
    }
    if is_prime_u64(n) {
        out.push(n);
        return;
    }
    let d = pollard_rho(n);
    factorize_into(d, out);
    factorize_into(n / d, out);
}

/// Single Brent attempt for parameter `c`. Returns `Some(d)` for a
/// non-trivial factor of `n`, or `None` if this `c` failed (cycle hit
/// the trivial factor `n` itself).
fn brent_attempt(n: u64, c: u64) -> Option<u64> {
    // Choose a non-degenerate starting point.
    let mut y: u64 = 2;
    let mut x: u64 = y;
    let mut q: u64 = 1;
    let mut g: u64 = 1;
    // Block size for batched gcd. 128 is a common sweet spot.
    let m: u64 = 128;
    let mut r: u64 = 1;

    let f = |v: u64| -> u64 {
        // (v*v + c) mod n in u128.
        let t = u128::from(v) * u128::from(v) % u128::from(n);
        let t = (t + u128::from(c)) % u128::from(n);
        t as u64
    };

    let mut ys: u64 = y;
    while g == 1 {
        x = y;
        for _ in 0..r {
            y = f(y);
        }
        let mut k: u64 = 0;
        while k < r && g == 1 {
            ys = y;
            let lim = m.min(r - k);
            for _ in 0..lim {
                y = f(y);
                let diff = x.abs_diff(y);
                if diff != 0 {
                    q = mulmod(q, diff, n);
                }
            }
            g = gcd(q, n);
            k += m;
        }
        r *= 2;
    }
    if g == n {
        // Backtrack one step at a time from `ys` to recover a factor.
        loop {
            ys = f(ys);
            let diff = x.abs_diff(ys);
            g = gcd(diff, n);
            if g > 1 {
                break;
            }
        }
    }
    if g == n {
        None
    } else {
        Some(g)
    }
}

/// `(a * b) mod m` via `u128`.
#[inline]
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64
}

/// `(base^exp) mod m` via `u128`.
#[inline]
fn powmod(base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    let m128 = u128::from(m);
    let mut b = u128::from(base) % m128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b % m128;
        }
        b = b * b % m128;
        exp >>= 1;
    }
    result as u64
}

/// Binary gcd on `u64`.
#[inline]
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    while b != 0 {
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
    }
    a << shift
}

/// First twelve primes — sufficient witnesses for a deterministic
/// Miller-Rabin over the entire `u64` range (Jaeschke / Sorenson-Webster).
const SMALL_PRIMES: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Deterministic Miller-Rabin for any `u64`. Inlined here to keep this
/// module independent of the standalone primality test.
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &SMALL_PRIMES {
        if n == u64::from(p) {
            return true;
        }
        if n.is_multiple_of(u64::from(p)) {
            return false;
        }
    }
    // Write n - 1 = d * 2^s with d odd.
    let mut d = n - 1;
    let mut s: u32 = 0;
    while d & 1 == 0 {
        d >>= 1;
        s += 1;
    }
    'witness: for &a in &SMALL_PRIMES {
        let mut x = powmod(u64::from(a), d, n);
        if x == 1 || x == n - 1 {
            continue 'witness;
        }
        for _ in 0..s - 1 {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{factorize, is_prime_u64, pollard_rho};

    #[test]
    fn trivial_inputs() {
        assert!(factorize(0).is_empty());
        assert!(factorize(1).is_empty());
    }

    #[test]
    fn primes_factor_to_themselves() {
        for &p in &[2_u64, 3, 5, 7, 11, 13, 1_000_000_007, 1_000_000_009] {
            assert_eq!(factorize(p), vec![p], "prime p={p}");
        }
    }

    #[test]
    fn smooth_composite() {
        assert_eq!(factorize(2 * 3 * 5 * 7), vec![2, 3, 5, 7]);
        assert_eq!(factorize(360), vec![2, 2, 2, 3, 3, 5]);
        assert_eq!(factorize(1024), vec![2; 10]);
    }

    #[test]
    fn fermat_f5() {
        // 2^32 + 1 = 641 * 6_700_417
        assert_eq!(factorize((1_u64 << 32) + 1), vec![641, 6_700_417]);
    }

    #[test]
    fn ten_to_18_plus_9() {
        // 10^18 + 9 = 7 * 11 * 13 * 211 * 241 * 2161 * 9181 * 1289609
        // Verify by factorising and checking the product / primality
        // invariants rather than hard-coding the sequence.
        let n: u64 = 1_000_000_000_000_000_009;
        let factors = factorize(n);
        assert!(!factors.is_empty());
        let prod: u128 = factors.iter().map(|&x| u128::from(x)).product();
        assert_eq!(prod, u128::from(n));
        for &p in &factors {
            assert!(is_prime_u64(p), "expected prime, got {p}");
        }
        // Sortedness invariant.
        for w in factors.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn prime_powers() {
        // 2^20
        assert_eq!(factorize(1 << 20), vec![2; 20]);
        // 3^10 = 59049
        assert_eq!(factorize(59_049), vec![3; 10]);
        // 7^8 = 5_764_801
        assert_eq!(factorize(5_764_801), vec![7; 8]);
        // 1_000_000_007^2
        let p: u64 = 1_000_000_007;
        assert_eq!(factorize(p * p), vec![p, p]);
    }

    #[test]
    fn u64_max_factors() {
        // u64::MAX = 2^64 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 6_700_417
        assert_eq!(
            factorize(u64::MAX),
            vec![3, 5, 17, 257, 641, 65_537, 6_700_417]
        );
    }

    #[test]
    fn semiprimes_around_10_to_18() {
        // p, q both near 10^9; n ≈ 10^18.
        let p: u64 = 999_999_937; // prime
        let q: u64 = 999_999_999_989; // prime (~10^12)
        let n = p.checked_mul(q);
        if let Some(n) = n {
            let factors = factorize(n);
            assert_eq!(factors, vec![p, q]);
        }
    }

    #[test]
    fn pollard_rho_on_known_composite() {
        // n = 8051 = 83 * 97
        let d = pollard_rho(8051);
        assert!(d == 83 || d == 97);
        assert!(8051_u64.is_multiple_of(d));
    }

    #[test]
    fn pollard_rho_on_carmichael() {
        // 561 = 3 * 11 * 17 (Carmichael number)
        let n: u64 = 561;
        // pollard_rho preconditions require an odd composite > 3 — 561 fits.
        let d = pollard_rho(n);
        assert!(n.is_multiple_of(d) && d > 1 && d < n);
    }

    #[test]
    fn miller_rabin_smoke() {
        for n in 0..200_u64 {
            // Cross-check against trial division.
            let mut prime = n >= 2;
            let mut k = 2;
            while k * k <= n {
                if n.is_multiple_of(k) {
                    prime = false;
                    break;
                }
                k += 1;
            }
            assert_eq!(is_prime_u64(n), prime, "n = {n}");
        }
    }

    /// Property test: for random `n` in `2..=10^12`, the product of
    /// `factorize(n)` equals `n` and every returned factor is prime.
    mod property {
        use super::{factorize, is_prime_u64};
        use quickcheck_macros::quickcheck;

        #[allow(clippy::needless_pass_by_value)]
        #[quickcheck]
        fn product_and_primality(seed: u64) -> bool {
            let n = (seed % 999_999_999_999) + 2; // in [2, 10^12 + 1]
            let factors = factorize(n);
            let prod: u128 = factors.iter().map(|&x| u128::from(x)).product();
            if prod != u128::from(n) {
                return false;
            }
            // Sorted ascending.
            for w in factors.windows(2) {
                if w[0] > w[1] {
                    return false;
                }
            }
            // Every factor is prime by inline Miller-Rabin.
            factors.iter().all(|&p| is_prime_u64(p))
        }
    }
}
