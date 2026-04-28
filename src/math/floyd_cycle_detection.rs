//! Floyd's cycle detection (the "tortoise and hare" algorithm).
//!
//! For a sequence defined by `x_0 = start` and `x_{i+1} = f(x_i)`, this module
//! finds the index `mu` where the cycle begins (the length of the leading tail)
//! and the cycle length `lambda`.
//!
//! # Algorithm
//! 1. **Meet inside the cycle.** Advance the tortoise one step at a time and
//!    the hare two steps at a time. The first index `i` at which they meet
//!    satisfies `x_i = x_{2i}`, which lies inside the cycle.
//! 2. **Find the cycle start.** Reset the tortoise to `start` and advance both
//!    pointers at speed 1. They meet at `x_mu`, the first repeated element.
//! 3. **Measure the cycle.** Hold one pointer fixed and advance the other until
//!    it returns; the number of steps taken is `lambda`.
//!
//! # Complexity
//! - Time:  O(mu + lambda) — at most a constant number of passes over the tail
//!   and the cycle.
//! - Space: O(1) — only the two pointers and a handful of counters.
//!
//! # Finite-domain note
//! When `f` is a pure function over `u64` (a finite domain) every orbit is
//! eventually periodic, so a cycle is guaranteed and `floyd_cycle` always
//! returns `Some`. The `Option` return type is preserved for symmetry with
//! callers that may want to plug in partial functions or saturating wrappers
//! in the future.

/// Detects the cycle in the sequence `x_0 = start`, `x_{i+1} = f(x_i)` and
/// returns `Some((mu, lambda))` where `mu` is the index at which the cycle
/// starts (`0` if the very first element repeats) and `lambda` is the cycle
/// length (`>= 1`).
///
/// Because `f: u64 -> u64` operates on a finite domain, a cycle always exists
/// and this function always returns `Some(_)`; see the module docs.
///
/// # Examples
/// ```
/// use rust_algorithms::math::floyd_cycle_detection::floyd_cycle;
/// // f(x) = (x*x + 1) mod 7, starting at 2.
/// // Sequence: 2, 5, 5, 5, ...  -> mu = 1, lambda = 1.
/// let (mu, lambda) = floyd_cycle(2, |x| (x * x + 1) % 7).unwrap();
/// assert_eq!((mu, lambda), (1, 1));
/// ```
#[allow(clippy::unnecessary_wraps)]
pub fn floyd_cycle<F: Fn(u64) -> u64>(start: u64, f: F) -> Option<(u64, u64)> {
    // Phase 1: tortoise moves one step, hare moves two; find a meeting point
    // inside the cycle.
    let mut tortoise = f(start);
    let mut hare = f(f(start));
    while tortoise != hare {
        tortoise = f(tortoise);
        hare = f(f(hare));
    }

    // Phase 2: reset tortoise to start; advance both at speed 1. They meet at
    // the cycle start `mu`.
    let mut mu: u64 = 0;
    tortoise = start;
    while tortoise != hare {
        tortoise = f(tortoise);
        hare = f(hare);
        mu += 1;
    }

    // Phase 3: hold tortoise fixed; advance hare until it returns to count the
    // cycle length `lambda`.
    let mut lambda: u64 = 1;
    hare = f(tortoise);
    while tortoise != hare {
        hare = f(hare);
        lambda += 1;
    }

    Some((mu, lambda))
}

#[cfg(test)]
mod tests {
    use super::floyd_cycle;
    use quickcheck_macros::quickcheck;
    use std::collections::HashMap;

    /// Brute-force reference: walk the sequence, recording each value's first
    /// index in a `HashMap`. The first repeat reveals `mu` (the index where
    /// the previously-seen value first appeared) and `lambda` (current index
    /// minus that previous index).
    fn brute_force<F: Fn(u64) -> u64>(start: u64, f: F, max_steps: u64) -> Option<(u64, u64)> {
        let mut seen: HashMap<u64, u64> = HashMap::new();
        let mut x = start;
        for i in 0..max_steps {
            if let Some(&prev) = seen.get(&x) {
                return Some((prev, i - prev));
            }
            seen.insert(x, i);
            x = f(x);
        }
        None
    }

    #[test]
    fn classic_quadratic_mod_7() {
        // f(x) = (x*x + 1) mod 7, starting at 2.
        // Sequence: 2, 5, 5, 5, ...  -> mu = 1, lambda = 1.
        let (mu, lambda) = floyd_cycle(2, |x| (x * x + 1) % 7).unwrap();
        assert_eq!((mu, lambda), (1, 1));
    }

    #[test]
    fn pure_cycle_fixed_point() {
        // f(x) = x  -> every starting value is a 1-cycle with mu = 0.
        let (mu, lambda) = floyd_cycle(42, |x| x).unwrap();
        assert_eq!((mu, lambda), (0, 1));
    }

    #[test]
    fn pure_cycle_no_tail() {
        // Three-cycle 0 -> 1 -> 2 -> 0; starting at 0 means mu = 0, lambda = 3.
        let (mu, lambda) = floyd_cycle(0, |x| (x + 1) % 3).unwrap();
        assert_eq!((mu, lambda), (0, 3));
    }

    #[test]
    fn long_tail_small_cycle() {
        // f sends every value < 100 to value+1 (a 100-step ramp), then 100 -> 101 -> 100.
        // Starting at 0: tail of 100 elements (indices 0..=99), cycle of length 2.
        use std::cmp::Ordering;
        let f = |x: u64| match x.cmp(&100) {
            Ordering::Less => x + 1,
            Ordering::Equal => 101,
            Ordering::Greater => 100,
        };
        let (mu, lambda) = floyd_cycle(0, f).unwrap();
        assert_eq!((mu, lambda), (100, 2));
    }

    #[test]
    fn cycle_starts_at_index_one() {
        // 0 -> 1 -> 2 -> 1; cycle is 1 -> 2 -> 1, mu = 1, lambda = 2.
        let f = |x: u64| if x == 1 { 2 } else { 1 };
        let (mu, lambda) = floyd_cycle(0, f).unwrap();
        assert_eq!((mu, lambda), (1, 2));
    }

    #[test]
    fn matches_brute_force_known_case() {
        let f = |x: u64| (x * x + 1) % 255;
        let from_floyd = floyd_cycle(3, f).unwrap();
        let from_brute = brute_force(3, f, 10_000).unwrap();
        assert_eq!(from_floyd, from_brute);
    }

    /// For a small modulus `n` and small PRNG-style coefficients, compare
    /// `floyd_cycle` against the brute-force reference. With `n = 8` the orbit
    /// length is bounded by 8, so 1024 brute-force steps are far more than
    /// enough.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_matches_brute_force(seed: u8, a: u8, c: u8) -> bool {
        const N: u64 = 8;
        let a = u64::from(a) % N;
        let c = u64::from(c) % N;
        let start = u64::from(seed) % N;
        let f = move |x: u64| (a * x + c) % N;

        let from_floyd = floyd_cycle(start, f).unwrap();
        let from_brute = brute_force(start, f, 1024).expect("orbit fits in 1024 steps");
        from_floyd == from_brute
    }

    /// Property: for any small finite-domain function, `lambda >= 1` and
    /// `mu + lambda` is at most the domain size, and `f` applied `lambda`
    /// times to `x_mu` returns `x_mu`.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_cycle_invariant(seed: u8, a: u8, c: u8) -> bool {
        const N: u64 = 8;
        let a = u64::from(a) % N;
        let c = u64::from(c) % N;
        let start = u64::from(seed) % N;
        let f = move |x: u64| (a * x + c) % N;

        let (mu, lambda) = floyd_cycle(start, f).unwrap();
        if lambda < 1 || mu + lambda > N {
            return false;
        }
        // Walk to x_mu, then walk lambda more steps and check it returns.
        let mut x = start;
        for _ in 0..mu {
            x = f(x);
        }
        let cycle_start = x;
        for _ in 0..lambda {
            x = f(x);
        }
        x == cycle_start
    }
}
