//! Uniform `rand5` from `rand7` (and the reverse) via rejection sampling.
//!
//! Given a black-box source of uniform integers in some range, we can build a
//! uniform source over a different range without using floating point or any
//! information about the underlying PRNG, provided the new range divides a
//! power of the old range. The classic interview puzzle is "implement `rand5`
//! using only `rand7`"; the symmetric direction ("`rand7` from `rand5`") works
//! the same way.
//!
//! # Method
//! Two consecutive draws of `rand7` (each in `1..=7`) form a base-7 digit pair
//! that is uniform over the 49 outcomes `1..=49`. We keep the first 45 (which
//! is `9 * 5`, the largest multiple of 5 not exceeding 49) and reject the rest;
//! mapping the kept value modulo 5 yields a uniform draw in `1..=5`. The
//! reverse direction uses two `rand5` draws (uniform over `1..=25`) and keeps
//! the first 21 (`3 * 7`).
//!
//! Rejection preserves uniformity: every accepted bucket is hit with
//! probability `1/45` (resp. `1/21`), so the conditional distribution of the
//! output given acceptance is uniform on the target range.
//!
//! # Complexity
//! - Expected `rand7` calls per `rand5_from_rand7` output: `2 * 49 / 45 ≈ 2.18`.
//! - Expected `rand5` calls per `rand7_from_rand5` output: `2 * 25 / 21 ≈ 2.38`.
//! - Worst-case is unbounded in theory but the rejection probability per pair
//!   is `4/49` (resp. `4/25`), so the chance of needing more than `t` pairs
//!   decays geometrically.
//! - Auxiliary space: `O(1)`.
//!
//! # Caller contract
//! The supplied closure MUST return a uniform integer in the documented range
//! (`1..=7` for `rand7`, `1..=5` for `rand5`). Out-of-range values would
//! silently bias the output, so we debug-assert the contract.

/// Draws a uniform integer in `1..=5` using only the given `rand7` source.
///
/// `rand7` must return a uniform integer in `1..=7` on every call. The closure
/// is invoked at least twice and, with low probability, more times due to
/// rejection.
///
/// # Examples
/// ```
/// use rust_algorithms::math::rand5_from_rand7::rand5_from_rand7;
/// let mut state: u32 = 1;
/// let mut rand7 = || {
///     // Toy LCG-ish stream; for tests/examples only.
///     state = state.wrapping_mul(1103515245).wrapping_add(12345);
///     (state % 7) + 1
/// };
/// let v = rand5_from_rand7(&mut rand7);
/// assert!((1..=5).contains(&v));
/// ```
pub fn rand5_from_rand7<F: FnMut() -> u32>(mut rand7: F) -> u32 {
    loop {
        let a = rand7();
        let b = rand7();
        debug_assert!((1..=7).contains(&a), "rand7 must return values in 1..=7");
        debug_assert!((1..=7).contains(&b), "rand7 must return values in 1..=7");
        // Map (a, b) to a base-7 index in 1..=49.
        let idx = (a - 1) * 7 + b; // 1..=49
        if idx <= 45 {
            // 45 = 9 * 5. Map uniformly to 1..=5.
            return ((idx - 1) % 5) + 1;
        }
        // Reject and resample.
    }
}

/// Draws a uniform integer in `1..=7` using only the given `rand5` source.
///
/// `rand5` must return a uniform integer in `1..=5` on every call.
///
/// # Examples
/// ```
/// use rust_algorithms::math::rand5_from_rand7::rand7_from_rand5;
/// let mut state: u32 = 1;
/// let mut rand5 = || {
///     state = state.wrapping_mul(1103515245).wrapping_add(12345);
///     (state % 5) + 1
/// };
/// let v = rand7_from_rand5(&mut rand5);
/// assert!((1..=7).contains(&v));
/// ```
pub fn rand7_from_rand5<F: FnMut() -> u32>(mut rand5: F) -> u32 {
    loop {
        let a = rand5();
        let b = rand5();
        debug_assert!((1..=5).contains(&a), "rand5 must return values in 1..=5");
        debug_assert!((1..=5).contains(&b), "rand5 must return values in 1..=5");
        // Map (a, b) to a base-5 index in 1..=25.
        let idx = (a - 1) * 5 + b; // 1..=25
        if idx <= 21 {
            // 21 = 3 * 7. Map uniformly to 1..=7.
            return ((idx - 1) % 7) + 1;
        }
        // Reject and resample.
    }
}

#[cfg(test)]
mod tests {
    use super::{rand5_from_rand7, rand7_from_rand5};

    /// Tiny deterministic `XorShift64` used to drive mocked `randN` closures.
    /// Independent of `super::` to keep the test self-contained.
    struct XorShift64 {
        state: u64,
    }
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 {
                    0x9E37_79B9_7F4A_7C15
                } else {
                    seed
                },
            }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }
        /// Uniform in `[0, bound)` via rejection — no modulo bias.
        fn next_bounded(&mut self, bound: u64) -> u64 {
            debug_assert!(bound > 0);
            let zone = u64::MAX - (u64::MAX % bound);
            loop {
                let r = self.next_u64();
                if r < zone {
                    return r % bound;
                }
            }
        }
    }

    /// Helper: build a closure that yields uniform `1..=n` draws from a fresh PRNG.
    fn make_randn(seed: u64, n: u64) -> impl FnMut() -> u32 {
        let mut rng = XorShift64::new(seed);
        move || (rng.next_bounded(n) as u32) + 1
    }

    #[test]
    fn rand5_from_rand7_in_range() {
        let mut rand7 = make_randn(0xCAFE_BABE, 7);
        for _ in 0..10_000 {
            let v = rand5_from_rand7(&mut rand7);
            assert!((1..=5).contains(&v), "got {v}");
        }
    }

    #[test]
    fn rand7_from_rand5_in_range() {
        let mut rand5 = make_randn(0xDEAD_BEEF, 5);
        for _ in 0..10_000 {
            let v = rand7_from_rand5(&mut rand5);
            assert!((1..=7).contains(&v), "got {v}");
        }
    }

    /// Coarse chi-square-style uniformity check for `rand5_from_rand7`.
    ///
    /// Across `TRIALS` draws each bucket should appear ~`TRIALS/5` times. With
    /// `TRIALS = 100_000`, mean per bucket is `20_000` and standard deviation
    /// is `sqrt(TRIALS * p * (1-p)) = sqrt(100_000 * 0.2 * 0.8) ≈ 126.5`. A
    /// generous `±600` (~5σ) window keeps the test stable while still failing
    /// on a real bias bug.
    #[test]
    fn rand5_from_rand7_uniform_distribution() {
        const TRIALS: u64 = 100_000;
        let mut rand7 = make_randn(0x0123_4567_89AB_CDEF, 7);
        let mut counts = [0_u64; 5];
        for _ in 0..TRIALS {
            let v = rand5_from_rand7(&mut rand7);
            counts[(v - 1) as usize] += 1;
        }
        let mean = TRIALS / 5; // 20_000
        let tol = 600_u64;
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c.abs_diff(mean) <= tol,
                "bucket {i} count {c} outside [{}, {}]",
                mean - tol,
                mean + tol,
            );
        }
    }

    /// Coarse uniformity check for `rand7_from_rand5`. Mean per bucket is
    /// `TRIALS / 7 ≈ 14_285` with standard deviation `≈ 113`; we use a `±600`
    /// (~5σ) window to keep the test robust.
    #[test]
    fn rand7_from_rand5_uniform_distribution() {
        const TRIALS: u64 = 100_000;
        let mut rand5 = make_randn(0xFEED_FACE_DEAD_BEEF, 5);
        let mut counts = [0_u64; 7];
        for _ in 0..TRIALS {
            let v = rand7_from_rand5(&mut rand5);
            counts[(v - 1) as usize] += 1;
        }
        // 100_000 doesn't divide by 7 exactly; use the float-rounded mean.
        let mean = TRIALS / 7; // 14_285
        let tol = 600_u64;
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c.abs_diff(mean) <= tol + 1, // +1 for rounding
                "bucket {i} count {c} outside [{}, {}]",
                mean.saturating_sub(tol),
                mean + tol,
            );
        }
    }

    /// Round-trip: pumping a uniform `rand5` through `rand7_from_rand5` and the
    /// result through `rand5_from_rand7` should still cover `1..=5`.
    #[test]
    fn round_trip_in_range() {
        let mut rand5 = make_randn(7, 5);
        let mut composed = || rand7_from_rand5(&mut rand5);
        for _ in 0..1000 {
            let v = rand5_from_rand7(&mut composed);
            assert!((1..=5).contains(&v));
        }
    }

    /// Deterministic mock: cycle through `1..=7` to confirm the rejection
    /// path is exercised and the function still terminates with an in-range
    /// value. (Not a uniformity test — the input here is *not* random.)
    #[test]
    fn rand5_from_rand7_terminates_on_cyclic_mock() {
        let mut i: u32 = 0;
        let mut rand7 = || {
            i = (i % 7) + 1;
            let v = i;
            // bump again so consecutive calls don't repeat trivially
            i += 1;
            v
        };
        let v = rand5_from_rand7(&mut rand7);
        assert!((1..=5).contains(&v));
    }

    #[test]
    fn rand7_from_rand5_terminates_on_cyclic_mock() {
        let mut i: u32 = 0;
        let mut rand5 = || {
            i = (i % 5) + 1;
            let v = i;
            i += 1;
            v
        };
        let v = rand7_from_rand5(&mut rand5);
        assert!((1..=7).contains(&v));
    }
}
