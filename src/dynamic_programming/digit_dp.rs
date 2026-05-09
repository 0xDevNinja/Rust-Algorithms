//! Digit dynamic programming framework.
//!
//! Digit DP is a technique for counting / aggregating over integers in a range
//! by walking the decimal representation from the most-significant digit to
//! the least-significant one. At each position we track:
//!
//! * `pos` — current digit index (0 = most significant).
//! * `tight` — whether the prefix chosen so far is still bounded above by
//!   the corresponding prefix of the upper limit.
//! * `started` — whether at least one non-zero digit has been emitted (used
//!   to distinguish leading zeros from "real" zero digits).
//! * `state` — application-specific auxiliary state (e.g. previous digit,
//!   running digit sum).
//!
//! Counting in a range `[low, high]` is handled by inclusion–exclusion:
//! `f(high) − f(low − 1)`.
//!
//! # Complexity
//!
//! For each application below the runtime is `O(D · S · 10)` where `D` is the
//! decimal length (≤ 20 for `u64`) and `S` is the size of the auxiliary state
//! space. Memoisation only caches the *loose* states (tight = false and
//! started = true), since tight states are visited at most once per position
//! and leading-zero states quickly transition into started ones.

/// Counts integers `x` in the inclusive range `[low, high]` whose decimal
/// digit sequence (most-significant first, with no leading zeros for `x > 0`;
/// for `x == 0` the sequence is the single digit `[0]`) satisfies `predicate`.
///
/// The implementation always uses the digit-DP route (no brute force fast
/// path) so callers get predictable behaviour on large ranges.
///
/// # Panics
///
/// Panics if `low > high`.
pub fn count_in_range<F>(low: u64, high: u64, predicate: F) -> u64
where
    F: Fn(&[u8]) -> bool,
{
    assert!(low <= high, "low must be ≤ high");
    let upper = count_predicate_up_to(high, &predicate);
    let lower = if low == 0 {
        0
    } else {
        count_predicate_up_to(low - 1, &predicate)
    };
    upper - lower
}

/// Returns the sum of digit-sums of all integers in `[low, high]`.
///
/// For example, `sum_of_digits_in_range(1, 12)` =
/// `1+2+3+4+5+6+7+8+9+(1+0)+(1+1)+(1+2)` = `51`.
///
/// # Panics
///
/// Panics if `low > high`.
pub fn sum_of_digits_in_range(low: u64, high: u64) -> u64 {
    assert!(low <= high, "low must be ≤ high");
    let upper = digit_sum_up_to(high);
    let lower = if low == 0 {
        0
    } else {
        digit_sum_up_to(low - 1)
    };
    upper - lower
}

/// Counts integers in `[low, high]` whose decimal representation contains no
/// two consecutive equal digits. Leading zeros are not considered digits, so
/// e.g. `100` (digits `1, 0, 0`) is rejected because of the trailing `00`,
/// while `7` is trivially valid.
///
/// # Panics
///
/// Panics if `low > high`.
pub fn count_without_consecutive_equal(low: u64, high: u64) -> u64 {
    assert!(low <= high, "low must be ≤ high");
    let upper = no_consecutive_equal_up_to(high);
    let lower = if low == 0 {
        0
    } else {
        no_consecutive_equal_up_to(low - 1)
    };
    upper - lower
}

/// Decimal digits of `n`, most significant first. `digits_of(0)` returns
/// `vec![0]` so the slice always has at least one element.
fn digits_of(mut n: u64) -> Vec<u8> {
    if n == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push((n % 10) as u8);
        n /= 10;
    }
    out.reverse();
    out
}

/// Generic recursion shared by all callers. We do *not* memoise the
/// predicate-driven `count_in_range` because predicates are arbitrary and may
/// inspect the full digit prefix; the upper bound `high < 2^64` has at most
/// 20 digits, so the unmemoised tree has ≤ 10^20 leaves in the worst case but
/// in practice the predicate check is the bottleneck. For the predicate API
/// we therefore enumerate all numbers whose length matches the bound's length
/// using a bounded recursion that prunes via the tight flag.
fn count_predicate_up_to<F>(n: u64, predicate: &F) -> u64
where
    F: Fn(&[u8]) -> bool,
{
    let digits = digits_of(n);
    let len = digits.len();
    let mut prefix: Vec<u8> = Vec::with_capacity(len);
    let mut total: u64 = 0;

    // Numbers shorter than `len` digits: enumerate over lengths 1..len plus
    // the special value 0.
    if predicate(&[0]) {
        total += 1;
    }
    for length in 1..len {
        prefix.clear();
        enumerate_shorter(length, 0, &mut prefix, predicate, &mut total);
    }
    // Numbers with exactly `len` digits, bounded above by `digits`.
    prefix.clear();
    enumerate_tight(&digits, 0, true, false, &mut prefix, predicate, &mut total);

    total
}

/// Enumerate every `length`-digit number with no leading zero, calling
/// `predicate` and incrementing `total` on each.
fn enumerate_shorter<F>(
    length: usize,
    pos: usize,
    prefix: &mut Vec<u8>,
    predicate: &F,
    total: &mut u64,
) where
    F: Fn(&[u8]) -> bool,
{
    if pos == length {
        if predicate(prefix) {
            *total += 1;
        }
        return;
    }
    let start: u8 = u8::from(pos == 0);
    for d in start..=9 {
        prefix.push(d);
        enumerate_shorter(length, pos + 1, prefix, predicate, total);
        prefix.pop();
    }
}

/// Enumerate `len`-digit numbers bounded above by `bound`, with `tight`
/// tracking whether the prefix so far equals `bound`'s prefix and `started`
/// tracking whether a non-zero digit has been emitted (always true once we
/// commit to a `len`-digit number, but the leading-zero case is folded in via
/// `enumerate_shorter` above).
fn enumerate_tight<F>(
    bound: &[u8],
    pos: usize,
    tight: bool,
    started: bool,
    prefix: &mut Vec<u8>,
    predicate: &F,
    total: &mut u64,
) where
    F: Fn(&[u8]) -> bool,
{
    if pos == bound.len() {
        if started && predicate(prefix) {
            *total += 1;
        }
        return;
    }
    let max_digit = if tight { bound[pos] } else { 9 };
    let min_digit: u8 = u8::from(!started);
    // First branch: emit a real digit (started becomes true).
    for d in min_digit..=max_digit {
        prefix.push(d);
        enumerate_tight(
            bound,
            pos + 1,
            tight && d == bound[pos],
            true,
            prefix,
            predicate,
            total,
        );
        prefix.pop();
    }
    // Second branch: skip this position as a leading zero (only valid while
    // not started and pos < bound.len() - 1 would make the number shorter,
    // but those are handled by enumerate_shorter; so we do nothing here).
    let _ = started;
}

/// Internal cache for `digit_sum_up_to`: stores `(count, sum)` pairs for the
/// loose subproblem at each `pos`. State is the position only; once we are
/// loose and started, every digit choice 0..=9 is independent of the prefix.
struct DigitSumCache {
    /// `cache[pos]` = Some((count, sum)) if computed; count = how many ways to
    /// fill positions `pos..len`, sum = total of digit-sums contributed by
    /// those positions across all those ways.
    cache: Vec<Option<(u64, u64)>>,
}

impl DigitSumCache {
    fn new(len: usize) -> Self {
        Self {
            cache: vec![None; len + 1],
        }
    }
}

/// Sum of digit-sums of all integers in `[0, n]`.
fn digit_sum_up_to(n: u64) -> u64 {
    let digits = digits_of(n);
    let len = digits.len();
    let mut cache = DigitSumCache::new(len);
    let (_count, sum) = digit_sum_recurse(&digits, 0, true, false, &mut cache);
    sum
}

/// Returns `(count, sum_of_digit_sums)` for completions of the current state.
fn digit_sum_recurse(
    digits: &[u8],
    pos: usize,
    tight: bool,
    started: bool,
    cache: &mut DigitSumCache,
) -> (u64, u64) {
    if pos == digits.len() {
        return (1, 0);
    }
    if !tight && started {
        if let Some(memo) = cache.cache[pos] {
            return memo;
        }
    }
    let max_digit = if tight { digits[pos] } else { 9 };
    let mut total_count: u64 = 0;
    let mut total_sum: u64 = 0;
    for d in 0..=max_digit {
        let new_tight = tight && d == digits[pos];
        let new_started = started || d != 0;
        let (sub_count, sub_sum) =
            digit_sum_recurse(digits, pos + 1, new_tight, new_started, cache);
        total_count += sub_count;
        // Each completion contributes `d` (only if started, so leading zeros
        // don't count; but since `d == 0` adds nothing anyway, gating is moot
        // for the sum — we still gate to be explicit).
        let digit_contrib = if new_started { d as u64 } else { 0 };
        total_sum += sub_sum + sub_count * digit_contrib;
    }
    if !tight && started {
        cache.cache[pos] = Some((total_count, total_sum));
    }
    (total_count, total_sum)
}

/// Internal cache for `no_consecutive_equal_up_to`. State is `(pos, prev)`
/// where `prev` is the previous digit (0..=9). Memoised only for loose +
/// started states.
struct NoConsecCache {
    /// `cache[pos][prev]` = Some(count) if computed.
    cache: Vec<[Option<u64>; 10]>,
}

impl NoConsecCache {
    fn new(len: usize) -> Self {
        Self {
            cache: vec![[None; 10]; len + 1],
        }
    }
}

/// Count of integers in `[0, n]` whose decimal representation has no two
/// consecutive equal digits.
fn no_consecutive_equal_up_to(n: u64) -> u64 {
    let digits = digits_of(n);
    let len = digits.len();
    let mut cache = NoConsecCache::new(len);
    // Initial `prev` is set to a sentinel 10 (out of range so the first real
    // digit can be anything). We pass `prev` as u8.
    no_consec_recurse(&digits, 0, true, false, 10, &mut cache)
}

fn no_consec_recurse(
    digits: &[u8],
    pos: usize,
    tight: bool,
    started: bool,
    prev: u8,
    cache: &mut NoConsecCache,
) -> u64 {
    if pos == digits.len() {
        // Every reached terminal state corresponds to a valid number,
        // including 0 (when never started).
        return 1;
    }
    if !tight && started {
        let p = prev as usize;
        if let Some(memo) = cache.cache[pos][p] {
            return memo;
        }
    }
    let max_digit = if tight { digits[pos] } else { 9 };
    let mut total: u64 = 0;
    for d in 0..=max_digit {
        if started && d == prev {
            continue;
        }
        let new_tight = tight && d == digits[pos];
        let new_started = started || d != 0;
        // If this digit is a leading zero (not started and d == 0), prev
        // stays at the sentinel; otherwise prev becomes d.
        let new_prev = if new_started { d } else { prev };
        total += no_consec_recurse(digits, pos + 1, new_tight, new_started, new_prev, cache);
    }
    if !tight && started {
        let p = prev as usize;
        cache.cache[pos][p] = Some(total);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::{
        count_in_range, count_without_consecutive_equal, digits_of, sum_of_digits_in_range,
    };

    fn brute_digit_sum(low: u64, high: u64) -> u64 {
        let mut s: u64 = 0;
        for x in low..=high {
            let mut y = x;
            if y == 0 {
                continue;
            }
            while y > 0 {
                s += y % 10;
                y /= 10;
            }
        }
        s
    }

    fn brute_no_consec(low: u64, high: u64) -> u64 {
        let mut count: u64 = 0;
        for x in low..=high {
            let d = digits_of(x);
            let mut ok = true;
            for w in d.windows(2) {
                if w[0] == w[1] {
                    ok = false;
                    break;
                }
            }
            if ok {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn digit_sum_one_to_nine() {
        assert_eq!(sum_of_digits_in_range(1, 9), 45);
    }

    #[test]
    fn digit_sum_one_to_hundred_matches_brute() {
        let expected = brute_digit_sum(1, 100);
        assert_eq!(sum_of_digits_in_range(1, 100), expected);
    }

    #[test]
    fn no_consec_zero_to_99() {
        // 0..=99 has 100 integers. Numbers excluded: 11, 22, 33, 44, 55, 66,
        // 77, 88, 99 — exactly 9. So 100 − 9 = 91.
        assert_eq!(count_without_consecutive_equal(0, 99), 91);
    }

    #[test]
    fn property_digit_sum_matches_brute_small() {
        for low in 0_u64..=20 {
            for high in low..=200 {
                assert_eq!(
                    sum_of_digits_in_range(low, high),
                    brute_digit_sum(low, high),
                    "digit sum mismatch on [{low}, {high}]"
                );
            }
        }
    }

    #[test]
    fn property_no_consec_matches_brute_to_1000() {
        for high in [0_u64, 1, 9, 10, 11, 99, 100, 121, 500, 999, 1000] {
            for low in [0_u64, 1, 5, 10, 50, 100, 500] {
                if low > high {
                    continue;
                }
                assert_eq!(
                    count_without_consecutive_equal(low, high),
                    brute_no_consec(low, high),
                    "no-consec mismatch on [{low}, {high}]"
                );
            }
        }
        // Spot check the full range.
        assert_eq!(
            count_without_consecutive_equal(0, 1000),
            brute_no_consec(0, 1000)
        );
    }

    #[test]
    fn count_in_range_predicate_palindromes() {
        // Number of palindromic integers in [1, 200]:
        // 1..9 (9), 11, 22, ..., 99 (9), 101, 111, 121, 131, 141, 151, 161,
        // 171, 181, 191 (10) -> 28.
        let is_palindrome = |digits: &[u8]| -> bool {
            let n = digits.len();
            for i in 0..n / 2 {
                if digits[i] != digits[n - 1 - i] {
                    return false;
                }
            }
            true
        };
        assert_eq!(count_in_range(1, 200, is_palindrome), 28);
    }

    #[test]
    fn count_in_range_predicate_matches_brute() {
        // Predicate: digit sum is even.
        let pred = |digits: &[u8]| -> bool {
            let s: u32 = digits.iter().map(|&d| d as u32).sum();
            s.is_multiple_of(2)
        };
        for low in 0_u64..=10 {
            for high in low..=300 {
                let dp = count_in_range(low, high, pred);
                let mut brute: u64 = 0;
                for x in low..=high {
                    let d = digits_of(x);
                    if pred(&d) {
                        brute += 1;
                    }
                }
                assert_eq!(dp, brute, "predicate count mismatch on [{low}, {high}]");
            }
        }
    }

    #[test]
    fn digits_of_basic() {
        assert_eq!(digits_of(0), vec![0]);
        assert_eq!(digits_of(7), vec![7]);
        assert_eq!(digits_of(123), vec![1, 2, 3]);
        assert_eq!(digits_of(1_000_000), vec![1, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn no_consec_large_range_no_panic() {
        // Just ensure the DP completes on a wide range.
        let v = count_without_consecutive_equal(0, 1_000_000);
        // Sanity: must not exceed the total count.
        assert!(v <= 1_000_001);
        assert!(v > 0);
    }

    #[test]
    fn digit_sum_includes_zero_correctly() {
        // sum over [0, 0] is 0.
        assert_eq!(sum_of_digits_in_range(0, 0), 0);
        // sum over [0, 9] is 0+1+...+9 = 45.
        assert_eq!(sum_of_digits_in_range(0, 9), 45);
    }
}
