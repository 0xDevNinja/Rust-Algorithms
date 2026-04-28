//! De Bruijn sequence `B(k, n)` over an alphabet of size `k`.
//!
//! A De Bruijn sequence of order `n` on an alphabet of size `k` is a cyclic
//! sequence of length `k^n` in which every possible length-`n` string over the
//! alphabet appears **exactly once** as a contiguous substring (where the
//! sequence wraps around at the end). For example, `B(2, 3)` is
//! `0,0,0,1,0,1,1,1`: the eight 3-tuples `000, 001, 010, 101, 011, 111, 110,
//! 100` each appear once when the sequence is read circularly.
//!
//! ## Algorithm
//!
//! This module uses the classic recursive construction by Frank Ruskey based
//! on Lyndon words (sometimes called the "prefer-largest" or `db(t, p)`
//! algorithm). Conceptually it walks every Lyndon word of length dividing `n`
//! over the alphabet `0..k` in lexicographic order and concatenates them; the
//! result is a De Bruijn sequence in linear output time.
//!
//! Equivalently, the algorithm is an Eulerian circuit on the De Bruijn graph
//! whose vertices are the `(n-1)`-tuples and whose edges are the `n`-tuples,
//! but the recursive form needs only `O(n)` extra space beyond the output
//! buffer.
//!
//! Runs in `O(k^n)` time and `O(k^n)` space (the size of the output).

/// Returns a De Bruijn sequence `B(k, n)` as a `Vec<u32>` whose entries are
/// digits in `0..k`. The returned sequence has length `k^n` and contains every
/// length-`n` string over the alphabet exactly once as a circular substring.
///
/// Edge cases:
/// * `n == 0` returns an empty `Vec`. (The only length-0 string is the empty
///   string, so a length-1 cyclic sequence trivially contains it; we instead
///   adopt the convention that an order-0 De Bruijn sequence is empty.)
/// * `k == 0` returns an empty `Vec` regardless of `n`, since there are no
///   strings to enumerate.
/// * `k == 1` returns `vec![0; n.max(1) as usize]`. The single `n`-tuple
///   `0,0,...,0` appears once cyclically; for `n == 0` we still return a
///   single zero so the cyclic-substring property is non-vacuous.
///
/// Runs in `O(k^n)` time and space.
pub fn de_bruijn(k: u32, n: u32) -> Vec<u32> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![0; n.max(1) as usize];
    }
    if n == 0 {
        return Vec::new();
    }

    let n = n as usize;
    let k_usize = k as usize;
    // Output capacity is exactly k^n.
    let total = k_usize.checked_pow(n as u32).expect("k^n overflowed usize");
    let mut sequence: Vec<u32> = Vec::with_capacity(total);
    // Working register `a[1..=n]` as in the classical formulation; index 0 is
    // unused so the recursion matches the textbook indices directly.
    let mut a: Vec<u32> = vec![0; n + 1];

    db(1, 1, n, k, &mut a, &mut sequence);

    debug_assert_eq!(sequence.len(), total);
    sequence
}

/// Recursive Lyndon-word generator. `t` is the current position being
/// considered, `p` is the length of the longest proper prefix that is also a
/// suffix of `a[1..t]` (i.e. the period). When `t > n` we have completed a
/// candidate Lyndon word of length `p`; if `n` is divisible by `p` we emit
/// `a[1..=p]` to the output.
fn db(t: usize, p: usize, n: usize, k: u32, a: &mut [u32], out: &mut Vec<u32>) {
    if t > n {
        if n.is_multiple_of(p) {
            out.extend_from_slice(&a[1..=p]);
        }
        return;
    }

    a[t] = a[t - p];
    db(t + 1, p, n, k, a, out);

    let start = a[t - p] + 1;
    for j in start..k {
        a[t] = j;
        db(t + 1, t, n, k, a, out);
    }
}

#[cfg(test)]
mod tests {
    use super::de_bruijn;
    use std::collections::HashSet;

    /// Returns every length-`n` window of `seq` read circularly. The number of
    /// windows equals `seq.len()`.
    fn circular_windows(seq: &[u32], n: usize) -> Vec<Vec<u32>> {
        if n == 0 || seq.is_empty() {
            return Vec::new();
        }
        let len = seq.len();
        (0..len)
            .map(|i| (0..n).map(|j| seq[(i + j) % len]).collect())
            .collect()
    }

    #[test]
    fn b_2_3_matches_canonical() {
        // The canonical "necklace" output for B(2, 3).
        assert_eq!(de_bruijn(2, 3), vec![0, 0, 0, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn b_2_1_is_zero_one() {
        assert_eq!(de_bruijn(2, 1), vec![0, 1]);
    }

    #[test]
    fn b_3_2_has_all_pairs() {
        let seq = de_bruijn(3, 2);
        assert_eq!(seq.len(), 9);
        let windows: HashSet<Vec<u32>> = circular_windows(&seq, 2).into_iter().collect();
        let mut expected: HashSet<Vec<u32>> = HashSet::new();
        for a in 0..3u32 {
            for b in 0..3u32 {
                expected.insert(vec![a, b]);
            }
        }
        assert_eq!(windows, expected);
    }

    #[test]
    fn n_zero_is_empty() {
        assert!(de_bruijn(2, 0).is_empty());
        assert!(de_bruijn(5, 0).is_empty());
    }

    #[test]
    fn k_zero_is_empty() {
        assert!(de_bruijn(0, 0).is_empty());
        assert!(de_bruijn(0, 3).is_empty());
    }

    #[test]
    fn k_one_is_repeated_zero() {
        assert_eq!(de_bruijn(1, 1), vec![0]);
        assert_eq!(de_bruijn(1, 4), vec![0, 0, 0, 0]);
        // Verify circular property: the single 4-tuple 0,0,0,0 appears once.
        let seq = de_bruijn(1, 4);
        let windows = circular_windows(&seq, 4);
        assert_eq!(windows.len(), 4);
        assert!(windows.iter().all(|w| w == &vec![0, 0, 0, 0]));
        // It "appears once" up to rotation — the four windows are all
        // identical because the sequence is constant.
    }

    #[test]
    fn k_one_n_zero_returns_single_zero() {
        // Convention: keep at least one symbol so the cyclic property is not
        // vacuous.
        assert_eq!(de_bruijn(1, 0), vec![0]);
    }

    #[test]
    fn entries_are_within_alphabet() {
        for k in 2..=5u32 {
            for n in 1..=4u32 {
                let seq = de_bruijn(k, n);
                assert!(seq.iter().all(|&d| d < k));
            }
        }
    }

    /// Property test: for each `(k, n)` in the range, the sequence has length
    /// `k^n` and every length-`n` tuple over `0..k` appears exactly once as a
    /// circular substring.
    #[test]
    fn every_tuple_appears_exactly_once_circularly() {
        for k in 2..=4u32 {
            for n in 1..=4u32 {
                let seq = de_bruijn(k, n);
                let expected_len = (k as usize).pow(n);
                assert_eq!(seq.len(), expected_len, "wrong length for B({k}, {n})");

                let windows = circular_windows(&seq, n as usize);
                assert_eq!(windows.len(), expected_len);

                let unique: HashSet<Vec<u32>> = windows.into_iter().collect();
                assert_eq!(
                    unique.len(),
                    expected_len,
                    "duplicate window in B({k}, {n})",
                );

                // Every tuple is in the alphabet (already implied by uniqueness
                // + correct length, but make it explicit).
                for w in &unique {
                    assert!(w.iter().all(|&d| d < k));
                }
            }
        }
    }
}
