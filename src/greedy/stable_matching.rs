//! Gale-Shapley deferred-acceptance algorithm for the stable marriage problem.
//!
//! Given two equal-size sides — `n` men and `n` women — each ranking every
//! member of the other side as a strict total order, a *matching* pairs each
//! man with a distinct woman. A pair `(m, w)` is *blocking* if `m` and `w`
//! prefer each other over their assigned partners; the matching is *stable*
//! when no blocking pair exists. Gale and Shapley (1962) proved that a stable
//! matching always exists and gave the following constructive algorithm.
//!
//! Algorithm: while some man `m` is free and has not yet proposed to every
//! woman, he proposes to the next woman `w` on his preference list. If `w` is
//! free she tentatively accepts; otherwise she compares `m` to her current
//! partner `m'` and keeps whichever she prefers, freeing the other. The
//! process terminates with a stable matching, and that matching is
//! man-optimal (each man gets the best partner he can in *any* stable
//! matching) and simultaneously woman-pessimal.
//!
//! Complexity: `O(n²)` time and `O(n²)` space. Each man proposes at most `n`
//! times (`O(n²)` proposals total); a precomputed inverse-rank table
//! `rank[w][m]` lets each woman compare two suitors in `O(1)`.
//!
//! Edge cases:
//! - `n == 0` returns an empty matching.
//! - `n == 1` returns `[0]`.

use std::collections::VecDeque;

/// Returns the man-optimal stable matching produced by Gale-Shapley.
///
/// `men_prefs[m]` is man `m`'s preference list — a permutation of `0..n`,
/// most preferred first. `women_prefs[w]` is the analogous list for woman
/// `w`. Both sides have the same size `n`. The returned vector
/// `match_man` has length `n` and satisfies `match_man[m] = w`, meaning
/// woman `w` is matched to man `m`.
///
/// Men propose; women hold the best current proposal. The result is the
/// unique man-optimal (and woman-pessimal) stable matching.
///
/// Time: `O(n²)`. Space: `O(n²)` for the inverse-rank table.
///
/// # Panics
///
/// Panics if `men_prefs.len() != women_prefs.len()` or if any preference
/// list does not have length `n`. Preference lists are assumed to be valid
/// permutations of `0..n`; malformed input may produce an incorrect matching
/// rather than a panic.
#[must_use]
pub fn gale_shapley(men_prefs: &[Vec<usize>], women_prefs: &[Vec<usize>]) -> Vec<usize> {
    let n = men_prefs.len();
    assert_eq!(
        women_prefs.len(),
        n,
        "men_prefs and women_prefs must have equal length"
    );
    if n == 0 {
        return Vec::new();
    }
    for prefs in men_prefs.iter().chain(women_prefs.iter()) {
        assert_eq!(prefs.len(), n, "every preference list must have length n");
    }

    // rank[w][m] = position of man m in woman w's preference list (lower is
    // better). This lets a woman compare two suitors in O(1).
    let mut rank = vec![vec![0_usize; n]; n];
    for (w, prefs) in women_prefs.iter().enumerate() {
        for (pos, &m) in prefs.iter().enumerate() {
            rank[w][m] = pos;
        }
    }

    // For each man, the index into his preference list of the next woman he
    // will propose to.
    let mut next_proposal = vec![0_usize; n];
    // match_woman[w] = Some(m) if woman w is currently engaged to m.
    let mut match_woman: Vec<Option<usize>> = vec![None; n];
    // match_man[m] = w if man m is engaged; sentinel n means "free".
    let mut match_man = vec![n; n];

    // Free men, in FIFO order. Initially every man is free.
    let mut free: VecDeque<usize> = (0..n).collect();

    while let Some(m) = free.pop_front() {
        // Man m proposes to the next woman on his list.
        let w = men_prefs[m][next_proposal[m]];
        next_proposal[m] += 1;

        match match_woman[w] {
            None => {
                // w is free: tentative engagement.
                match_woman[w] = Some(m);
                match_man[m] = w;
            }
            Some(current) => {
                if rank[w][m] < rank[w][current] {
                    // w prefers m over her current partner: switch.
                    match_woman[w] = Some(m);
                    match_man[m] = w;
                    match_man[current] = n;
                    free.push_back(current);
                } else {
                    // w rejects m; he stays free and will try the next woman.
                    free.push_back(m);
                }
            }
        }
    }

    match_man
}

#[cfg(test)]
mod tests {
    use super::gale_shapley;

    /// Verify that a candidate matching is actually a permutation, i.e. each
    /// woman is matched to exactly one man.
    fn is_perfect_matching(match_man: &[usize]) -> bool {
        let n = match_man.len();
        let mut seen = vec![false; n];
        for &w in match_man {
            if w >= n || seen[w] {
                return false;
            }
            seen[w] = true;
        }
        true
    }

    /// Returns true iff the matching has no blocking pair under the given
    /// preferences. A blocking pair `(m, w)` is one where `m` prefers `w`
    /// over his match and `w` prefers `m` over hers.
    fn is_stable(
        men_prefs: &[Vec<usize>],
        women_prefs: &[Vec<usize>],
        match_man: &[usize],
    ) -> bool {
        let n = match_man.len();

        // Precompute ranks for O(1) preference lookups.
        let mut rank_man = vec![vec![0_usize; n]; n];
        for (m, prefs) in men_prefs.iter().enumerate() {
            for (pos, &w) in prefs.iter().enumerate() {
                rank_man[m][w] = pos;
            }
        }
        let mut rank_woman = vec![vec![0_usize; n]; n];
        for (w, prefs) in women_prefs.iter().enumerate() {
            for (pos, &m) in prefs.iter().enumerate() {
                rank_woman[w][m] = pos;
            }
        }
        let mut match_woman = vec![0_usize; n];
        for (m, &w) in match_man.iter().enumerate() {
            match_woman[w] = m;
        }

        for m in 0..n {
            for w in 0..n {
                let m_partner = match_man[m];
                let w_partner = match_woman[w];
                // (m, w) is a blocking pair iff each strictly prefers the
                // other over their current partner.
                if rank_man[m][w] < rank_man[m][m_partner]
                    && rank_woman[w][m] < rank_woman[w][w_partner]
                {
                    return false;
                }
            }
        }
        true
    }

    /// Brute-force enumerate every stable matching by trying all `n!`
    /// permutations and keeping those with no blocking pair.
    fn all_stable_matchings(
        men_prefs: &[Vec<usize>],
        women_prefs: &[Vec<usize>],
    ) -> Vec<Vec<usize>> {
        let n = men_prefs.len();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut out = Vec::new();
        permute(&mut perm, 0, &mut |p| {
            if is_stable(men_prefs, women_prefs, p) {
                out.push(p.to_vec());
            }
        });
        out
    }

    fn permute(perm: &mut Vec<usize>, k: usize, visit: &mut dyn FnMut(&[usize])) {
        if k == perm.len() {
            visit(perm);
            return;
        }
        for i in k..perm.len() {
            perm.swap(k, i);
            permute(perm, k + 1, visit);
            perm.swap(k, i);
        }
    }

    /// Tiny xorshift64 PRNG so property tests are deterministic without a
    /// dependency on `rand`.
    struct XorShift(u64);
    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn shuffle(&mut self, v: &mut [usize]) {
            // Fisher-Yates.
            for i in (1..v.len()).rev() {
                let j = (self.next_u64() as usize) % (i + 1);
                v.swap(i, j);
            }
        }
    }

    fn random_prefs(n: usize, rng: &mut XorShift) -> Vec<Vec<usize>> {
        (0..n)
            .map(|_| {
                let mut p: Vec<usize> = (0..n).collect();
                rng.shuffle(&mut p);
                p
            })
            .collect()
    }

    #[test]
    fn n_one() {
        let men = vec![vec![0]];
        let women = vec![vec![0]];
        assert_eq!(gale_shapley(&men, &women), vec![0]);
    }

    #[test]
    fn n_zero() {
        let men: Vec<Vec<usize>> = vec![];
        let women: Vec<Vec<usize>> = vec![];
        assert!(gale_shapley(&men, &women).is_empty());
    }

    #[test]
    fn n_two_both_prefer_zero() {
        // Both men list woman 0 first; both women list man 0 first. Man 0
        // wins the contested pair.
        let men = vec![vec![0, 1], vec![0, 1]];
        let women = vec![vec![0, 1], vec![0, 1]];
        assert_eq!(gale_shapley(&men, &women), vec![0, 1]);
    }

    #[test]
    fn classic_3x3_textbook() {
        // Standard 3-couple example. Men's preferences:
        //   M0: W0 W1 W2
        //   M1: W1 W0 W2
        //   M2: W0 W1 W2
        // Women's preferences:
        //   W0: M1 M0 M2
        //   W1: M0 M1 M2
        //   W2: M0 M1 M2
        // Man-optimal stable matching: M0-W0, M1-W1, M2-W2.
        let men = vec![vec![0, 1, 2], vec![1, 0, 2], vec![0, 1, 2]];
        let women = vec![vec![1, 0, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let result = gale_shapley(&men, &women);
        assert_eq!(result, vec![0, 1, 2]);
        assert!(is_perfect_matching(&result));
        assert!(is_stable(&men, &women, &result));
    }

    #[test]
    fn knuth_4x4_example() {
        // Knuth's classic 4x4 example (adapted from "Mariages stables").
        // Men's preferences:
        //   M0: W0 W1 W2 W3
        //   M1: W1 W0 W2 W3
        //   M2: W0 W1 W2 W3
        //   M3: W3 W0 W1 W2
        // Women's preferences:
        //   W0: M3 M0 M1 M2
        //   W1: M0 M2 M1 M3
        //   W2: M0 M1 M2 M3
        //   W3: M3 M0 M1 M2
        let men = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 2, 3],
            vec![0, 1, 2, 3],
            vec![3, 0, 1, 2],
        ];
        let women = vec![
            vec![3, 0, 1, 2],
            vec![0, 2, 1, 3],
            vec![0, 1, 2, 3],
            vec![3, 0, 1, 2],
        ];
        let result = gale_shapley(&men, &women);
        assert!(is_perfect_matching(&result));
        assert!(is_stable(&men, &women, &result));
        // Man-optimality: each man's match is at least as good (lower rank
        // index) as his match in any other stable matching.
        let stable = all_stable_matchings(&men, &women);
        assert!(stable.contains(&result));
        for other in &stable {
            for m in 0..men.len() {
                let mine = men[m].iter().position(|&w| w == result[m]).unwrap();
                let theirs = men[m].iter().position(|&w| w == other[m]).unwrap();
                assert!(
                    mine <= theirs,
                    "not man-optimal: man {m} got rank {mine} but rank {theirs} exists"
                );
            }
        }
    }

    #[test]
    fn multiple_stable_matchings_man_optimal_dominates() {
        // Cyclic example: each man's top choice is a different woman, but the
        // women rank the men in the opposite cycle — so the man-optimal
        // matching gives every man his first choice, while the woman-optimal
        // matching gives every woman her first choice.
        // Men:   M0: W0 W1 W2   Women: W0: M2 M1 M0
        //        M1: W1 W2 W0          W1: M0 M2 M1
        //        M2: W2 W0 W1          W2: M1 M0 M2
        let men = vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]];
        let women = vec![vec![2, 1, 0], vec![0, 2, 1], vec![1, 0, 2]];
        let result = gale_shapley(&men, &women);
        assert_eq!(result, vec![0, 1, 2]);
        assert!(is_stable(&men, &women, &result));

        let stable = all_stable_matchings(&men, &women);
        assert!(stable.len() >= 2, "expected at least two stable matchings");
        assert!(stable.contains(&vec![0, 1, 2]));
        assert!(stable.contains(&vec![1, 2, 0]));
        // Result must dominate every other stable matching for every man.
        for other in &stable {
            for m in 0..men.len() {
                let mine = men[m].iter().position(|&w| w == result[m]).unwrap();
                let theirs = men[m].iter().position(|&w| w == other[m]).unwrap();
                assert!(mine <= theirs);
            }
        }
    }

    #[test]
    fn property_no_blocking_pair_random_small_inputs() {
        let mut rng = XorShift::new(0x00C0_FFEE_u64);
        for n in 1..=6 {
            for _ in 0..40 {
                let men = random_prefs(n, &mut rng);
                let women = random_prefs(n, &mut rng);
                let result = gale_shapley(&men, &women);
                assert!(
                    is_perfect_matching(&result),
                    "not a perfect matching for n={n}"
                );
                assert!(
                    is_stable(&men, &women, &result),
                    "blocking pair found for n={n}, men={men:?}, women={women:?}, result={result:?}"
                );
            }
        }
    }

    #[test]
    fn property_man_optimal_woman_pessimal_random_small_inputs() {
        let mut rng = XorShift::new(0xDEAD_BEEF_u64);
        for n in 1..=5 {
            for _ in 0..30 {
                let men = random_prefs(n, &mut rng);
                let women = random_prefs(n, &mut rng);
                let result = gale_shapley(&men, &women);
                let stable = all_stable_matchings(&men, &women);
                assert!(stable.contains(&result));

                // Man-optimal: every man's partner in `result` is ranked at
                // least as high (lower index) as in any other stable
                // matching. Woman-pessimal: every woman's partner in
                // `result` is ranked at least as low as in any other.
                let mut match_woman_result = vec![0_usize; n];
                for (m, &w) in result.iter().enumerate() {
                    match_woman_result[w] = m;
                }
                for other in &stable {
                    let mut match_woman_other = vec![0_usize; n];
                    for (m, &w) in other.iter().enumerate() {
                        match_woman_other[w] = m;
                    }
                    for m in 0..n {
                        let mine = men[m].iter().position(|&w| w == result[m]).unwrap();
                        let theirs = men[m].iter().position(|&w| w == other[m]).unwrap();
                        assert!(mine <= theirs, "not man-optimal");
                    }
                    for w in 0..n {
                        let mine = women[w]
                            .iter()
                            .position(|&m| m == match_woman_result[w])
                            .unwrap();
                        let theirs = women[w]
                            .iter()
                            .position(|&m| m == match_woman_other[w])
                            .unwrap();
                        assert!(mine >= theirs, "not woman-pessimal");
                    }
                }
            }
        }
    }
}
