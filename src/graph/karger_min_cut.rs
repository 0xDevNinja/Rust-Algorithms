//! Karger's randomized minimum cut algorithm for undirected multigraphs.
//!
//! A single trial repeatedly picks a uniformly random edge from the
//! current multigraph and contracts it (merging its endpoints into one
//! supervertex and discarding any self-loops created in the process)
//! until only two supervertices remain. The number of edges still
//! running between those two supervertices is a candidate for the
//! global minimum cut. Each contraction is O(α(V)) via union–find, so
//! one trial is O(E · α(V)).
//!
//! ## Probabilistic correctness
//!
//! For a graph with `n` vertices any one trial returns the true minimum
//! cut with probability ≥ `2 / (n · (n - 1))`. Running `T` independent
//! trials and keeping the smallest cut therefore drives the failure
//! probability to `(1 - 2 / (n · (n - 1)))^T`. The classical advice is
//! `T = Θ(n² · ln n)`, which makes the failure probability `O(1/n)`. We
//! leave the trial count to the caller so they can trade accuracy for
//! runtime; the tests in this module use a generous fixed count to keep
//! them deterministic.
//!
//! ## Determinism
//!
//! Randomness comes from a small `XorShift64` PRNG seeded by the
//! caller. The same `(n, edges, iterations, seed)` tuple therefore
//! always produces the same answer, which makes Karger's behaviour
//! reproducible in unit tests and downstream callers without pulling in
//! the `rand` crate.

use crate::data_structures::union_find::UnionFind;

/// Tiny deterministic `XorShift64` PRNG. Adequate for randomized
/// contraction; not for cryptographic use.
struct XorShift64(u64);

impl XorShift64 {
    const fn new(seed: u64) -> Self {
        // XorShift collapses to zero on a zero seed, which would jam
        // the generator at zero forever — fall back to a fixed nonzero
        // constant in that case.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `0..bound`. `bound` must be non-zero.
    const fn gen_range(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Estimates the minimum cut of an undirected multigraph using Karger's
/// randomized contraction algorithm.
///
/// `n` is the vertex count (vertices are `0..n`). `edges` is an
/// undirected edge multiset; parallel edges are honoured (and matter
/// for the cut count) and self-loops are filtered out before any
/// trials run. `iterations` independent contraction trials are
/// executed and the smallest cut observed across all of them is
/// returned. `seed` seeds a deterministic PRNG so tests and downstream
/// callers get reproducible answers.
///
/// Returns `0` for `n < 2` or when no usable edges remain after
/// self-loop filtering. With high probability the returned value is
/// the true minimum cut once `iterations` is `Ω(n² · ln n)`.
pub fn karger_min_cut(n: usize, edges: &[(usize, usize)], iterations: usize, seed: u64) -> usize {
    if n < 2 {
        return 0;
    }

    // Strip self-loops once up front — they can never cross any cut
    // and would otherwise waste contraction draws.
    let usable: Vec<(usize, usize)> = edges
        .iter()
        .copied()
        .filter(|&(u, v)| u != v && u < n && v < n)
        .collect();

    if usable.is_empty() || iterations == 0 {
        return 0;
    }

    let mut rng = XorShift64::new(seed);
    let mut best = usize::MAX;

    for _ in 0..iterations {
        let cut = single_trial(n, &usable, &mut rng);
        if cut < best {
            best = cut;
        }
    }

    if best == usize::MAX {
        0
    } else {
        best
    }
}

/// Runs one independent Karger contraction trial and returns the size
/// of the cut between the two surviving supervertices.
fn single_trial(n: usize, edges: &[(usize, usize)], rng: &mut XorShift64) -> usize {
    let mut dsu = UnionFind::new(n);
    let mut remaining = n;

    // Repeatedly pick a random edge whose endpoints still belong to
    // distinct supervertices and contract it. We sample uniformly from
    // the original edge list and skip already-collapsed self-loops; the
    // expected number of skipped draws stays O(E) because each
    // successful contraction reduces `remaining` by one.
    while remaining > 2 {
        let (u, v) = edges[rng.gen_range(edges.len())];
        if dsu.union(u, v) {
            remaining -= 1;
        }
    }

    // Cut size = number of original edges still crossing the two
    // surviving supervertices.
    edges
        .iter()
        .filter(|&&(u, v)| dsu.find(u) != dsu.find(v))
        .count()
}

#[cfg(test)]
mod tests {
    use super::karger_min_cut;

    #[test]
    fn trivial_sizes() {
        assert_eq!(karger_min_cut(0, &[], 10, 1), 0);
        assert_eq!(karger_min_cut(1, &[], 10, 1), 0);
        // n >= 2 but no edges → no cut.
        assert_eq!(karger_min_cut(5, &[], 10, 1), 0);
    }

    #[test]
    fn triangle_k3_min_cut_is_two() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        assert_eq!(karger_min_cut(3, &edges, 50, 0x00C0_FFEE), 2);
    }

    #[test]
    fn complete_k4_min_cut_is_three() {
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        assert_eq!(karger_min_cut(4, &edges, 200, 0xDEAD_BEEF), 3);
    }

    #[test]
    fn bridge_between_two_cliques_is_one() {
        // Two K4 cliques on {0,1,2,3} and {4,5,6,7}, joined by the
        // single bridge edge (3, 4) — global min cut is 1.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for a in 0..4 {
            for b in (a + 1)..4 {
                edges.push((a, b));
            }
        }
        for a in 4..8 {
            for b in (a + 1)..8 {
                edges.push((a, b));
            }
        }
        edges.push((3, 4));
        assert_eq!(karger_min_cut(8, &edges, 400, 0x1234_5678), 1);
    }

    #[test]
    fn parallel_edges_count_toward_cut() {
        // Three vertices in a line with multiplicity. The 0-1 boundary
        // has two parallel edges and 1-2 has three; the min cut is the
        // smaller multiplicity, i.e. 2.
        let edges = vec![(0, 1), (0, 1), (1, 2), (1, 2), (1, 2)];
        assert_eq!(karger_min_cut(3, &edges, 100, 42), 2);
    }

    #[test]
    fn self_loops_are_filtered() {
        // The triangle's true min cut is 2; sprinkling self-loops in
        // must not change the answer (they cross no cut and must be
        // dropped before contraction).
        let edges = vec![(0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (2, 2)];
        assert_eq!(karger_min_cut(3, &edges, 80, 7), 2);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        // Same inputs and seed must produce the same answer across
        // calls — the whole point of seeding our own PRNG.
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
        let a = karger_min_cut(4, &edges, 50, 99);
        let b = karger_min_cut(4, &edges, 50, 99);
        assert_eq!(a, b);
    }
}
