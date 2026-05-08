//! Edmonds' blossom algorithm for maximum cardinality matching in
//! **general** (not necessarily bipartite) graphs.
//!
//! Given an undirected simple graph on `n` vertices and an edge list, returns
//! a maximum-cardinality matching: a largest set of edges no two of which
//! share an endpoint. Unlike Hopcroft–Karp this works on graphs containing
//! odd cycles, where a naive augmenting-path search can get stuck on a
//! "blossom" — an odd alternating cycle through the search tree.
//!
//! # Algorithm
//! For every currently unmatched vertex `r`, run a BFS on the alternating
//! tree rooted at `r`. The BFS labels each visited vertex either *even*
//! (reachable from `r` by an even-length alternating path) or *odd*
//! (reachable by an odd-length alternating path). When the BFS finds:
//!   - an unmatched even neighbour: an **augmenting path** has been found;
//!     flip every edge on it to grow the matching by one;
//!   - an even–even edge: an **odd cycle** through the tree has been found
//!     (a "blossom"). Contract every vertex on the cycle into a single
//!     super-vertex, represented by their lowest common ancestor in the
//!     tree, and continue the BFS from that super-vertex.
//!
//! Augmentation walks back to the root using two arrays:
//!   - `mate[v]`: matched partner of `v` (the matching edge);
//!   - `parent[v]`: predecessor of `v` along a non-matching tree edge.
//!     Inside a blossom, `parent` is rewritten during contraction so that
//!     the augmenting walk traverses the cycle in the right direction.
//!
//! # Complexity
//! - Time:  `O(V^3)` in the worst case (an outer loop over `V` BFS roots,
//!   each contracting at most `O(V)` blossoms, each blossom traversing
//!   `O(V)` parents). Clarity is preferred over the asymptotically faster
//!   `O(E · α(V) · √V)` Micali–Vazirani variant.
//! - Space: `O(V + E)` for adjacency list and per-BFS bookkeeping.
//!
//! # Preconditions
//! - Vertices are `0..n`. Edges with endpoints `≥ n` will panic on
//!   out-of-bounds access.
//! - Self-loops are silently ignored. Parallel edges are tolerated.

use std::collections::VecDeque;

/// Maximum-cardinality matching of the undirected graph on `n` vertices
/// described by `edges`. Returns `mate` where `mate[v] = Some(u)` means
/// `v` is matched to `u` and `mate[v] = None` means `v` is unmatched.
///
/// The returned matching is symmetric: `mate[u] = Some(v)` iff
/// `mate[v] = Some(u)`. The matching size equals
/// `mate.iter().filter(|m| m.is_some()).count() / 2`.
///
/// # Panics
/// Panics if any edge endpoint is `>= n`.
#[must_use]
pub fn maximum_matching(n: usize, edges: &[(usize, usize)]) -> Vec<Option<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        assert!(u < n && v < n, "edge endpoint out of range");
        if u == v {
            // Self-loops cannot participate in a matching.
            continue;
        }
        adj[u].push(v);
        adj[v].push(u);
    }
    Solver::new(adj).run()
}

/// Convenience wrapper returning the matched pairs `(u, v)` with `u < v`.
#[must_use]
pub fn matching_pairs(n: usize, edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mate = maximum_matching(n, edges);
    let mut out = Vec::new();
    for (u, m) in mate.iter().enumerate() {
        if let Some(v) = *m {
            if u < v {
                out.push((u, v));
            }
        }
    }
    out
}

/// Sentinel meaning "no vertex". Using `usize::MAX` keeps the per-BFS
/// arrays as plain `Vec<usize>`, which is what most published Edmonds
/// implementations use; the wrapper API still exposes `Option<usize>`.
const NIL: usize = usize::MAX;

struct Solver {
    n: usize,
    adj: Vec<Vec<usize>>,
    mate: Vec<usize>,
    /// Predecessor on the *non-matching* edge in the alternating tree.
    /// Mutated during blossom contraction.
    parent: Vec<usize>,
    /// Representative of the blossom currently containing each vertex.
    /// Outside any blossom, `base[v] = v`.
    base: Vec<usize>,
    /// True if the vertex is part of the alternating tree at all.
    in_tree: Vec<bool>,
    /// True once a vertex has been queued during the current BFS.
    in_queue: Vec<bool>,
    queue: VecDeque<usize>,
}

impl Solver {
    fn new(adj: Vec<Vec<usize>>) -> Self {
        let n = adj.len();
        Self {
            n,
            adj,
            mate: vec![NIL; n],
            parent: vec![NIL; n],
            base: (0..n).collect(),
            in_tree: vec![false; n],
            in_queue: vec![false; n],
            queue: VecDeque::new(),
        }
    }

    fn run(mut self) -> Vec<Option<usize>> {
        for root in 0..self.n {
            if self.mate[root] == NIL {
                self.augment_from(root);
            }
        }
        self.mate
            .iter()
            .map(|&v| if v == NIL { None } else { Some(v) })
            .collect()
    }

    /// Returns the LCA of `a` and `b` in the alternating tree, walking up
    /// alternately by `mate` then `parent`.
    fn lca(&self, a: usize, b: usize) -> usize {
        let mut seen = vec![false; self.n];
        // Walk from a until we hit the root, marking every base on the way.
        let mut x = a;
        loop {
            let bx = self.base[x];
            seen[bx] = true;
            let m = self.mate[bx];
            if m == NIL {
                break;
            }
            let p = self.parent[m];
            if p == NIL {
                break;
            }
            x = p;
        }
        // Walk from b and stop at the first base seen from a.
        let mut y = b;
        loop {
            let by = self.base[y];
            if seen[by] {
                return by;
            }
            let m = self.mate[by];
            // Both endpoints are even, so b's walk must reach a base
            // already visited from a (worst case: the BFS root itself).
            debug_assert!(m != NIL, "even vertex has no mate above the LCA");
            let p = self.parent[m];
            debug_assert!(p != NIL, "even vertex's mate has no parent above the LCA");
            y = p;
        }
    }

    /// Marks the path from `v` up to the blossom base `lca_base` as part of
    /// the new blossom, rerouting non-matching parents through `child` so
    /// that augmentation can later traverse the cycle in the right
    /// direction. Returns the marker array.
    fn mark_path(&mut self, in_blossom: &mut [bool], v: usize, lca_base: usize, child: usize) {
        let mut cur = v;
        let mut child = child;
        while self.base[cur] != lca_base {
            in_blossom[self.base[cur]] = true;
            let m = self.mate[cur];
            in_blossom[self.base[m]] = true;
            // After contraction the cycle vertex `cur` should be reachable
            // from `child` via a non-matching edge.
            self.parent[cur] = child;
            child = m;
            cur = self.parent[m];
        }
    }

    /// Attempts to find an augmenting path from `root`. If found, flips
    /// the matching along it.
    fn augment_from(&mut self, root: usize) {
        // Reset per-BFS state.
        for i in 0..self.n {
            self.parent[i] = NIL;
            self.base[i] = i;
            self.in_tree[i] = false;
            self.in_queue[i] = false;
        }
        self.queue.clear();

        self.in_tree[root] = true;
        self.in_queue[root] = true;
        self.queue.push_back(root);

        while let Some(v) = self.queue.pop_front() {
            // Iterate by index since we may mutate `self.adj`-adjacent
            // arrays inside the loop.
            for i in 0..self.adj[v].len() {
                let w = self.adj[v][i];
                if self.base[v] == self.base[w] || self.mate[v] == w {
                    continue;
                }
                // `w` is even when it is the root or its mate is already in
                // the tree as an even ("outer") vertex.
                let w_is_even =
                    w == root || (self.mate[w] != NIL && self.parent[self.mate[w]] != NIL);
                if w_is_even {
                    // Blossom case: contract the v–w cycle.
                    let lca_base = self.lca(v, w);
                    let mut in_blossom = vec![false; self.n];
                    self.mark_path(&mut in_blossom, v, lca_base, w);
                    self.mark_path(&mut in_blossom, w, lca_base, v);
                    for k in 0..self.n {
                        if in_blossom[self.base[k]] {
                            self.base[k] = lca_base;
                            // Inner blossom vertices become even after
                            // contraction; queue them so their outgoing
                            // edges get explored.
                            if !self.in_queue[k] {
                                self.in_queue[k] = true;
                                self.queue.push_back(k);
                            }
                        }
                    }
                } else if !self.in_tree[w] {
                    self.parent[w] = v;
                    self.in_tree[w] = true;
                    if self.mate[w] == NIL {
                        // Augmenting path found ending at w.
                        self.augment(w);
                        return;
                    }
                    let mw = self.mate[w];
                    self.in_tree[mw] = true;
                    if !self.in_queue[mw] {
                        self.in_queue[mw] = true;
                        self.queue.push_back(mw);
                    }
                }
            }
        }
    }

    /// Walks back from `v` to the BFS root, flipping matching status of
    /// every edge along the way.
    fn augment(&mut self, mut v: usize) {
        while v != NIL {
            let pv = self.parent[v];
            let ppv = self.mate[pv];
            self.mate[v] = pv;
            self.mate[pv] = v;
            v = ppv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{matching_pairs, maximum_matching};

    /// Validates the returned `mate` array is a proper matching:
    /// every matched edge appears in `edges`, every vertex is matched at
    /// most once, and the symmetry `mate[u] = Some(v) ⇔ mate[v] = Some(u)`
    /// holds. Returns the matching size.
    fn check(n: usize, edges: &[(usize, usize)], mate: &[Option<usize>]) -> usize {
        assert_eq!(mate.len(), n);
        let mut size = 0;
        for (u, m) in mate.iter().enumerate() {
            if let Some(v) = *m {
                assert!(v < n, "mate index out of range");
                assert_eq!(mate[v], Some(u), "mate is not symmetric for ({u},{v})");
                assert_ne!(u, v, "self-match for vertex {u}");
                let in_edges = edges
                    .iter()
                    .any(|&(a, b)| (a == u && b == v) || (a == v && b == u));
                assert!(in_edges, "matched edge ({u},{v}) is not in input edge list");
                size += 1;
            }
        }
        assert!(
            size % 2 == 0,
            "total matched count must be even, got {size}"
        );
        size / 2
    }

    /// Reference brute force: tries every subset of edges and returns the
    /// largest one that forms a valid matching. Exponential in `|E|`, only
    /// safe for tiny graphs.
    fn brute_force(n: usize, edges: &[(usize, usize)]) -> usize {
        let m = edges.len();
        assert!(m <= 20, "brute force only feasible for <= 20 edges");
        let mut best = 0;
        for mask in 0u32..(1u32 << m) {
            let mut used = vec![false; n];
            let mut size = 0;
            let mut ok = true;
            for (i, &(a, b)) in edges.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    if used[a] || used[b] || a == b {
                        ok = false;
                        break;
                    }
                    used[a] = true;
                    used[b] = true;
                    size += 1;
                }
            }
            if ok && size > best {
                best = size;
            }
        }
        best
    }

    #[test]
    fn empty_graph() {
        let mate = maximum_matching(5, &[]);
        assert_eq!(mate, vec![None; 5]);
        assert_eq!(check(5, &[], &mate), 0);
    }

    #[test]
    fn single_edge() {
        let edges = [(0, 1)];
        let mate = maximum_matching(2, &edges);
        assert_eq!(check(2, &edges, &mate), 1);
    }

    #[test]
    fn triangle_size_one() {
        // C3 = K3: any one of the three edges is a maximum matching.
        let edges = [(0, 1), (1, 2), (2, 0)];
        let mate = maximum_matching(3, &edges);
        assert_eq!(check(3, &edges, &mate), 1);
    }

    #[test]
    fn k4_perfect_matching() {
        // K4 has a perfect matching of size 2.
        let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let mate = maximum_matching(4, &edges);
        assert_eq!(check(4, &edges, &mate), 2);
    }

    #[test]
    fn odd_cycle_c5_canonical_blossom() {
        // C5 — the smallest non-bipartite graph that forces a blossom
        // contraction. Maximum matching = 2.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
        let mate = maximum_matching(5, &edges);
        assert_eq!(check(5, &edges, &mate), 2);
    }

    #[test]
    fn path_chain_six_vertices() {
        // 0-1-2-3-4-5 path: maximum matching {0-1, 2-3, 4-5}, size 3.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        let mate = maximum_matching(6, &edges);
        assert_eq!(check(6, &edges, &mate), 3);
    }

    #[test]
    fn petersen_graph_perfect_matching() {
        // Petersen graph: 10 vertices, 15 edges, 3-regular, contains odd
        // cycles. Has perfect matchings of size 5.
        // Outer 5-cycle: 0-1-2-3-4-0
        // Inner 5-pointed star: 5-7-9-6-8-5
        // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
        ];
        let mate = maximum_matching(10, &edges);
        assert_eq!(check(10, &edges, &mate), 5);
    }

    #[test]
    fn isolated_vertices() {
        // 6 vertices, only one edge.
        let edges = [(2, 4)];
        let mate = maximum_matching(6, &edges);
        assert_eq!(check(6, &edges, &mate), 1);
        assert_eq!(mate[0], None);
        assert_eq!(mate[1], None);
        assert_eq!(mate[3], None);
        assert_eq!(mate[5], None);
    }

    #[test]
    fn matching_pairs_returns_normalised() {
        let edges = [(0, 1), (2, 3)];
        let pairs = matching_pairs(4, &edges);
        let mut got: Vec<(usize, usize)> = pairs;
        got.sort();
        assert_eq!(got, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn k7_max_matching_size_three() {
        // K7 has odd order ⇒ no perfect matching, but max matching = 3
        // (uses 6 of the 7 vertices; one is left out).
        let mut edges = Vec::new();
        for u in 0..7 {
            for v in (u + 1)..7 {
                edges.push((u, v));
            }
        }
        let mate = maximum_matching(7, &edges);
        assert_eq!(check(7, &edges, &mate), 3);
    }

    #[test]
    fn two_triangles_share_vertex_size_one_per_triangle() {
        // Bowtie: two triangles sharing vertex 0.
        // 0-1-2-0 and 0-3-4-0. Max matching = 2 (e.g. {1-2, 3-4}).
        let edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)];
        let mate = maximum_matching(5, &edges);
        assert_eq!(check(5, &edges, &mate), 2);
    }

    #[test]
    fn flower_of_blossoms() {
        // C5 with a tail attached. Vertices 0..5 form C5; vertex 5 attaches
        // to 0 via edge (0,5). Optimal matching = 3.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)];
        let mate = maximum_matching(6, &edges);
        assert_eq!(check(6, &edges, &mate), 3);
    }

    /// Deterministic xorshift used to generate random small graphs for the
    /// property test below.
    fn random_graph(n: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if next().trailing_zeros() >= 2 {
                    // ~25% density
                    edges.push((u, v));
                }
            }
        }
        edges
    }

    #[test]
    fn property_random_small_graphs_match_brute_force() {
        for seed in 0u64..40 {
            for n in 1..=6 {
                let edges = random_graph(n, seed);
                let mate = maximum_matching(n, &edges);
                let size = check(n, &edges, &mate);
                let expected = brute_force(n, &edges);
                assert_eq!(
                    size, expected,
                    "blossom got {size} expected {expected} (n={n} seed={seed} edges={edges:?})"
                );
            }
        }
    }

    #[test]
    fn property_no_vertex_matched_twice() {
        // Larger random graphs: only checks the *internal consistency*
        // properties (no double match, every matched edge in input). The
        // `check` helper asserts all of those.
        for seed in 0u64..10 {
            let n = 12;
            let edges = random_graph(n, seed);
            let mate = maximum_matching(n, &edges);
            let _ = check(n, &edges, &mate);
        }
    }

    #[test]
    fn parallel_edges_and_self_loops_handled() {
        // Self-loops (0,0) must be ignored; parallel (1,2) edges must not
        // cause double-matching.
        let edges = [(0, 0), (1, 2), (1, 2), (2, 3), (3, 3)];
        let mate = maximum_matching(4, &edges);
        let size = check(4, &edges, &mate);
        // Underlying simple graph is the path 1-2-3, so max matching = 1.
        assert_eq!(size, 1);
    }
}
