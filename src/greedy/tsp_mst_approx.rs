//! 2-approximation algorithm for metric Travelling Salesman via MST.
//!
//! Given a complete graph specified by a square symmetric distance matrix
//! satisfying the triangle inequality (`d(i,k) <= d(i,j) + d(j,k)` for all
//! `i,j,k`, with `d(i,i) == 0` and `d(i,j) >= 0`), this routine produces a
//! Hamiltonian cycle whose total length is at most twice the optimum.
//!
//! Algorithm:
//! 1. Build a minimum spanning tree of the complete graph using Prim's
//!    algorithm on the dense distance matrix (`O(n^2)`).
//! 2. Run a pre-order DFS on the MST starting from vertex 0 and emit each
//!    vertex the first time it is visited (this is the "shortcut walk" — an
//!    Eulerian walk on the doubled MST with repeats removed via the triangle
//!    inequality).
//! 3. Append vertex 0 at the end to close the cycle.
//!
//! Time complexity: `O(n^2)` (dominated by dense Prim).
//! Space complexity: `O(n^2)` for the implicit input plus `O(n)` working sets.
//!
//! Approximation guarantee (sketch). Let `OPT` be the cost of an optimal
//! Hamiltonian tour. Removing any one edge from `OPT` yields a spanning path,
//! whose cost is at least the MST cost: `MST <= OPT`. Doubling every MST edge
//! gives an Eulerian multigraph; an Euler tour costs `2 * MST <= 2 * OPT`.
//! Shortcutting repeated visits cannot increase cost under the triangle
//! inequality, so the emitted Hamiltonian cycle costs at most `2 * OPT`.
//!
//! Edge cases:
//! - `n == 0` panics — there is no tour over an empty vertex set.
//! - `n == 1` returns `vec![0, 0]` (the trivial degenerate cycle).
//! - The input is validated for squareness, symmetry, non-negativity, and
//!   zero diagonal; violations panic.

/// Returns a Hamiltonian cycle over the `n` cities described by `weights`,
/// using the MST-based 2-approximation under the metric (triangle-inequality)
/// assumption.
///
/// `weights` is an `n x n` symmetric distance matrix with `weights[i][i] == 0`
/// and `weights[i][j] >= 0`. The returned vector has length `n + 1` and starts
/// and ends at city `0` (i.e. `result[0] == result[n] == 0`).
///
/// Time: `O(n^2)`. Space: `O(n)` auxiliary.
///
/// # Panics
/// - if `n == 0`,
/// - if `weights` is not square,
/// - if any diagonal entry is non-zero,
/// - if any off-diagonal entry is negative or non-finite,
/// - if the matrix is not symmetric.
#[must_use]
pub fn tsp_mst_2approx(weights: &[Vec<f64>]) -> Vec<usize> {
    let n = weights.len();
    assert!(n > 0, "tsp_mst_2approx requires at least one city");
    validate_matrix(weights);

    if n == 1 {
        return vec![0, 0];
    }

    let parents = dense_prim_parents(weights);
    let children = children_from_parents(&parents);

    let mut tour = Vec::with_capacity(n + 1);
    preorder_dfs(0, &children, &mut tour);
    tour.push(0);
    tour
}

/// Validates that `weights` is square, has a zero diagonal, is non-negative,
/// finite, and symmetric. Panics on any violation.
fn validate_matrix(weights: &[Vec<f64>]) {
    let n = weights.len();
    for (i, row) in weights.iter().enumerate() {
        assert!(
            row.len() == n,
            "weights must be square: row {i} has length {} (expected {n})",
            row.len()
        );
        for (j, &w) in row.iter().enumerate() {
            assert!(w.is_finite(), "weights[{i}][{j}] is not finite");
            if i == j {
                assert!(w == 0.0, "weights[{i}][{i}] must be zero, got {w}");
            } else {
                assert!(w >= 0.0, "weights[{i}][{j}] must be non-negative, got {w}");
            }
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                (weights[i][j] - weights[j][i]).abs() == 0.0,
                "weights must be symmetric: weights[{i}][{j}] != weights[{j}][{i}]"
            );
        }
    }
}

/// Dense-graph Prim's algorithm. Returns the parent array of the MST rooted
/// at `0`; `parents[0] == 0` by convention. `O(n^2)`.
fn dense_prim_parents(weights: &[Vec<f64>]) -> Vec<usize> {
    let n = weights.len();
    let mut in_tree = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut parent = vec![0usize; n];
    best[0] = 0.0;
    for _ in 0..n {
        // Pick the not-yet-included vertex with the smallest connecting edge.
        let mut u = usize::MAX;
        let mut min_w = f64::INFINITY;
        for v in 0..n {
            if !in_tree[v] && best[v] < min_w {
                min_w = best[v];
                u = v;
            }
        }
        in_tree[u] = true;
        // Relax edges out of `u`.
        for v in 0..n {
            if !in_tree[v] && weights[u][v] < best[v] {
                best[v] = weights[u][v];
                parent[v] = u;
            }
        }
    }
    parent
}

/// Inverts a parent array into per-node child lists. Children are emitted in
/// ascending vertex order, which makes the DFS pre-order deterministic.
fn children_from_parents(parents: &[usize]) -> Vec<Vec<usize>> {
    let n = parents.len();
    let mut children = vec![Vec::new(); n];
    for (v, &p) in parents.iter().enumerate() {
        if v != p {
            children[p].push(v);
        }
    }
    children
}

/// Iterative pre-order DFS that pushes each vertex on first visit.
fn preorder_dfs(root: usize, children: &[Vec<usize>], out: &mut Vec<usize>) {
    let mut stack = vec![root];
    while let Some(u) = stack.pop() {
        out.push(u);
        // Push children in reverse so the smallest is popped first → ascending
        // pre-order.
        for &c in children[u].iter().rev() {
            stack.push(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tsp_mst_2approx;

    /// Total length of a closed tour over the given distance matrix.
    fn tour_length(weights: &[Vec<f64>], tour: &[usize]) -> f64 {
        tour.windows(2).map(|w| weights[w[0]][w[1]]).sum()
    }

    /// Brute-force optimal Hamiltonian tour cost (n! permutations, fine for
    /// the n ≤ 7 test instances).
    fn brute_force_opt(weights: &[Vec<f64>]) -> f64 {
        let n = weights.len();
        if n <= 1 {
            return 0.0;
        }
        // Fix vertex 0 as the start to cut the symmetry — every cyclic shift
        // of a tour has the same length.
        let mut rest: Vec<usize> = (1..n).collect();
        let mut best = f64::INFINITY;
        permute(&mut rest, 0, &mut |perm| {
            let mut cost = weights[0][perm[0]];
            for w in perm.windows(2) {
                cost += weights[w[0]][w[1]];
            }
            cost += weights[*perm.last().unwrap()][0];
            if cost < best {
                best = cost;
            }
        });
        best
    }

    fn permute<F: FnMut(&[usize])>(arr: &mut [usize], start: usize, f: &mut F) {
        if start + 1 >= arr.len() {
            f(arr);
            return;
        }
        for i in start..arr.len() {
            arr.swap(start, i);
            permute(arr, start + 1, f);
            arr.swap(start, i);
        }
    }

    /// Builds a Euclidean distance matrix from 2D points. Euclidean distances
    /// satisfy the triangle inequality.
    fn euclidean_matrix(points: &[(f64, f64)]) -> Vec<Vec<f64>> {
        let n = points.len();
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                m[i][j] = dx.hypot(dy);
            }
        }
        m
    }

    /// Checks that a tour is a valid closed Hamiltonian cycle over `n` cities:
    /// length is `n + 1`, starts and ends at the same city, and each city
    /// appears exactly once when the closing repeat is excluded.
    fn assert_valid_tour(tour: &[usize], n: usize) {
        assert_eq!(tour.len(), n + 1, "tour length must be n + 1");
        assert_eq!(tour[0], tour[n], "tour must close on its starting city");
        let mut seen = vec![false; n];
        for &c in &tour[..n] {
            assert!(c < n, "city index {c} out of range");
            assert!(!seen[c], "city {c} visited more than once");
            seen[c] = true;
        }
        assert!(seen.iter().all(|&s| s), "every city must be visited");
    }

    #[test]
    fn n_one_returns_trivial_cycle() {
        let weights = vec![vec![0.0]];
        assert_eq!(tsp_mst_2approx(&weights), vec![0, 0]);
    }

    #[test]
    fn n_two_returns_round_trip() {
        let weights = vec![vec![0.0, 4.2], vec![4.2, 0.0]];
        assert_eq!(tsp_mst_2approx(&weights), vec![0, 1, 0]);
    }

    #[test]
    fn unit_square_within_two_times_opt() {
        // Corners of a unit square; optimal tour cost is 4.0.
        let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let weights = euclidean_matrix(&pts);
        let tour = tsp_mst_2approx(&weights);
        assert_valid_tour(&tour, 4);
        let approx = tour_length(&weights, &tour);
        let opt = brute_force_opt(&weights);
        assert!(
            (opt - 4.0).abs() < 1e-9,
            "expected unit-square OPT 4.0, got {opt}"
        );
        let bound = 2.0f64.mul_add(opt, 1e-9);
        assert!(approx <= bound, "approx {approx} exceeds 2 * OPT ({bound})");
    }

    #[test]
    fn each_city_appears_exactly_once_excluding_close() {
        let pts = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0), (1.5, 2.0)];
        let weights = euclidean_matrix(&pts);
        let tour = tsp_mst_2approx(&weights);
        assert_valid_tour(&tour, pts.len());
    }

    /// Tiny linear-congruential PRNG so the property test is deterministic
    /// without pulling in a `rand` dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map the upper 53 bits into [0, 1).
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn random_metric_instances_within_two_times_opt() {
        let mut rng = Lcg(0x00C0_FFEE_D00D_BEEF_u64);
        for _ in 0..30 {
            let n = 5 + (rng.next_f64() * 3.0) as usize; // 5, 6, or 7
            let pts: Vec<(f64, f64)> = (0..n)
                .map(|_| (rng.next_f64() * 100.0, rng.next_f64() * 100.0))
                .collect();
            let weights = euclidean_matrix(&pts);
            let tour = tsp_mst_2approx(&weights);
            assert_valid_tour(&tour, n);
            let approx = tour_length(&weights, &tour);
            let opt = brute_force_opt(&weights);
            let bound = 2.0f64.mul_add(opt, 1e-9);
            assert!(
                approx <= bound,
                "n={n} approx {approx} exceeds 2 * OPT ({bound})"
            );
        }
    }

    #[test]
    #[should_panic(expected = "at least one city")]
    fn empty_input_panics() {
        let weights: Vec<Vec<f64>> = vec![];
        let _ = tsp_mst_2approx(&weights);
    }

    #[test]
    #[should_panic(expected = "must be square")]
    fn non_square_panics() {
        let weights = vec![vec![0.0, 1.0], vec![1.0, 0.0, 2.0]];
        let _ = tsp_mst_2approx(&weights);
    }

    #[test]
    #[should_panic(expected = "must be symmetric")]
    fn asymmetric_panics() {
        let weights = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 0.0, 4.0],
            vec![2.0, 4.0, 0.0],
        ];
        let _ = tsp_mst_2approx(&weights);
    }

    #[test]
    #[should_panic(expected = "must be non-negative")]
    fn negative_weight_panics() {
        let weights = vec![vec![0.0, -1.0], vec![-1.0, 0.0]];
        let _ = tsp_mst_2approx(&weights);
    }

    #[test]
    #[should_panic(expected = "must be zero")]
    fn nonzero_diagonal_panics() {
        let weights = vec![vec![1.0, 2.0], vec![2.0, 0.0]];
        let _ = tsp_mst_2approx(&weights);
    }
}
