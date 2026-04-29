//! Block-cut tree (a.k.a. block tree) of an undirected graph.
//!
//! Given an undirected (multi-)graph `G = (V, E)`, a **biconnected component**
//! (also called a *block*) is a maximal subgraph with no articulation point
//! of its own — equivalently, a maximal set of edges such that any two edges
//! lie on a common simple cycle, plus their incident vertices. An isolated
//! vertex with no incident edges forms a degenerate block of size one.
//!
//! The **block-cut tree** is a bipartite forest whose nodes are
//! `{cut vertices} ∪ {blocks}`. There is an edge between cut vertex `v` and
//! block `B` whenever `v ∈ B`. The result is a tree per connected component
//! of `G` (a forest overall when `G` is disconnected).
//!
//! This module reports more than the strict bipartite tree: every original
//! vertex retains an index in `tree_adj` (`0..n`), so callers can index into
//! the result directly using their own vertex IDs. Block nodes occupy indices
//! `n..n + blocks.len()`. Only cut-vertex nodes have outgoing tree edges; the
//! remaining vertex indices are present but have empty adjacency lists. Each
//! vertex's full block membership is reported separately via
//! [`BlockCutTree::block_of_vertex`].
//!
//! ## Algorithm
//!
//! Tarjan's classical biconnected-components DFS. We maintain discovery /
//! low-link timestamps and an auxiliary **edge stack**: every tree-edge or
//! back-edge is pushed onto the stack as it is first traversed. Whenever the
//! DFS finishes a child `v` of `u` and observes `low[v] >= disc[u]`, the
//! edges on top of the stack down to and including `(u, v)` form a complete
//! biconnected component, so we pop them and collect their vertices. Cut
//! vertices are detected with the standard rule (root with >= 2 children, or
//! non-root with some child whose `low[v] >= disc[u]`).
//!
//! Isolated vertices (no incident edges) form singleton blocks recorded
//! after the main DFS sweep.
//!
//! Runs in `O(V + E)` time and `O(V + E)` space.

/// Block-cut tree of an undirected graph computed by Tarjan's biconnected
/// components DFS.
///
/// All inputs are interpreted as **undirected**: every edge `{u, v}` must
/// appear in both adjacency lists. Self-loops are ignored. Parallel edges
/// (multi-edges) are permitted and contribute correctly to biconnectivity.
#[derive(Debug, Clone)]
pub struct BlockCutTree {
    /// Vertices of each biconnected component. The list of indices for a
    /// block is sorted ascending and contains no duplicates.
    pub blocks: Vec<Vec<usize>>,
    /// `cut_vertices[v]` is `true` iff `v` is an articulation point of the
    /// underlying graph.
    pub cut_vertices: Vec<bool>,
    /// Adjacency of the bipartite block-cut tree. Indices `0..n` are
    /// original vertices and indices `n..n + blocks.len()` are block nodes.
    /// Only cut-vertex indices and block indices have non-empty entries.
    pub tree_adj: Vec<Vec<usize>>,
    /// `block_of_vertex[v]` lists every block containing vertex `v`, sorted
    /// ascending and free of duplicates. A non-cut vertex appears in exactly
    /// one block; a cut vertex appears in two or more.
    pub block_of_vertex: Vec<Vec<usize>>,
}

impl BlockCutTree {
    /// Build the block-cut tree of an undirected graph given as an adjacency
    /// list `adj`. The graph may be disconnected; the result then represents
    /// a forest. Time and space `O(V + E)`.
    #[must_use]
    pub fn build(adj: &[Vec<usize>]) -> Self {
        let n = adj.len();
        let mut disc = vec![usize::MAX; n];
        let mut low = vec![0_usize; n];
        let mut cut_vertices = vec![false; n];
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        // Edge stack stores `(u, v)` pairs as discovered during the DFS.
        let mut edge_stack: Vec<(usize, usize)> = Vec::new();
        let mut timer = 0_usize;

        for root in 0..n {
            if disc[root] == usize::MAX {
                if adj[root].iter().all(|&v| v == root) {
                    // Vertex with no non-self edges: emit a singleton block.
                    disc[root] = timer;
                    low[root] = timer;
                    timer += 1;
                    blocks.push(vec![root]);
                    continue;
                }
                Self::dfs(
                    adj,
                    root,
                    usize::MAX,
                    &mut disc,
                    &mut low,
                    &mut cut_vertices,
                    &mut blocks,
                    &mut edge_stack,
                    &mut timer,
                );
            }
        }

        // Build the auxiliary tree.
        let mut tree_adj: Vec<Vec<usize>> = vec![Vec::new(); n + blocks.len()];
        let mut block_of_vertex: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (b_idx, block) in blocks.iter().enumerate() {
            let block_node = n + b_idx;
            for &v in block {
                block_of_vertex[v].push(b_idx);
                if cut_vertices[v] {
                    tree_adj[v].push(block_node);
                    tree_adj[block_node].push(v);
                }
            }
        }
        for list in &mut block_of_vertex {
            list.sort_unstable();
            list.dedup();
        }
        for list in &mut tree_adj {
            list.sort_unstable();
            list.dedup();
        }

        Self {
            blocks,
            cut_vertices,
            tree_adj,
            block_of_vertex,
        }
    }

    /// Number of biconnected components.
    #[must_use]
    pub const fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Tree-adjacency index of block `b` (offset by the number of original
    /// vertices). Useful when traversing [`Self::tree_adj`].
    #[must_use]
    pub const fn block_node(&self, b: usize) -> usize {
        self.block_of_vertex.len() + b
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        adj: &[Vec<usize>],
        u: usize,
        parent: usize,
        disc: &mut [usize],
        low: &mut [usize],
        cut: &mut [bool],
        blocks: &mut Vec<Vec<usize>>,
        edge_stack: &mut Vec<(usize, usize)>,
        timer: &mut usize,
    ) {
        disc[u] = *timer;
        low[u] = *timer;
        *timer += 1;
        let mut child_count = 0_usize;
        let mut parent_used = false;
        for &v in &adj[u] {
            if v == u {
                // Self-loop: cannot lie on any simple cycle and does not
                // affect biconnectivity. Skip.
                continue;
            }
            if disc[v] == usize::MAX {
                child_count += 1;
                edge_stack.push((u, v));
                Self::dfs(adj, v, u, disc, low, cut, blocks, edge_stack, timer);
                low[u] = low[u].min(low[v]);
                if low[v] >= disc[u] {
                    // Pop one biconnected component off the stack.
                    let mut verts: Vec<usize> = Vec::new();
                    while let Some(&(a, b)) = edge_stack.last() {
                        edge_stack.pop();
                        verts.push(a);
                        verts.push(b);
                        if (a, b) == (u, v) || (a, b) == (v, u) {
                            break;
                        }
                    }
                    verts.sort_unstable();
                    verts.dedup();
                    blocks.push(verts);
                    if parent != usize::MAX {
                        cut[u] = true;
                    }
                }
            } else if v != parent || parent_used {
                // Back-edge to an ancestor we have already discovered.
                if disc[v] < disc[u] {
                    edge_stack.push((u, v));
                }
                low[u] = low[u].min(disc[v]);
            } else {
                // First sighting of the parent edge from `u`'s side: ignore,
                // but mark so a parallel edge to the same parent counts.
                parent_used = true;
            }
        }
        if parent == usize::MAX && child_count > 1 {
            cut[u] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BlockCutTree;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;

    fn undirected(edges: &[(usize, usize)], n: usize) -> Vec<Vec<usize>> {
        let mut g: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in edges {
            g[u].push(v);
            g[v].push(u);
        }
        g
    }

    fn block_set(bct: &BlockCutTree) -> BTreeSet<Vec<usize>> {
        bct.blocks.iter().cloned().collect()
    }

    #[test]
    fn empty_graph() {
        let bct = BlockCutTree::build(&[]);
        assert!(bct.blocks.is_empty());
        assert!(bct.cut_vertices.is_empty());
        assert!(bct.tree_adj.is_empty());
        assert!(bct.block_of_vertex.is_empty());
    }

    #[test]
    fn single_vertex_is_singleton_block() {
        let bct = BlockCutTree::build(&[vec![]]);
        assert_eq!(bct.blocks, vec![vec![0_usize]]);
        assert_eq!(bct.cut_vertices, vec![false]);
        assert_eq!(bct.block_of_vertex, vec![vec![0_usize]]);
        assert!(bct.tree_adj[0].is_empty());
    }

    #[test]
    fn k2_one_block_no_cuts() {
        let g = undirected(&[(0, 1)], 2);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.blocks, vec![vec![0, 1]]);
        assert_eq!(bct.cut_vertices, vec![false, false]);
        for v in 0..2 {
            assert_eq!(bct.block_of_vertex[v], vec![0]);
            assert!(bct.tree_adj[v].is_empty());
        }
    }

    #[test]
    fn k3_triangle_one_block_no_cuts() {
        let g = undirected(&[(0, 1), (1, 2), (2, 0)], 3);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.blocks, vec![vec![0, 1, 2]]);
        assert_eq!(bct.cut_vertices, vec![false, false, false]);
    }

    #[test]
    fn butterfly_two_triangles_share_vertex() {
        // Triangle 0-1-2 and triangle 2-3-4 share cut vertex 2.
        let g = undirected(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.num_blocks(), 2);
        let bs = block_set(&bct);
        assert!(bs.contains(&vec![0, 1, 2]));
        assert!(bs.contains(&vec![2, 3, 4]));
        assert_eq!(bct.cut_vertices, vec![false, false, true, false, false]);
        // Cut vertex 2 connects to both block nodes.
        assert_eq!(bct.tree_adj[2].len(), 2);
        for &nbr in &bct.tree_adj[2] {
            assert!((5..5 + 2).contains(&nbr));
        }
    }

    #[test]
    fn path_each_edge_is_block() {
        // 0 - 1 - 2 - 3
        let g = undirected(&[(0, 1), (1, 2), (2, 3)], 4);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.num_blocks(), 3);
        let bs = block_set(&bct);
        assert!(bs.contains(&vec![0, 1]));
        assert!(bs.contains(&vec![1, 2]));
        assert!(bs.contains(&vec![2, 3]));
        assert_eq!(bct.cut_vertices, vec![false, true, true, false]);
        assert_eq!(bct.tree_adj[1].len(), 2);
        assert_eq!(bct.tree_adj[2].len(), 2);
    }

    #[test]
    fn star_graph_centre_is_cut() {
        // 0 connected to 1, 2, 3.
        let g = undirected(&[(0, 1), (0, 2), (0, 3)], 4);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.num_blocks(), 3);
        assert_eq!(bct.cut_vertices, vec![true, false, false, false]);
        assert_eq!(bct.tree_adj[0].len(), 3);
        let bs = block_set(&bct);
        assert!(bs.contains(&vec![0, 1]));
        assert!(bs.contains(&vec![0, 2]));
        assert!(bs.contains(&vec![0, 3]));
    }

    #[test]
    fn binary_tree_every_edge_block_internal_cuts() {
        //         0
        //        / \
        //       1   2
        //      /|   |
        //     3 4   5
        let g = undirected(&[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)], 6);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.num_blocks(), 5);
        assert_eq!(
            bct.cut_vertices,
            vec![true, true, true, false, false, false]
        );
    }

    #[test]
    fn disconnected_graph_forest() {
        // Component A: triangle 0-1-2. Component B: edge 3-4. Component C: isolated 5.
        let g = undirected(&[(0, 1), (1, 2), (2, 0), (3, 4)], 6);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.num_blocks(), 3);
        let bs = block_set(&bct);
        assert!(bs.contains(&vec![0, 1, 2]));
        assert!(bs.contains(&vec![3, 4]));
        assert!(bs.contains(&vec![5]));
        assert!(bct.cut_vertices.iter().all(|&b| !b));
    }

    #[test]
    fn parallel_edges_dont_create_cut() {
        // Two parallel edges between 0 and 1: still one block, no cut.
        let g: Vec<Vec<usize>> = vec![vec![1, 1], vec![0, 0]];
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.blocks, vec![vec![0, 1]]);
        assert!(bct.cut_vertices.iter().all(|&b| !b));
    }

    #[test]
    fn self_loop_ignored() {
        // Self-loop at 0, plus edge 0-1.
        let g = undirected(&[(0, 0), (0, 1)], 2);
        let bct = BlockCutTree::build(&g);
        assert_eq!(bct.blocks, vec![vec![0, 1]]);
        assert!(bct.cut_vertices.iter().all(|&b| !b));
    }

    // ---------- Brute-force reference + quickcheck ----------

    /// Brute-force articulation points: remove each vertex and count
    /// connected components of the remainder.
    fn brute_articulation(adj: &[Vec<usize>]) -> Vec<bool> {
        let n = adj.len();
        let mut cuts = vec![false; n];
        let base = count_components(adj, usize::MAX);
        for v in 0..n {
            // A vertex is an articulation point iff removing it strictly
            // increases the number of connected components.
            //
            // Excluding `v` removes one component if `v` was isolated, so we
            // compare to the count restricted to `V \ {v}`. Equivalently,
            // count components among the remaining `n - 1` vertices and
            // compare to (`base` minus 1 if `v` was isolated, else `base`).
            let isolated = adj[v].iter().all(|&u| u == v);
            let expected = if isolated { base - 1 } else { base };
            let after = count_components(adj, v);
            if after > expected {
                cuts[v] = true;
            }
        }
        cuts
    }

    fn count_components(adj: &[Vec<usize>], excluded: usize) -> usize {
        let n = adj.len();
        let mut seen = vec![false; n];
        if excluded != usize::MAX {
            seen[excluded] = true;
        }
        let mut comps = 0_usize;
        for s in 0..n {
            if !seen[s] {
                comps += 1;
                let mut stack = vec![s];
                seen[s] = true;
                while let Some(u) = stack.pop() {
                    for &w in &adj[u] {
                        if !seen[w] {
                            seen[w] = true;
                            stack.push(w);
                        }
                    }
                }
            }
        }
        comps
    }

    fn find_uf(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    /// Brute-force biconnected-component vertex sets by edge equivalence:
    /// two edges are in the same block iff they lie on a common simple
    /// cycle. We enumerate simple cycles by DFS and union edges on each.
    fn brute_blocks(adj: &[Vec<usize>]) -> BTreeSet<Vec<usize>> {
        let n = adj.len();
        let mut result: BTreeSet<Vec<usize>> = BTreeSet::new();

        // Canonical edges (u < v), self-loops dropped, duplicates removed.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for u in 0..n {
            for &v in &adj[u] {
                if u < v {
                    edges.push((u, v));
                }
            }
        }
        edges.sort_unstable();
        let m = edges.len();

        // Vertices with no incident (non-self) edge become singleton blocks.
        let mut has_incident = vec![false; n];
        for &(u, v) in &edges {
            has_incident[u] = true;
            has_incident[v] = true;
        }
        for v in 0..n {
            if !has_incident[v] {
                result.insert(vec![v]);
            }
        }

        // Edge index lookup.
        let edge_index = |a: usize, b: usize| -> Option<usize> {
            let key = if a < b { (a, b) } else { (b, a) };
            edges.binary_search(&key).ok()
        };

        // Union-find over edges.
        let mut parent: Vec<usize> = (0..m).collect();
        let mut union = |parent: &mut [usize], a: usize, b: usize| {
            let ra = find_uf(parent, a);
            let rb = find_uf(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        };

        // Each edge is its own block at minimum (covers bridges).
        // The simple-cycle enumeration handles the rest.

        // Enumerate every simple cycle starting at every vertex.
        for start in 0..n {
            let mut on_path = vec![false; n];
            let mut path_edges: Vec<usize> = Vec::new();
            dfs_cycles(
                adj,
                &edge_index,
                start,
                start,
                &mut on_path,
                &mut path_edges,
                &mut parent,
                &mut union,
                usize::MAX,
            );
        }

        // Cluster edges and emit each cluster as a block.
        let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for ei in 0..m {
            let r = find_uf(&mut parent, ei);
            clusters.entry(r).or_default().push(ei);
        }
        for eis in clusters.into_values() {
            let mut verts: Vec<usize> = Vec::new();
            for ei in eis {
                let (a, b) = edges[ei];
                verts.push(a);
                verts.push(b);
            }
            verts.sort_unstable();
            verts.dedup();
            result.insert(verts);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs_cycles<F, U>(
        adj: &[Vec<usize>],
        edge_index: &F,
        start: usize,
        current: usize,
        on_path: &mut [bool],
        path_edges: &mut Vec<usize>,
        parent: &mut [usize],
        union: &mut U,
        last_edge: usize,
    ) where
        F: Fn(usize, usize) -> Option<usize>,
        U: FnMut(&mut [usize], usize, usize),
    {
        on_path[current] = true;
        for &nxt in &adj[current] {
            if nxt == current {
                continue;
            }
            let Some(ei) = edge_index(current, nxt) else {
                continue;
            };
            if ei == last_edge {
                continue;
            }
            if nxt == start && path_edges.len() >= 2 {
                // Found a simple cycle: union all its edges.
                let mut prev = path_edges[0];
                for &cur in path_edges.iter().skip(1) {
                    union(parent, prev, cur);
                    prev = cur;
                }
                union(parent, prev, ei);
                continue;
            }
            if !on_path[nxt] && nxt > start {
                // Restrict to nxt > start to enumerate each cycle once
                // per starting orientation; lower-numbered vertices were
                // already handled when they were `start`. We still need
                // to permit reaching `start` itself to close a cycle,
                // which is handled above.
                path_edges.push(ei);
                dfs_cycles(
                    adj, edge_index, start, nxt, on_path, path_edges, parent, union, ei,
                );
                path_edges.pop();
            }
        }
        on_path[current] = false;
    }

    #[quickcheck]
    fn quickcheck_brute_force_matches(seed_edges: Vec<(u8, u8)>) -> bool {
        // Restrict to n = 6 vertices.
        let n: usize = 6;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (a, b) in seed_edges {
            let u = (a as usize) % n;
            let v = (b as usize) % n;
            if u != v {
                let (lo, hi) = if u < v { (u, v) } else { (v, u) };
                edges.push((lo, hi));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let g = undirected(&edges, n);

        let bct = BlockCutTree::build(&g);

        // Property 1: union of all blocks is exactly {0..n}.
        let mut covered: BTreeSet<usize> = BTreeSet::new();
        for block in &bct.blocks {
            for &v in block {
                covered.insert(v);
            }
        }
        if covered != (0..n).collect::<BTreeSet<usize>>() {
            return false;
        }

        // Property 2: matches brute-force block decomposition.
        let ours = block_set(&bct);
        let theirs = brute_blocks(&g);
        if ours != theirs {
            return false;
        }

        // Property 3: cut-vertex flag matches brute-force articulation.
        let brute_cuts = brute_articulation(&g);
        if bct.cut_vertices != brute_cuts {
            return false;
        }

        // Property 4: a vertex is a cut vertex iff it appears in >= 2 blocks.
        for v in 0..n {
            let count = bct.block_of_vertex[v].len();
            if bct.cut_vertices[v] != (count >= 2) {
                return false;
            }
        }

        true
    }
}
