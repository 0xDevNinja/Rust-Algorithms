# Rust-Algorithms

[![CI](https://github.com/0xDevNinja/Rust-Algorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/0xDevNinja/Rust-Algorithms/actions/workflows/ci.yml)
[![Coverage](https://github.com/0xDevNinja/Rust-Algorithms/actions/workflows/coverage.yml/badge.svg)](https://github.com/0xDevNinja/Rust-Algorithms/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/0xDevNinja/Rust-Algorithms/branch/main/graph/badge.svg)](https://codecov.io/gh/0xDevNinja/Rust-Algorithms)

Classical algorithms implemented in idiomatic Rust, each paired with a thorough
test suite. The goal of this repository is twofold:

1. Provide reference implementations that prioritise correctness and clarity
   over micro-optimisation.
2. Document the trade-offs (time / space / stability / preconditions) of each
   algorithm so the code reads as a study companion as well as a library.

## Status

Active. New algorithms are added regularly. Open issues describe the next
batch of work; pull requests are welcome.

## Layout

```
src/
├── backtracking/        recursive search with pruning
├── bit_manipulation/    bit tricks and subset enumeration
├── data_structures/     union-find, Fenwick, segment, sparse table, etc.
├── dynamic_programming/ classic DP recurrences
├── geometry/            computational geometry
├── graph/               traversal, shortest paths, flow, matching
├── greedy/              greedy-choice algorithms
├── math/                number theory and elementary math
├── searching/           ordered- and unordered-collection searches
├── sorting/             comparison and non-comparison sorts
├── string/              substring search and suffix structures
└── lib.rs               re-exports
```

Every algorithm lives in its own file with inline `#[cfg(test)]` tests. Most
modules also have property-based tests via
[`quickcheck`](https://docs.rs/quickcheck).

## Build

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Minimum supported Rust version: 1.74 (edition 2021).

A flat index of every implementation lives in [DIRECTORY.md](DIRECTORY.md);
it is regenerated automatically on every push to `main`.

## Algorithms

### Sorting
| Algorithm | Time (avg) | Time (worst) | Space | Stable |
|-----------|------------|--------------|-------|--------|
| Bubble    | O(n²)      | O(n²)        | O(1)  | yes    |
| Selection | O(n²)      | O(n²)        | O(1)  | no     |
| Insertion | O(n²)      | O(n²)        | O(1)  | yes    |
| Merge     | O(n log n) | O(n log n)   | O(n)  | yes    |
| Quick     | O(n log n) | O(n²)        | O(log n) | no |
| Randomized Quick | O(n log n) | O(n²) | O(log n) | no |
| Heap      | O(n log n) | O(n log n)   | O(1)  | no     |
| Counting  | O(n + k)   | O(n + k)     | O(k)  | yes    |
| Radix     | O(d·(n+b)) | O(d·(n+b))   | O(n+b)| yes    |
| Shell     | O(n^1.3)*  | O(n²)        | O(1)  | no     |
| Tim       | O(n log n) | O(n log n)   | O(n)  | yes    |
| Bucket    | O(n + k)*  | O(n²)        | O(n+k)| yes    |
| Gnome     | O(n²)      | O(n²)        | O(1)  | yes    |
| Comb      | ~O(n log n)| O(n²)        | O(1)  | no     |
| Pigeonhole| O(n + r)   | O(n + r)     | O(r)  | yes    |

Selection / order statistics:
- Quickselect — kth order statistic, O(n) avg / O(n²) worst
- Median-of-medians (BFPRT) — kth order statistic, O(n) worst case
- External k-way merge — heap-based merge of k sorted runs, O(N log k)

### Searching
- Linear, Binary, Jump, Exponential, Interpolation, Ternary, Fibonacci
- Sublist (subarray) search — naive O(n·m) substring match

### Graph
Traversal & shortest path:
- BFS, DFS, 0-1 BFS, Dijkstra, Bellman–Ford, A* search
- DAG single-source shortest path (toposort + relax)
- Floyd–Warshall (all-pairs), Johnson's all-pairs (reweighted Dijkstra)
- Tree diameter (two-BFS, O(N))

MST & connectivity:
- Kruskal, Prim, Borůvka
- Tarjan SCC, Kosaraju SCC
- Bridges & articulation points
- LCA via binary lifting, heavy-light decomposition, centroid decomposition

Flows & matching:
- Edmonds–Karp, Dinic's, push-relabel max-flow
- Min-cost max-flow (successive shortest paths)
- Hopcroft–Karp bipartite matching, König's theorem (vertex cover)

Cuts & cycles:
- Karger's randomized min-cut, Stoer–Wagner global min-cut
- Bipartite check (BFS 2-coloring), 2-SAT solver
- Eulerian path / circuit (Hierholzer)
- Functional-graph cycle detection (rho structure)

Specialised:
- Topological sort, minimum path cover on DAG (Dilworth)
- De Bruijn sequence, Bron–Kerbosch maximum clique
- Kirchhoff's matrix-tree theorem (spanning tree count)

### Greedy
- Activity selection / interval scheduling
- Boyer-Moore majority vote
- Fractional knapsack
- Job sequencing with deadlines

### Data Structures
- Union-find (disjoint set) — union by rank + path compression
- Fenwick tree (binary indexed tree) — point update / prefix-sum query in O(log n)
- Segment tree with lazy propagation — range add / range sum in O(log n)
- Trie (prefix tree) — insert / contains / starts_with in O(L)
- LRU cache — slab + HashMap, O(1) amortised get/put
- Sqrt decomposition — point update / range sum in O(√n)
- Coordinate compression — rank values into dense indices, O(n log n)
- Sparse table — idempotent range queries (min/max/gcd) in O(1) after O(n log n) build
- Cartesian tree — min-heap with in-order property, O(n) build
- Block-cut tree — biconnected components + articulation tree

### String
Substring search:
- KMP — O(n + m), longest-proper-prefix table
- Boyer-Moore (full bad-char + good-suffix) — O(n) avg
- Boyer-Moore-Horspool — bad-char only, simpler
- Rabin–Karp — polynomial rolling hash
- Finite-automaton match — DFA-based, O(n) match after O(m·256) build
- Aho-Corasick — multi-pattern, O(n + m + z)
- Z-algorithm — Z-array in O(n)

Structures:
- Suffix array — sorted suffixes, O(n log² n)
- LCP array (Kasai) — O(n) given suffix array
- Suffix automaton — Blumer's online construction
- Palindromic tree (Eertree) — online O(n)
- Manacher's algorithm — longest palindromic substring, O(n)
- Polynomial string hashing — O(1) substring hash after O(n) prep

Other:
- Booth's least rotation, Lyndon decomposition (Duval)
- Anagram detection / signatures, run-length encoding
- Roman numeral conversion

### Backtracking
- N-queens — all solutions or count, by column / diagonal bookkeeping
- Sudoku 9×9 solver — backtracking with row/col/box bitmask bookkeeping
- Permutations and combinations generators
- Knight's tour (Warnsdorff's heuristic)
- Hamiltonian path / cycle

### Bit Manipulation
- Cookbook — count_set_bits, is_power_of_two, next_power_of_two,
  lowest_set_bit, clear_lowest_set_bit, parity, swap_bits
- Subset enumeration — submasks via `s = (s-1) & mask`,
  k-subsets via Gosper's hack

### Math
- Sieve of Eratosthenes — primes up to N in O(N log log N)
- Modular exponentiation — (base^exp) mod m in O(log exp), u128 intermediates
- Extended Euclidean — Bezout coefficients + modular inverse
- GCD and LCM — iterative Euclidean (`const fn`), overflow-safe LCM
- Fast-doubling Fibonacci — O(log n)
- Modular nCr with precomputed factorials — Fermat's little theorem
- Catalan numbers — DP recurrence, O(n²)
- Floyd's cycle detection (tortoise & hare) — O(1) space
- Modular linear equation solver — `a·x ≡ b (mod m)`
- Arbitrary-base conversion — bases 2..=36
- Reservoir sampling (Algorithm R)
- Zeller's congruence — day-of-week from Gregorian date

### Geometry
- Polygon area (Shoelace formula)
- Polygon centroid
- Point-in-polygon (ray casting)
- Convex hull (Andrew's monotone chain) — O(n log n)
- Closest pair of points — divide & conquer, O(n log n)
- Line-segment intersection
- Rotating calipers — polygon diameter, O(n)
- Welzl's smallest enclosing circle — expected O(n)

### Dynamic Programming
- Fibonacci (memoised), 0/1 Knapsack, Longest Common Subsequence,
  Longest Common Substring, Longest Increasing Subsequence, Edit Distance,
  Coin Change, Matrix-Chain Multiplication, Rod Cutting,
  Kadane (max subarray sum), Subset-sum, Longest Palindromic Subsequence,
  Weighted Interval Scheduling, Matrix Exponentiation, Counting Tilings (2×N / 4×N grids)

## Contributing

1. Pick an open issue (or open a new one describing the algorithm).
2. Create a file `src/<category>/<algorithm>.rs` containing the implementation
   and an inline test module.
3. Run `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`.
4. Open a pull request.

## License

[MIT](LICENSE).
