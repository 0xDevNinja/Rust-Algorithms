# Rust-Algorithms

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
├── sorting/          comparison and non-comparison sorts
├── searching/        ordered- and unordered-collection searches
├── graph/            traversal, shortest paths, MST
├── dynamic_programming/
│                     classic DP recurrences
└── lib.rs            re-exports
```

Every algorithm lives in its own file with inline `#[cfg(test)]` tests. Most
sorts and DP routines additionally have property-based tests via
[`quickcheck`](https://docs.rs/quickcheck).

## Build

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Minimum supported Rust version: 1.74 (edition 2021).

## Algorithms

### Sorting
| Algorithm | Time (avg) | Time (worst) | Space | Stable |
|-----------|------------|--------------|-------|--------|
| Bubble    | O(n²)      | O(n²)        | O(1)  | yes    |
| Selection | O(n²)      | O(n²)        | O(1)  | no     |
| Insertion | O(n²)      | O(n²)        | O(1)  | yes    |
| Merge     | O(n log n) | O(n log n)   | O(n)  | yes    |
| Quick     | O(n log n) | O(n²)        | O(log n) | no |
| Heap      | O(n log n) | O(n log n)   | O(1)  | no     |
| Counting  | O(n + k)   | O(n + k)     | O(k)  | yes    |
| Radix     | O(d·(n+b)) | O(d·(n+b))   | O(n+b)| yes    |
| Shell     | O(n^1.3)*  | O(n²)        | O(1)  | no     |
| Tim       | O(n log n) | O(n log n)   | O(n)  | yes    |
| Bucket    | O(n + k)*  | O(n²)        | O(n+k)| yes    |
| Gnome     | O(n²)      | O(n²)        | O(1)  | yes    |
| Comb      | ~O(n log n)| O(n²)        | O(1)  | no     |
| Pigeonhole| O(n + r)   | O(n + r)     | O(r)  | yes    |

### Searching
- Linear, Binary, Jump, Exponential, Interpolation, Ternary, Fibonacci
- Sublist (subarray) search — naive O(n·m) substring match

### Graph
- Breadth-first, Depth-first, Dijkstra, Bellman–Ford, Kruskal, Prim,
  Topological sort, Floyd–Warshall, A* search, Tarjan SCC, Kosaraju SCC,
  Edmonds–Karp max-flow

### Dynamic Programming
- Fibonacci (memoised), 0/1 Knapsack, Longest Common Subsequence,
  Longest Increasing Subsequence, Edit Distance, Coin Change,
  Matrix-Chain Multiplication, Rod Cutting

## Contributing

1. Pick an open issue (or open a new one describing the algorithm).
2. Create a file `src/<category>/<algorithm>.rs` containing the implementation
   and an inline test module.
3. Run `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`.
4. Open a pull request.

## License

[MIT](LICENSE).
