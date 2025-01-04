# Contributing

Thanks for your interest in improving Rust-Algorithms.

## Before opening a PR

1. Run the local check suite:
   ```sh
   cargo fmt --check
   cargo clippy --all-targets -- -D warnings
   cargo test
   ```
2. New algorithms must include:
   - An implementation in `src/<category>/<algorithm>.rs`.
   - An inline `#[cfg(test)] mod tests { ... }` covering empty input, single
     element, sorted/reverse-sorted/random, and (where feasible) a
     `quickcheck` property comparing against `slice::sort` or another
     reference.
   - A short doc comment naming the algorithm and citing complexity.
3. Keep modules focused — one algorithm per file.

## Style

- Follow `rustfmt` defaults.
- Prefer iterators and slice methods over manual indexing where readable.
- Avoid `unsafe` in this repository.

## Commit messages

Conventional Commits:

- `feat(sorting): add radix sort`
- `fix(graph): correct edge weight in dijkstra test`
- `refactor(searching): unify binary search variants`
- `test(dp): add edit-distance property tests`
