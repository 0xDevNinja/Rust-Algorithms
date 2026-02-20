//! Fenwick tree (binary-indexed tree) over `i64`. Point update + prefix-sum
//! query in O(log n). Internally 1-indexed; the public API is 0-indexed.

/// Binary-indexed tree supporting point updates and prefix-sum queries.
pub struct FenwickTree {
    tree: Vec<i64>,
}

impl FenwickTree {
    /// Creates a new tree of size `n` filled with zeroes.
    pub fn new(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }

    /// Adds `delta` to index `idx` (0-indexed). O(log n).
    pub fn update(&mut self, idx: usize, delta: i64) {
        let mut i = idx + 1;
        while i < self.tree.len() {
            self.tree[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    /// Returns the sum of `arr[0..=idx]` (0-indexed inclusive). O(log n).
    pub fn prefix_sum(&self, idx: usize) -> i64 {
        let mut i = idx + 1;
        let mut sum = 0_i64;
        while i > 0 {
            sum += self.tree[i];
            i -= i & i.wrapping_neg();
        }
        sum
    }

    /// Returns the sum of `arr[lo..=hi]` (0-indexed inclusive). O(log n).
    pub fn range_sum(&self, lo: usize, hi: usize) -> i64 {
        let upper = self.prefix_sum(hi);
        if lo == 0 {
            upper
        } else {
            upper - self.prefix_sum(lo - 1)
        }
    }

    /// Number of elements the tree was created over.
    pub const fn len(&self) -> usize {
        self.tree.len() - 1
    }

    /// True if the tree is empty.
    pub const fn is_empty(&self) -> bool {
        self.tree.len() <= 1
    }
}

#[cfg(test)]
mod tests {
    use super::FenwickTree;

    #[test]
    fn empty() {
        let ft = FenwickTree::new(0);
        assert!(ft.is_empty());
        assert_eq!(ft.len(), 0);
    }

    #[test]
    fn single_element_updates() {
        let mut ft = FenwickTree::new(1);
        ft.update(0, 5);
        assert_eq!(ft.prefix_sum(0), 5);
        ft.update(0, 7);
        assert_eq!(ft.prefix_sum(0), 12);
    }

    #[test]
    fn prefix_sum_after_updates() {
        let mut ft = FenwickTree::new(8);
        // arr = [1, 2, 3, 4, 5, 6, 7, 8]
        for (i, v) in (1..=8).enumerate() {
            ft.update(i, v);
        }
        assert_eq!(ft.prefix_sum(0), 1);
        assert_eq!(ft.prefix_sum(3), 10);
        assert_eq!(ft.prefix_sum(7), 36);
    }

    #[test]
    fn range_sum() {
        let mut ft = FenwickTree::new(5);
        for (i, v) in [3, 1, 4, 1, 5].iter().enumerate() {
            ft.update(i, *v);
        }
        assert_eq!(ft.range_sum(0, 4), 14);
        assert_eq!(ft.range_sum(1, 3), 6); // 1 + 4 + 1
        assert_eq!(ft.range_sum(2, 2), 4);
    }

    #[test]
    fn negative_deltas() {
        let mut ft = FenwickTree::new(4);
        ft.update(0, 10);
        ft.update(1, 5);
        ft.update(0, -3);
        assert_eq!(ft.prefix_sum(1), 12);
    }
}
