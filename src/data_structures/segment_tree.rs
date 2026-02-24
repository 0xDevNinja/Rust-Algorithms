//! Segment tree with lazy propagation supporting range-add updates and
//! range-sum queries over `i64`. O(log n) per operation, O(n) space.

/// Range-add / range-sum segment tree.
pub struct SegmentTree {
    n: usize,
    tree: Vec<i64>,
    lazy: Vec<i64>,
}

impl SegmentTree {
    /// Builds a tree of length `n` initialised to zeros.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            tree: vec![0; 4 * n.max(1)],
            lazy: vec![0; 4 * n.max(1)],
        }
    }

    /// Builds a tree initialised from `data`.
    pub fn from_slice(data: &[i64]) -> Self {
        let mut t = Self::new(data.len());
        if !data.is_empty() {
            t.build(1, 0, data.len() - 1, data);
        }
        t
    }

    fn build(&mut self, node: usize, lo: usize, hi: usize, data: &[i64]) {
        if lo == hi {
            self.tree[node] = data[lo];
            return;
        }
        let mid = lo.midpoint(hi);
        self.build(2 * node, lo, mid, data);
        self.build(2 * node + 1, mid + 1, hi, data);
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1];
    }

    fn push_down(&mut self, node: usize, lo: usize, hi: usize) {
        if self.lazy[node] != 0 {
            let mid = lo.midpoint(hi);
            let left = 2 * node;
            let right = 2 * node + 1;
            let pending = self.lazy[node];
            self.tree[left] += pending * (mid - lo + 1) as i64;
            self.lazy[left] += pending;
            self.tree[right] += pending * (hi - mid) as i64;
            self.lazy[right] += pending;
            self.lazy[node] = 0;
        }
    }

    fn update_inner(
        &mut self,
        node: usize,
        lo: usize,
        hi: usize,
        ql: usize,
        qh: usize,
        delta: i64,
    ) {
        if qh < lo || hi < ql {
            return;
        }
        if ql <= lo && hi <= qh {
            self.tree[node] += delta * (hi - lo + 1) as i64;
            self.lazy[node] += delta;
            return;
        }
        self.push_down(node, lo, hi);
        let mid = lo.midpoint(hi);
        self.update_inner(2 * node, lo, mid, ql, qh, delta);
        self.update_inner(2 * node + 1, mid + 1, hi, ql, qh, delta);
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1];
    }

    /// Adds `delta` to every position in `[lo, hi]` (inclusive, 0-indexed).
    pub fn range_add(&mut self, lo: usize, hi: usize, delta: i64) {
        if self.n == 0 || lo > hi || hi >= self.n {
            return;
        }
        self.update_inner(1, 0, self.n - 1, lo, hi, delta);
    }

    fn query_inner(&mut self, node: usize, lo: usize, hi: usize, ql: usize, qh: usize) -> i64 {
        if qh < lo || hi < ql {
            return 0;
        }
        if ql <= lo && hi <= qh {
            return self.tree[node];
        }
        self.push_down(node, lo, hi);
        let mid = lo.midpoint(hi);
        self.query_inner(2 * node, lo, mid, ql, qh)
            + self.query_inner(2 * node + 1, mid + 1, hi, ql, qh)
    }

    /// Returns the inclusive range sum of `[lo, hi]`.
    pub fn range_sum(&mut self, lo: usize, hi: usize) -> i64 {
        if self.n == 0 || lo > hi || hi >= self.n {
            return 0;
        }
        self.query_inner(1, 0, self.n - 1, lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::SegmentTree;

    #[test]
    fn empty() {
        let mut t = SegmentTree::new(0);
        assert_eq!(t.range_sum(0, 0), 0);
    }

    #[test]
    fn single_element() {
        let mut t = SegmentTree::from_slice(&[42]);
        assert_eq!(t.range_sum(0, 0), 42);
        t.range_add(0, 0, 8);
        assert_eq!(t.range_sum(0, 0), 50);
    }

    #[test]
    fn pure_query_no_updates() {
        let mut t = SegmentTree::from_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(t.range_sum(0, 4), 15);
        assert_eq!(t.range_sum(1, 3), 9);
    }

    #[test]
    fn range_add_then_query() {
        let mut t = SegmentTree::from_slice(&[0; 10]);
        t.range_add(2, 5, 3);
        assert_eq!(t.range_sum(0, 9), 12); // 4 elements * 3
        assert_eq!(t.range_sum(2, 5), 12);
        assert_eq!(t.range_sum(0, 1), 0);
    }

    #[test]
    fn overlapping_updates() {
        let mut t = SegmentTree::from_slice(&[0; 8]);
        t.range_add(0, 7, 1);
        t.range_add(3, 5, 2);
        // arr = [1,1,1,3,3,3,1,1]
        assert_eq!(t.range_sum(0, 7), 14);
        assert_eq!(t.range_sum(3, 5), 9);
    }

    #[test]
    fn against_brute_force() {
        let mut data = vec![0_i64; 16];
        let mut t = SegmentTree::from_slice(&data);
        let updates: &[(usize, usize, i64)] = &[(0, 4, 5), (3, 9, -2), (10, 15, 7), (5, 12, 1)];
        for &(lo, hi, d) in updates {
            t.range_add(lo, hi, d);
            for x in &mut data[lo..=hi] {
                *x += d;
            }
        }
        for lo in 0..data.len() {
            for hi in lo..data.len() {
                let expected: i64 = data[lo..=hi].iter().sum();
                assert_eq!(t.range_sum(lo, hi), expected);
            }
        }
    }
}
