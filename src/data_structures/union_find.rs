//! Union–find / disjoint-set data structure with union by rank and path
//! compression. Each operation runs in amortised inverse-Ackermann time
//! (effectively constant for any reasonable input size).

/// Disjoint set forest over `n` elements `0..n`.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
    components: usize,
}

impl UnionFind {
    /// Creates a new disjoint-set forest with `n` singleton components.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            components: n,
        }
    }

    /// Returns the canonical representative of `x`'s component, applying
    /// path compression along the way.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    /// Unions the components containing `a` and `b`. Returns `true` if a
    /// merge occurred, `false` if they were already in the same set.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        self.components -= 1;
        true
    }

    /// Returns `true` if `a` and `b` are in the same component.
    pub fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    /// Number of disjoint components.
    pub const fn component_count(&self) -> usize {
        self.components
    }

    /// Total number of elements.
    pub const fn len(&self) -> usize {
        self.parent.len()
    }

    /// True if there are no elements.
    pub const fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::UnionFind;

    #[test]
    fn empty() {
        let dsu = UnionFind::new(0);
        assert!(dsu.is_empty());
        assert_eq!(dsu.component_count(), 0);
    }

    #[test]
    fn singletons() {
        let dsu = UnionFind::new(5);
        assert_eq!(dsu.component_count(), 5);
        assert_eq!(dsu.len(), 5);
    }

    #[test]
    fn union_and_find() {
        let mut dsu = UnionFind::new(6);
        assert!(dsu.union(0, 1));
        assert!(dsu.union(2, 3));
        assert!(dsu.union(1, 2));
        assert!(dsu.connected(0, 3));
        assert!(!dsu.connected(0, 4));
        assert_eq!(dsu.component_count(), 3); // {0,1,2,3}, {4}, {5}
    }

    #[test]
    fn duplicate_union_returns_false() {
        let mut dsu = UnionFind::new(3);
        assert!(dsu.union(0, 1));
        assert!(!dsu.union(1, 0));
        assert_eq!(dsu.component_count(), 2);
    }

    #[test]
    fn path_compression_does_not_corrupt() {
        let mut dsu = UnionFind::new(8);
        for i in 1..8 {
            dsu.union(i - 1, i);
        }
        for i in 0..8 {
            assert_eq!(dsu.find(i), dsu.find(0));
        }
        assert_eq!(dsu.component_count(), 1);
    }
}
