//! Skip list (Pugh 1990) — probabilistic ordered set.
//!
//! A skip list is a multi-level linked list where each node is promoted to the
//! next level with probability `p = 0.5`.  It achieves **O(log n) expected**
//! time for `insert`, `remove`, and `contains`, and **O(n log n) expected**
//! space, without the rotations required by balanced BSTs.
//!
//! # Layout
//! Nodes live in an arena (`Vec<Node<K>>`).  Links are `Option<usize>` indices
//! into that arena — no raw pointers, no `unsafe`.  Node 0 is the sentinel
//! head whose `key` is `None`; all real nodes have `key = Some(k)`.
//!
//! # Complexity
//! - `insert` / `remove` / `contains`: **O(log n)** expected time.
//! - `iter`: **O(n)**.
//! - Space: **O(n · `MAX_LEVEL` / 2)** expected ≈ **O(n)**.
//!
//! # Preconditions
//! Keys must implement `Ord`.  Duplicate keys are silently rejected
//! (`insert` returns `false`).
//!
//! # Capacity
//! `MAX_LEVEL = 16` supports up to ≈ 65 536 elements before the level cap
//! becomes a practical bottleneck.  Raise this constant if you need more.

/// Maximum number of levels in the skip list.
/// With `p = 0.5` this supports roughly `2^MAX_LEVEL` = 65 536 keys before
/// the probabilistic height bound starts degrading expected performance.
const MAX_LEVEL: usize = 16;

/// A single node stored in the arena.
struct Node<K> {
    /// `None` only for the sentinel head (index 0).
    key: Option<K>,
    /// `forward[i]` is the index of the next node at level `i`, or `None`.
    forward: Vec<Option<usize>>,
}

/// A probabilistic ordered set backed by a skip list.
///
/// Set semantics: duplicate keys are rejected. Keys are iterated in ascending
/// order via [`SkipList::iter`].
pub struct SkipList<K: Ord> {
    /// Node arena.  Index 0 is always the sentinel head.
    nodes: Vec<Node<K>>,
    /// Current effective height (1-indexed).  Starts at 1.
    level: usize,
    /// Number of real (non-sentinel) keys stored.
    len: usize,
    /// `XorShift64` state for a fast, seedable, dependency-free PRNG.
    rng_state: u64,
}

// ---- internal helpers ----------------------------------------------------- //

impl<K: Ord> SkipList<K> {
    /// `XorShift64` PRNG — one step.
    const fn next_rand(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Flip a fair coin using the internal PRNG.
    const fn coin(&mut self) -> bool {
        self.next_rand() & 1 == 0
    }

    /// Draw a random level in `[1, MAX_LEVEL]`.
    const fn random_level(&mut self) -> usize {
        let mut lvl = 1;
        while lvl < MAX_LEVEL && self.coin() {
            lvl += 1;
        }
        lvl
    }

    /// Allocate a new node and return its arena index.
    fn alloc(&mut self, key: Option<K>, height: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            key,
            forward: vec![None; height],
        });
        idx
    }

    /// Build the `update` array: for each level `i`, `update[i]` is the index
    /// of the rightmost node whose forward pointer at level `i` must be
    /// patched when inserting / removing `key`.
    fn find_predecessors(&self, key: &K) -> [usize; MAX_LEVEL] {
        let mut update = [0usize; MAX_LEVEL];
        let mut cur = 0usize; // start at head
                              // Walk from the highest active level downward.
        let top = self.level; // 1-indexed; levels are 0..top-1
        for i in (0..top).rev() {
            while let Some(nxt) = self.nodes[cur].forward[i] {
                // Only real nodes have Some(key).
                let nxt_key = self.nodes[nxt].key.as_ref().expect("real node");
                if nxt_key < key {
                    cur = nxt;
                } else {
                    break;
                }
            }
            update[i] = cur;
        }
        update
    }
}

// ---- public API ----------------------------------------------------------- //

impl<K: Ord> SkipList<K> {
    /// Creates an empty skip list seeded from a platform entropy source.
    pub fn new() -> Self {
        // Mix in a pseudo-random seed from a compile-time constant plus a
        // per-invocation value.  Not cryptographically strong, but adequate
        // for a probabilistic data structure.
        let seed = {
            // Start from a well-distributed constant.
            let mut s: u64 = 0x853c_49e6_748f_ea9b;
            // XOR with the address of a local variable for ASLR entropy.
            #[allow(clippy::borrow_as_ptr)]
            let stack_val: u64 = (&raw const s) as u64;
            s ^= stack_val;
            s ^= s.wrapping_mul(0x6c62_272e_07bb_0142);
            if s == 0 {
                0xdead_beef_cafe_1234
            } else {
                s
            }
        };
        Self::with_seed(seed)
    }

    /// Creates an empty skip list with the given PRNG seed.  Use this for
    /// deterministic tests.
    pub fn with_seed(seed: u64) -> Self {
        let mut sl = Self {
            nodes: Vec::new(),
            level: 1,
            len: 0,
            rng_state: if seed == 0 { 1 } else { seed }, // XorShift must not be 0
        };
        // Allocate sentinel head at index 0.
        sl.alloc(None, MAX_LEVEL);
        sl
    }

    /// Returns the number of keys stored.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the set contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if `key` is present in the set.
    pub fn contains(&self, key: &K) -> bool {
        let mut cur = 0usize;
        for i in (0..self.level).rev() {
            while let Some(nxt) = self.nodes[cur].forward[i] {
                let nxt_key = self.nodes[nxt].key.as_ref().expect("real node");
                match nxt_key.cmp(key) {
                    std::cmp::Ordering::Less => cur = nxt,
                    std::cmp::Ordering::Equal => return true,
                    std::cmp::Ordering::Greater => break,
                }
            }
        }
        false
    }

    /// Inserts `key` into the set.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was already
    /// present (duplicate rejected).
    pub fn insert(&mut self, key: K) -> bool {
        let update = self.find_predecessors(&key);

        // Check for duplicate: peek at the level-0 successor of update[0].
        if let Some(nxt) = self.nodes[update[0]].forward[0] {
            if self.nodes[nxt].key.as_ref().expect("real node") == &key {
                return false; // duplicate
            }
        }

        let new_level = self.random_level();

        // Raise effective level if the new node is taller.
        if new_level > self.level {
            // update[self.level..new_level] already defaults to 0 (head).
            self.level = new_level;
        }

        let new_idx = self.alloc(Some(key), new_level);

        // Splice into each level.
        for (i, &pred) in update.iter().enumerate().take(new_level) {
            let old_fwd = self.nodes[pred].forward[i];
            self.nodes[new_idx].forward[i] = old_fwd;
            self.nodes[pred].forward[i] = Some(new_idx);
        }

        self.len += 1;
        true
    }

    /// Removes `key` from the set.
    ///
    /// Returns `true` if the key was present and removed, `false` otherwise.
    pub fn remove(&mut self, key: &K) -> bool {
        let update = self.find_predecessors(key);

        // The candidate node is the level-0 successor of update[0].
        let Some(target) = self.nodes[update[0]].forward[0] else {
            return false;
        };

        if self.nodes[target].key.as_ref().expect("real node") != key {
            return false; // not found
        }

        // Unlink from every level where this node appears.
        for i in 0..self.level {
            if self.nodes[update[i]].forward[i] == Some(target) {
                self.nodes[update[i]].forward[i] = self.nodes[target].forward[i];
            } else {
                break; // levels above this one don't contain the target
            }
        }

        // Shrink effective level if the top level(s) of the head are now empty.
        while self.level > 1 && self.nodes[0].forward[self.level - 1].is_none() {
            self.level -= 1;
        }

        self.len -= 1;
        true
    }

    /// Returns an iterator that yields keys in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = &K> {
        SkipListIter {
            nodes: &self.nodes,
            cur: self.nodes[0].forward[0], // first real node
        }
    }
}

impl<K: Ord> Default for SkipList<K> {
    fn default() -> Self {
        Self::new()
    }
}

// ---- iterator ------------------------------------------------------------- //

struct SkipListIter<'a, K> {
    nodes: &'a [Node<K>],
    cur: Option<usize>,
}

impl<'a, K> Iterator for SkipListIter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.cur?;
        let node = &self.nodes[idx];
        self.cur = node.forward[0];
        node.key.as_ref() // always Some for real nodes
    }
}

// ---- tests ---------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::SkipList;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;

    // ---- unit tests -------------------------------------------------------

    #[test]
    fn empty_state() {
        let sl: SkipList<i32> = SkipList::new();
        assert_eq!(sl.len(), 0);
        assert!(sl.is_empty());
        assert!(!sl.contains(&0));
        assert_eq!(sl.iter().count(), 0);
    }

    #[test]
    fn remove_absent_returns_false_on_empty() {
        let mut sl: SkipList<i32> = SkipList::new();
        assert!(!sl.remove(&42));
    }

    #[test]
    fn single_insert_and_contains() {
        let mut sl = SkipList::new();
        assert!(sl.insert(7i32));
        assert!(sl.contains(&7));
        assert!(!sl.contains(&0));
        assert_eq!(sl.len(), 1);
        assert!(!sl.is_empty());
    }

    #[test]
    fn duplicate_insert_returns_false() {
        let mut sl = SkipList::new();
        assert!(sl.insert(3i32));
        assert!(!sl.insert(3i32)); // duplicate
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn iter_yields_ascending_order() {
        let mut sl = SkipList::with_seed(1);
        let keys = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        for &k in &keys {
            sl.insert(k);
        }
        let collected: Vec<i32> = sl.iter().copied().collect();
        let mut sorted = keys;
        sorted.sort_unstable();
        assert_eq!(collected, sorted);
    }

    #[test]
    fn remove_and_contains() {
        let mut sl = SkipList::with_seed(2);
        sl.insert(10i32);
        sl.insert(20);
        sl.insert(30);
        assert!(sl.remove(&20));
        assert!(!sl.contains(&20));
        assert!(sl.contains(&10));
        assert!(sl.contains(&30));
        assert_eq!(sl.len(), 2);
    }

    #[test]
    fn remove_absent_key_returns_false() {
        let mut sl = SkipList::with_seed(3);
        sl.insert(1i32);
        assert!(!sl.remove(&99));
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn large_deterministic_insert_and_remove() {
        // Insert 1000 distinct keys, then remove 500, verify sorted iter and len.
        let mut sl = SkipList::with_seed(42);

        // Use a simple LCG to produce 1000 distinct values for reproducibility.
        let mut lcg_state = 12345u64;
        let mut lcg = || -> i64 {
            lcg_state = lcg_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (lcg_state >> 33) as i64
        };

        let mut keys: Vec<i64> = (0..1000).map(|_| lcg()).collect();
        // Deduplicate so we get exactly distinct keys.
        keys.sort_unstable();
        keys.dedup();
        let n = keys.len();

        for &k in &keys {
            assert!(sl.insert(k));
        }
        assert_eq!(sl.len(), n);

        // Remove the first 500 keys.
        let to_remove = &keys[..500.min(n)];
        let removed = to_remove.len();
        for &k in to_remove {
            assert!(sl.remove(&k));
        }
        assert_eq!(sl.len(), n - removed);

        // Remaining iter must be sorted.
        let v: Vec<i64> = sl.iter().copied().collect();
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(v, sorted);
    }

    // ---- property / model-checked test -----------------------------------

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32),
        Remove(i32),
        Contains(i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            // Key space restricted to 0..=31 to force collisions.
            // Use wrapping_abs to avoid overflow when i32::MIN is generated.
            let k = i32::arbitrary(g).wrapping_abs() % 32;
            match u8::arbitrary(g) % 3 {
                0 => Self::Insert(k),
                1 => Self::Remove(k),
                _ => Self::Contains(k),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn model_matches_btreeset(ops: Vec<Op>) -> bool {
        let mut sl: SkipList<i32> = SkipList::with_seed(0xdead_beef);
        let mut oracle: BTreeSet<i32> = BTreeSet::new();

        for op in ops {
            match op {
                Op::Insert(k) => {
                    let sl_res = sl.insert(k);
                    let bt_res = oracle.insert(k);
                    if sl_res != bt_res {
                        return false;
                    }
                }
                Op::Remove(k) => {
                    let sl_res = sl.remove(&k);
                    let bt_res = oracle.remove(&k);
                    if sl_res != bt_res {
                        return false;
                    }
                }
                Op::Contains(k) => {
                    if sl.contains(&k) != oracle.contains(&k) {
                        return false;
                    }
                }
            }

            // After every operation, iter must match BTreeSet::iter.
            let sl_keys: Vec<i32> = sl.iter().copied().collect();
            let bt_keys: Vec<i32> = oracle.iter().copied().collect();
            if sl_keys != bt_keys {
                return false;
            }

            // len must also agree.
            if sl.len() != oracle.len() {
                return false;
            }
        }
        true
    }
}
