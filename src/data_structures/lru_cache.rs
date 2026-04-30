//! Least-Recently-Used (LRU) cache.
//!
//! A bounded-capacity key/value cache that evicts the **least recently used**
//! entry when full. "Use" means either inserting / updating a key (`put`) or
//! looking it up (`get`); both operations move the affected entry to the
//! most-recently-used end.
//!
//! # Design
//! A doubly-linked list tracks recency: most-recently-used at the head,
//! least-recently-used at the tail. The list is stored in a `Vec` of nodes
//! (a slab) so links are `Option<usize>` indices, side-stepping `Rc`/`RefCell`
//! and keeping the implementation entirely in safe Rust. A `HashMap<K, usize>`
//! maps each live key to its slab index for O(1) lookup. Slots freed by
//! eviction are pushed onto a free-list and reused by subsequent inserts, so
//! the slab grows to at most `capacity` nodes.
//!
//! # Complexity
//! - `get`, `put`, `contains_key`: **O(1)** amortized (one `HashMap` probe
//!   plus constant-time list relinking).
//! - `len`, `is_empty`, `capacity`: **O(1)**.
//! - Space: **O(capacity)**.
//!
//! # Capacity zero
//! A cache constructed with `capacity == 0` never stores anything; every
//! `put(k, v)` returns `Some(v)` (the just-inserted value, reported as if it
//! were immediately evicted) and the cache stays empty.

use std::collections::HashMap;
use std::hash::Hash;

/// Internal slab node. `prev`/`next` are indices into the slab, or `None` for
/// list endpoints. Free slots also live here, linked through `next`.
struct Node<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// A fixed-capacity LRU cache.
///
/// Generic over key type `K` (which must be `Eq + Hash + Clone` so the
/// `HashMap` can own a copy of every live key) and value type `V`.
pub struct LruCache<K: Eq + Hash + Clone, V> {
    capacity: usize,
    map: HashMap<K, usize>,
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    /// Creates an empty cache that holds at most `capacity` entries.
    ///
    /// `capacity == 0` is allowed; see the module docs for its semantics.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            nodes: Vec::with_capacity(capacity),
            free: Vec::new(),
            head: None,
            tail: None,
        }
    }

    /// Returns the maximum number of entries the cache will hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns `true` if `key` is present in the cache.
    ///
    /// Does **not** count as a "use" — recency order is unchanged.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Looks up `key`, marking the entry as most-recently-used on a hit.
    ///
    /// Returns `None` if `key` is absent.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let idx = *self.map.get(key)?;
        self.move_to_head(idx);
        Some(&self.nodes[idx].as_ref().expect("live node").value)
    }

    /// Inserts or updates `key -> value`, marking the entry as
    /// most-recently-used.
    ///
    /// Returns:
    /// - `Some(old_value)` if `key` was already present (the value it
    ///   replaced).
    /// - `Some(evicted_value)` if the insertion pushed the cache over
    ///   capacity, evicting the LRU entry.
    /// - `Some(value)` immediately if `capacity == 0` (the value is reported
    ///   as evicted and the cache stays empty).
    /// - `None` otherwise.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 {
            return Some(value);
        }

        if let Some(&idx) = self.map.get(&key) {
            // Update existing entry and bump to MRU.
            let node = self.nodes[idx].as_mut().expect("live node");
            let old = std::mem::replace(&mut node.value, value);
            self.move_to_head(idx);
            return Some(old);
        }

        // Fresh insert. Evict tail first if we'd otherwise exceed capacity.
        let evicted = if self.map.len() == self.capacity {
            self.pop_tail()
        } else {
            None
        };

        let idx = self.alloc_node(Node {
            key: key.clone(),
            value,
            prev: None,
            next: self.head,
        });
        if let Some(h) = self.head {
            self.nodes[h].as_mut().expect("live node").prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
        self.map.insert(key, idx);

        evicted
    }

    /// Allocates a slab slot for `node`, reusing a freed index if one exists.
    fn alloc_node(&mut self, node: Node<K, V>) -> usize {
        if let Some(idx) = self.free.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            self.nodes.push(Some(node));
            self.nodes.len() - 1
        }
    }

    /// Detaches `idx` from the recency list (does not free the slot).
    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let node = self.nodes[idx].as_ref().expect("live node");
            (node.prev, node.next)
        };
        match prev {
            Some(p) => self.nodes[p].as_mut().expect("live node").next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.nodes[n].as_mut().expect("live node").prev = prev,
            None => self.tail = prev,
        }
        let node = self.nodes[idx].as_mut().expect("live node");
        node.prev = None;
        node.next = None;
    }

    /// Moves `idx` to the head (most-recently-used) of the recency list.
    fn move_to_head(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.detach(idx);
        let node = self.nodes[idx].as_mut().expect("live node");
        node.prev = None;
        node.next = self.head;
        if let Some(h) = self.head {
            self.nodes[h].as_mut().expect("live node").prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    /// Removes the LRU entry (tail) and returns its value.
    fn pop_tail(&mut self) -> Option<V> {
        let idx = self.tail?;
        self.detach(idx);
        let node = self.nodes[idx].take().expect("live node");
        self.map.remove(&node.key);
        self.free.push(idx);
        Some(node.value)
    }
}

impl<K: Eq + Hash + Clone, V> Default for LruCache<K, V> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::LruCache;
    use quickcheck_macros::quickcheck;
    use std::collections::HashMap;

    #[test]
    fn empty_cache() {
        let mut c: LruCache<i32, i32> = LruCache::new(4);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.capacity(), 4);
        assert!(!c.contains_key(&1));
        assert_eq!(c.get(&1), None);
    }

    #[test]
    fn capacity_zero_never_stores() {
        let mut c: LruCache<i32, i32> = LruCache::new(0);
        assert_eq!(c.put(1, 10), Some(10));
        assert_eq!(c.put(2, 20), Some(20));
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.capacity(), 0);
        assert!(!c.contains_key(&1));
        assert_eq!(c.get(&1), None);
    }

    #[test]
    fn capacity_one_keeps_newest_only() {
        let mut c: LruCache<i32, i32> = LruCache::new(1);
        assert_eq!(c.put(1, 10), None);
        assert_eq!(c.len(), 1);
        // Inserting a second key evicts the first.
        assert_eq!(c.put(2, 20), Some(10));
        assert_eq!(c.len(), 1);
        assert!(!c.contains_key(&1));
        assert_eq!(c.get(&2), Some(&20));
        // Updating the only key returns the old value, no eviction.
        assert_eq!(c.put(2, 22), Some(20));
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&2), Some(&22));
    }

    #[test]
    fn basic_put_and_get() {
        let mut c: LruCache<&str, i32> = LruCache::new(3);
        assert_eq!(c.put("a", 1), None);
        assert_eq!(c.put("b", 2), None);
        assert_eq!(c.put("c", 3), None);
        assert_eq!(c.len(), 3);
        assert_eq!(c.get(&"a"), Some(&1));
        assert_eq!(c.get(&"b"), Some(&2));
        assert_eq!(c.get(&"c"), Some(&3));
        assert!(c.contains_key(&"a"));
        assert!(!c.contains_key(&"z"));
    }

    #[test]
    fn lru_eviction_order() {
        // 3-capacity cache, insert 4 items: oldest untouched item is evicted.
        let mut c: LruCache<i32, i32> = LruCache::new(3);
        assert_eq!(c.put(1, 10), None);
        assert_eq!(c.put(2, 20), None);
        assert_eq!(c.put(3, 30), None);
        // Inserting 4 evicts the LRU, which is key 1.
        assert_eq!(c.put(4, 40), Some(10));
        assert!(!c.contains_key(&1));
        assert!(c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert!(c.contains_key(&4));
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn get_promotes_to_mru() {
        let mut c: LruCache<i32, i32> = LruCache::new(3);
        c.put(1, 10);
        c.put(2, 20);
        c.put(3, 30);
        // Touching key 1 makes key 2 the LRU.
        assert_eq!(c.get(&1), Some(&10));
        // Inserting 4 must now evict key 2, not key 1.
        assert_eq!(c.put(4, 40), Some(20));
        assert!(c.contains_key(&1));
        assert!(!c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert!(c.contains_key(&4));
    }

    #[test]
    fn updating_existing_key_bumps_to_mru_and_does_not_evict() {
        let mut c: LruCache<i32, i32> = LruCache::new(3);
        c.put(1, 10);
        c.put(2, 20);
        c.put(3, 30);
        // Update key 1 — should return old value 10, not evict anyone.
        assert_eq!(c.put(1, 11), Some(10));
        assert_eq!(c.len(), 3);
        // Now key 2 is LRU; inserting 4 evicts 2.
        assert_eq!(c.put(4, 40), Some(20));
        assert!(c.contains_key(&1));
        assert!(!c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert!(c.contains_key(&4));
        assert_eq!(c.get(&1), Some(&11));
    }

    #[test]
    fn get_absent_key_returns_none() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        c.put(1, 10);
        assert_eq!(c.get(&99), None);
        assert_eq!(c.get(&1), Some(&10));
    }

    #[test]
    fn len_is_empty_capacity_correctness() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        assert!(c.is_empty());
        assert_eq!(c.capacity(), 2);
        c.put(1, 10);
        assert!(!c.is_empty());
        assert_eq!(c.len(), 1);
        c.put(2, 20);
        assert_eq!(c.len(), 2);
        // Eviction keeps len at capacity.
        c.put(3, 30);
        assert_eq!(c.len(), 2);
        assert_eq!(c.capacity(), 2);
    }

    // ---- property test: brute-force LRU oracle ----

    /// Reference implementation: O(n) per op, recency tracked by Vec order
    /// (front = LRU, back = MRU).
    struct BruteForceLru<K: Eq + Clone, V: Clone> {
        capacity: usize,
        items: Vec<(K, V)>,
    }

    impl<K: Eq + Clone, V: Clone> BruteForceLru<K, V> {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                items: Vec::new(),
            }
        }

        fn get(&mut self, key: &K) -> Option<V> {
            let pos = self.items.iter().position(|(k, _)| k == key)?;
            let entry = self.items.remove(pos);
            let value = entry.1.clone();
            self.items.push(entry);
            Some(value)
        }

        fn put(&mut self, key: K, value: V) {
            if self.capacity == 0 {
                return;
            }
            if let Some(pos) = self.items.iter().position(|(k, _)| k == &key) {
                self.items.remove(pos);
            } else if self.items.len() == self.capacity {
                self.items.remove(0);
            }
            self.items.push((key, value));
        }

        fn keys(&self) -> Vec<K> {
            self.items.iter().map(|(k, _)| k.clone()).collect()
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Get(u8),
        Put(u8, u8),
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(cap: u8, raw_ops: Vec<(bool, u8, u8)>) -> bool {
        let cap = (cap as usize) % 9; // capacity in 0..=8
        let ops: Vec<Op> = raw_ops
            .into_iter()
            .take(50)
            .map(|(is_get, k, v)| if is_get { Op::Get(k) } else { Op::Put(k, v) })
            .collect();

        let mut fast: LruCache<u8, u8> = LruCache::new(cap);
        let mut slow: BruteForceLru<u8, u8> = BruteForceLru::new(cap);

        for op in &ops {
            match *op {
                Op::Get(k) => {
                    let fast_val = fast.get(&k).copied();
                    let slow_val = slow.get(&k);
                    if fast_val != slow_val {
                        return false;
                    }
                }
                Op::Put(k, v) => {
                    fast.put(k, v);
                    slow.put(k, v);
                }
            }
            if fast.len() != slow.keys().len() {
                return false;
            }
        }

        // Final key set must match (order-independent).
        let mut fast_keys: Vec<u8> = (0u8..=u8::MAX).filter(|k| fast.contains_key(k)).collect();
        let mut slow_keys = slow.keys();
        fast_keys.sort_unstable();
        slow_keys.sort_unstable();
        if fast_keys != slow_keys {
            return false;
        }

        // Final stored values for each live key must match.
        let mut fast_map: HashMap<u8, u8> = HashMap::new();
        for k in &fast_keys {
            fast_map.insert(*k, *fast.get(k).expect("live key"));
        }
        let mut slow_map: HashMap<u8, u8> = HashMap::new();
        for (k, v) in &slow.items {
            slow_map.insert(*k, *v);
        }
        fast_map == slow_map
    }
}
