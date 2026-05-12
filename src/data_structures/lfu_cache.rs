//! Least-Frequently-Used (LFU) cache.
//!
//! A bounded-capacity key/value cache that, when full, evicts the entry with
//! the **lowest access frequency**. Ties between equally-cold entries are
//! broken by **least-recent use** (the staler of the two goes first). Both
//! `get` (a hit) and `put` (insert or update) count as a "use" and bump the
//! affected entry's frequency by one.
//!
//! # Design
//! Three pieces, all O(1):
//! * `entries: HashMap<K, EntryIndex>` — maps each live key to a slot in a
//!   slab of [`Entry`] records holding `(value, freq, node_idx)`.
//! * `buckets: HashMap<usize, Bucket>` — one indexed doubly-linked list per
//!   distinct frequency. The list stores key-slot indices in MRU-first order:
//!   the head is the most-recently-used member of that frequency class, the
//!   tail is the eviction candidate.
//! * `min_freq: usize` — the smallest frequency currently populated, so the
//!   eviction bucket is found in O(1).
//!
//! Each linked list lives inside a single `Vec<Node>` arena (no `Rc` /
//! `RefCell` / `unsafe`), with `prev`/`next` stored as `Option<usize>` indices
//! and freed slots reused via a free-list. A bump or insert at frequency `f`
//! detaches the node from bucket `f-1` (if any) and pushes it on the head of
//! bucket `f`; if the source bucket empties and equalled `min_freq`, the
//! tracker advances. Eviction pops the tail of `buckets[min_freq]`.
//!
//! # Complexity
//! - `get`, `put`: **O(1)** amortized — one hash lookup plus constant-time
//!   relinking.
//! - `len`, `is_empty`, `capacity`: **O(1)**.
//! - Space: **O(capacity)**.
//!
//! # Capacity zero
//! A cache constructed with `capacity == 0` ignores every `put` and returns
//! `None` from every `get`. This matches the `LeetCode` problem statement.

use std::collections::HashMap;
use std::hash::Hash;

/// Slab node for a per-frequency doubly-linked list of key-slot indices.
struct Node {
    /// Index into [`LFUCache::entries_slab`] identifying the cached key.
    entry_idx: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// One frequency class, holding the key-slots accessed exactly `freq` times.
struct Bucket {
    head: Option<usize>,
    tail: Option<usize>,
    len: usize,
}

impl Bucket {
    const fn new() -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
        }
    }
}

/// Slab record for a single cached key: its value, current frequency, and the
/// linked-list node that represents it inside `buckets[freq]`.
struct Entry<K, V> {
    key: K,
    value: V,
    freq: usize,
    node_idx: usize,
}

/// A fixed-capacity LFU cache with O(1) `get` and `put`.
///
/// Generic over key type `K` (`Eq + Hash + Clone` so the lookup map can own a
/// copy of every live key) and value type `V` (`Clone` because `get` returns
/// the stored value by clone).
pub struct LFUCache<K: Eq + Hash + Clone, V: Clone> {
    capacity: usize,
    /// Live-key index: `K -> entries_slab` slot.
    entries: HashMap<K, usize>,
    /// Slab of live entries; freed slots are recycled via `free_entries`.
    entries_slab: Vec<Option<Entry<K, V>>>,
    free_entries: Vec<usize>,
    /// Shared arena for all bucket linked-list nodes; freed slots recycled via
    /// `free_nodes`.
    nodes: Vec<Option<Node>>,
    free_nodes: Vec<usize>,
    /// Per-frequency doubly-linked lists.
    buckets: HashMap<usize, Bucket>,
    /// Smallest populated frequency, or `0` when the cache is empty.
    min_freq: usize,
}

impl<K: Eq + Hash + Clone, V: Clone> LFUCache<K, V> {
    /// Creates an empty cache that holds at most `capacity` entries.
    ///
    /// `capacity == 0` is allowed; see the module docs for its semantics.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            entries_slab: Vec::with_capacity(capacity),
            free_entries: Vec::new(),
            nodes: Vec::with_capacity(capacity),
            free_nodes: Vec::new(),
            buckets: HashMap::new(),
            min_freq: 0,
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
        self.entries.len()
    }

    /// Returns `true` if the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if `key` is present in the cache.
    ///
    /// Does **not** count as a "use" — frequency and recency order are
    /// unchanged.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Looks up `key`. On a hit, bumps the entry's frequency by one and
    /// returns a clone of the stored value. Returns `None` on a miss.
    pub fn get(&mut self, key: &K) -> Option<V> {
        let entry_idx = *self.entries.get(key)?;
        self.bump(entry_idx);
        let value = self.entries_slab[entry_idx]
            .as_ref()
            .expect("live entry")
            .value
            .clone();
        Some(value)
    }

    /// Inserts or updates `key -> value`.
    ///
    /// On a hit the value is overwritten and the entry's frequency bumped by
    /// one. On a miss, if the cache is at capacity the LFU entry (ties broken
    /// by least-recent use) is evicted first; the new entry then enters at
    /// frequency `1`. With `capacity == 0` the call is a no-op.
    pub fn put(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }

        if let Some(&entry_idx) = self.entries.get(&key) {
            // Overwrite the value and bump frequency.
            self.entries_slab[entry_idx]
                .as_mut()
                .expect("live entry")
                .value = value;
            self.bump(entry_idx);
            return;
        }

        // Fresh insert: evict the LFU+LRU entry if at capacity.
        if self.entries.len() == self.capacity {
            self.evict_lfu();
        }

        let node_idx = self.alloc_node(Node {
            entry_idx: 0, // patched after entry slot is known
            prev: None,
            next: None,
        });
        let entry_idx = self.alloc_entry(Entry {
            key: key.clone(),
            value,
            freq: 1,
            node_idx,
        });
        self.nodes[node_idx].as_mut().expect("live node").entry_idx = entry_idx;

        self.entries.insert(key, entry_idx);
        self.bucket_push_head(1, node_idx);
        self.min_freq = 1;
    }

    // ---- internal helpers -------------------------------------------------

    /// Bumps the entry at `entry_idx` from frequency `f` to `f + 1`, moving
    /// its node to the head of the new bucket and advancing `min_freq` if the
    /// old bucket emptied.
    fn bump(&mut self, entry_idx: usize) {
        let (old_freq, node_idx) = {
            let entry = self.entries_slab[entry_idx].as_ref().expect("live entry");
            (entry.freq, entry.node_idx)
        };
        let new_freq = old_freq + 1;

        self.bucket_detach(old_freq, node_idx);
        let old_bucket_empty = self.buckets.get(&old_freq).is_none_or(|b| b.len == 0);
        if old_bucket_empty {
            self.buckets.remove(&old_freq);
            if self.min_freq == old_freq {
                self.min_freq = new_freq;
            }
        }

        self.entries_slab[entry_idx]
            .as_mut()
            .expect("live entry")
            .freq = new_freq;
        self.bucket_push_head(new_freq, node_idx);
    }

    /// Removes the LFU+LRU entry: tail of `buckets[min_freq]`.
    fn evict_lfu(&mut self) {
        let bucket = self
            .buckets
            .get_mut(&self.min_freq)
            .expect("non-empty cache has populated min_freq");
        let victim_node = bucket.tail.expect("bucket with len>0 has a tail");
        let victim_entry = self.nodes[victim_node]
            .as_ref()
            .expect("live node")
            .entry_idx;

        self.bucket_detach(self.min_freq, victim_node);
        if self.buckets.get(&self.min_freq).is_none_or(|b| b.len == 0) {
            self.buckets.remove(&self.min_freq);
        }

        let entry = self.entries_slab[victim_entry].take().expect("live entry");
        self.entries.remove(&entry.key);
        self.free_entries.push(victim_entry);

        self.nodes[victim_node] = None;
        self.free_nodes.push(victim_node);
    }

    /// Pushes `node_idx` onto the head of `buckets[freq]`, creating the
    /// bucket if it doesn't yet exist.
    fn bucket_push_head(&mut self, freq: usize, node_idx: usize) {
        let bucket = self.buckets.entry(freq).or_insert_with(Bucket::new);
        let old_head = bucket.head;
        bucket.head = Some(node_idx);
        if bucket.tail.is_none() {
            bucket.tail = Some(node_idx);
        }
        bucket.len += 1;

        let node = self.nodes[node_idx].as_mut().expect("live node");
        node.prev = None;
        node.next = old_head;
        if let Some(h) = old_head {
            self.nodes[h].as_mut().expect("live node").prev = Some(node_idx);
        }
    }

    /// Unlinks `node_idx` from `buckets[freq]` (does not free the node slot).
    fn bucket_detach(&mut self, freq: usize, node_idx: usize) {
        let (prev, next) = {
            let node = self.nodes[node_idx].as_ref().expect("live node");
            (node.prev, node.next)
        };
        match prev {
            Some(p) => self.nodes[p].as_mut().expect("live node").next = next,
            None => {
                if let Some(b) = self.buckets.get_mut(&freq) {
                    b.head = next;
                }
            }
        }
        match next {
            Some(n) => self.nodes[n].as_mut().expect("live node").prev = prev,
            None => {
                if let Some(b) = self.buckets.get_mut(&freq) {
                    b.tail = prev;
                }
            }
        }
        if let Some(b) = self.buckets.get_mut(&freq) {
            b.len -= 1;
        }
        let node = self.nodes[node_idx].as_mut().expect("live node");
        node.prev = None;
        node.next = None;
    }

    /// Allocates a slab slot for `node`, reusing a freed index if one exists.
    fn alloc_node(&mut self, node: Node) -> usize {
        if let Some(idx) = self.free_nodes.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            self.nodes.push(Some(node));
            self.nodes.len() - 1
        }
    }

    /// Allocates an entries-slab slot for `entry`, reusing a freed index if
    /// one exists.
    fn alloc_entry(&mut self, entry: Entry<K, V>) -> usize {
        if let Some(idx) = self.free_entries.pop() {
            self.entries_slab[idx] = Some(entry);
            idx
        } else {
            self.entries_slab.push(Some(entry));
            self.entries_slab.len() - 1
        }
    }
}

impl<K: Eq + Hash + Clone, V: Clone> Default for LFUCache<K, V> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::LFUCache;

    #[test]
    fn empty_cache() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(4);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.capacity(), 4);
        assert!(!c.contains_key(&1));
        assert_eq!(c.get(&1), None);
    }

    #[test]
    fn capacity_zero_ignores_puts() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(0);
        c.put(1, 10);
        c.put(2, 20);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.capacity(), 0);
        assert!(!c.contains_key(&1));
        assert_eq!(c.get(&1), None);
    }

    #[test]
    fn basic_put_and_get() {
        let mut c: LFUCache<&str, i32> = LFUCache::new(3);
        c.put("a", 1);
        c.put("b", 2);
        c.put("c", 3);
        assert_eq!(c.len(), 3);
        assert_eq!(c.get(&"a"), Some(1));
        assert_eq!(c.get(&"b"), Some(2));
        assert_eq!(c.get(&"c"), Some(3));
        assert!(c.contains_key(&"a"));
        assert!(!c.contains_key(&"z"));
    }

    #[test]
    fn updating_existing_key_overwrites_and_bumps_freq() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(2);
        c.put(1, 10);
        c.put(1, 11); // overwrite + bump freq to 2
        c.put(2, 20); // freq 1
                      // Inserting a third key must evict key 2 (lower freq).
        c.put(3, 30);
        assert!(c.contains_key(&1));
        assert!(!c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert_eq!(c.get(&1), Some(11));
    }

    #[test]
    fn evicts_least_frequent() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        // Bump key 1 to freq 2; key 2 stays at freq 1.
        assert_eq!(c.get(&1), Some(10));
        // Inserting a third key evicts key 2 (the LFU entry).
        c.put(3, 30);
        assert!(c.contains_key(&1));
        assert!(!c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn frequency_tie_broken_by_least_recent() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(3);
        c.put(1, 10); // freq 1, oldest
        c.put(2, 20); // freq 1
        c.put(3, 30); // freq 1, newest
                      // All three sit at freq 1: key 1 is the LRU. Inserting 4 evicts 1.
        c.put(4, 40);
        assert!(!c.contains_key(&1));
        assert!(c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert!(c.contains_key(&4));
    }

    #[test]
    fn frequency_tie_broken_by_least_recent_after_gets() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(3);
        c.put(1, 10);
        c.put(2, 20);
        c.put(3, 30);
        // Bump key 2 to freq 2; bump key 3 to freq 2.
        assert_eq!(c.get(&2), Some(20));
        assert_eq!(c.get(&3), Some(30));
        // Now: key 1 at freq 1, keys 2 & 3 at freq 2 (3 more recent than 2).
        // Inserting 4 must evict key 1 (sole freq-1 entry).
        c.put(4, 40);
        assert!(!c.contains_key(&1));
        assert!(c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert!(c.contains_key(&4));
        // Bump key 4 by getting it so it joins freq 2 last (most recent).
        assert_eq!(c.get(&4), Some(40)); // freq 2
                                         // freq 2 holds {2, 3, 4} with 4 newest, 2 oldest.
                                         // Inserting 5 enters at freq 1 — capacity is 3, eviction needed.
                                         // No freq-1 keys remain, so the LFU bucket is freq 2; victim is key 2.
        c.put(5, 50);
        assert!(!c.contains_key(&2));
        assert!(c.contains_key(&3));
        assert!(c.contains_key(&4));
        assert!(c.contains_key(&5));
    }

    /// Classic `LeetCode` 460 example.
    /// `LFUCache(2); put(1,1); put(2,2); get(1)=1; put(3,3) -> evicts 2;
    /// get(2)=null; get(3)=3; put(4,4) -> evicts 1; get(1)=null; get(3)=3;
    /// get(4)=4`.
    #[test]
    fn leetcode_classic_example() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(2);
        c.put(1, 1);
        c.put(2, 2);
        assert_eq!(c.get(&1), Some(1));
        c.put(3, 3); // evicts key 2 (freq 1, key 1 is freq 2)
        assert_eq!(c.get(&2), None);
        assert_eq!(c.get(&3), Some(3));
        c.put(4, 4); // freq map: 1->freq 2, 3->freq 2, both tied; key 1 is LRU.
        assert_eq!(c.get(&1), None);
        assert_eq!(c.get(&3), Some(3));
        assert_eq!(c.get(&4), Some(4));
    }

    #[test]
    fn capacity_one_keeps_newest_only() {
        let mut c: LFUCache<i32, i32> = LFUCache::new(1);
        c.put(1, 10);
        assert_eq!(c.len(), 1);
        // Inserting a second key evicts the first regardless of frequency.
        c.put(2, 20);
        assert!(!c.contains_key(&1));
        assert_eq!(c.get(&2), Some(20));
        // Even after bumping key 2's freq, replacing it with a brand-new key
        // evicts it (only one slot).
        c.put(3, 30);
        assert!(!c.contains_key(&2));
        assert_eq!(c.get(&3), Some(30));
    }
}
