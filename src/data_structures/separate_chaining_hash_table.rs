//! Separate-chaining hash table.
//!
//! A hash table that resolves collisions by storing all keys that map to the
//! same bucket in a `Vec` of `(K, V)` pairs (a "chain"). This is the canonical
//! textbook variant: simple to understand, safe, and correct.
//!
//! # Complexity
//! - `insert`, `get`, `remove`, `contains_key`: **O(1)** amortised under a
//!   uniform hash; **O(n)** worst-case when all keys collide into one bucket.
//! - Space: **O(n)** where n is the number of stored entries.
//!
//! # Resize policy
//! When the load factor (entries / buckets) exceeds **0.75** after an insert,
//! the bucket array is doubled and all entries are rehashed. This keeps the
//! expected chain length below 1.5 and preserves O(1) amortised behaviour.
//!
//! # Preconditions
//! - Keys must implement `Eq + Hash + Clone`.
//! - Values must implement `Clone`.
//! - `with_capacity(0)` is allowed; the first insert triggers an initial
//!   allocation to [`DEFAULT_CAPACITY`] buckets.

use std::hash::{BuildHasher, Hash, RandomState};

/// Default number of buckets when the table is created with [`SeparateChainingHashTable::new`].
pub const DEFAULT_CAPACITY: usize = 16;

/// Load-factor threshold that triggers a resize (doubles bucket count).
const MAX_LOAD_FACTOR: f64 = 0.75;

/// A hash table that resolves collisions using separate chaining.
///
/// Each bucket is a `Vec<(K, V)>`; on collision the new pair is appended to
/// the bucket's `Vec`. See the module docs for complexity and resize policy.
pub struct SeparateChainingHashTable<K, V, S = RandomState> {
    buckets: Vec<Vec<(K, V)>>,
    len: usize,
    hash_builder: S,
}

// ── construction ─────────────────────────────────────────────────────────────

impl<K: Eq + Hash + Clone, V: Clone> SeparateChainingHashTable<K, V> {
    /// Creates an empty table with [`DEFAULT_CAPACITY`] buckets and a
    /// randomly-seeded hasher.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates an empty table with at least `n` buckets.
    ///
    /// If `n` is 0 the table allocates no buckets until the first insert.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        let bucket_count = if n == 0 { 0 } else { n };
        Self {
            buckets: vec![Vec::new(); bucket_count],
            len: 0,
            hash_builder: RandomState::new(),
        }
    }
}

impl<K: Eq + Hash + Clone, V: Clone, S: BuildHasher> SeparateChainingHashTable<K, V, S> {
    // ── private helpers ───────────────────────────────────────────────────────

    /// Hashes `key` to a bucket index in `[0, bucket_count)`.
    fn bucket_index(&self, key: &K, bucket_count: usize) -> usize {
        (self.hash_builder.hash_one(key) as usize) % bucket_count
    }

    /// Returns the current load factor, or 0.0 when no buckets exist.
    fn load_factor(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        self.len as f64 / self.buckets.len() as f64
    }

    /// Doubles the bucket array and rehashes every entry.
    fn resize(&mut self) {
        let new_bucket_count = if self.buckets.is_empty() {
            DEFAULT_CAPACITY
        } else {
            self.buckets.len() * 2
        };

        let mut new_buckets: Vec<Vec<(K, V)>> = vec![Vec::new(); new_bucket_count];

        for chain in self.buckets.drain(..) {
            for (key, value) in chain {
                let idx = (self.hash_builder.hash_one(&key) as usize) % new_bucket_count;
                new_buckets[idx].push((key, value));
            }
        }

        self.buckets = new_buckets;
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Inserts `key → value`.
    ///
    /// If `key` was already present the value is updated and the **previous**
    /// value is returned as `Some(old_value)`. Otherwise `None` is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Ensure there are buckets before hashing.
        if self.buckets.is_empty() {
            self.resize();
        }

        let idx = self.bucket_index(&key, self.buckets.len());

        // Update in place if key already exists.
        for &mut (ref k, ref mut v) in &mut self.buckets[idx] {
            if k == &key {
                let old = v.clone();
                *v = value;
                return Some(old);
            }
        }

        // Fresh insert.
        self.buckets[idx].push((key, value));
        self.len += 1;

        // Resize after update so we don't invalidate the index we just used.
        if self.load_factor() > MAX_LOAD_FACTOR {
            self.resize();
        }

        None
    }

    /// Returns a reference to the value associated with `key`, or `None`.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.buckets.is_empty() {
            return None;
        }
        let idx = self.bucket_index(key, self.buckets.len());
        self.buckets[idx]
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Removes `key` from the table and returns its value, or `None` if absent.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.buckets.is_empty() {
            return None;
        }
        let idx = self.bucket_index(key, self.buckets.len());
        let chain = &mut self.buckets[idx];
        if let Some(pos) = chain.iter().position(|(k, _)| k == key) {
            let (_, value) = chain.swap_remove(pos);
            self.len -= 1;
            Some(value)
        } else {
            None
        }
    }

    /// Returns `true` if `key` is present in the table.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns the number of key-value pairs stored in the table.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the table contains no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<K: Eq + Hash + Clone, V: Clone> Default for SeparateChainingHashTable<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::SeparateChainingHashTable;
    use quickcheck_macros::quickcheck;
    use std::collections::HashMap;

    // ---- unit tests ----------------------------------------------------------

    #[test]
    fn empty_table() {
        let t: SeparateChainingHashTable<i32, i32> = SeparateChainingHashTable::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(!t.contains_key(&42));
        assert_eq!(t.get(&42), None);
    }

    #[test]
    fn insert_and_get() {
        let mut t = SeparateChainingHashTable::new();
        assert_eq!(t.insert(1, "one"), None);
        assert_eq!(t.insert(2, "two"), None);
        assert_eq!(t.get(&1), Some(&"one"));
        assert_eq!(t.get(&2), Some(&"two"));
        assert_eq!(t.get(&3), None);
    }

    #[test]
    fn overwrite_returns_old_value() {
        let mut t = SeparateChainingHashTable::new();
        t.insert("a", 10);
        let prev = t.insert("a", 20);
        assert_eq!(prev, Some(10));
        assert_eq!(t.get(&"a"), Some(&20));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_returns_value() {
        let mut t = SeparateChainingHashTable::new();
        t.insert(99, "hello");
        let removed = t.remove(&99);
        assert_eq!(removed, Some("hello"));
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn remove_absent_key_returns_none() {
        let mut t: SeparateChainingHashTable<i32, i32> = SeparateChainingHashTable::new();
        assert_eq!(t.remove(&7), None);
    }

    #[test]
    fn get_missing_key_returns_none() {
        let mut t = SeparateChainingHashTable::new();
        t.insert(1, 100);
        assert_eq!(t.get(&2), None);
    }

    #[test]
    fn len_tracking() {
        let mut t = SeparateChainingHashTable::new();
        assert_eq!(t.len(), 0);
        t.insert("x", 1);
        assert_eq!(t.len(), 1);
        t.insert("y", 2);
        assert_eq!(t.len(), 2);
        t.insert("x", 99); // overwrite — len stays 2
        assert_eq!(t.len(), 2);
        t.remove(&"x");
        assert_eq!(t.len(), 1);
        t.remove(&"y");
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn contains_key() {
        let mut t = SeparateChainingHashTable::new();
        t.insert(5, "five");
        assert!(t.contains_key(&5));
        assert!(!t.contains_key(&6));
        t.remove(&5);
        assert!(!t.contains_key(&5));
    }

    #[test]
    fn resize_on_heavy_insertion_all_retrievable() {
        let mut t = SeparateChainingHashTable::new();
        for i in 0..1000 {
            t.insert(i, i * 2);
        }
        assert_eq!(t.len(), 1000);
        for i in 0..1000 {
            assert_eq!(t.get(&i), Some(&(i * 2)), "key {i} not found after resize");
        }
    }

    #[test]
    fn string_keys() {
        let mut t: SeparateChainingHashTable<String, usize> = SeparateChainingHashTable::new();
        let words = ["apple", "banana", "cherry", "date", "elderberry"];
        for (idx, w) in words.iter().enumerate() {
            t.insert(w.to_string(), idx);
        }
        assert_eq!(t.len(), words.len());
        for (idx, w) in words.iter().enumerate() {
            assert_eq!(t.get(&w.to_string()), Some(&idx));
        }
        t.remove(&"banana".to_string());
        assert!(!t.contains_key(&"banana".to_string()));
        assert_eq!(t.len(), words.len() - 1);
    }

    #[test]
    fn with_capacity_zero_works() {
        let mut t: SeparateChainingHashTable<i32, i32> =
            SeparateChainingHashTable::with_capacity(0);
        assert!(t.is_empty());
        t.insert(1, 1);
        assert_eq!(t.get(&1), Some(&1));
    }

    #[test]
    fn default_is_same_as_new() {
        let t: SeparateChainingHashTable<i32, i32> = SeparateChainingHashTable::default();
        assert!(t.is_empty());
    }

    // ---- property test -------------------------------------------------------

    /// Operations applied to the hash table must produce the same results as
    /// the standard library `HashMap`.
    #[derive(Clone, Debug)]
    enum Op {
        Insert(u8, u8),
        Remove(u8),
        Get(u8),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            match u8::arbitrary(g) % 3 {
                0 => Self::Insert(u8::arbitrary(g), u8::arbitrary(g)),
                1 => Self::Remove(u8::arbitrary(g)),
                _ => Self::Get(u8::arbitrary(g)),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn qc_matches_std_hashmap(ops: Vec<Op>) -> bool {
        let mut ours: SeparateChainingHashTable<u8, u8> = SeparateChainingHashTable::new();
        let mut std_map: HashMap<u8, u8> = HashMap::new();

        for op in ops.iter().take(200) {
            match *op {
                Op::Insert(k, v) => {
                    let ours_prev = ours.insert(k, v);
                    let std_prev = std_map.insert(k, v);
                    if ours_prev != std_prev {
                        return false;
                    }
                }
                Op::Remove(k) => {
                    let ours_val = ours.remove(&k);
                    let std_val = std_map.remove(&k);
                    if ours_val != std_val {
                        return false;
                    }
                }
                Op::Get(k) => {
                    if ours.get(&k) != std_map.get(&k) {
                        return false;
                    }
                }
            }
            if ours.len() != std_map.len() {
                return false;
            }
        }

        // Final state: every key in std_map must be reachable in ours with same value.
        for (k, v) in &std_map {
            if ours.get(k) != Some(v) {
                return false;
            }
        }
        ours.len() == std_map.len()
    }
}
