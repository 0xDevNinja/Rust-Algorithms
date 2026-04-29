//! Open-addressing hash table with linear probing.
//!
//! # Algorithm
//! An **open-addressing hash table** stores all entries directly in a flat
//! array (the *slot array*) instead of separate linked lists per bucket.
//! Collisions are resolved by **linear probing**: when slot `h` is occupied,
//! the table checks `h+1`, `h+2`, … (modulo capacity) until it finds a free
//! slot.
//!
//! # Tombstones
//! Deleting an entry cannot simply clear the slot because doing so would break
//! probe chains for keys that hashed to an earlier slot and had to skip past
//! the now-cleared slot during insertion. Instead the slot is marked
//! [`Slot::Tombstone`]. During lookup, tombstones are skipped (the probe
//! continues). During resize the entire array is rebuilt, discarding all
//! tombstones and starting fresh probe chains.
//!
//! # Complexity
//! | Operation | Average | Worst case |
//! |-----------|---------|------------|
//! | `insert`  | O(1)    | O(n)       |
//! | `get`     | O(1)    | O(n)       |
//! | `remove`  | O(1)    | O(n)       |
//!
//! Space: O(n) — the slot array never exceeds twice the number of live entries
//! because of the 0.5 load-factor threshold.
//!
//! # Cache locality vs. separate chaining
//! Because all data lives in one contiguous array, sequential probes hit the
//! same cache lines; separate-chaining follows heap pointers that may be
//! scattered in memory. This makes open addressing faster in practice at low
//! load factors. The trade-off is **primary clustering**: long runs of occupied
//! slots can form, raising the constant in O(1) as the load factor approaches
//! the resize threshold. Tombstones additionally cause "phantom load": they
//! count toward the threshold but hold no live data, so a workload with heavy
//! delete churn can trigger earlier resizes than expected.
//!
//! # Resize policy
//! When `(occupied_slots + tombstone_slots) / capacity > 0.5` the table is
//! doubled and all live entries are reinserted, clearing tombstones.
//! Default initial capacity is 16 (always rounded up to the next power of two
//! so that the modulo reduction never skews the probe distribution).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// One slot in the backing array.
enum Slot<K, V> {
    /// Never written to; terminates a probe on lookup.
    Empty,
    /// Holds a live key-value pair.
    Occupied(K, V),
    /// A previously occupied slot whose entry was deleted.
    ///
    /// Probes skip tombstones rather than stopping, so that entries that were
    /// inserted after the deleted key (and that collided with it) remain
    /// reachable.
    Tombstone,
}

/// Open-addressing hash table with linear probing and tombstone deletion.
///
/// Generic over key type `K` (`Eq + Hash + Clone`) and value type `V`
/// (`Clone`).
pub struct OpenAddressingHashTable<K, V> {
    slots: Vec<Slot<K, V>>,
    /// Number of `Occupied` slots.
    len: usize,
    /// Number of `Tombstone` slots (counts toward load factor).
    tombstones: usize,
}

// ── helpers ────────────────────────────────────────────────────────────────

fn hash_index<K: Hash>(key: &K, capacity: usize) -> usize {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    (h.finish() as usize) % capacity
}

/// Returns the next capacity: at least `min_cap`, and always a power of two
/// (so the probe distribution is uniform).
const fn next_power_of_two_capacity(min_cap: usize) -> usize {
    if min_cap <= 16 {
        return 16;
    }
    // `next_power_of_two` is a const fn on usize in stable Rust.
    min_cap.next_power_of_two()
}

// ── public API ─────────────────────────────────────────────────────────────

impl<K: Eq + Hash + Clone, V: Clone> OpenAddressingHashTable<K, V> {
    /// Creates an empty table with the default initial capacity (16 slots).
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    /// Creates an empty table with at least `n` slots pre-allocated.
    ///
    /// The actual capacity is rounded up to the nearest power of two and never
    /// below 16.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        let cap = next_power_of_two_capacity(n);
        let slots = (0..cap).map(|_| Slot::Empty).collect();
        Self {
            slots,
            len: 0,
            tombstones: 0,
        }
    }

    /// Returns the number of live key-value pairs.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no live pairs are stored.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if `key` has a live entry.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Looks up `key` and returns a reference to its value, or `None`.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        let cap = self.slots.len();
        let start = hash_index(key, cap);
        for i in 0..cap {
            let idx = (start + i) % cap;
            match &self.slots[idx] {
                Slot::Empty => return None,
                Slot::Occupied(k, v) if k == key => return Some(v),
                // Tombstone or different key: skip and continue probing.
                Slot::Tombstone | Slot::Occupied(..) => {}
            }
        }
        None
    }

    /// Inserts `key -> value`.
    ///
    /// Returns the previous value if `key` was already present, or `None` on a
    /// fresh insert. Triggers a resize when the combined occupied + tombstone
    /// count exceeds half the capacity.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Resize *before* inserting so that we always have room.
        if (self.len + self.tombstones) * 2 >= self.slots.len() {
            self.resize(self.slots.len() * 2);
        }

        let cap = self.slots.len();
        let start = hash_index(&key, cap);

        // Track the first tombstone we encounter; we can reuse it for the insert.
        let mut first_tombstone: Option<usize> = None;

        for i in 0..cap {
            let idx = (start + i) % cap;
            match &self.slots[idx] {
                Slot::Empty => {
                    // Insert here (or at the earlier tombstone).
                    let target = first_tombstone.unwrap_or(idx);
                    if first_tombstone.is_some() {
                        // We're replacing a tombstone.
                        self.tombstones -= 1;
                    }
                    self.slots[target] = Slot::Occupied(key, value);
                    self.len += 1;
                    return None;
                }
                Slot::Tombstone => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(idx);
                    }
                }
                Slot::Occupied(k, _) if k == &key => {
                    // Key already exists — overwrite and return old value.
                    let Slot::Occupied(_, old) =
                        std::mem::replace(&mut self.slots[idx], Slot::Occupied(key, value))
                    else {
                        unreachable!()
                    };
                    return Some(old);
                }
                Slot::Occupied(..) => {}
            }
        }

        // The only way to exhaust all slots without finding Empty or the key is
        // if the table is entirely tombstones and occupied entries with no Empty
        // slot — impossible after the resize guard above, but handle gracefully.
        if let Some(ts_idx) = first_tombstone {
            self.tombstones -= 1;
            self.slots[ts_idx] = Slot::Occupied(key, value);
            self.len += 1;
        }
        None
    }

    /// Removes `key` and returns its value, or `None` if absent.
    ///
    /// The vacated slot is marked [`Slot::Tombstone`] to preserve the probe
    /// chains of keys that hash to earlier slots.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let cap = self.slots.len();
        let start = hash_index(key, cap);
        for i in 0..cap {
            let idx = (start + i) % cap;
            match &self.slots[idx] {
                Slot::Empty => return None,
                Slot::Occupied(k, _) if k == key => {
                    let Slot::Occupied(_, old) =
                        std::mem::replace(&mut self.slots[idx], Slot::Tombstone)
                    else {
                        unreachable!()
                    };
                    self.len -= 1;
                    self.tombstones += 1;
                    return Some(old);
                }
                // Tombstone or different key: skip and continue probing.
                Slot::Tombstone | Slot::Occupied(..) => {}
            }
        }
        None
    }

    // ── internal ──────────────────────────────────────────────────────────

    /// Rebuilds the slot array with `new_cap` slots, discarding tombstones.
    ///
    /// All live entries are reinserted into the new array via fresh probe
    /// chains.
    fn resize(&mut self, new_cap: usize) {
        let new_cap = next_power_of_two_capacity(new_cap);
        let old_slots =
            std::mem::replace(&mut self.slots, (0..new_cap).map(|_| Slot::Empty).collect());
        self.len = 0;
        self.tombstones = 0;

        for slot in old_slots {
            if let Slot::Occupied(k, v) = slot {
                // Safe: we just doubled capacity so there is guaranteed room.
                self.insert(k, v);
            }
        }
    }
}

impl<K: Eq + Hash + Clone, V: Clone> Default for OpenAddressingHashTable<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::OpenAddressingHashTable;
    use quickcheck_macros::quickcheck;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    // ── deterministic unit tests ──────────────────────────────────────────

    #[test]
    fn empty_table() {
        let t: OpenAddressingHashTable<i32, i32> = OpenAddressingHashTable::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(!t.contains_key(&1));
        assert_eq!(t.get(&1), None);
    }

    #[test]
    fn insert_and_get() {
        let mut t = OpenAddressingHashTable::new();
        assert_eq!(t.insert(1, 10), None);
        assert_eq!(t.insert(2, 20), None);
        assert_eq!(t.get(&1), Some(&10));
        assert_eq!(t.get(&2), Some(&20));
        assert_eq!(t.get(&3), None);
        assert!(t.contains_key(&1));
        assert!(!t.contains_key(&99));
        assert_eq!(t.len(), 2);
        assert!(!t.is_empty());
    }

    #[test]
    fn overwrite_returns_old_value() {
        let mut t = OpenAddressingHashTable::new();
        assert_eq!(t.insert("hello", 1), None);
        // Second insert with same key returns old value.
        assert_eq!(t.insert("hello", 2), Some(1));
        // Length stays 1.
        assert_eq!(t.len(), 1);
        assert_eq!(t.get(&"hello"), Some(&2));
    }

    #[test]
    fn remove_returns_value_and_creates_tombstone_probe_chain_survives() {
        // We need two keys that land in the same initial bucket so that the
        // second key's probe chain runs through the first key's slot.
        //
        // Strategy: insert enough keys with the same hash-mod-capacity that we
        // can guarantee a probe chain, then remove the first and verify the
        // second is still accessible.
        //
        // We use `with_capacity(16)` (capacity stays 16 for small inserts).
        // DefaultHasher is deterministic within a build but may differ between
        // runs.  We find a colliding pair at runtime.
        let capacity = 16usize;

        // Search for two keys whose initial bucket is identical.
        let bucket = |k: i32| {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            (h.finish() as usize) % capacity
        };

        let base = 0i32;
        let base_bucket = bucket(base);
        // Find a key that maps to the same bucket.
        let collider = (1..10_000i32)
            .find(|&k| bucket(k) == base_bucket)
            .expect("collision must exist in 10_000 candidates");

        let mut t: OpenAddressingHashTable<i32, i32> = OpenAddressingHashTable::with_capacity(16);
        t.insert(base, 100);
        t.insert(collider, 200);
        assert_eq!(t.len(), 2);

        // Remove the first key — leaves a tombstone.
        let removed = t.remove(&base);
        assert_eq!(removed, Some(100));
        assert_eq!(t.len(), 1);

        // The first key is gone.
        assert_eq!(t.get(&base), None);
        assert!(!t.contains_key(&base));

        // The colliding key must still be reachable through the tombstone.
        assert_eq!(t.get(&collider), Some(&200));
        assert!(t.contains_key(&collider));
    }

    #[test]
    fn remove_absent_key_returns_none() {
        let mut t: OpenAddressingHashTable<i32, i32> = OpenAddressingHashTable::new();
        t.insert(1, 10);
        assert_eq!(t.remove(&99), None);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn len_tracking() {
        let mut t = OpenAddressingHashTable::new();
        assert_eq!(t.len(), 0);
        t.insert(1, 1);
        assert_eq!(t.len(), 1);
        t.insert(2, 2);
        assert_eq!(t.len(), 2);
        t.insert(1, 99); // overwrite, no length change
        assert_eq!(t.len(), 2);
        t.remove(&1);
        assert_eq!(t.len(), 1);
        t.remove(&1); // already removed
        assert_eq!(t.len(), 1);
        t.remove(&2);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn resize_on_heavy_insertion_all_keys_retrievable() {
        let mut t = OpenAddressingHashTable::new();
        for i in 0..1000i32 {
            t.insert(i, i * 2);
        }
        assert_eq!(t.len(), 1000);
        for i in 0..1000i32 {
            assert_eq!(
                t.get(&i),
                Some(&(i * 2)),
                "key {i} not found after heavy insert"
            );
        }
    }

    #[test]
    fn string_keys() {
        let mut t: OpenAddressingHashTable<String, usize> = OpenAddressingHashTable::new();
        for (i, word) in ["alpha", "beta", "gamma", "delta", "epsilon"]
            .iter()
            .enumerate()
        {
            t.insert((*word).to_string(), i);
        }
        assert_eq!(t.len(), 5);
        assert_eq!(t.get(&"gamma".to_string()), Some(&2));
        assert_eq!(t.get(&"zeta".to_string()), None);
    }

    #[test]
    fn insert_remove_churn_exercises_tombstone_reclamation() {
        // Alternate inserts and removes to generate many tombstones, forcing
        // at least one resize that reclaims them.
        let mut t: OpenAddressingHashTable<i32, i32> = OpenAddressingHashTable::new();
        for round in 0..10i32 {
            // Insert 100 entries.
            for i in 0..100i32 {
                t.insert(round * 1000 + i, i);
            }
            // Remove 90 of them, leaving 10 live + 90 tombstones.
            for i in 0..90i32 {
                t.remove(&(round * 1000 + i));
            }
        }
        // 10 rounds × 10 survivors = 100 live entries.
        assert_eq!(t.len(), 100);
        for round in 0..10i32 {
            for i in 90..100i32 {
                assert_eq!(
                    t.get(&(round * 1000 + i)),
                    Some(&i),
                    "key {round}_{i} missing after churn"
                );
            }
        }
    }

    #[test]
    fn with_capacity_constructor() {
        let mut t: OpenAddressingHashTable<i32, i32> = OpenAddressingHashTable::with_capacity(64);
        t.insert(1, 2);
        assert_eq!(t.get(&1), Some(&2));
    }

    // ── quickcheck property test ──────────────────────────────────────────

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32, i32),
        Remove(i32),
        Get(i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            match u8::arbitrary(g) % 3 {
                0 => Self::Insert(i32::arbitrary(g) % 200, i32::arbitrary(g)),
                1 => Self::Remove(i32::arbitrary(g) % 200),
                _ => Self::Get(i32::arbitrary(g) % 200),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_matches_std_hashmap(ops: Vec<Op>) -> bool {
        let mut ours: OpenAddressingHashTable<i32, i32> = OpenAddressingHashTable::new();
        let mut std_map: HashMap<i32, i32> = HashMap::new();

        for op in &ops {
            match *op {
                Op::Insert(k, v) => {
                    let a = ours.insert(k, v);
                    let b = std_map.insert(k, v);
                    if a != b {
                        return false;
                    }
                }
                Op::Remove(k) => {
                    let a = ours.remove(&k);
                    let b = std_map.remove(&k);
                    if a != b {
                        return false;
                    }
                }
                Op::Get(k) => {
                    let a = ours.get(&k);
                    let b = std_map.get(&k);
                    if a != b {
                        return false;
                    }
                }
            }
            if ours.len() != std_map.len() {
                return false;
            }
        }
        true
    }
}
