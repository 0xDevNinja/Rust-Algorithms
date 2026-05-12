//! K closest points to the origin (Euclidean distance).
//!
//! Given a slice of 2D points and an integer `k`, return `k` points whose
//! Euclidean distance from the origin is smallest. Two implementations are
//! provided that target different operating points:
//!
//! - [`k_closest_heap`]: maintain a bounded **max-heap of size `k`** keyed by
//!   squared distance. Time `O(n log k)`, extra space `O(k)`. Streaming-
//!   friendly and stable across repeated runs.
//! - [`k_closest_quickselect`]: in-place Hoare-style **quickselect** that
//!   partitions a working buffer so the `k` smallest squared distances end up
//!   in the first `k` slots. Time `O(n)` on average and `O(n^2)` worst case,
//!   extra space `O(n)` for the working buffer. The pivot index is drawn from
//!   a deterministic `xorshift64*` PRNG seeded by the input length, so the
//!   choice of pivots is reproducible and independent of any global RNG state
//!   while still avoiding the classic adversarial inputs that pin a fixed
//!   pivot to a quadratic path.
//!
//! Both routines compare **squared** distances to avoid an unnecessary
//! `sqrt` and to keep all comparisons in exact integer-like arithmetic over
//! `f64` (no rounding from a square root). They return *some* `k` points
//! with the smallest distances; the **order of the returned vector is
//! unspecified** and ties between equidistant points are broken by their
//! position in the input slice (whichever candidate the algorithm happens to
//! encounter first wins). Callers that need a canonical ordering should sort
//! the output themselves.
//!
//! Edge cases:
//!
//! - `k == 0` returns an empty vector.
//! - `k >= points.len()` returns a copy of every input point.
//! - An empty input always returns an empty vector regardless of `k`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Heap entry pairing a squared distance with the originating point. Pulled
/// out of [`k_closest_heap`] so clippy's `items_after_statements` lint stays
/// happy and so the `Ord` impl is reusable if we add a streaming variant.
#[derive(Copy, Clone)]
struct Entry {
    d2: f64,
    p: (f64, f64),
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        cmp_f64(self.d2, other.d2) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_f64(self.d2, other.d2)
    }
}

/// Squared Euclidean distance from the origin. Using the squared form keeps
/// comparisons exact relative to the inputs and avoids a `sqrt` call per
/// point, which is the only operation the algorithms need. `mul_add` is used
/// for the fused multiply-add path on platforms where it is cheaper and a hair
/// more accurate than the separate ops.
#[inline]
fn sq_dist(p: (f64, f64)) -> f64 {
    p.0.mul_add(p.0, p.1 * p.1)
}

/// Total order over `f64` distances. Distances are non-negative and finite
/// for finite inputs, but we still fall back to `Equal` on `NaN` to keep the
/// heap and partition routines well-defined when the caller passes pathological
/// floats. `NaN`-bearing inputs simply produce an unspecified-but-valid output.
#[inline]
fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// Returns `k` points from `points` whose Euclidean distance from the origin
/// is smallest, computed via a bounded max-heap of size `k`.
///
/// The heap is keyed by squared distance so the largest of the `k` current
/// candidates sits at the top. For each subsequent point we compare against
/// that top; if the new point is closer we pop the top and push the newcomer,
/// which keeps the heap size at exactly `k`. After processing all points the
/// heap holds the `k` smallest distances.
///
/// - Time: `O(n log k)`.
/// - Extra space: `O(k)`.
///
/// The returned vector contains the points in **unspecified order** (the
/// natural draining order of the heap). Ties in distance are broken in favor
/// of whichever point reached the heap first; later equidistant points do not
/// displace earlier ones because the strict `<` comparison rejects equal keys.
///
/// # Examples
///
/// ```
/// use rust_algorithms::searching::k_closest_points::k_closest_heap;
///
/// let pts = [(1.0, 3.0), (-2.0, 2.0), (5.0, 8.0), (0.0, 1.0)];
/// let mut got = k_closest_heap(&pts, 2);
/// got.sort_by(|a, b| (a.0 * a.0 + a.1 * a.1)
///     .partial_cmp(&(b.0 * b.0 + b.1 * b.1)).unwrap());
/// assert_eq!(got, vec![(0.0, 1.0), (-2.0, 2.0)]);
/// ```
pub fn k_closest_heap(points: &[(f64, f64)], k: usize) -> Vec<(f64, f64)> {
    if k == 0 || points.is_empty() {
        return Vec::new();
    }
    if k >= points.len() {
        return points.to_vec();
    }

    // Max-heap by squared distance. The `Entry` wrapper supplies a
    // NaN-tolerant `Ord` impl so `BinaryHeap` (which requires `Ord`) is happy
    // with `f64` keys.
    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);
    for &p in points {
        let d2 = sq_dist(p);
        if heap.len() < k {
            heap.push(Entry { d2, p });
        } else if let Some(top) = heap.peek() {
            if cmp_f64(d2, top.d2) == Ordering::Less {
                heap.pop();
                heap.push(Entry { d2, p });
            }
        }
    }
    heap.into_iter().map(|e| e.p).collect()
}

/// Deterministic xorshift64* generator. Seeded by the input length so the
/// output is reproducible for a given input size while still varying enough
/// to dodge worst-case partitions on inputs that would defeat a fixed pivot.
#[inline]
const fn xorshift_next(state: &mut u64) -> u64 {
    // Avoid the absorbing zero state — fall back to a non-zero seed if the
    // caller passed a length that hashed down to 0.
    if *state == 0 {
        *state = 0x9E37_79B9_7F4A_7C15;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Hoare-style partition over `buf[lo..=hi]` keyed by `d2`. Returns an index
/// `j` such that every element in `lo..=j` has `d2 <= pivot` and every
/// element in `j+1..=hi` has `d2 >= pivot`. Hoare's scheme places no special
/// constraint on the pivot's final position, only on the boundary, which is
/// the property quickselect needs to recurse on the correct side.
fn hoare_partition(buf: &mut [(f64, (f64, f64))], lo: usize, hi: usize, rng: &mut u64) -> usize {
    let span = hi - lo + 1;
    let pivot_off = (xorshift_next(rng) as usize) % span;
    let pivot = buf[lo + pivot_off].0;

    // Use signed sentinels so we can step past the ends without underflow.
    let mut i = lo as isize - 1;
    let mut j = hi as isize + 1;
    loop {
        loop {
            i += 1;
            if cmp_f64(buf[i as usize].0, pivot) != Ordering::Less {
                break;
            }
        }
        loop {
            j -= 1;
            if cmp_f64(buf[j as usize].0, pivot) != Ordering::Greater {
                break;
            }
        }
        if i >= j {
            return j as usize;
        }
        buf.swap(i as usize, j as usize);
    }
}

/// Returns `k` points from `points` whose Euclidean distance from the origin
/// is smallest, computed via in-place Hoare quickselect.
///
/// We copy the input into a scratch buffer of `(d2, point)` pairs so the
/// algorithm can rearrange the points freely without disturbing the caller's
/// slice. Quickselect recurses on whichever side of the partition contains
/// the `k`-th element; once the boundary is established the first `k` slots
/// of the buffer hold (in unspecified internal order) the `k` closest points.
///
/// - Time: `O(n)` average, `O(n^2)` worst case.
/// - Extra space: `O(n)` for the scratch buffer.
///
/// The pivot is drawn from a `xorshift64*` PRNG seeded by `points.len()`,
/// which gives reproducible results across runs while still randomizing the
/// pivot away from adversarial input orderings.
///
/// As with [`k_closest_heap`], the returned order is unspecified and ties on
/// distance are broken by the partition's incidental layout.
///
/// # Examples
///
/// ```
/// use rust_algorithms::searching::k_closest_points::k_closest_quickselect;
///
/// let pts = [(1.0, 3.0), (-2.0, 2.0), (5.0, 8.0), (0.0, 1.0)];
/// let mut got = k_closest_quickselect(&pts, 2);
/// got.sort_by(|a, b| (a.0 * a.0 + a.1 * a.1)
///     .partial_cmp(&(b.0 * b.0 + b.1 * b.1)).unwrap());
/// assert_eq!(got, vec![(0.0, 1.0), (-2.0, 2.0)]);
/// ```
pub fn k_closest_quickselect(points: &[(f64, f64)], k: usize) -> Vec<(f64, f64)> {
    if k == 0 || points.is_empty() {
        return Vec::new();
    }
    if k >= points.len() {
        return points.to_vec();
    }

    let mut buf: Vec<(f64, (f64, f64))> = points.iter().map(|&p| (sq_dist(p), p)).collect();

    // Seed the PRNG from the length so the partition sequence is reproducible
    // for a given input size yet still varies across sizes.
    let mut rng: u64 =
        (points.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_F00D;

    // Iterative quickselect: locate the boundary at index `k - 1` so the
    // first `k` entries of `buf` are the smallest. Iteration avoids any risk
    // of stack overflow on adversarial inputs that would otherwise produce
    // deep recursion.
    let target = k - 1;
    let mut lo = 0usize;
    let mut hi = buf.len() - 1;
    while lo < hi {
        let p = hoare_partition(&mut buf, lo, hi, &mut rng);
        if target <= p {
            hi = p;
        } else {
            lo = p + 1;
        }
    }

    buf.into_iter().take(k).map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::{k_closest_heap, k_closest_quickselect, sq_dist};

    /// Sort a result vector by squared distance so two outputs of the same
    /// multiset compare equal. Using squared distance keeps the comparator
    /// exact and matches the algorithms' internal key.
    fn sorted_by_dist(mut v: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        v.sort_by(|a, b| sq_dist(*a).partial_cmp(&sq_dist(*b)).unwrap());
        v
    }

    #[test]
    fn k_zero_returns_empty() {
        let pts = [(1.0, 2.0), (3.0, 4.0)];
        assert!(k_closest_heap(&pts, 0).is_empty());
        assert!(k_closest_quickselect(&pts, 0).is_empty());
    }

    #[test]
    fn empty_input_returns_empty() {
        let pts: [(f64, f64); 0] = [];
        assert!(k_closest_heap(&pts, 3).is_empty());
        assert!(k_closest_quickselect(&pts, 3).is_empty());
    }

    #[test]
    fn k_larger_than_n_returns_all() {
        let pts = [(1.0, 2.0), (3.0, 4.0)];
        assert_eq!(
            sorted_by_dist(k_closest_heap(&pts, 5)),
            sorted_by_dist(pts.to_vec())
        );
        assert_eq!(
            sorted_by_dist(k_closest_quickselect(&pts, 5)),
            sorted_by_dist(pts.to_vec())
        );
    }

    #[test]
    fn k_equals_n_returns_all() {
        let pts = [(1.0, 2.0), (3.0, 4.0), (-1.0, 0.0)];
        assert_eq!(
            sorted_by_dist(k_closest_heap(&pts, 3)),
            sorted_by_dist(pts.to_vec())
        );
        assert_eq!(
            sorted_by_dist(k_closest_quickselect(&pts, 3)),
            sorted_by_dist(pts.to_vec())
        );
    }

    #[test]
    fn simple_known_example() {
        // Distances^2: (1,3)=10, (-2,2)=8, (5,8)=89, (0,1)=1.
        // The two closest are (0,1) and (-2,2).
        let pts = [(1.0, 3.0), (-2.0, 2.0), (5.0, 8.0), (0.0, 1.0)];
        let expected = vec![(0.0, 1.0), (-2.0, 2.0)];

        assert_eq!(sorted_by_dist(k_closest_heap(&pts, 2)), expected);
        assert_eq!(sorted_by_dist(k_closest_quickselect(&pts, 2)), expected);
    }

    #[test]
    fn single_point() {
        let pts = [(7.0, -3.0)];
        assert_eq!(k_closest_heap(&pts, 1), vec![(7.0, -3.0)]);
        assert_eq!(k_closest_quickselect(&pts, 1), vec![(7.0, -3.0)]);
    }

    #[test]
    fn implementations_agree_on_small_inputs() {
        // A handful of hand-rolled inputs covering negatives, ties, and
        // duplicates. We compare the two algorithms by sorting their outputs
        // by distance — the spec only guarantees the multiset of returned
        // points, so order-insensitive equality is the right check.
        let cases: Vec<(Vec<(f64, f64)>, usize)> = vec![
            (vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0)], 2),
            (vec![(3.0, 4.0), (-3.0, -4.0), (1.0, 0.0), (0.0, 1.0)], 3),
            (vec![(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)], 2), // duplicates
            (vec![(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)], 4), // tied
            (
                vec![
                    (10.0, 0.0),
                    (0.0, -10.0),
                    (5.0, 5.0),
                    (-5.0, 5.0),
                    (1.0, 1.0),
                ],
                1,
            ),
        ];

        for (pts, k) in cases {
            let a = sorted_by_dist(k_closest_heap(&pts, k));
            let b = sorted_by_dist(k_closest_quickselect(&pts, k));
            // The two outputs must match position-for-position once sorted by
            // distance — equal-distance ties are broken differently between
            // the heap and partition, but the *distances* of the returned
            // points are identical.
            assert_eq!(a.len(), b.len(), "lengths differ for k={k}");
            for (pa, pb) in a.iter().zip(b.iter()) {
                assert!(
                    (sq_dist(*pa) - sq_dist(*pb)).abs() < 1e-12,
                    "distance mismatch: {pa:?} vs {pb:?} (k={k})"
                );
            }
        }
    }

    #[test]
    fn returns_correct_count() {
        let pts: Vec<(f64, f64)> = (0..50).map(|i| (i as f64, (50 - i) as f64)).collect();
        for k in 0..=pts.len() {
            assert_eq!(k_closest_heap(&pts, k).len(), k);
            assert_eq!(k_closest_quickselect(&pts, k).len(), k);
        }
    }

    #[test]
    fn returned_points_are_the_closest() {
        // Brute-force the k smallest distances and confirm the algorithms'
        // outputs share the same distance multiset.
        let pts: Vec<(f64, f64)> = (0..30)
            .map(|i| {
                let x = ((i * 7) % 11) as f64 - 5.0;
                let y = ((i * 13) % 17) as f64 - 8.0;
                (x, y)
            })
            .collect();
        let mut all_d: Vec<f64> = pts.iter().map(|&p| sq_dist(p)).collect();
        all_d.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for k in [1usize, 5, 10, 20, 30] {
            let expected_d = &all_d[..k];

            let mut got_h: Vec<f64> = k_closest_heap(&pts, k)
                .iter()
                .map(|&p| sq_dist(p))
                .collect();
            got_h.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(got_h.as_slice(), expected_d, "heap mismatch at k={k}");

            let mut got_q: Vec<f64> = k_closest_quickselect(&pts, k)
                .iter()
                .map(|&p| sq_dist(p))
                .collect();
            got_q.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(
                got_q.as_slice(),
                expected_d,
                "quickselect mismatch at k={k}"
            );
        }
    }

    #[test]
    fn ties_documented_behavior() {
        // All four points are at distance 1 from the origin. With k=2 both
        // algorithms must return *some* two of them; the spec is that ties
        // are broken by encounter order in the input but only as an
        // implementation detail. We assert the weaker, contractual property:
        // every returned point has the tied distance.
        let pts = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
        for got in [k_closest_heap(&pts, 2), k_closest_quickselect(&pts, 2)] {
            assert_eq!(got.len(), 2);
            for p in got {
                assert!((sq_dist(p) - 1.0).abs() < 1e-12);
            }
        }
    }
}
