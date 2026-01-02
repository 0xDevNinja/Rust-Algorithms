//! Bucket sort for non-negative `f64` keys in `[0, 1)`. Average O(n + k) when
//! input is uniform; O(n²) worst case (all elements in one bucket). Stable
//! when each bucket is sorted with a stable algorithm.

use super::insertion_sort::insertion_sort;

/// Sorts `slice` of `f64` values assumed to lie in `[0.0, 1.0)`.
///
/// Out-of-range values are clamped into the last bucket; if you need a
/// general-range bucket sort, scale your input first.
pub fn bucket_sort(slice: &mut [f64]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    let mut buckets: Vec<Vec<OrdF64>> = (0..n).map(|_| Vec::new()).collect();
    for &x in slice.iter() {
        let mut idx = (x * n as f64) as usize;
        if idx >= n {
            idx = n - 1;
        }
        buckets[idx].push(OrdF64(x));
    }
    let mut pos = 0;
    for mut b in buckets {
        insertion_sort(&mut b);
        for OrdF64(v) in b {
            slice[pos] = v;
            pos += 1;
        }
    }
}

/// Wrapper to give a total order to `f64` (treats NaN-free input).
#[derive(Copy, Clone, PartialEq, PartialOrd)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::bucket_sort;

    fn approx_sorted(v: &[f64]) -> bool {
        v.windows(2).all(|w| w[0] <= w[1])
    }

    #[test]
    fn empty() {
        let mut v: Vec<f64> = vec![];
        bucket_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn single() {
        let mut v = vec![0.42];
        bucket_sort(&mut v);
        assert_eq!(v, vec![0.42]);
    }

    #[test]
    fn uniform_distribution() {
        let mut v = vec![0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68];
        bucket_sort(&mut v);
        assert!(approx_sorted(&v));
    }

    #[test]
    fn already_sorted() {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let expected = v.clone();
        bucket_sort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn duplicates() {
        let mut v = vec![0.5, 0.5, 0.5, 0.5];
        bucket_sort(&mut v);
        assert_eq!(v, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn boundary_values() {
        let mut v = vec![0.999, 0.0, 0.5];
        bucket_sort(&mut v);
        assert!(approx_sorted(&v));
        assert_eq!(v[0], 0.0);
    }
}
