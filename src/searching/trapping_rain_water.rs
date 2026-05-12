//! Trapping Rain Water — two-pointer linear sweep.
//!
//! Given a histogram of non-negative bar heights, compute the total volume of
//! water that would be trapped between the bars after rain. The water above
//! any column `i` equals
//! `min(max(heights[..=i]), max(heights[i..])) - heights[i]` (clamped at 0).
//!
//! The two-pointer technique avoids the auxiliary prefix/suffix-max arrays:
//! maintain `lo`/`hi` indices and `lmax`/`rmax` so far. At each step move the
//! lower side inward; whichever side is lower is bounded by its own running
//! max (the other side is known to be at least as tall), so we can safely
//! accumulate `running_max - current` for that column.
//!
//! - Time: `O(n)`.
//! - Space: `O(1)`.

/// Total trapped water for the given histogram of bar heights.
///
/// Returns `0` for empty or singleton inputs and for any monotone or flat
/// profile. Uses `u64` arithmetic throughout; never overflows for inputs that
/// fit the elevation map model (per-column water is bounded by the larger of
/// the two side maxima).
#[must_use]
pub fn trap(heights: &[u64]) -> u64 {
    if heights.len() < 3 {
        return 0;
    }
    let (mut lo, mut hi) = (0_usize, heights.len() - 1);
    let (mut lmax, mut rmax) = (0_u64, 0_u64);
    let mut water = 0_u64;
    while lo < hi {
        if heights[lo] <= heights[hi] {
            if heights[lo] >= lmax {
                lmax = heights[lo];
            } else {
                water += lmax - heights[lo];
            }
            lo += 1;
        } else {
            if heights[hi] >= rmax {
                rmax = heights[hi];
            } else {
                water += rmax - heights[hi];
            }
            hi -= 1;
        }
    }
    water
}

#[cfg(test)]
mod tests {
    use super::trap;

    #[test]
    fn empty_returns_zero() {
        assert_eq!(trap(&[]), 0);
    }

    #[test]
    fn single_bar_returns_zero() {
        assert_eq!(trap(&[5]), 0);
    }

    #[test]
    fn two_bars_returns_zero() {
        assert_eq!(trap(&[3, 7]), 0);
    }

    #[test]
    fn classic_example_six() {
        assert_eq!(trap(&[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6);
    }

    #[test]
    fn descending_then_ascending_nine() {
        assert_eq!(trap(&[4, 2, 0, 3, 2, 5]), 9);
    }

    #[test]
    fn monotonic_increasing_traps_nothing() {
        assert_eq!(trap(&[1, 2, 3, 4, 5, 6]), 0);
    }

    #[test]
    fn monotonic_decreasing_traps_nothing() {
        assert_eq!(trap(&[6, 5, 4, 3, 2, 1]), 0);
    }

    #[test]
    fn plateau_traps_nothing() {
        assert_eq!(trap(&[3, 3, 3, 3, 3]), 0);
    }

    #[test]
    fn all_zero_traps_nothing() {
        assert_eq!(trap(&[0, 0, 0, 0]), 0);
    }

    #[test]
    fn simple_well() {
        // walls of height 2 around a single empty cell -> 2 units
        assert_eq!(trap(&[2, 0, 2]), 2);
    }

    #[test]
    fn matches_brute_force_small() {
        let cases: &[&[u64]] = &[
            &[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
            &[4, 2, 0, 3, 2, 5],
            &[2, 0, 2],
            &[5, 4, 1, 2],
            &[0, 7, 1, 4, 6],
            &[3, 0, 0, 2, 0, 4],
        ];
        for &h in cases {
            assert_eq!(trap(h), brute(h), "mismatch on {h:?}");
        }
    }

    fn brute(h: &[u64]) -> u64 {
        let n = h.len();
        let mut total = 0_u64;
        for i in 0..n {
            let lmax = h[..=i].iter().copied().max().unwrap_or(0);
            let rmax = h[i..].iter().copied().max().unwrap_or(0);
            total += lmax.min(rmax) - h[i];
        }
        total
    }
}
