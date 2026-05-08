//! Integer partition enumeration and counting.
//!
//! `partitions(n)` enumerates every partition of `n` as a non-increasing
//! `Vec<u32>`. The recursion fixes a maximum allowed part and emits each
//! choice greedily; the number of leaves equals `p(n)`.
//!
//! `partition_count(n)` evaluates Euler's pentagonal-number recurrence
//! `p(n) = Sum_{k>=1} (-1)^{k-1} (p(n - k(3k-1)/2) + p(n - k(3k+1)/2))`,
//! which terminates in `O(sqrt(n))` terms per step.
//!
//! Complexity:
//! - `partitions(n)`: `O(p(n) * n)` time, output-sensitive.
//! - `partition_count(n)`: `O(n * sqrt(n))` time, `O(n)` space.

/// Returns every partition of `n` as a non-increasing `Vec<u32>`.
///
/// `partitions(0)` is `vec![vec![]]` (the empty partition).
pub fn partitions(n: u32) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    enumerate(n, n, &mut current, &mut out);
    out
}

fn enumerate(remaining: u32, max_part: u32, current: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
    if remaining == 0 {
        out.push(current.clone());
        return;
    }
    let upper = remaining.min(max_part);
    for part in (1..=upper).rev() {
        current.push(part);
        enumerate(remaining - part, part, current, out);
        current.pop();
    }
}

/// Returns `p(n)`, the number of integer partitions of `n`, via Euler's
/// pentagonal-number recurrence.
pub fn partition_count(n: u32) -> u64 {
    let n = n as usize;
    let mut p = vec![0_u64; n + 1];
    p[0] = 1;
    for m in 1..=n {
        let mut total: i64 = 0;
        let mut k: i64 = 1;
        loop {
            let g1 = (k * (3 * k - 1) / 2) as usize;
            let g2 = (k * (3 * k + 1) / 2) as usize;
            if g1 > m {
                break;
            }
            let sign = if k % 2 == 1 { 1 } else { -1 };
            total += sign * p[m - g1] as i64;
            if g2 <= m {
                total += sign * p[m - g2] as i64;
            }
            k += 1;
        }
        p[m] = total as u64;
    }
    p[n]
}

#[cfg(test)]
mod tests {
    use super::{partition_count, partitions};

    #[test]
    fn empty_partition_for_zero() {
        assert_eq!(partitions(0), vec![Vec::<u32>::new()]);
    }

    #[test]
    fn single_partition_for_one() {
        assert_eq!(partitions(1), vec![vec![1]]);
    }

    #[test]
    fn canonical_partitions_of_four() {
        let expected = vec![
            vec![4],
            vec![3, 1],
            vec![2, 2],
            vec![2, 1, 1],
            vec![1, 1, 1, 1],
        ];
        assert_eq!(partitions(4), expected);
    }

    #[test]
    fn enumeration_matches_count_small() {
        for n in 0..=12 {
            let parts = partitions(n);
            assert_eq!(parts.len() as u64, partition_count(n), "mismatch at n={n}");
        }
    }

    #[test]
    fn each_partition_sums_and_is_non_increasing() {
        for n in 0..=10 {
            for part in partitions(n) {
                assert_eq!(part.iter().sum::<u32>(), n);
                assert!(part.windows(2).all(|w| w[0] >= w[1]));
            }
        }
    }

    #[test]
    fn partition_count_50() {
        assert_eq!(partition_count(50), 204_226);
    }

    #[test]
    fn partition_count_100() {
        assert_eq!(partition_count(100), 190_569_292);
    }

    #[test]
    fn partition_count_zero_and_one() {
        assert_eq!(partition_count(0), 1);
        assert_eq!(partition_count(1), 1);
    }
}
