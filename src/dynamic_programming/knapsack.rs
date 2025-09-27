//! 0/1 Knapsack: pick a subset of items maximising value subject to a weight
//! capacity. O(n · W) time, O(W) space.

/// Returns the maximum total value selectable without exceeding `capacity`.
/// `weights[i]` and `values[i]` describe item `i`. Item count must equal
/// the length of both slices.
pub fn knapsack_01(capacity: usize, weights: &[usize], values: &[u64]) -> u64 {
    assert_eq!(
        weights.len(),
        values.len(),
        "weights and values must have the same length"
    );
    let mut dp = vec![0_u64; capacity + 1];
    for i in 0..weights.len() {
        let w = weights[i];
        let v = values[i];
        if w > capacity {
            continue;
        }
        for cap in (w..=capacity).rev() {
            let candidate = dp[cap - w] + v;
            if candidate > dp[cap] {
                dp[cap] = candidate;
            }
        }
    }
    dp[capacity]
}

#[cfg(test)]
mod tests {
    use super::knapsack_01;

    #[test]
    fn classic_example() {
        // weights / values from CLRS-style example.
        let w = vec![2, 3, 4, 5];
        let v = vec![3, 4, 5, 6];
        assert_eq!(knapsack_01(5, &w, &v), 7);
    }

    #[test]
    fn item_too_heavy() {
        assert_eq!(knapsack_01(3, &[5], &[100]), 0);
    }

    #[test]
    fn zero_capacity() {
        assert_eq!(knapsack_01(0, &[1, 2], &[3, 4]), 0);
    }

    #[test]
    fn empty() {
        assert_eq!(knapsack_01(10, &[], &[]), 0);
    }
}
