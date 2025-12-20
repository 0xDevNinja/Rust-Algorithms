//! Rod cutting: maximise revenue from cutting a rod of length `n` into
//! integer-length pieces. O(n²).

/// Returns the maximum revenue obtainable. `prices[i]` is the price of a
/// piece of length `i + 1`. Pieces of length > `prices.len()` are not sold.
pub fn rod_cutting(n: usize, prices: &[u64]) -> u64 {
    let mut dp = vec![0_u64; n + 1];
    for length in 1..=n {
        let mut best = 0_u64;
        for cut in 1..=length.min(prices.len()) {
            let candidate = prices[cut - 1] + dp[length - cut];
            if candidate > best {
                best = candidate;
            }
        }
        dp[length] = best;
    }
    dp[n]
}

#[cfg(test)]
mod tests {
    use super::rod_cutting;

    #[test]
    fn clrs_example() {
        // Length:  1 2 3 4 5  6  7  8  9 10
        // Price:   1 5 8 9 10 17 17 20 24 30
        let prices = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30];
        assert_eq!(rod_cutting(4, &prices), 10);
        assert_eq!(rod_cutting(8, &prices), 22);
    }

    #[test]
    fn zero_length() {
        assert_eq!(rod_cutting(0, &[1, 5, 8]), 0);
    }

    #[test]
    fn no_prices() {
        assert_eq!(rod_cutting(5, &[]), 0);
    }

    #[test]
    fn longer_than_table_uses_subcuts() {
        let prices = [3];
        assert_eq!(rod_cutting(5, &prices), 15);
    }
}
