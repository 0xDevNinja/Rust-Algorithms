//! Best time to buy and sell stock — single transaction. O(n).
//!
//! Given a slice of daily prices, find the maximum profit obtainable by
//! buying once and selling once on a later day. If no profitable trade
//! exists, the profit is `0`.
//!
//! The single-pass algorithm tracks the minimum price seen so far and the
//! best profit achievable when selling on the current day.

/// Returns the maximum profit from a single buy-then-sell transaction over
/// `prices`. Returns `0` if no profitable trade exists (including empty or
/// monotonically non-increasing inputs).
pub fn max_profit(prices: &[i64]) -> i64 {
    let mut min_price = i64::MAX;
    let mut best = 0_i64;
    for &p in prices {
        if p < min_price {
            min_price = p;
        } else {
            let profit = p - min_price;
            if profit > best {
                best = profit;
            }
        }
    }
    best
}

/// Returns `(profit, Some((buy_idx, sell_idx)))` for the optimal single
/// buy-then-sell transaction, or `(0, None)` if no profitable pair exists.
pub fn max_profit_with_indices(prices: &[i64]) -> (i64, Option<(usize, usize)>) {
    let mut min_price = i64::MAX;
    let mut min_idx = 0_usize;
    let mut best = 0_i64;
    let mut best_pair: Option<(usize, usize)> = None;
    for (i, &p) in prices.iter().enumerate() {
        if p < min_price {
            min_price = p;
            min_idx = i;
        } else {
            let profit = p - min_price;
            if profit > best {
                best = profit;
                best_pair = Some((min_idx, i));
            }
        }
    }
    (best, best_pair)
}

#[cfg(test)]
mod tests {
    use super::{max_profit, max_profit_with_indices};

    #[test]
    fn empty_input() {
        assert_eq!(max_profit(&[]), 0);
        assert_eq!(max_profit_with_indices(&[]), (0, None));
    }

    #[test]
    fn single_price() {
        assert_eq!(max_profit(&[42]), 0);
        assert_eq!(max_profit_with_indices(&[42]), (0, None));
    }

    #[test]
    fn monotonic_decreasing() {
        let v = [9_i64, 7, 5, 3, 1];
        assert_eq!(max_profit(&v), 0);
        assert_eq!(max_profit_with_indices(&v), (0, None));
    }

    #[test]
    fn classic_example() {
        // [7,1,5,3,6,4] -> buy at index 1 (price 1), sell at index 4 (price 6)
        let v = [7_i64, 1, 5, 3, 6, 4];
        assert_eq!(max_profit(&v), 5);
        assert_eq!(max_profit_with_indices(&v), (5, Some((1, 4))));
    }

    #[test]
    fn monotonic_increasing() {
        // [1,2,3,4,5] -> buy at 0, sell at 4, profit 4
        let v = [1_i64, 2, 3, 4, 5];
        assert_eq!(max_profit(&v), 4);
        assert_eq!(max_profit_with_indices(&v), (4, Some((0, 4))));
    }

    #[test]
    fn all_equal() {
        let v = [3_i64, 3, 3, 3];
        assert_eq!(max_profit(&v), 0);
        assert_eq!(max_profit_with_indices(&v), (0, None));
    }

    #[test]
    fn dip_then_recover() {
        // Lowest price comes after an early peak; ensure we use the later min.
        let v = [5_i64, 4, 3, 2, 10];
        assert_eq!(max_profit(&v), 8);
        assert_eq!(max_profit_with_indices(&v), (8, Some((3, 4))));
    }

    #[test]
    fn two_elements_profitable() {
        let v = [1_i64, 100];
        assert_eq!(max_profit(&v), 99);
        assert_eq!(max_profit_with_indices(&v), (99, Some((0, 1))));
    }

    #[test]
    fn two_elements_not_profitable() {
        let v = [100_i64, 1];
        assert_eq!(max_profit(&v), 0);
        assert_eq!(max_profit_with_indices(&v), (0, None));
    }
}
