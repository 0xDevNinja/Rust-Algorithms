//! Fractional knapsack via greedy value-density (value/weight) selection.
//!
//! Given items as `(value, weight)` pairs and a real-valued knapsack capacity,
//! returns the maximum total value achievable when items may be split into
//! arbitrary fractions. Sorts the input indices by value-to-weight ratio in
//! descending order, takes whole items while they fit, and finally takes a
//! fraction of the next item to exactly fill the remaining capacity.
//!
//! Time complexity: `O(n log n)` (dominated by the sort).
//! Space complexity: `O(n)` for the index permutation.
//!
//! Greedy optimality: for the *fractional* (continuous) knapsack the
//! highest-density-first rule is provably optimal — any feasible solution can
//! be transformed into the greedy one by swapping mass from a lower-density
//! item to a higher-density one without decreasing total value. This is in
//! contrast to the 0/1 knapsack, where greedy is only a 2-approximation.
//!
//! Edge cases:
//! - Empty input or `capacity <= 0.0` returns `0.0`.
//! - Items with `weight <= 0.0` are treated as zero-weight: their full `value`
//!   is included if `value > 0.0`, and they are skipped otherwise. They never
//!   consume capacity, so they are processed before any fractional split. (The
//!   `capacity <= 0.0` short-circuit takes priority over zero-weight items;
//!   the convention is that an empty knapsack carries nothing at all.)
//! - Items with negative `value` are skipped (taking zero of them is feasible
//!   and always at least as good as taking any positive amount).
//! - Non-finite inputs (`NaN`, `±∞`) in `items` or `capacity` are not
//!   supported; behaviour is unspecified.

/// Returns the maximum total value achievable by packing `items` into a
/// knapsack of the given `capacity`, allowing fractional pieces of items.
///
/// `items` is a slice of `(value, weight)` pairs. The output is the optimal
/// total value as an `f64`.
///
/// Time: `O(n log n)`. Space: `O(n)`.
///
/// See the module-level documentation for edge-case conventions.
#[must_use]
pub fn fractional_knapsack(items: &[(f64, f64)], capacity: f64) -> f64 {
    // Treat capacity <= 0 (and NaN) the same: nothing can be packed.
    if items.is_empty() || capacity.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }

    // Sort indices by value/weight ratio descending. Zero-weight positive-value
    // items have effectively infinite density and sort first.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&i, &j| {
        density(items[j])
            .partial_cmp(&density(items[i]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut remaining = capacity;
    let mut total = 0.0_f64;

    for i in order {
        let (value, weight) = items[i];
        if value <= 0.0 {
            // Skip items that cannot improve the objective.
            continue;
        }
        if weight <= 0.0 {
            // Zero/negative-weight item with positive value: take it fully,
            // it consumes no capacity.
            total += value;
            continue;
        }
        if remaining <= 0.0 {
            break;
        }
        if weight <= remaining {
            total += value;
            remaining -= weight;
        } else {
            // Take the fraction that exactly fills the remaining capacity.
            total += value * (remaining / weight);
            break;
        }
    }

    total
}

/// Value density (value-per-unit-weight). Zero-or-negative weights with
/// positive value sort to the top via `f64::INFINITY`.
fn density((value, weight): (f64, f64)) -> f64 {
    if weight <= 0.0 {
        if value > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        value / weight
    }
}

#[cfg(test)]
mod tests {
    use super::fractional_knapsack;
    use quickcheck_macros::quickcheck;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn empty_items_returns_zero() {
        assert_eq!(fractional_knapsack(&[], 50.0), 0.0);
    }

    #[test]
    fn zero_capacity_returns_zero() {
        let items = [(60.0, 10.0), (100.0, 20.0)];
        assert_eq!(fractional_knapsack(&items, 0.0), 0.0);
    }

    #[test]
    fn negative_capacity_returns_zero() {
        let items = [(60.0, 10.0)];
        assert_eq!(fractional_knapsack(&items, -5.0), 0.0);
    }

    #[test]
    fn single_item_under_capacity_takes_full_value() {
        // weight (10) fits inside capacity (50): take the whole item.
        let items = [(60.0, 10.0)];
        assert!(approx_eq(fractional_knapsack(&items, 50.0), 60.0));
    }

    #[test]
    fn single_item_at_capacity_takes_full_value() {
        let items = [(42.0, 7.0)];
        assert!(approx_eq(fractional_knapsack(&items, 7.0), 42.0));
    }

    #[test]
    fn single_item_over_capacity_takes_fraction() {
        // weight 10, capacity 4 -> fraction 0.4 -> value 60 * 0.4 = 24.
        let items = [(60.0, 10.0)];
        assert!(approx_eq(fractional_knapsack(&items, 4.0), 24.0));
    }

    #[test]
    fn classic_textbook_example() {
        // CLRS / standard textbook example: items by (value, weight)
        //   (60, 10), (100, 20), (120, 30); capacity 50 -> 240.0.
        // Densities: 6.0, 5.0, 4.0. Take items 0 and 1 fully (weight 30,
        // value 160), then 20/30 of item 2 = value 80. Total = 240.
        let items = [(60.0, 10.0), (100.0, 20.0), (120.0, 30.0)];
        let result = fractional_knapsack(&items, 50.0);
        assert!(
            (result - 240.0).abs() < 1e-9,
            "expected 240.0, got {result}"
        );
    }

    #[test]
    fn classic_example_input_order_independent() {
        // Same items reversed should still produce 240 — verifies ratio sort
        // rather than insertion order.
        let items = [(120.0, 30.0), (100.0, 20.0), (60.0, 10.0)];
        let result = fractional_knapsack(&items, 50.0);
        assert!((result - 240.0).abs() < 1e-9);
    }

    #[test]
    fn ratio_sort_beats_input_order() {
        // Greedy by input order would take (10, 10) fully then 40/50 of (40, 50)
        // = 10 + 32 = 42. Optimal (ratio sort) takes (40, 50) at 50/50 = 40,
        // wait — recompute: capacity 50, items [(10,10), (40,50)].
        //   input-order: take (10,10) fully -> 10, remaining 40, take 40/50 of
        //     (40,50) -> 32. Total 42.
        //   ratio-sort: density 1.0 vs 0.8 -> take (10,10) first anyway.
        // Construct a case where input order is strictly worse:
        //   items [(10, 50), (40, 10)], capacity 20.
        //   input-order: 20/50 of (10,50) = 4. Total 4.
        //   ratio-sort: density 0.2 vs 4.0 -> take (40,10) fully -> 40,
        //     remaining 10, take 10/50 of (10,50) = 2. Total 42.
        let items = [(10.0, 50.0), (40.0, 10.0)];
        let result = fractional_knapsack(&items, 20.0);
        assert!((result - 42.0).abs() < 1e-9, "expected 42.0, got {result}");
    }

    #[test]
    fn all_zero_value_returns_zero() {
        let items = [(0.0, 5.0), (0.0, 10.0), (0.0, 1.0)];
        assert_eq!(fractional_knapsack(&items, 100.0), 0.0);
    }

    #[test]
    fn zero_weight_positive_value_taken_fully() {
        // Zero-weight items consume no capacity and contribute their full
        // value. Combined with a normal item that fills the knapsack exactly.
        let items = [(7.0, 0.0), (10.0, 5.0)];
        let result = fractional_knapsack(&items, 5.0);
        assert!((result - 17.0).abs() < 1e-9);
    }

    #[test]
    fn zero_weight_zero_value_skipped() {
        let items = [(0.0, 0.0), (10.0, 5.0)];
        let result = fractional_knapsack(&items, 5.0);
        assert!((result - 10.0).abs() < 1e-9);
    }

    #[test]
    fn negative_value_items_skipped() {
        // A negative-value item should never be taken.
        let items = [(-100.0, 5.0), (10.0, 5.0)];
        let result = fractional_knapsack(&items, 100.0);
        assert!((result - 10.0).abs() < 1e-9);
    }

    #[test]
    fn capacity_far_exceeds_total_weight() {
        // When capacity dwarfs total weight, result is the sum of all values.
        let items = [(3.0, 1.0), (5.0, 2.0), (8.0, 4.0)];
        let result = fractional_knapsack(&items, 1000.0);
        assert!((result - 16.0).abs() < 1e-9);
    }

    /// Take items in input order, fractionally filling the knapsack — a
    /// non-greedy heuristic that the optimal greedy must always match or beat.
    fn input_order_fill(items: &[(f64, f64)], capacity: f64) -> f64 {
        let mut remaining = capacity.max(0.0);
        let mut total = 0.0_f64;
        for &(value, weight) in items {
            if value <= 0.0 {
                continue;
            }
            if weight <= 0.0 {
                total += value;
                continue;
            }
            if remaining <= 0.0 {
                break;
            }
            if weight <= remaining {
                total += value;
                remaining -= weight;
            } else {
                total += value * (remaining / weight);
                break;
            }
        }
        total
    }

    fn sum_positive_values(items: &[(f64, f64)]) -> f64 {
        items
            .iter()
            .filter(|&&(v, _)| v > 0.0)
            .map(|&(v, _)| v)
            .sum()
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn greedy_at_least_input_order_and_at_most_unlimited(raw: Vec<(u8, u8)>) -> bool {
        // Cap n at 8 and use small unsigned values cast to f64 so we stay in
        // exact-representable territory and avoid pathological FP cases.
        let items: Vec<(f64, f64)> = raw
            .into_iter()
            .take(8)
            // Map to strictly positive weights to keep the bounds meaningful.
            .map(|(v, w)| (f64::from(v), f64::from(w).max(1.0)))
            .collect();

        // Capacity drawn from the items themselves to keep it in range.
        let total_weight: f64 = items.iter().map(|&(_, w)| w).sum();
        // Try several capacities: 0, half, all, and a generous over-cap.
        for &cap in &[0.0, total_weight * 0.5, total_weight, total_weight + 10.0] {
            let g = fractional_knapsack(&items, cap);
            let lower = input_order_fill(&items, cap);
            let upper = sum_positive_values(&items);
            // Greedy is never worse than the input-order heuristic and never
            // better than taking every positive-value item entirely.
            if g + 1e-9 < lower {
                return false;
            }
            if g > upper + 1e-9 {
                return false;
            }
        }
        true
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn greedy_matches_independent_reference(raw: Vec<(u8, u8)>) -> bool {
        // For fractional knapsack, greedy IS the optimum, so an independent
        // implementation that sorts and fills must agree exactly (mod EPS).
        let items: Vec<(f64, f64)> = raw
            .into_iter()
            .take(8)
            .map(|(v, w)| (f64::from(v), f64::from(w).max(1.0)))
            .collect();
        let total_weight: f64 = items.iter().map(|&(_, w)| w).sum();

        for &cap in &[0.0, total_weight * 0.25, total_weight, total_weight * 2.0] {
            let g = fractional_knapsack(&items, cap);
            let r = reference_fill(&items, cap);
            if (g - r).abs() > 1e-9 {
                return false;
            }
        }
        true
    }

    /// Independent sort-and-fill reference — same algorithm, written from
    /// scratch so a regression in the production version does not silently
    /// agree with itself.
    fn reference_fill(items: &[(f64, f64)], capacity: f64) -> f64 {
        if capacity <= 0.0 {
            return 0.0;
        }
        let mut filtered: Vec<(f64, f64)> =
            items.iter().copied().filter(|&(v, _)| v > 0.0).collect();
        // Highest density first; treat zero/negative weight as +inf density.
        filtered.sort_by(|&(va, wa), &(vb, wb)| {
            let da = if wa <= 0.0 { f64::INFINITY } else { va / wa };
            let db = if wb <= 0.0 { f64::INFINITY } else { vb / wb };
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut remaining = capacity;
        let mut total = 0.0;
        for (v, w) in filtered {
            if w <= 0.0 {
                total += v;
                continue;
            }
            if remaining <= 0.0 {
                break;
            }
            if w <= remaining {
                total += v;
                remaining -= w;
            } else {
                total += v * (remaining / w);
                break;
            }
        }
        total
    }
}
