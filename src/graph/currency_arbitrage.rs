//! Currency arbitrage detection via Bellman–Ford on `-ln(rate)` weights.
//!
//! Given a square exchange-rate matrix where `rates[i][j]` is the multiplier
//! for converting one unit of currency `i` into currency `j`, an arbitrage
//! opportunity exists iff there is a cycle whose product of rates exceeds 1.
//! Taking `-ln` of each rate turns multiplicative cycles into additive ones:
//! the product `r1 * r2 * ... * rk > 1` becomes `(-ln r1) + ... + (-ln rk) < 0`,
//! i.e. a negative cycle in the resulting weighted digraph.
//!
//! We run a single Bellman–Ford relaxation pass from a virtual source connected
//! to every node with weight `0` (equivalently, initialise every distance to
//! `0`). If any edge can still be relaxed after `V - 1` rounds, a negative
//! cycle — and therefore an arbitrage — exists.
//!
//! Complexity: O(V·E) where V is the number of currencies and E = V² edges
//! (one per non-zero entry of the matrix), giving O(V³) overall.
//!
//! Non-positive rates (`<= 0`) and non-finite rates are skipped: they have no
//! meaningful logarithm and cannot form a valid trade edge.
//!
//! No `unsafe`, no external dependencies.

/// Returns `true` if the exchange-rate matrix admits an arbitrage cycle.
///
/// `rates` must be square. `rates[i][j]` is the multiplier when converting
/// currency `i` into currency `j`. Self-loops (`rates[i][i]`) and entries
/// that are `<= 0` or non-finite are ignored.
pub fn has_arbitrage(rates: &[Vec<f64>]) -> bool {
    let n = rates.len();
    if n < 2 {
        return false;
    }
    for row in rates {
        if row.len() != n {
            return false;
        }
    }

    // Build edge list with weight = -ln(rate). Skip invalid / self edges.
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let r = rates[i][j];
            if !r.is_finite() || r <= 0.0 {
                continue;
            }
            edges.push((i, j, -r.ln()));
        }
    }

    // Initialise every distance to 0 — equivalent to a virtual source with
    // zero-weight edges to all vertices. This guarantees every node is
    // reachable so any negative cycle anywhere will trigger relaxation.
    let mut dist = vec![0.0_f64; n];

    for _ in 0..n.saturating_sub(1) {
        let mut updated = false;
        for &(u, v, w) in &edges {
            let candidate = dist[u] + w;
            if candidate < dist[v] {
                dist[v] = candidate;
                updated = true;
            }
        }
        if !updated {
            break;
        }
    }

    // One more pass: any further relaxation proves a negative cycle exists.
    edges.iter().any(|&(u, v, w)| dist[u] + w < dist[v])
}

#[cfg(test)]
mod tests {
    use super::has_arbitrage;

    #[test]
    fn single_currency_no_arbitrage() {
        let rates = vec![vec![1.0]];
        assert!(!has_arbitrage(&rates));
    }

    #[test]
    fn empty_matrix_no_arbitrage() {
        let rates: Vec<Vec<f64>> = vec![];
        assert!(!has_arbitrage(&rates));
    }

    #[test]
    fn two_currency_profitable_cycle() {
        // USD -> EUR -> USD with product 1.1 * 1.0 = 1.1 > 1 (arbitrage).
        // rates[0][1] = 1.1 (USD->EUR), rates[1][0] = 1.0 (EUR->USD).
        let rates = vec![vec![1.0, 1.1], vec![1.0, 1.0]];
        assert!(has_arbitrage(&rates));
    }

    #[test]
    fn two_currency_no_arbitrage() {
        // Reciprocal rates, product == 1, no arbitrage.
        let rates = vec![vec![1.0, 2.0], vec![0.5, 1.0]];
        assert!(!has_arbitrage(&rates));
    }

    #[test]
    fn classic_four_currency_arbitrage() {
        // USD, EUR, GBP, JPY. Construct a cycle USD->EUR->GBP->USD with
        // product 0.9 * 0.88 * 1.30 = 1.0296 > 1 (arbitrage).
        let rates = vec![
            // USD: ->USD ->EUR  ->GBP  ->JPY
            vec![1.0, 0.9, 0.75, 110.0],
            // EUR
            vec![1.11, 1.0, 0.88, 122.0],
            // GBP — GBP->USD = 1.30 closes the cycle profitably.
            vec![1.30, 1.13, 1.0, 138.0],
            // JPY
            vec![0.009, 0.0082, 0.0072, 1.0],
        ];
        assert!(has_arbitrage(&rates));
    }

    #[test]
    fn four_currency_consistent_no_arbitrage() {
        // Rates derived from a single price vector p = [1, 2, 4, 8]:
        // rates[i][j] = p[j] / p[i]. Every cycle product == 1.
        let p = [1.0_f64, 2.0, 4.0, 8.0];
        let n = p.len();
        let mut rates = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                rates[i][j] = p[j] / p[i];
            }
        }
        assert!(!has_arbitrage(&rates));
    }

    #[test]
    fn ignores_non_positive_entries() {
        // Zero / negative entries should be treated as missing edges, not
        // crash the algorithm. With only the trivial self-loops and one
        // valid edge the graph cannot form a cycle.
        let rates = vec![vec![1.0, 0.0], vec![-1.0, 1.0]];
        assert!(!has_arbitrage(&rates));
    }

    #[test]
    fn non_square_rejected() {
        let rates = vec![vec![1.0, 2.0], vec![0.5]];
        assert!(!has_arbitrage(&rates));
    }
}
