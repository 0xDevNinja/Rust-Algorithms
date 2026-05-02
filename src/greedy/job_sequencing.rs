//! Job sequencing with deadlines via highest-profit-first greedy slot assignment.
//!
//! Each job is a `(deadline, profit)` pair, takes one unit of time, and must
//! complete by its deadline (i.e. it occupies some integer time slot
//! `t ∈ 1..=deadline`). At most one job runs per slot. The goal is to pick a
//! subset of jobs and an assignment to slots that maximises the total profit.
//!
//! Algorithm: sort job indices by profit descending, then for each job in turn
//! place it in the *latest* still-free slot `≤ deadline`. If no such slot
//! exists, the job is dropped. Greedy correctness follows from an exchange
//! argument — given any optimal schedule, repeatedly swapping a lower-profit
//! job for a not-yet-scheduled higher-profit one (using a free slot at or
//! before its deadline) never decreases total profit, and the latest-free-slot
//! rule preserves the most flexibility for later (lower-profit) jobs.
//!
//! Complexity: `O(n²)` time and `O(n + D)` space, where `D` is the maximum
//! deadline. The inner "find latest free slot ≤ deadline" scan is the
//! quadratic factor; an `O(n α(n))` variant exists using a disjoint-set union
//! over slots, but the linear scan is kept here for clarity.
//!
//! Edge cases:
//! - Empty input returns `(0, vec![])`.
//! - Jobs with `deadline == 0` cannot occupy any slot in `1..=0` and are
//!   skipped.
//! - Jobs with `profit <= 0` are skipped — taking them is never strictly
//!   better than taking nothing in that slot, and skipping keeps the
//!   selected-index list to *strictly* improving picks.

/// Returns `(max_profit, selected_indices)` for the maximum-profit job
/// sequence, where `selected_indices` are the indices of chosen jobs into the
/// original `jobs` slice, sorted ascending.
///
/// `jobs[i] = (deadline, profit)`. Each job takes one unit of time and may be
/// scheduled in any free integer slot `t ∈ 1..=deadline`.
///
/// Empty input yields `(0, vec![])`. Jobs with `deadline == 0` or
/// `profit <= 0` are skipped (see module docs).
///
/// Time: `O(n²)`. Space: `O(n + D)` where `D` is the maximum deadline.
#[must_use]
pub fn job_sequencing(jobs: &[(usize, i64)]) -> (i64, Vec<usize>) {
    if jobs.is_empty() {
        return (0, Vec::new());
    }

    let max_deadline = jobs.iter().map(|&(d, _)| d).max().unwrap_or(0);
    if max_deadline == 0 {
        return (0, Vec::new());
    }

    // Sort job indices by profit descending. Stable sort keeps original input
    // order among ties, which makes the scheduled set deterministic.
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&i, &j| jobs[j].1.cmp(&jobs[i].1));

    // slots[t] holds the index of the job assigned to time slot t (1..=D).
    // Index 0 is unused so deadlines map directly to slot numbers.
    let mut slots: Vec<Option<usize>> = vec![None; max_deadline + 1];
    let mut total_profit: i64 = 0;

    for i in order {
        let (deadline, profit) = jobs[i];
        if deadline == 0 || profit <= 0 {
            continue;
        }
        // Find the latest free slot at or before this job's deadline.
        let upper = deadline.min(max_deadline);
        let mut t = upper;
        while t >= 1 {
            if slots[t].is_none() {
                slots[t] = Some(i);
                total_profit += profit;
                break;
            }
            t -= 1;
        }
    }

    let mut selected: Vec<usize> = slots.into_iter().flatten().collect();
    selected.sort_unstable();
    (total_profit, selected)
}

#[cfg(test)]
mod tests {
    use super::job_sequencing;
    use quickcheck_macros::quickcheck;

    /// Brute-force the maximum profit over all 2^n subsets: a subset is
    /// feasible iff there is some assignment of its jobs to distinct integer
    /// slots, each ≤ that job's deadline. By Hall's theorem (or the classical
    /// EDF result), a subset is feasible iff sorting it by deadline ascending
    /// and assigning slots `1, 2, 3, ...` keeps every assigned slot ≤ that
    /// job's deadline.
    fn brute_force_max_profit(jobs: &[(usize, i64)]) -> i64 {
        let n = jobs.len();
        let mut best: i64 = 0;
        for mask in 0_u32..(1_u32 << n) {
            let mut chosen: Vec<(usize, i64)> = Vec::new();
            for i in 0..n {
                if mask & (1 << i) != 0 {
                    chosen.push(jobs[i]);
                }
            }
            // Skip subsets containing a deadline-0 job — never feasible.
            if chosen.iter().any(|&(d, _)| d == 0) {
                continue;
            }
            chosen.sort_by_key(|&(d, _)| d);
            let feasible = chosen.iter().enumerate().all(|(idx, &(d, _))| idx < d);
            if !feasible {
                continue;
            }
            let profit: i64 = chosen.iter().map(|&(_, p)| p).sum();
            if profit > best {
                best = profit;
            }
        }
        best
    }

    /// Verify that the indices returned by `job_sequencing` correspond to a
    /// feasible schedule whose total profit matches the reported value.
    fn schedule_is_feasible(jobs: &[(usize, i64)], indices: &[usize], reported: i64) -> bool {
        // Indices must be sorted ascending and unique.
        if indices.windows(2).any(|w| w[0] >= w[1]) {
            return false;
        }
        // Profits must sum to the reported total and all be positive.
        let mut total: i64 = 0;
        for &i in indices {
            let (_, p) = jobs[i];
            if p <= 0 {
                return false;
            }
            total += p;
        }
        if total != reported {
            return false;
        }
        // Feasibility: sort by deadline ascending and check slot assignment.
        let mut chosen: Vec<(usize, i64)> = indices.iter().map(|&i| jobs[i]).collect();
        chosen.sort_by_key(|&(d, _)| d);
        chosen
            .iter()
            .enumerate()
            .all(|(idx, &(d, _))| d > 0 && idx < d)
    }

    #[test]
    fn empty_input() {
        let (profit, picks) = job_sequencing(&[]);
        assert_eq!(profit, 0);
        assert!(picks.is_empty());
    }

    #[test]
    fn single_job_positive_profit() {
        let jobs = [(1_usize, 50_i64)];
        let (profit, picks) = job_sequencing(&jobs);
        assert_eq!(profit, 50);
        assert_eq!(picks, vec![0]);
    }

    #[test]
    fn single_job_zero_profit_skipped() {
        let jobs = [(3_usize, 0_i64)];
        let (profit, picks) = job_sequencing(&jobs);
        assert_eq!(profit, 0);
        assert!(picks.is_empty());
    }

    #[test]
    fn single_job_negative_profit_skipped() {
        let jobs = [(2_usize, -10_i64)];
        let (profit, picks) = job_sequencing(&jobs);
        assert_eq!(profit, 0);
        assert!(picks.is_empty());
    }

    #[test]
    fn deadline_zero_skipped() {
        // Two jobs with deadline 0 (cannot be scheduled) and one with deadline 1.
        let jobs = [(0_usize, 100_i64), (0, 50), (1, 7)];
        let (profit, picks) = job_sequencing(&jobs);
        assert_eq!(profit, 7);
        assert_eq!(picks, vec![2]);
    }

    #[test]
    fn all_same_deadline_picks_highest_profit() {
        // Five jobs all due by slot 1: only one can run; greedy picks max.
        let jobs = [(1_usize, 10_i64), (1, 30), (1, 20), (1, 40), (1, 5)];
        let (profit, picks) = job_sequencing(&jobs);
        assert_eq!(profit, 40);
        assert_eq!(picks, vec![3]);
    }

    #[test]
    fn classic_textbook_example() {
        // Jobs: (deadline, profit). Optimum is profit 142 by scheduling the
        // jobs at indices {0, 2, 4} into slots 2, 1, 3 respectively.
        let jobs = [(2_usize, 100_i64), (1, 19), (2, 27), (1, 25), (3, 15)];
        let (profit, picks) = job_sequencing(&jobs);
        assert_eq!(profit, 142);
        assert_eq!(picks, vec![0, 2, 4]);
        assert!(schedule_is_feasible(&jobs, &picks, profit));
    }

    #[test]
    fn latest_slot_rule_keeps_low_profit_short_deadline_jobs() {
        // (deadline, profit). If a high-profit far-deadline job greedily took
        // slot 1, the deadline-1 job would be evicted. Latest-free-slot rule
        // places the far-deadline job in slot 3 instead, preserving slot 1.
        let jobs = [(3_usize, 100_i64), (1, 50), (2, 30)];
        let (profit, picks) = job_sequencing(&jobs);
        // Optimal: take all three jobs in slots 1, 2, 3.
        assert_eq!(profit, 180);
        assert_eq!(picks, vec![0, 1, 2]);
        assert!(schedule_is_feasible(&jobs, &picks, profit));
    }

    #[test]
    fn output_indices_sorted_and_unique() {
        let jobs = [(4_usize, 70_i64), (1, 80), (2, 50), (4, 60), (3, 40)];
        let (_profit, picks) = job_sequencing(&jobs);
        assert!(picks.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn brute_force_agreement_on_handpicked_inputs() {
        let cases: &[&[(usize, i64)]] = &[
            &[(2, 100), (1, 19), (2, 27), (1, 25), (3, 15)],
            &[(1, 5), (1, 5), (1, 5)],
            &[(3, 10), (3, 20), (3, 30), (3, 40)],
            &[(0, 1000), (1, 1)],
            &[(2, -7), (1, 10), (2, 20)],
        ];
        for jobs in cases {
            let (profit, picks) = job_sequencing(jobs);
            let optimum = brute_force_max_profit(jobs);
            assert_eq!(profit, optimum, "profit mismatch for {jobs:?}");
            assert!(
                schedule_is_feasible(jobs, &picks, profit),
                "infeasible schedule for {jobs:?}: picks {picks:?}"
            );
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn greedy_matches_brute_force(raw: Vec<(u8, i8)>) -> bool {
        // Cap n at 8 to keep the 2^n brute force fast. Bound deadlines to
        // 0..=4 so feasibility constraints actually bite, and let profits
        // include negatives via i8 (skipped by the algorithm).
        let jobs: Vec<(usize, i64)> = raw
            .into_iter()
            .take(8)
            .map(|(d, p)| (usize::from(d % 5), i64::from(p)))
            .collect();

        let (profit, picks) = job_sequencing(&jobs);
        let optimum = brute_force_max_profit(&jobs);

        profit == optimum && schedule_is_feasible(&jobs, &picks, profit)
    }
}
