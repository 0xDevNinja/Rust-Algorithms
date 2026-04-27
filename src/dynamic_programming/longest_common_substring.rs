//! Longest common substring between two slices. O(n · m) time, O(n · m) space.

/// Returns the longest common substring as a `Vec<T>`.
pub fn longest_common_substring<T: Clone + Eq>(a: &[T], b: &[T]) -> Vec<T> {
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0_usize; m + 1]; n + 1];
    let mut best_len = 0;
    let mut best_end_a = 0;
    for i in 0..n {
        for j in 0..m {
            dp[i + 1][j + 1] = if a[i] == b[j] {
                dp[i][j] + 1
            } else {
                0 // no match -> reset the counter
            };
            if dp[i + 1][j + 1] > best_len {
                best_len = dp[i + 1][j + 1];
                best_end_a = i + 1;
            }
        }
    }
    a[best_end_a - best_len..best_end_a].to_vec()
}

/// Returns just the length of the longest common substring (lighter alternative).
pub fn longest_common_substring_length<T: Eq>(a: &[T], b: &[T]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev = vec![0_usize; m + 1];
    let mut curr = vec![0_usize; m + 1];
    let mut best = 0;
    for i in 0..n {
        for j in 0..m {
            curr[j + 1] = if a[i] == b[j] { prev[j] + 1 } else { 0 };
            best = best.max(curr[j + 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{longest_common_substring_length, longest_common_substring};
    use quickcheck_macros::quickcheck;

    fn brute_force(a: &[u8], b: &[u8]) -> usize {
        let mut best = 0;
        for start in 0..a.len() {
            for end in (start + 1)..=a.len() {
                let sub = &a[start..end];
                if b.windows(sub.len()).any(|w| w == sub) {
                    best = best.max(sub.len());
                }
            }
        }
        best
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_brute_force(a: Vec<u8>, b: Vec<u8>) -> bool {
        let bf_len = brute_force(&a, &b);
        longest_common_substring_length(&a, &b) == bf_len && longest_common_substring(&a, &b).len() == bf_len
    }

    #[test]
    fn classic() {
        let a: Vec<char> = "ABCBDAB".chars().collect();
        let b: Vec<char> = "DCABCBDD".chars().collect();
        let lcs = longest_common_substring(&a, &b);
        assert_eq!(lcs.len(), 5);
        let s: String = lcs.into_iter().collect();
        assert_eq!(s, "ABCBD");
    }

    #[test]
    fn no_overlap() {
        let a: Vec<char> = "abc".chars().collect();
        let b: Vec<char> = "xyz".chars().collect();
        assert_eq!(longest_common_substring_length(&a, &b), 0);
        assert!(longest_common_substring(&a, &b).is_empty());
    }

    #[test]
    fn identical() {
        let a: Vec<i32> = vec![1, 2, 3];
        assert_eq!(longest_common_substring(&a, &a), a);
    }

    #[test]
    fn empty() {
        let a: Vec<i32> = vec![];
        let b: Vec<i32> = vec![1, 2];
        assert_eq!(longest_common_substring_length(&a, &b), 0);
        assert_eq!(longest_common_substring(&a, &b).len(), 0);
    }

    #[test]
    fn single_element() {
        let a: Vec<i32> = vec![1];
        let b: Vec<i32> = vec![2];
        assert_eq!(longest_common_substring(&a, &a), vec![1]);
        assert_eq!(longest_common_substring(&a, &b), Vec::<i32>::new());
    }
}
