//! Rabin–Karp substring search using a polynomial rolling hash. Average
//! O(n + m) when collisions are rare; worst case O(n · m) when every window
//! collides. Hash collisions are handled by an explicit char-by-char
//! verification so the result is always exact.

const BASE: u64 = 257;
const MODULUS: u64 = 1_000_000_007;

/// Returns all start indices at which `pattern` occurs in `text`.
///
/// Operates on chars (Unicode-aware). Empty pattern matches at every index.
pub fn rabin_karp(text: &str, pattern: &str) -> Vec<usize> {
    let text: Vec<char> = text.chars().collect();
    let pat: Vec<char> = pattern.chars().collect();
    let n = text.len();
    let m = pat.len();
    if m == 0 {
        return (0..=n).collect();
    }
    if m > n {
        return Vec::new();
    }

    let mut high_power = 1_u64;
    for _ in 0..m - 1 {
        high_power = (high_power * BASE) % MODULUS;
    }

    let mut pat_hash = 0_u64;
    let mut window_hash = 0_u64;
    for i in 0..m {
        pat_hash = (pat_hash * BASE + pat[i] as u64) % MODULUS;
        window_hash = (window_hash * BASE + text[i] as u64) % MODULUS;
    }

    let mut matches = Vec::new();
    for i in 0..=n - m {
        if window_hash == pat_hash && text[i..i + m] == pat[..] {
            matches.push(i);
        }
        if i + m < n {
            // Drop text[i], add text[i + m].
            let leading = (text[i] as u64 * high_power) % MODULUS;
            window_hash = (window_hash + MODULUS - leading) % MODULUS;
            window_hash = (window_hash * BASE + text[i + m] as u64) % MODULUS;
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::rabin_karp;

    #[test]
    fn empty_text() {
        assert_eq!(rabin_karp("", "a"), Vec::<usize>::new());
    }

    #[test]
    fn empty_pattern() {
        assert_eq!(rabin_karp("abc", ""), vec![0, 1, 2, 3]);
    }

    #[test]
    fn pattern_longer_than_text() {
        assert_eq!(rabin_karp("ab", "abc"), Vec::<usize>::new());
    }

    #[test]
    fn single_match() {
        assert_eq!(rabin_karp("hello world", "world"), vec![6]);
    }

    #[test]
    fn overlapping_matches() {
        assert_eq!(rabin_karp("aaaaa", "aaa"), vec![0, 1, 2]);
    }

    #[test]
    fn no_match() {
        assert_eq!(rabin_karp("abcdef", "xyz"), Vec::<usize>::new());
    }

    #[test]
    fn classic_example() {
        let result = rabin_karp("ABABDABACDABABCABAB", "ABABCABAB");
        assert_eq!(result, vec![10]);
    }

    #[test]
    fn many_matches() {
        let text = "aabaaabaaab";
        let pattern = "aab";
        assert_eq!(rabin_karp(text, pattern), vec![0, 4, 8]);
    }
}
