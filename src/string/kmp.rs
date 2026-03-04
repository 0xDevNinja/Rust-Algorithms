//! Knuth–Morris–Pratt substring search. O(n + m) preprocessing + matching.

/// Returns all start indices at which `pattern` occurs in `text`.
///
/// An empty pattern matches at every index from 0 to `text.len()` (consistent
/// with most reference KMP implementations).
pub fn kmp_search(text: &str, pattern: &str) -> Vec<usize> {
    let text: Vec<char> = text.chars().collect();
    let pat: Vec<char> = pattern.chars().collect();
    if pat.is_empty() {
        return (0..=text.len()).collect();
    }
    let lps = build_lps(&pat);
    let (mut i, mut j) = (0_usize, 0_usize);
    let mut matches = Vec::new();
    while i < text.len() {
        if text[i] == pat[j] {
            i += 1;
            j += 1;
            if j == pat.len() {
                matches.push(i - j);
                j = lps[j - 1];
            }
        } else if j > 0 {
            j = lps[j - 1];
        } else {
            i += 1;
        }
    }
    matches
}

/// Builds the longest-proper-prefix-which-is-also-suffix table for `pat`.
fn build_lps(pat: &[char]) -> Vec<usize> {
    let m = pat.len();
    let mut lps = vec![0_usize; m];
    let mut len = 0_usize;
    let mut i = 1_usize;
    while i < m {
        if pat[i] == pat[len] {
            len += 1;
            lps[i] = len;
            i += 1;
        } else if len > 0 {
            len = lps[len - 1];
        } else {
            lps[i] = 0;
            i += 1;
        }
    }
    lps
}

#[cfg(test)]
mod tests {
    use super::kmp_search;

    #[test]
    fn empty_text_nonempty_pattern() {
        assert_eq!(kmp_search("", "a"), Vec::<usize>::new());
    }

    #[test]
    fn empty_pattern_matches_every_index() {
        assert_eq!(kmp_search("abc", ""), vec![0, 1, 2, 3]);
    }

    #[test]
    fn single_match() {
        assert_eq!(kmp_search("hello world", "world"), vec![6]);
    }

    #[test]
    fn no_match() {
        assert_eq!(kmp_search("abcdef", "xyz"), Vec::<usize>::new());
    }

    #[test]
    fn overlapping_matches() {
        // pattern "aaa" appears at 0, 1, 2 in "aaaaa".
        assert_eq!(kmp_search("aaaaa", "aaa"), vec![0, 1, 2]);
    }

    #[test]
    fn classic_example_with_partial_failure() {
        // The classic KMP failure-function exercise.
        let result = kmp_search("ABABDABACDABABCABAB", "ABABCABAB");
        assert_eq!(result, vec![10]);
    }

    #[test]
    fn unicode() {
        let text = "café au lait, café noir";
        assert_eq!(kmp_search(text, "café"), vec![0, 14]);
    }
}
