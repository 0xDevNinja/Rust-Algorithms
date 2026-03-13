//! Z-algorithm. Computes the Z-array of a string in O(n) where `Z[i]` is
//! the length of the longest substring starting at `i` that matches a
//! prefix of the string. Useful as a building block for substring search,
//! periodicity, and pattern analysis.

/// Returns the Z-array of `s` (operates on chars). `Z[0]` is conventionally
/// set to `s.len()` (or 0 for the empty string).
pub fn z_array(s: &str) -> Vec<usize> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut z = vec![0_usize; n];
    if n == 0 {
        return z;
    }
    z[0] = n;
    let (mut l, mut r) = (0_usize, 0_usize);
    for i in 1..n {
        if i < r {
            z[i] = (r - i).min(z[i - l]);
        }
        while i + z[i] < n && chars[z[i]] == chars[i + z[i]] {
            z[i] += 1;
        }
        if i + z[i] > r {
            l = i;
            r = i + z[i];
        }
    }
    z
}

/// Returns all start indices at which `pattern` occurs in `text` using the
/// Z-array of `pattern + sentinel + text`.
pub fn z_search(text: &str, pattern: &str) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..=text.chars().count()).collect();
    }
    let combined = format!("{pattern}\u{0}{text}");
    let z = z_array(&combined);
    let m = pattern.chars().count();
    let mut matches = Vec::new();
    for (i, &val) in z.iter().enumerate().skip(m + 1) {
        if val >= m {
            matches.push(i - m - 1);
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::{z_array, z_search};

    #[test]
    fn empty_string() {
        assert_eq!(z_array(""), Vec::<usize>::new());
    }

    #[test]
    fn z_array_simple() {
        // s = "aabxaabxcaabxaabxay" — classic Z-array exercise.
        let z = z_array("aabxaabxcaabxaabxay");
        assert_eq!(z[0], 19);
        // Z[4] matches "aabx" (prefix length 4).
        assert_eq!(z[4], 4);
        // Z[9] matches "aabxaabx" (prefix length 8).
        assert_eq!(z[9], 8);
    }

    #[test]
    fn z_array_all_distinct() {
        let z = z_array("abcdef");
        assert_eq!(z, vec![6, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn z_array_periodic() {
        let z = z_array("aaaaa");
        assert_eq!(z, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn search_empty_pattern() {
        assert_eq!(z_search("abc", ""), vec![0, 1, 2, 3]);
    }

    #[test]
    fn search_single_match() {
        assert_eq!(z_search("hello world", "world"), vec![6]);
    }

    #[test]
    fn search_overlapping_matches() {
        assert_eq!(z_search("aaaaa", "aaa"), vec![0, 1, 2]);
    }

    #[test]
    fn search_no_match() {
        assert_eq!(z_search("abcdef", "xyz"), Vec::<usize>::new());
    }
}
