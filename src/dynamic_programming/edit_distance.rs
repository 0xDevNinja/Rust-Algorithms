//! Levenshtein edit distance (insertion, deletion, substitution all cost 1).
//! O(n · m) time, O(min(n, m)) space.

/// Returns the minimum number of edits to transform `a` into `b`.
pub fn edit_distance<T: Eq>(a: &[T], b: &[T]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let (short, long) = if n < m { (a, b) } else { (b, a) };
    let s = short.len();
    let l = long.len();
    let mut prev: Vec<usize> = (0..=s).collect();
    let mut curr = vec![0_usize; s + 1];
    for i in 1..=l {
        curr[0] = i;
        for j in 1..=s {
            curr[j] = if long[i - 1] == short[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[s]
}

#[cfg(test)]
mod tests {
    use super::edit_distance;

    fn s(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn classic() {
        assert_eq!(edit_distance(&s("kitten"), &s("sitting")), 3);
        assert_eq!(edit_distance(&s("flaw"), &s("lawn")), 2);
    }

    #[test]
    fn identical() {
        assert_eq!(edit_distance(&s("abc"), &s("abc")), 0);
    }

    #[test]
    fn one_empty() {
        assert_eq!(edit_distance::<char>(&[], &s("hello")), 5);
        assert_eq!(edit_distance(&s("hello"), &[]), 5);
    }

    #[test]
    fn both_empty() {
        let a: Vec<char> = vec![];
        let b: Vec<char> = vec![];
        assert_eq!(edit_distance(&a, &b), 0);
    }
}
