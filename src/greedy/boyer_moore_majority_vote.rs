//! O(n) time, O(1) space majority-element detector. Verifies the candidate with a second pass.

/// returns the majority element if found, `None` if none found
pub fn majority_vote<T: Eq + Clone>(list: &[T]) -> Option<&T> {
    let mut candidate = None;
    let mut count = 0_usize;
    for x in list {
        match count {
            0 => {
                candidate = Some(x);
                count = 1;
            }
            _ if candidate == Some(x) => count += 1,
            _ => count -= 1,
        }
    }
    // second pass: verify candidate is actually a majority
    candidate.filter(|&c| list.iter().filter(|x| *x == c).count() > list.len() / 2)
}

#[cfg(test)]
mod tests {
    use super::majority_vote;
    use quickcheck_macros::quickcheck;

    #[test]
    fn clear_majority() {
        assert_eq!(majority_vote(&[1, 1, 2, 1, 3]), Some(&1));
    }

    #[test]
    fn no_majority() {
        assert_eq!(majority_vote(&[1, 2, 1, 2, 3, 3]), None);
    }

    #[test]
    fn single_element() {
        assert_eq!(majority_vote(&[42]), Some(&42));
    }

    #[test]
    fn empty() {
        assert_eq!(majority_vote::<i32>(&[]), None);
    }

    #[allow(clippy::needless_pass_by_value, clippy::naive_bytecount)]
    #[quickcheck]
    fn result_is_majority_or_none(list: Vec<u8>) -> bool {
        majority_vote(&list).map_or_else(
            || {
                // if majority_vote(&list) returns `None`, then we confirm that there
                // is no majority by checking every possibility in 0..255
                let n = list.len();
                (0u8..=255).all(|v| list.iter().filter(|&&x| x == v).count() <= n / 2)
            },
            // otherwise, ensure that the second pass occurred and the return element is actually the majority
            |m| list.iter().filter(|x| *x == m).count() > list.len() / 2,
        )
    }
}
