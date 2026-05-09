//! Bit-vector sort, the opening trick from Jon Bentley's _Programming Pearls_
//! Column 1.
//!
//! Given a bounded set of distinct non-negative integers, allocate a bitmap
//! large enough to cover the universe, mark each input value's bit, then walk
//! the bitmap in order to emit the sorted output. The technique trades space
//! for time when the universe is small relative to `usize::MAX` and the inputs
//! are guaranteed unique.
//!
//! Complexity: `O(universe + n)` time, `O(universe / 64)` space.

const BITS_PER_WORD: u32 = u64::BITS;

/// Sorts a slice of distinct `u32` values drawn from `[0, universe)`.
///
/// The input domain must be known up front via `universe`. Every value must be
/// strictly less than `universe`, and no value may appear twice.
///
/// # Panics
///
/// Panics if any element is `>= universe`, or if duplicate values are present.
#[must_use]
pub fn bit_vector_sort(values: &[u32], universe: u32) -> Vec<u32> {
    if universe == 0 {
        assert!(
            values.is_empty(),
            "bit_vector_sort: empty universe cannot contain any values",
        );
        return Vec::new();
    }

    let words = (universe / BITS_PER_WORD) as usize + 1;
    let mut bitmap = vec![0_u64; words];

    for &value in values {
        assert!(
            value < universe,
            "bit_vector_sort: value {value} is outside the universe [0, {universe})",
        );
        let word = (value / BITS_PER_WORD) as usize;
        let bit = value % BITS_PER_WORD;
        let mask = 1_u64 << bit;
        assert!(
            bitmap[word] & mask == 0,
            "bit_vector_sort: duplicate value {value}",
        );
        bitmap[word] |= mask;
    }

    let mut sorted = Vec::with_capacity(values.len());
    for (word_idx, &word) in bitmap.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros();
            let value = (word_idx as u32) * BITS_PER_WORD + bit;
            if value < universe {
                sorted.push(value);
            }
            remaining &= remaining - 1;
        }
    }
    sorted
}

#[cfg(test)]
mod tests {
    use super::bit_vector_sort;
    use quickcheck_macros::quickcheck;
    use std::collections::HashSet;

    #[test]
    fn empty_input() {
        let sorted = bit_vector_sort(&[], 100);
        assert!(sorted.is_empty());
    }

    #[test]
    fn empty_universe_empty_input() {
        let sorted = bit_vector_sort(&[], 0);
        assert!(sorted.is_empty());
    }

    #[test]
    fn single_value() {
        assert_eq!(bit_vector_sort(&[42], 100), vec![42]);
    }

    #[test]
    fn classic_seven_numbers() {
        // Bentley's Column 1 walks through sorting a handful of distinct ints
        // from a bounded domain; this is a representative seven-number case.
        let input = [63, 7, 14, 0, 99, 27, 41];
        let sorted = bit_vector_sort(&input, 100);
        assert_eq!(sorted, vec![0, 7, 14, 27, 41, 63, 99]);
    }

    #[test]
    fn full_domain_identity() {
        let input: Vec<u32> = (0..256).collect();
        let mut shuffled = input.clone();
        shuffled.reverse();
        let sorted = bit_vector_sort(&shuffled, 256);
        assert_eq!(sorted, input);
    }

    #[test]
    fn boundary_values_kept() {
        // Includes 0 and universe - 1 to exercise both endpoints.
        let input = [0, 63, 64, 127, 128];
        let sorted = bit_vector_sort(&input, 129);
        assert_eq!(sorted, vec![0, 63, 64, 127, 128]);
    }

    #[test]
    #[should_panic(expected = "duplicate value")]
    fn duplicate_panics() {
        let _ = bit_vector_sort(&[3, 5, 3], 10);
    }

    #[test]
    #[should_panic(expected = "outside the universe")]
    fn out_of_range_panics() {
        let _ = bit_vector_sort(&[3, 5, 10], 10);
    }

    #[test]
    #[should_panic(expected = "outside the universe")]
    fn way_out_of_range_panics() {
        let _ = bit_vector_sort(&[1_000_000], 100);
    }

    #[quickcheck]
    fn matches_std_sort_on_distinct_inputs(raw: Vec<u32>) -> bool {
        const UNIVERSE: u32 = 1024;
        let mut seen = HashSet::new();
        let input: Vec<u32> = raw
            .into_iter()
            .map(|x| x % UNIVERSE)
            .filter(|x| seen.insert(*x))
            .collect();
        let mut expected = input.clone();
        expected.sort_unstable();
        let actual = bit_vector_sort(&input, UNIVERSE);
        actual == expected
    }
}
