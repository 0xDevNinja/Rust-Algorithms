//! In-place 90 degree clockwise rotation of an N x N matrix.
//!
//! The rotation is performed via the classical *transpose then reverse rows*
//! technique:
//!
//! 1. Transpose the matrix in place by swapping `m[i][j]` with `m[j][i]` for
//!    every `i < j`.
//! 2. Reverse each row.
//!
//! The composition of these two operations maps the element originally at
//! `(i, j)` to position `(j, n - 1 - i)`, which is exactly a 90 degree
//! clockwise rotation.
//!
//! Runs in O(n^2) time and O(1) extra space (no auxiliary matrix is
//! allocated). The element type only needs to be `Clone` to satisfy
//! [`Vec::reverse`] / [`<[T]>::swap`] requirements through safe APIs.

/// Rotates a square `n x n` matrix 90 degrees clockwise in place.
///
/// Uses the transpose-then-reverse-rows technique. Runs in O(n^2) time and
/// O(1) extra space.
///
/// # Panics
///
/// Panics if `matrix` is not square, i.e. if any row's length differs from
/// the number of rows.
#[allow(clippy::ptr_arg)]
pub fn rotate_90_cw<T: Clone>(matrix: &mut Vec<Vec<T>>) {
    let n = matrix.len();
    for (i, row) in matrix.iter().enumerate() {
        assert!(
            row.len() == n,
            "rotate_90_cw requires a square matrix: row {} has length {} but expected {}",
            i,
            row.len(),
            n
        );
    }
    if n < 2 {
        return;
    }

    // Step 1: transpose in place — swap across the main diagonal.
    for i in 0..n {
        for j in (i + 1)..n {
            // Take the two rows as disjoint mutable slices, then swap the
            // single elements safely without `unsafe`.
            let (top, bottom) = matrix.split_at_mut(j);
            std::mem::swap(&mut top[i][j], &mut bottom[0][i]);
        }
    }

    // Step 2: reverse each row.
    for row in matrix.iter_mut() {
        row.reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::rotate_90_cw;

    #[test]
    fn rotate_1x1_is_noop() {
        let mut m = vec![vec![42]];
        rotate_90_cw(&mut m);
        assert_eq!(m, vec![vec![42]]);
    }

    #[test]
    fn rotate_2x2() {
        // 1 2      3 1
        // 3 4  ->  4 2
        let mut m = vec![vec![1, 2], vec![3, 4]];
        rotate_90_cw(&mut m);
        assert_eq!(m, vec![vec![3, 1], vec![4, 2]]);
    }

    #[test]
    fn rotate_3x3_hand_example() {
        // 1 2 3      7 4 1
        // 4 5 6  ->  8 5 2
        // 7 8 9      9 6 3
        let mut m = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        rotate_90_cw(&mut m);
        assert_eq!(m, vec![vec![7, 4, 1], vec![8, 5, 2], vec![9, 6, 3]]);
    }

    #[test]
    fn rotate_4x4() {
        //  1  2  3  4        13  9  5  1
        //  5  6  7  8   ->   14 10  6  2
        //  9 10 11 12        15 11  7  3
        // 13 14 15 16        16 12  8  4
        let mut m = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        rotate_90_cw(&mut m);
        assert_eq!(
            m,
            vec![
                vec![13, 9, 5, 1],
                vec![14, 10, 6, 2],
                vec![15, 11, 7, 3],
                vec![16, 12, 8, 4],
            ]
        );
    }

    #[test]
    fn double_rotation_is_180() {
        // Rotating twice should equal a 180-degree rotation, which is the
        // matrix with both rows and columns reversed.
        let original = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        let mut got = original.clone();
        rotate_90_cw(&mut got);
        rotate_90_cw(&mut got);

        let mut expected: Vec<Vec<i32>> = original.iter().rev().cloned().collect();
        for row in &mut expected {
            row.reverse();
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn four_rotations_is_identity() {
        let original = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25],
        ];
        let mut m = original.clone();
        for _ in 0..4 {
            rotate_90_cw(&mut m);
        }
        assert_eq!(m, original);
    }

    #[test]
    #[should_panic(expected = "requires a square matrix")]
    fn panics_on_non_square() {
        let mut m = vec![vec![1, 2, 3], vec![4, 5, 6]];
        rotate_90_cw(&mut m);
    }
}
