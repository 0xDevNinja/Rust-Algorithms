//! Spiral matrix traversal.
//!
//! Walks a rectangular `m x n` matrix in clockwise spiral order, starting at
//! the top-left corner and shrinking the active boundary on each pass.
//!
//! Complexity: `O(m * n)` time, `O(m * n)` output space.

/// Returns the elements of `matrix` in clockwise spiral order starting from
/// the top-left corner.
///
/// An empty outer slice yields an empty `Vec`. Any non-empty matrix must be
/// rectangular: every row must share the same non-zero length.
///
/// # Panics
///
/// Panics if `matrix` is ragged (rows of differing lengths) or if any row is
/// empty.
pub fn spiral_order<T: Clone>(matrix: &[Vec<T>]) -> Vec<T> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let cols = matrix[0].len();
    assert!(cols > 0, "spiral_order: rows must be non-empty");
    for row in matrix {
        assert!(
            row.len() == cols,
            "spiral_order: matrix must be rectangular"
        );
    }

    let rows = matrix.len();
    let mut out = Vec::with_capacity(rows * cols);

    let mut top = 0usize;
    let mut bottom = rows - 1;
    let mut left = 0usize;
    let mut right = cols - 1;

    loop {
        // Top row, left -> right.
        for c in left..=right {
            out.push(matrix[top][c].clone());
        }
        if top == bottom {
            break;
        }
        top += 1;

        // Right column, top -> bottom.
        for r in top..=bottom {
            out.push(matrix[r][right].clone());
        }
        if left == right {
            break;
        }
        right -= 1;

        // Bottom row, right -> left.
        for c in (left..=right).rev() {
            out.push(matrix[bottom][c].clone());
        }
        if top > bottom {
            break;
        }
        bottom -= 1;

        // Left column, bottom -> top.
        for r in (top..=bottom).rev() {
            out.push(matrix[r][left].clone());
        }
        if left >= right {
            break;
        }
        left += 1;

        if top > bottom || left > right {
            break;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::spiral_order;

    #[test]
    fn empty_matrix() {
        let m: Vec<Vec<i32>> = Vec::new();
        assert_eq!(spiral_order(&m), Vec::<i32>::new());
    }

    #[test]
    fn single_row() {
        let m = vec![vec![1, 2, 3, 4]];
        assert_eq!(spiral_order(&m), vec![1, 2, 3, 4]);
    }

    #[test]
    fn single_column() {
        let m = vec![vec![1], vec![2], vec![3], vec![4]];
        assert_eq!(spiral_order(&m), vec![1, 2, 3, 4]);
    }

    #[test]
    fn one_by_one() {
        let m = vec![vec![42]];
        assert_eq!(spiral_order(&m), vec![42]);
    }

    #[test]
    fn three_by_three() {
        let m = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        assert_eq!(spiral_order(&m), vec![1, 2, 3, 6, 9, 8, 7, 4, 5]);
    }

    #[test]
    fn two_by_three() {
        let m = vec![vec![1, 2, 3], vec![4, 5, 6]];
        assert_eq!(spiral_order(&m), vec![1, 2, 3, 6, 5, 4]);
    }

    #[test]
    fn three_by_four() {
        let m = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];
        assert_eq!(
            spiral_order(&m),
            vec![1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
        );
    }

    #[test]
    fn four_by_three() {
        let m = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];
        assert_eq!(
            spiral_order(&m),
            vec![1, 2, 3, 6, 9, 12, 11, 10, 7, 4, 5, 8]
        );
    }

    #[test]
    fn clones_non_copy() {
        let m = vec![
            vec![String::from("a"), String::from("b")],
            vec![String::from("c"), String::from("d")],
        ];
        assert_eq!(
            spiral_order(&m),
            vec![
                String::from("a"),
                String::from("b"),
                String::from("d"),
                String::from("c"),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "rectangular")]
    fn panics_on_ragged() {
        let m = vec![vec![1, 2, 3], vec![4, 5]];
        let _ = spiral_order(&m);
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn panics_on_empty_row() {
        let m: Vec<Vec<i32>> = vec![vec![]];
        let _ = spiral_order(&m);
    }
}
