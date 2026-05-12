//! Reverse Polish Notation (RPN) evaluator for integer expressions.
//!
//! Evaluates a slice of tokens written in postfix (RPN) form, where each
//! token is either a decimal integer literal (optionally signed with a
//! leading `-` or `+`) or one of the binary operators `+`, `-`, `*`, `/`.
//!
//! The algorithm is the classical stack-based evaluator: scan tokens left
//! to right, push numeric tokens onto a stack, and on each operator pop
//! the top two operands, apply the operation, and push the result. After
//! consuming all tokens the stack must contain exactly one value, which is
//! the result.
//!
//! Arithmetic is performed on `i64`. Integer division truncates toward
//! zero (matching Rust's `/` operator on `i64`). Division by zero,
//! malformed expressions (operator without enough operands, leftover
//! operands), unrecognised tokens, empty input, and arithmetic overflow
//! all return `Err(String)` describing the failure.
//!
//! # Complexity
//!
//! `O(n)` time and `O(n)` space, where `n` is the number of tokens. Each
//! token is processed in constant time and the stack holds at most `n`
//! values.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::rpn_evaluator::evaluate_rpn;
//!
//! assert_eq!(evaluate_rpn(&["2", "1", "+", "3", "*"]), Ok(9));
//! assert_eq!(evaluate_rpn(&["4", "13", "5", "/", "+"]), Ok(6));
//! assert_eq!(evaluate_rpn(&["42"]), Ok(42));
//! assert!(evaluate_rpn(&[]).is_err());
//! ```

/// Evaluate an expression in Reverse Polish Notation.
///
/// `tokens` is a sequence of whitespace-free tokens. Each token is either
/// an integer literal parseable as `i64` or one of the operators `+`,
/// `-`, `*`, `/`. Division truncates toward zero.
///
/// Returns `Err(String)` on empty input, unknown tokens, missing
/// operands, leftover operands, division by zero, or arithmetic overflow.
///
/// # Complexity
///
/// `O(n)` time and `O(n)` extra space in the number of tokens.
pub fn evaluate_rpn(tokens: &[&str]) -> Result<i64, String> {
    if tokens.is_empty() {
        return Err("empty token list".to_string());
    }

    let mut stack: Vec<i64> = Vec::with_capacity(tokens.len());

    for &token in tokens {
        match token {
            "+" | "-" | "*" | "/" => {
                let rhs = stack
                    .pop()
                    .ok_or_else(|| format!("operator {token:?} missing right operand"))?;
                let lhs = stack
                    .pop()
                    .ok_or_else(|| format!("operator {token:?} missing left operand"))?;
                let result = match token {
                    "+" => lhs
                        .checked_add(rhs)
                        .ok_or_else(|| "integer overflow in addition".to_string())?,
                    "-" => lhs
                        .checked_sub(rhs)
                        .ok_or_else(|| "integer overflow in subtraction".to_string())?,
                    "*" => lhs
                        .checked_mul(rhs)
                        .ok_or_else(|| "integer overflow in multiplication".to_string())?,
                    "/" => {
                        if rhs == 0 {
                            return Err("division by zero".to_string());
                        }
                        lhs.checked_div(rhs)
                            .ok_or_else(|| "integer overflow in division".to_string())?
                    }
                    _ => unreachable!(),
                };
                stack.push(result);
            }
            _ => {
                let value: i64 = token
                    .parse()
                    .map_err(|_| format!("invalid token {token:?}"))?;
                stack.push(value);
            }
        }
    }

    if stack.len() != 1 {
        return Err(format!(
            "malformed expression: {} values left on stack",
            stack.len()
        ));
    }
    Ok(stack[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_addition_then_multiplication() {
        // (2 + 1) * 3 = 9
        assert_eq!(evaluate_rpn(&["2", "1", "+", "3", "*"]), Ok(9));
    }

    #[test]
    fn division_truncates_then_addition() {
        // 4 + (13 / 5) = 4 + 2 = 6
        assert_eq!(evaluate_rpn(&["4", "13", "5", "/", "+"]), Ok(6));
    }

    #[test]
    fn single_number() {
        assert_eq!(evaluate_rpn(&["42"]), Ok(42));
    }

    #[test]
    fn single_negative_number() {
        assert_eq!(evaluate_rpn(&["-7"]), Ok(-7));
    }

    #[test]
    fn negative_operand_addition() {
        // -1 + 2 = 1
        assert_eq!(evaluate_rpn(&["-1", "2", "+"]), Ok(1));
    }

    #[test]
    fn empty_input_errors() {
        assert!(evaluate_rpn(&[]).is_err());
    }

    #[test]
    fn division_by_zero_errors() {
        assert!(evaluate_rpn(&["1", "0", "/"]).is_err());
    }

    #[test]
    fn operator_without_operands_errors() {
        assert!(evaluate_rpn(&["+"]).is_err());
    }

    #[test]
    fn operator_with_one_operand_errors() {
        assert!(evaluate_rpn(&["1", "+"]).is_err());
    }

    #[test]
    fn leftover_operands_errors() {
        // Two values, no operator -> malformed.
        assert!(evaluate_rpn(&["1", "2"]).is_err());
    }

    #[test]
    fn unknown_token_errors() {
        assert!(evaluate_rpn(&["1", "2", "&"]).is_err());
    }

    #[test]
    fn subtraction_order_is_lhs_minus_rhs() {
        // 10 3 - = 10 - 3 = 7
        assert_eq!(evaluate_rpn(&["10", "3", "-"]), Ok(7));
    }

    #[test]
    fn division_truncates_toward_zero_for_negatives() {
        // -7 / 2 = -3 (truncation toward zero)
        assert_eq!(evaluate_rpn(&["-7", "2", "/"]), Ok(-3));
        assert_eq!(evaluate_rpn(&["7", "-2", "/"]), Ok(-3));
    }

    #[test]
    fn longer_expression() {
        // ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
        // = ((10 * (6 / (12 * -11))) + 17) + 5
        // = ((10 * (6 / -132)) + 17) + 5
        // = ((10 * 0) + 17) + 5
        // = 22
        let tokens = [
            "10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+",
        ];
        assert_eq!(evaluate_rpn(&tokens), Ok(22));
    }

    #[test]
    fn overflow_errors() {
        let max = i64::MAX.to_string();
        let tokens = [max.as_str(), "1", "+"];
        assert!(evaluate_rpn(&tokens).is_err());
    }
}
