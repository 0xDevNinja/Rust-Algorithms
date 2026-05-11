//! Arithmetic expression evaluator for infix integer expressions.
//!
//! Evaluates expressions composed of:
//!
//! - non-negative decimal integer literals,
//! - the binary operators `+`, `-`, `*`, `/` (integer division, truncating
//!   toward zero, with division-by-zero reported as an error),
//! - the unary operator `-` (and `+`),
//! - parentheses `(` and `)`.
//!
//! Arithmetic is performed on `i64`. Whitespace between tokens is ignored.
//! Any malformed input (mismatched parentheses, missing operands, unknown
//! characters, etc.) returns `Err(String)` describing the problem.
//!
//! The implementation is a straightforward recursive-descent parser following
//! the grammar:
//!
//! ```text
//! expr   := term   (('+' | '-') term)*
//! term   := factor (('*' | '/') factor)*
//! factor := ('+' | '-') factor | '(' expr ')' | integer
//! ```
//!
//! # Complexity
//!
//! `O(n)` time and `O(d)` extra stack space, where `n` is the length of the
//! input and `d` is the maximum parenthesis nesting depth. Each input byte
//! is consumed at most a constant number of times.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::expression_evaluator::evaluate;
//!
//! assert_eq!(evaluate("1+2"), Ok(3));
//! assert_eq!(evaluate("2 + 3 * 4"), Ok(14));
//! assert_eq!(evaluate("(1+2)*3"), Ok(9));
//! assert_eq!(evaluate("-5+3"), Ok(-2));
//! assert_eq!(evaluate("10/3"), Ok(3));
//! assert!(evaluate("5/0").is_err());
//! ```

/// Evaluate an infix integer arithmetic expression.
///
/// Supports `+`, `-`, `*`, `/`, parentheses, integer literals, and unary
/// `+`/`-`. Whitespace is tolerated anywhere between tokens. All arithmetic
/// is performed on `i64`; division truncates toward zero.
///
/// Returns `Err(String)` describing the failure on syntax errors, division
/// by zero, or arithmetic overflow.
///
/// # Complexity
///
/// `O(n)` time, `O(d)` space, where `n = expr.len()` and `d` is the maximum
/// parenthesis nesting depth.
pub fn evaluate(expr: &str) -> Result<i64, String> {
    let bytes = expr.as_bytes();
    let mut parser = Parser { bytes, pos: 0 };

    parser.skip_whitespace();
    if parser.pos >= bytes.len() {
        return Err("empty expression".to_string());
    }

    let value = parser.parse_expr()?;
    parser.skip_whitespace();
    if parser.pos != bytes.len() {
        return Err(format!(
            "unexpected character {:?} at position {}",
            bytes[parser.pos] as char, parser.pos
        ));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        self.bytes.get(self.pos).copied()
    }

    fn parse_expr(&mut self) -> Result<i64, String> {
        let mut acc = self.parse_term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    acc = acc
                        .checked_add(rhs)
                        .ok_or_else(|| "integer overflow in addition".to_string())?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    acc = acc
                        .checked_sub(rhs)
                        .ok_or_else(|| "integer overflow in subtraction".to_string())?;
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    fn parse_term(&mut self) -> Result<i64, String> {
        let mut acc = self.parse_factor()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    acc = acc
                        .checked_mul(rhs)
                        .ok_or_else(|| "integer overflow in multiplication".to_string())?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    if rhs == 0 {
                        return Err("division by zero".to_string());
                    }
                    acc = acc
                        .checked_div(rhs)
                        .ok_or_else(|| "integer overflow in division".to_string())?;
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    fn parse_factor(&mut self) -> Result<i64, String> {
        match self.peek() {
            Some(b'+') => {
                self.pos += 1;
                self.parse_factor()
            }
            Some(b'-') => {
                self.pos += 1;
                let v = self.parse_factor()?;
                v.checked_neg()
                    .ok_or_else(|| "integer overflow in negation".to_string())
            }
            Some(b'(') => {
                self.pos += 1;
                let v = self.parse_expr()?;
                self.skip_whitespace();
                match self.bytes.get(self.pos) {
                    Some(&b')') => {
                        self.pos += 1;
                        Ok(v)
                    }
                    Some(&c) => Err(format!(
                        "expected ')' at position {}, found {:?}",
                        self.pos, c as char
                    )),
                    None => Err("expected ')' but reached end of input".to_string()),
                }
            }
            Some(c) if c.is_ascii_digit() => self.parse_integer(),
            Some(c) => Err(format!(
                "unexpected character {:?} at position {}",
                c as char, self.pos
            )),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn parse_integer(&mut self) -> Result<i64, String> {
        let start = self.pos;
        let mut value: i64 = 0;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            let digit = (self.bytes[self.pos] - b'0') as i64;
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add(digit))
                .ok_or_else(|| format!("integer literal overflows i64 at position {start}"))?;
            self.pos += 1;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_addition() {
        assert_eq!(evaluate("1+2"), Ok(3));
    }

    #[test]
    fn precedence_left_to_right() {
        assert_eq!(evaluate("2*3+4"), Ok(10));
    }

    #[test]
    fn precedence_mul_over_add() {
        assert_eq!(evaluate("2+3*4"), Ok(14));
    }

    #[test]
    fn parentheses_group() {
        assert_eq!(evaluate("(1+2)*3"), Ok(9));
    }

    #[test]
    fn unary_minus_leading() {
        assert_eq!(evaluate("-5+3"), Ok(-2));
    }

    #[test]
    fn integer_division_truncates() {
        assert_eq!(evaluate("10/3"), Ok(3));
    }

    #[test]
    fn division_by_zero_errors() {
        assert!(evaluate("5/0").is_err());
    }

    #[test]
    fn unmatched_open_paren_errors() {
        assert!(evaluate("1+(2*3").is_err());
    }

    #[test]
    fn empty_input_errors() {
        assert!(evaluate("").is_err());
    }

    #[test]
    fn whitespace_only_errors() {
        assert!(evaluate("   \t").is_err());
    }

    #[test]
    fn whitespace_tolerated_between_tokens() {
        assert_eq!(evaluate(" 1 +  2 * ( 3 + 4 ) "), Ok(15));
    }

    #[test]
    fn deeply_nested_parens() {
        assert_eq!(evaluate("(((1)))"), Ok(1));
    }

    #[test]
    fn unary_minus_after_operator() {
        assert_eq!(evaluate("3*-2"), Ok(-6));
        assert_eq!(evaluate("3+-2"), Ok(1));
    }

    #[test]
    fn unary_plus_is_identity() {
        assert_eq!(evaluate("+7"), Ok(7));
        assert_eq!(evaluate("3 + +4"), Ok(7));
    }

    #[test]
    fn double_unary_minus_is_positive() {
        assert_eq!(evaluate("--5"), Ok(5));
    }

    #[test]
    fn subtraction_is_left_associative() {
        assert_eq!(evaluate("10-3-2"), Ok(5));
    }

    #[test]
    fn division_truncates_toward_zero_for_negatives() {
        assert_eq!(evaluate("-10/3"), Ok(-3));
        assert_eq!(evaluate("10/-3"), Ok(-3));
    }

    #[test]
    fn unmatched_close_paren_errors() {
        assert!(evaluate("1+2)").is_err());
    }

    #[test]
    fn missing_operand_errors() {
        assert!(evaluate("1+").is_err());
        assert!(evaluate("*3").is_err());
    }

    #[test]
    fn unknown_character_errors() {
        assert!(evaluate("1$2").is_err());
    }

    #[test]
    fn multidigit_literals() {
        assert_eq!(evaluate("123+456"), Ok(579));
    }

    #[test]
    fn leading_zeros_in_literal() {
        assert_eq!(evaluate("007+3"), Ok(10));
    }

    #[test]
    fn complex_expression() {
        // ((2 + 3) * (4 - 1)) / 5 + -2 = 15/5 + -2 = 3 - 2 = 1
        assert_eq!(evaluate("((2+3)*(4-1))/5 + -2"), Ok(1));
    }

    #[test]
    fn empty_parens_errors() {
        assert!(evaluate("()").is_err());
    }
}
