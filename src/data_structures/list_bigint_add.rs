//! Big-integer addition over singly-linked lists of decimal digits.
//!
//! Each [`DigitList`] stores a non-negative integer as a chain of nodes whose
//! `digit` field is in `0..10` and whose first node (the head) holds the
//! **least-significant** digit. Storing digits LSB-first lets [`add`]
//! propagate carries in a single forward sweep without first walking the
//! list to find its tail or aligning lengths.
//!
//! - [`from_u128`] / [`to_u128`]: `O(d)` time, `O(d)` space, where `d` is the
//!   number of decimal digits.
//! - [`add`]: `O(max(da, db))` time, `O(max(da, db))` space — a single
//!   iterative pass over both lists with a `0/1` carry. The result is
//!   stripped of leading zeros (the value zero keeps a single `0` digit).
//!
//! [`from_u128`]: DigitList::from_u128
//! [`to_u128`]: DigitList::to_u128
//! [`add`]: DigitList::add
//!
//! No `unsafe` is used: lists are owned `Box<Node>` chains and built by
//! pushing onto the front of the result while iterating, which keeps the
//! natural LSB-first order.
//!
//! # Example
//! ```
//! use rust_algorithms::data_structures::list_bigint_add::DigitList;
//!
//! let a = DigitList::from_u128(99);
//! let b = DigitList::from_u128(1);
//! assert_eq!(DigitList::add(&a, &b).to_u128(), 100);
//! ```

/// One decimal digit in the linked list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Decimal digit in `0..10`.
    pub digit: u8,
    /// Pointer to the next-more-significant digit, or `None` at the MSB end.
    pub next: Option<Box<Self>>,
}

/// A non-negative integer represented as an LSB-first linked list of decimal
/// digits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DigitList {
    /// Head holds the least-significant digit; `None` represents the value
    /// zero alongside the canonical single-zero list.
    pub head: Option<Box<Node>>,
}

impl DigitList {
    /// Builds an empty list (representing the value `0`).
    ///
    /// Time: `O(1)`. Space: `O(1)`.
    pub const fn new() -> Self {
        Self { head: None }
    }

    /// Builds a `DigitList` from a `u128`. The value `0` produces a list with
    /// a single zero digit so that round-tripping `0` through
    /// [`DigitList::to_u128`] preserves the canonical "0" form.
    ///
    /// Time: `O(d)`. Space: `O(d)`, where `d` is the number of decimal digits
    /// of `n` (or `1` when `n == 0`).
    pub fn from_u128(mut n: u128) -> Self {
        if n == 0 {
            return Self {
                head: Some(Box::new(Node {
                    digit: 0,
                    next: None,
                })),
            };
        }
        let mut head: Option<Box<Node>> = None;
        let mut tail: &mut Option<Box<Node>> = &mut head;
        while n > 0 {
            let d = (n % 10) as u8;
            n /= 10;
            *tail = Some(Box::new(Node {
                digit: d,
                next: None,
            }));
            tail = &mut tail.as_mut().unwrap().next;
        }
        Self { head }
    }

    /// Decodes the list back to a `u128`. Panics if the represented value
    /// does not fit in `u128`.
    ///
    /// Time: `O(d)`. Space: `O(1)`.
    pub fn to_u128(&self) -> u128 {
        let mut value: u128 = 0;
        let mut place: u128 = 1;
        let mut cur = self.head.as_deref();
        while let Some(node) = cur {
            value = value
                .checked_add((node.digit as u128) * place)
                .expect("DigitList value overflows u128");
            cur = node.next.as_deref();
            if cur.is_some() {
                place = place
                    .checked_mul(10)
                    .expect("DigitList value overflows u128");
            }
        }
        value
    }

    /// Adds two `DigitList`s and returns a new list, also LSB-first, with no
    /// leading zeros (except the canonical single zero for `0 + 0`).
    ///
    /// Walks both inputs in lockstep, propagating a `0/1` carry; when one
    /// list is exhausted, continues with the remaining digits of the other.
    /// A final carry produces a leading `1` digit.
    ///
    /// Time: `O(max(da, db))`. Space: `O(max(da, db) + 1)`.
    pub fn add(a: &Self, b: &Self) -> Self {
        let mut head: Option<Box<Node>> = None;
        let mut tail: &mut Option<Box<Node>> = &mut head;
        let mut pa = a.head.as_deref();
        let mut pb = b.head.as_deref();
        let mut carry: u8 = 0;

        while pa.is_some() || pb.is_some() || carry != 0 {
            let da = pa.map_or(0u8, |n| n.digit);
            let db = pb.map_or(0u8, |n| n.digit);
            let sum = da + db + carry;
            let digit = sum % 10;
            carry = sum / 10;

            *tail = Some(Box::new(Node { digit, next: None }));
            tail = &mut tail.as_mut().unwrap().next;

            if let Some(n) = pa {
                pa = n.next.as_deref();
            }
            if let Some(n) = pb {
                pb = n.next.as_deref();
            }
        }

        let mut out = Self { head };
        out.strip_leading_zeros();
        out
    }

    /// Strips leading zeros from the MSB end while keeping a single zero for
    /// the value `0`.
    fn strip_leading_zeros(&mut self) {
        // Find the index of the last non-zero digit.
        let mut last_nonzero: Option<usize> = None;
        let mut cur = self.head.as_deref();
        let mut idx = 0usize;
        while let Some(node) = cur {
            if node.digit != 0 {
                last_nonzero = Some(idx);
            }
            cur = node.next.as_deref();
            idx += 1;
        }

        match last_nonzero {
            None => {
                // All zeros (or empty): keep canonical single zero.
                self.head = Some(Box::new(Node {
                    digit: 0,
                    next: None,
                }));
            }
            Some(keep_through) => {
                // Truncate the chain after position `keep_through`.
                let mut cur = self.head.as_deref_mut();
                let mut i = 0usize;
                while let Some(node) = cur {
                    if i == keep_through {
                        node.next = None;
                        return;
                    }
                    cur = node.next.as_deref_mut();
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DigitList;
    use quickcheck_macros::quickcheck;

    #[test]
    fn round_trip_zero() {
        let z = DigitList::from_u128(0);
        // canonical single-zero digit
        assert!(z.head.is_some());
        assert_eq!(z.head.as_ref().unwrap().digit, 0);
        assert!(z.head.as_ref().unwrap().next.is_none());
        assert_eq!(z.to_u128(), 0);
    }

    #[test]
    fn round_trip_small() {
        for v in [1u128, 9, 10, 11, 100, 12345, 99999] {
            assert_eq!(DigitList::from_u128(v).to_u128(), v);
        }
    }

    #[test]
    fn round_trip_large() {
        let v: u128 = u64::MAX as u128 + 12345;
        assert_eq!(DigitList::from_u128(v).to_u128(), v);
    }

    #[test]
    fn add_zero_zero() {
        let a = DigitList::from_u128(0);
        let b = DigitList::from_u128(0);
        let s = DigitList::add(&a, &b);
        assert_eq!(s.to_u128(), 0);
        // canonical single-zero output
        assert_eq!(s.head.as_ref().unwrap().digit, 0);
        assert!(s.head.as_ref().unwrap().next.is_none());
    }

    #[test]
    fn add_zero_x() {
        let x = DigitList::from_u128(123_456_789);
        let z = DigitList::from_u128(0);
        assert_eq!(DigitList::add(&x, &z).to_u128(), 123_456_789);
        assert_eq!(DigitList::add(&z, &x).to_u128(), 123_456_789);
    }

    #[test]
    fn add_seven_eight() {
        let a = DigitList::from_u128(7);
        let b = DigitList::from_u128(8);
        let s = DigitList::add(&a, &b);
        assert_eq!(s.to_u128(), 15);
        // Two digits: 5 then 1.
        let h = s.head.as_ref().unwrap();
        assert_eq!(h.digit, 5);
        let n = h.next.as_ref().unwrap();
        assert_eq!(n.digit, 1);
        assert!(n.next.is_none());
    }

    #[test]
    fn add_carry_chain() {
        let a = DigitList::from_u128(99);
        let b = DigitList::from_u128(1);
        let s = DigitList::add(&a, &b);
        assert_eq!(s.to_u128(), 100);
        // Digits LSB-first: 0, 0, 1.
        let d0 = s.head.as_ref().unwrap();
        let d1 = d0.next.as_ref().unwrap();
        let d2 = d1.next.as_ref().unwrap();
        assert_eq!((d0.digit, d1.digit, d2.digit), (0, 0, 1));
        assert!(d2.next.is_none());
    }

    #[test]
    fn add_uneven_lengths() {
        let a = DigitList::from_u128(9_999);
        let b = DigitList::from_u128(1);
        assert_eq!(DigitList::add(&a, &b).to_u128(), 10_000);

        let a = DigitList::from_u128(1);
        let b = DigitList::from_u128(9_999_999_999);
        assert_eq!(DigitList::add(&a, &b).to_u128(), 10_000_000_000);
    }

    #[test]
    fn add_large_within_u128() {
        let a: u128 = u64::MAX as u128;
        let b: u128 = u64::MAX as u128;
        let la = DigitList::from_u128(a);
        let lb = DigitList::from_u128(b);
        assert_eq!(DigitList::add(&la, &lb).to_u128(), a + b);
    }

    #[test]
    fn no_leading_zeros_in_result() {
        // Sum that ends in trailing-MSB zeros internally still must not
        // expose leading zeros (it never would, by construction, but assert
        // the invariant explicitly).
        let s = DigitList::add(&DigitList::from_u128(50), &DigitList::from_u128(50));
        assert_eq!(s.to_u128(), 100);
        // last digit (MSB) must be non-zero
        let mut cur = s.head.as_deref();
        let mut last = 0u8;
        while let Some(n) = cur {
            last = n.digit;
            cur = n.next.as_deref();
        }
        assert_ne!(last, 0);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn add_matches_native(a: u64, b: u64) -> bool {
        let expected = a as u128 + b as u128;
        let got = DigitList::add(
            &DigitList::from_u128(a as u128),
            &DigitList::from_u128(b as u128),
        )
        .to_u128();
        got == expected
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn round_trip_u64(a: u64) -> bool {
        DigitList::from_u128(a as u128).to_u128() == a as u128
    }
}
