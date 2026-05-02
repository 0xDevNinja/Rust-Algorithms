//! Zeller's congruence — compute the day of the week for a Gregorian date.
//!
//! Uses the Gregorian variant of Zeller's formula. Supported range is
//! `[1583-01-01, 9999-12-31]`: the Gregorian calendar took effect on
//! 1582-10-15, so 1583-01-01 is the first full Gregorian year.
//!
//! # Algorithm
//! Treat January and February as months 13 and 14 of the previous year. With
//! `q = day`, `m = adjusted month`, `K = year mod 100`, `J = year / 100`:
//!
//! ```text
//! h = (q + 13(m + 1)/5 + K + K/4 + J/4 - 2*J) mod 7
//! ```
//!
//! Zeller maps `h = 0` to Saturday, `1` to Sunday, ..., `6` to Friday.
//!
//! # Complexity
//! Time and space O(1) — pure integer arithmetic, no allocation.

/// Days of the week, ordered with Sunday first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DayOfWeek {
    Sunday,
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
}

impl DayOfWeek {
    /// Returns the English name of the weekday.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sunday => "Sunday",
            Self::Monday => "Monday",
            Self::Tuesday => "Tuesday",
            Self::Wednesday => "Wednesday",
            Self::Thursday => "Thursday",
            Self::Friday => "Friday",
            Self::Saturday => "Saturday",
        }
    }
}

/// Returns `true` iff `year` is a Gregorian leap year.
const fn is_gregorian_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Returns the number of days in `month` of `year` (Gregorian).
/// Returns `None` if `month` is not in `1..=12`.
const fn days_in_month(year: i32, month: u32) -> Option<u32> {
    let d = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_gregorian_leap(year) {
                29
            } else {
                28
            }
        }
        _ => return None,
    };
    Some(d)
}

/// Returns the day of the week for the Gregorian date `year-month-day`, or
/// `None` if the date is outside the supported range or not a real calendar
/// date (invalid month, day out of range for the month, e.g. Feb 29 of a
/// non-leap year).
///
/// Supported range: `1583 <= year <= 9999`, `1 <= month <= 12`,
/// `1 <= day <= days_in_month(year, month)`.
///
/// # Examples
/// ```
/// use rust_algorithms::math::zellers_congruence::{day_of_week, DayOfWeek};
/// assert_eq!(day_of_week(1776, 7, 4), Some(DayOfWeek::Thursday));
/// assert_eq!(day_of_week(2000, 1, 1), Some(DayOfWeek::Saturday));
/// ```
pub const fn day_of_week(year: i32, month: u32, day: u32) -> Option<DayOfWeek> {
    if year < 1583 || year > 9999 {
        return None;
    }
    if month < 1 || month > 12 {
        return None;
    }
    let Some(max_day) = days_in_month(year, month) else {
        return None;
    };
    if day < 1 || day > max_day {
        return None;
    }

    // Shift Jan/Feb to months 13/14 of the previous year.
    let (m, y) = if month < 3 {
        (month as i32 + 12, year - 1)
    } else {
        (month as i32, year)
    };
    let q = day as i32;
    let k = y.rem_euclid(100);
    let j = y.div_euclid(100);

    // Zeller's Gregorian formula. All quantities are non-negative within the
    // supported range, so `rem_euclid` matches the mathematical mod.
    let h = (q + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 - 2 * j).rem_euclid(7);

    // Zeller: 0 = Saturday, 1 = Sunday, ..., 6 = Friday.
    let dow = match h {
        0 => DayOfWeek::Saturday,
        1 => DayOfWeek::Sunday,
        2 => DayOfWeek::Monday,
        3 => DayOfWeek::Tuesday,
        4 => DayOfWeek::Wednesday,
        5 => DayOfWeek::Thursday,
        6 => DayOfWeek::Friday,
        _ => return None, // unreachable: rem_euclid(7) is in 0..=6
    };
    Some(dow)
}

#[cfg(test)]
mod tests {
    use super::{day_of_week, is_gregorian_leap, DayOfWeek};
    use quickcheck_macros::quickcheck;

    // --- Known-date sanity checks ---

    #[test]
    fn us_independence_day_1776() {
        assert_eq!(day_of_week(1776, 7, 4), Some(DayOfWeek::Thursday));
    }

    #[test]
    fn pearl_harbor_1941() {
        assert_eq!(day_of_week(1941, 12, 7), Some(DayOfWeek::Sunday));
    }

    #[test]
    fn millennium_new_year() {
        assert_eq!(day_of_week(2000, 1, 1), Some(DayOfWeek::Saturday));
    }

    #[test]
    fn leap_day_2000() {
        // 2000 is a leap year (divisible by 400).
        assert_eq!(day_of_week(2000, 2, 29), Some(DayOfWeek::Tuesday));
    }

    #[test]
    fn non_leap_day_2001_is_invalid() {
        // 2001 is not a leap year.
        assert_eq!(day_of_week(2001, 2, 29), None);
    }

    #[test]
    fn century_non_leap_1900() {
        // 1900 is divisible by 100 but not 400 — not a leap year.
        assert_eq!(day_of_week(1900, 2, 29), None);
    }

    #[test]
    fn century_leap_2400() {
        // 2400 is divisible by 400 — a leap year.
        assert_eq!(day_of_week(2400, 2, 29), Some(DayOfWeek::Tuesday));
    }

    // --- Edge cases ---

    #[test]
    fn invalid_month_zero() {
        assert_eq!(day_of_week(2000, 0, 15), None);
    }

    #[test]
    fn invalid_month_thirteen() {
        assert_eq!(day_of_week(2000, 13, 15), None);
    }

    #[test]
    fn invalid_day_zero() {
        assert_eq!(day_of_week(2000, 6, 0), None);
    }

    #[test]
    fn invalid_day_thirty_two_in_january() {
        assert_eq!(day_of_week(2000, 1, 32), None);
    }

    #[test]
    fn day_thirty_in_february_invalid() {
        assert_eq!(day_of_week(2000, 2, 30), None);
        assert_eq!(day_of_week(2001, 2, 30), None);
    }

    #[test]
    fn day_thirty_one_in_april_invalid() {
        assert_eq!(day_of_week(2024, 4, 31), None);
    }

    #[test]
    fn year_below_supported_range() {
        assert_eq!(day_of_week(1582, 12, 31), None);
        assert_eq!(day_of_week(1, 1, 1), None);
    }

    #[test]
    fn year_above_supported_range() {
        assert_eq!(day_of_week(10_000, 1, 1), None);
    }

    #[test]
    fn supported_range_endpoints() {
        // Both endpoints should produce a valid weekday.
        assert!(day_of_week(1583, 1, 1).is_some());
        assert!(day_of_week(9999, 12, 31).is_some());
    }

    #[test]
    fn as_str_round_trip() {
        assert_eq!(DayOfWeek::Sunday.as_str(), "Sunday");
        assert_eq!(DayOfWeek::Monday.as_str(), "Monday");
        assert_eq!(DayOfWeek::Tuesday.as_str(), "Tuesday");
        assert_eq!(DayOfWeek::Wednesday.as_str(), "Wednesday");
        assert_eq!(DayOfWeek::Thursday.as_str(), "Thursday");
        assert_eq!(DayOfWeek::Friday.as_str(), "Friday");
        assert_eq!(DayOfWeek::Saturday.as_str(), "Saturday");
    }

    // --- Reference implementation for cross-checking ---

    /// Number of days in `month` of `year`, assuming `1 <= month <= 12` and
    /// `year >= 1`.
    fn days_in_month_ref(year: i32, month: u32) -> u32 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if is_gregorian_leap(year) {
                    29
                } else {
                    28
                }
            }
            _ => 0,
        }
    }

    /// Reference: count days from a known anchor and reduce mod 7.
    /// Anchor: 2000-01-01 was a Saturday.
    fn day_of_week_ref(year: i32, month: u32, day: u32) -> DayOfWeek {
        let anchor_year = 2000_i32;
        let mut days: i64 = 0;

        if year >= anchor_year {
            for y in anchor_year..year {
                days += if is_gregorian_leap(y) { 366 } else { 365 };
            }
        } else {
            for y in year..anchor_year {
                days -= if is_gregorian_leap(y) { 366 } else { 365 };
            }
        }
        for m in 1..month {
            days += i64::from(days_in_month_ref(year, m));
        }
        days += i64::from(day) - 1;

        // 2000-01-01 was Saturday — index 6 in [Sun..Sat].
        let idx = (days + 6).rem_euclid(7);
        match idx {
            0 => DayOfWeek::Sunday,
            1 => DayOfWeek::Monday,
            2 => DayOfWeek::Tuesday,
            3 => DayOfWeek::Wednesday,
            4 => DayOfWeek::Thursday,
            5 => DayOfWeek::Friday,
            6 => DayOfWeek::Saturday,
            _ => unreachable!(),
        }
    }

    #[test]
    fn reference_anchor_self_consistency() {
        // Anchor sanity: 2000-01-01 is Saturday.
        assert_eq!(day_of_week_ref(2000, 1, 1), DayOfWeek::Saturday);
    }

    #[test]
    fn matches_reference_across_decade() {
        // Walk every day from 1995 through 2005 and compare.
        for year in 1995..=2005 {
            for month in 1..=12u32 {
                let max_day = days_in_month_ref(year, month);
                for day in 1..=max_day {
                    let got = day_of_week(year, month, day).unwrap();
                    let expected = day_of_week_ref(year, month, day);
                    assert_eq!(got, expected, "mismatch on {year}-{month:02}-{day:02}");
                }
            }
        }
    }

    #[test]
    fn matches_reference_at_century_boundaries() {
        // Centuries around the 100/400 leap rule.
        for year in [1700, 1800, 1900, 2000, 2100, 2400] {
            for month in 1..=12u32 {
                let max_day = days_in_month_ref(year, month);
                for day in 1..=max_day {
                    let got = day_of_week(year, month, day).unwrap();
                    let expected = day_of_week_ref(year, month, day);
                    assert_eq!(got, expected, "mismatch on {year}-{month:02}-{day:02}");
                }
            }
        }
    }

    // --- Property tests ---

    /// Random valid Gregorian dates in the supported range agree with the
    /// reference implementation.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_matches_reference(year: u16, month: u8, day: u8) -> bool {
        // Map into the supported range deterministically.
        let year = 1583 + i32::from(year % (9999 - 1583 + 1));
        let month = (u32::from(month) % 12) + 1;
        let max_day = days_in_month_ref(year, month);
        let day = (u32::from(day) % max_day) + 1;

        day_of_week(year, month, day) == Some(day_of_week_ref(year, month, day))
    }

    /// Successive days advance the weekday by exactly one.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_consecutive_days_advance_by_one(year: u16, month: u8, day: u8) -> bool {
        let year = 1583 + i32::from(year % (9999 - 1583 + 1));
        let month = (u32::from(month) % 12) + 1;
        let max_day = days_in_month_ref(year, month);
        // Skip the last day of the month — handling month/year rollover is
        // the reference's job and not what this property tests.
        if max_day < 2 {
            return true;
        }
        let day = (u32::from(day) % (max_day - 1)) + 1;

        let today = day_of_week(year, month, day).unwrap();
        let tomorrow = day_of_week(year, month, day + 1).unwrap();
        let expected_idx = (today as u8 + 1) % 7;
        tomorrow as u8 == expected_idx
    }

    /// `as_str` is consistent with the enum order.
    #[test]
    fn enum_discriminants_in_order() {
        assert_eq!(DayOfWeek::Sunday as u8, 0);
        assert_eq!(DayOfWeek::Saturday as u8, 6);
    }
}
