//! Iterative radix-2 Cooley-Tukey fast Fourier transform.
//!
//! The transform runs in `O(n log n)` time with `O(1)` extra storage on top
//! of the input buffer (apart from the bit-reversal index swap). Input length
//! must be a power of two; callers are responsible for zero-padding.
//!
//! A small in-file `Complex` type backed by `f64` components avoids pulling
//! in `num-complex`. Because the transform operates on floating point, results
//! are subject to round-off; expect roughly `1e-9` absolute error on inputs of
//! moderate size and larger error as `n` grows. When the inputs are known to
//! be integer-valued (e.g. polynomial multiplication of integer coefficients),
//! the caller may round the real component of the output.
//!
//! Time complexity: `O(n log n)` for `fft_in_place`; polynomial multiplication
//! is `O((n + m) log(n + m))` where `n`, `m` are the input degrees.

use std::ops::{Add, Div, Mul, Sub};

/// Minimal complex number with `f64` components used by the FFT routines.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    /// Constructs a complex number from real and imaginary parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            im: self.re.mul_add(rhs.im, self.im * rhs.re),
        }
    }
}

impl Div<f64> for Complex {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

/// In-place iterative Cooley-Tukey FFT (radix 2, decimation in time).
///
/// `a.len()` must be a power of two. When `invert` is `true` the inverse
/// transform is computed and the result is divided by `n` so a forward
/// transform followed by an inverse transform recovers the original
/// (up to floating-point round-off). Lengths of 0 or 1 are no-ops.
#[allow(clippy::ptr_arg)]
pub fn fft_in_place(a: &mut Vec<Complex>, invert: bool) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "fft length must be a power of two");

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    // Butterfly stages.
    let mut len = 2usize;
    while len <= n {
        let angle = if invert {
            -2.0 * std::f64::consts::PI / (len as f64)
        } else {
            2.0 * std::f64::consts::PI / (len as f64)
        };
        let wlen = Complex::new(angle.cos(), angle.sin());
        let half = len / 2;
        let mut i = 0usize;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = a[i + k];
                let v = a[i + k + half] * w;
                a[i + k] = u + v;
                a[i + k + half] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }

    if invert {
        let n_f = n as f64;
        for x in a.iter_mut() {
            *x = *x / n_f;
        }
    }
}

/// Multiplies two polynomials given by their coefficient vectors using FFT.
///
/// `a` and `b` are interpreted as polynomials with `a[i]` and `b[i]` being
/// the coefficient of `x^i`. The returned vector has length
/// `a.len() + b.len() - 1` (or empty when either input is empty) and contains
/// the convolution `c[k] = sum_{i + j = k} a[i] * b[j]`.
///
/// Internally pads both inputs with zeros to the next power of two `>=`
/// `a.len() + b.len() - 1`, runs FFT/iFFT, and returns the real components.
/// The result is **not** rounded to integers; callers operating over integer
/// coefficients should `.round()` themselves. Expect floating-point error on
/// the order of `1e-9` for moderate-sized inputs.
pub fn multiply_polynomials(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let result_len = a.len() + b.len() - 1;
    let mut n = 1usize;
    while n < result_len {
        n <<= 1;
    }

    let mut fa: Vec<Complex> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut fb: Vec<Complex> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fa.resize(n, Complex::new(0.0, 0.0));
    fb.resize(n, Complex::new(0.0, 0.0));

    fft_in_place(&mut fa, false);
    fft_in_place(&mut fb, false);

    for i in 0..n {
        fa[i] = fa[i] * fb[i];
    }

    fft_in_place(&mut fa, true);

    fa.into_iter().take(result_len).map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::{fft_in_place, multiply_polynomials, Complex};
    use std::f64::consts::PI;

    /// Brute-force discrete Fourier transform used as ground truth for tests.
    fn dft(a: &[Complex]) -> Vec<Complex> {
        let n = a.len();
        let mut out = vec![Complex::new(0.0, 0.0); n];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut acc = Complex::new(0.0, 0.0);
            for (j, &x) in a.iter().enumerate() {
                let angle = 2.0 * PI * (j as f64) * (k as f64) / (n as f64);
                let w = Complex::new(angle.cos(), angle.sin());
                acc = acc + x * w;
            }
            *slot = acc;
        }
        out
    }

    fn close(a: Complex, b: Complex, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn matches_dft_brute_force() {
        let input: Vec<Complex> = [1.0_f64, 1.0, 1.0, 1.0]
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let expected = dft(&input);
        let mut actual = input;
        fft_in_place(&mut actual, false);
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!(close(*a, *b, 1e-9), "fft mismatch: {a:?} vs {b:?}");
        }
        // [1,1,1,1] -> [4, 0, 0, 0].
        assert!(close(actual[0], Complex::new(4.0, 0.0), 1e-9));
        for v in actual.iter().skip(1) {
            assert!(close(*v, Complex::new(0.0, 0.0), 1e-9));
        }
    }

    #[test]
    fn inverse_round_trip() {
        let original: Vec<Complex> = [3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let mut buf = original.clone();
        fft_in_place(&mut buf, false);
        fft_in_place(&mut buf, true);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!(close(*a, *b, 1e-9), "round trip mismatch: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn multiply_two_quadratics() {
        // (1 + 2x + 3x^2)(4 + 5x + 6x^2) = 4 + 13x + 28x^2 + 27x^3 + 18x^4
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let expected = [4.0, 13.0, 28.0, 27.0, 18.0];
        let got = multiply_polynomials(&a, &b);
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got {g}, expected {e}");
        }
    }

    fn naive_convolution(a: &[f64], b: &[f64]) -> Vec<f64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] += x * y;
            }
        }
        out
    }

    #[test]
    fn matches_naive_convolution() {
        // Deterministic small "random" sequences to keep the test reproducible.
        let a = [0.5_f64, -1.5, 2.25, 3.0, -0.75, 1.125];
        let b = [-2.0_f64, 1.0, 0.25, 4.0, -3.5];
        let expected = naive_convolution(&a, &b);
        let got = multiply_polynomials(&a, &b);
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "fft conv {g} vs naive {e}");
        }
    }

    #[test]
    fn n_one_is_noop() {
        let mut buf = vec![Complex::new(7.5, -2.25)];
        fft_in_place(&mut buf, false);
        assert_eq!(buf, vec![Complex::new(7.5, -2.25)]);
        fft_in_place(&mut buf, true);
        assert_eq!(buf, vec![Complex::new(7.5, -2.25)]);

        // Single-coefficient polynomial product.
        let got = multiply_polynomials(&[3.0], &[4.0]);
        assert_eq!(got.len(), 1);
        assert!((got[0] - 12.0).abs() < 1e-9);
    }

    #[test]
    fn empty_polynomial_product_is_empty() {
        assert!(multiply_polynomials(&[], &[1.0, 2.0]).is_empty());
        assert!(multiply_polynomials(&[1.0], &[]).is_empty());
    }
}
