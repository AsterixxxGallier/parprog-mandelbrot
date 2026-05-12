#![allow(unused)]

use num::complex::{c64, Complex64 as c64};
use rayon::prelude::*;
use std::time::Instant;

mod gpu;

const X_RESOLUTION: usize = 1923;
const Y_RESOLUTION: usize = 1447;
const TOTAL_RESOLUTION: usize = X_RESOLUTION * Y_RESOLUTION;

// we start with Z = C to skip the first iteration, hence the - 1
const ITERS: usize = 223 - 1;

fn mandelbrot(c: c64, exp: f64) -> bool {
    let mut z = c;
    for _ in 0..ITERS {
        z = z.powf(exp) + c;
        if z.norm_sqr() > 4.0 {
            return false;
        }
    }
    true
}

fn mandelbrot_explicit(c_re: f64, c_im: f64, exp: f64) -> bool {
    let mut z_re = c_re;
    let mut z_im = c_im;
    for _ in 0..ITERS {
        let z_norm = z_re.hypot(z_im);
        let z_arg = z_im.atan2(z_re);
        let pow_norm = z_norm.powf(exp);
        let pow_arg = z_arg * exp;
        let pow_re = pow_norm * pow_arg.cos();
        let pow_im = pow_norm * pow_arg.sin();
        z_re = pow_re + c_re;
        z_im = pow_im + c_im;
        if z_re * z_re + z_im * z_im > 4.0 {
            return false;
        }
    }
    true
}

fn mandelbrot_count(exp: f64) -> usize {
    let mut count = 0;
    for x in 0..X_RESOLUTION {
        let re = (x as f64 / X_RESOLUTION as f64) * 2.0 - 1.0;
        count += (0..Y_RESOLUTION).into_par_iter().map(|y| {
            let im = (y as f64 / Y_RESOLUTION as f64) * 2.0 - 1.0;
            mandelbrot_explicit(re, im, exp)
        }).filter(|in_set| *in_set).count();
    }
    count
}

fn main() {
    // gpu::main();

    let start = Instant::now();
    let count = mandelbrot_count(2.5 + 1e-16);
    println!("took {:?}", start.elapsed());
    println!("count: {count}");
    assert_eq!(count, 1313923);
}
