#![allow(unused)]

use num::complex::{c64, Complex64 as c64};
use rayon::prelude::*;
use std::time::Instant;

mod gpu;

const X_RESOLUTION: u32 = 1923;
const Y_RESOLUTION: u32 = 1447;
const TOTAL_RESOLUTION: u32 = X_RESOLUTION * Y_RESOLUTION;

// we start with Z = C to skip the first iteration, hence the - 1
const ITERS: u32 = 223 - 1;

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

fn mandelbrot_explicit_half(c_re: f32, c_im: f32, exp: f32) -> bool {
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

fn mandelbrot_count_half(exp: f32) -> usize {
    let mut count = 0;
    for x in 0..X_RESOLUTION {
        let re = (x as f32 / X_RESOLUTION as f32) * 2.0 - 1.0;
        count += (0..Y_RESOLUTION).into_par_iter().map(|y| {
            let im = (y as f32 / Y_RESOLUTION as f32) * 2.0 - 1.0;
            mandelbrot_explicit_half(re, im, exp)
        }).filter(|in_set| *in_set).count();
    }
    count
}

fn main() {
    gpu::main();

    // let start = Instant::now();
    // println!("diff for 2.0: {}", mandelbrot_count(2.0).abs_diff(mandelbrot_count_half(2.0)));
    // println!("diff for 2.1: {}", mandelbrot_count(2.1).abs_diff(mandelbrot_count_half(2.1)));
    // println!("diff for 2.2: {}", mandelbrot_count(2.2).abs_diff(mandelbrot_count_half(2.2)));
    // println!("diff for 2.3: {}", mandelbrot_count(2.3).abs_diff(mandelbrot_count_half(2.3)));
    // println!("diff for 2.4: {}", mandelbrot_count(2.4).abs_diff(mandelbrot_count_half(2.4)));
    // println!("diff for 2.5: {}", mandelbrot_count(2.5).abs_diff(mandelbrot_count_half(2.5)));
    // println!("diff for 2.6: {}", mandelbrot_count(2.6).abs_diff(mandelbrot_count_half(2.6)));
    // println!("diff for 2.7: {}", mandelbrot_count(2.7).abs_diff(mandelbrot_count_half(2.7)));
    // println!("diff for 2.8: {}", mandelbrot_count(2.8).abs_diff(mandelbrot_count_half(2.8)));
    // println!("diff for 2.9: {}", mandelbrot_count(2.9).abs_diff(mandelbrot_count_half(2.9)));

    // let count = mandelbrot_count_half(2.5);
    // println!("took {:?}", start.elapsed());
    // println!("count: {count}");
    // assert_eq!(count, 1313923);
}
