use std::time::Instant;
use num::complex::{c64, Complex64 as c64};

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

fn main() {
    let start = Instant::now();
    let mut count = 0;
    for i in 0..1_000_000 {
        let c = c64(i as f64 / 1_000_000_000.0, 0.0);
        count += mandelbrot(c, 2.5) as u32;
    }
    println!("took {:?}", start.elapsed());
    println!("count: {count}");
}
