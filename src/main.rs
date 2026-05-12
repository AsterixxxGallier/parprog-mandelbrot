use std::time::Instant;
use num::complex::{c64, Complex64 as c64};

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

fn main() {
    gpu::main();

    /*let start = Instant::now();
    let mut count = 0;
    for x in 0..1_000 {
        let re = (x as f64 / 1_000.0) * 2.0 - 1.0;
        for y in 0..1_000 {
            let im = (y as f64 / 1_000.0) * 2.0 - 1.0;
            let c = c64(re, im);
            count += mandelbrot(c, 2.5) as u32;
        }
    }
    println!("took {:?}", start.elapsed());
    println!("count: {count}");
    assert_eq!(count, 472328);*/
}
