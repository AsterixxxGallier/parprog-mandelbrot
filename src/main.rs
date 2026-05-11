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

fn mandelbrot_(c: c64, exp: f64) -> bool {
    let mut z = c;
    for _ in 0..ITERS {
        z = z.powc(c64(exp, 0.0)) + c;
        if z.norm_sqr() > 4.0 {
            return false;
        }
    }
    true
}

fn mandelbrot_exp(c: c64, exp: f64) -> bool {
    // let exp = c64(exp, 0.0);
    let mut z = c;
    for _ in 0..ITERS {
        // z = (exp * z.ln()).exp() + c;
        // let (r, theta) = self.to_polar();
        // Self::new(r.ln(), theta)
        // let z_ln = c64(z.norm().ln(), z.arg());
        // let base = z_ln * exp;
        let base = c64(z.norm().ln() * exp, z.arg());
        z = c64::from_polar(base.re.exp(), base.im) + c;
        if z.norm_sqr() > 4.0 {
            return false;
        }
    }
    true
}

fn mandelbrot_polar(c: c64, exp: f64) -> bool {
    let (mut r, mut theta) = c.to_polar();
    for _ in 0..ITERS {
        r = r.powf(exp);
        theta *= exp;
        (r, theta) = (c64::from_polar(r, theta) + c).to_polar();
        if r > 2.0 {
            return false;
        }
    }
    true
}

fn main() {
    let start = Instant::now();
    let mut count = 0;
    for x in 0..1_000 {
        let re = (x as f64 / 1_000.0) * 2.0 - 1.0;
        for y in 0..1_000 {
            let im = (y as f64 / 1_000.0) * 2.0 - 1.0;
            let c = c64(re, im);
            count += mandelbrot_(c, 2.5) as u32;
        }
    }
    // for i in 0..1_000_000 {
    //     let c = c64(i as f64 / 1_000_000_000.0, 0.0);
    //     count += mandelbrot_exp(c, 2.5) as u32;
    // }
    println!("took {:?}", start.elapsed());
    println!("count: {count}");
    assert_eq!(count, 472328);
}
