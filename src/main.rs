// Sieht aus wie ein Käfer. Mit 'nem sehr fetten Arsch.

#![allow(unused)]

use std::slice;
use num::complex::{c64, Complex64 as c64};
use rayon::prelude::*;
use std::time::Instant;
use crate::gpu::{decompress, TOTAL_BLOCK_COUNT, TOTAL_BLOCK_SIZE, X_BLOCK_COUNT, X_BLOCK_SIZE, Y_BLOCK_SIZE};

mod gpu;
mod data;

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
    // let (full_count, interesting_blocks) = load_data(exp);
    // let compressed: &[u16] = &[0x2ab1, 0x803f, 0xe0, 0xb1ff, 0xda, 0xffcf, 0x80ff, 0xd3, 0xffef, 0x81ff, 0xd4, 0xfff3, 0x80ff, 0xd6, 0xffff, 0x803f, 0xd3, 0xffff, 0x807f, 0xd3, 0xffff, 0x807f, 0xd3, 0xffff, 0x80ff, 0xd3, 0xffff, 0x81ff, 0xd4, 0xffff, 0x80ff, 0xc000, 0xc2, 0xfffd, 0x83ff, 0x10, 0x8003, 0xb7, 0xffff, 0x803f, 0xa0f0, 0xc000, 0x8007, 0xa7, 0xffff, 0xfc0f, 0xbfff, 0xf000, 0x8007, 0xa8, 0xffff, 0xfff1, 0xffff, 0xfe01, 0x8001, 0xa8, 0xdfff, 0xffff, 0xffff, 0xbf83, 0xb9, 0xff0f, 0xffff, 0xffff, 0x83f8, 0xbb, 0xffff, 0xffff, 0xe7ff, 0x800f, 0xb3, 0xffff, 0xffff, 0xffff, 0xc2, 0xffff, 0xffff, 0xffff, 0x8007, 0xb4, 0xffff, 0xffff, 0xffff, 0x803f, 0xb3, 0xffff, 0xffff, 0xffff, 0x81ff, 0xb4, 0xffff, 0xffff, 0xffff, 0x87ff, 0xb3, 0xffff, 0xffff, 0xffff, 0xbfff, 0xb4, 0xffff, 0xffff, 0xffff, 0xffff, 0x8001, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x8003, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x800f, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x803f, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x877f, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x8dff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x83ff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x87ff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x8fff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x8fff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0xbfff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0xbfff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0xbfff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x87ff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x83ff, 0xa4, 0xffff, 0xffff, 0xffff, 0xffff, 0x83ff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x83ff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x83ff, 0xa8, 0xffff, 0xffff, 0xffff, 0xffff, 0x83ff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x87ff, 0xa5, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0xbfff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0xbfff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0xbfff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x9fff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x8fff, 0xa6, 0xffff, 0xffff, 0xffff, 0xffff, 0x8fff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x87ff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x83ff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x8dff, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x877f, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x803f, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x800f, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x8003, 0xa7, 0xffff, 0xffff, 0xffff, 0xffff, 0x8001, 0xa7, 0xffff, 0xffff, 0xffff, 0xbfff, 0xb7, 0xffff, 0xffff, 0xffff, 0x87ff, 0xb6, 0xffff, 0xffff, 0xffff, 0x81ff, 0xb7, 0xffff, 0xffff, 0xffff, 0x803f, 0xb6, 0xffff, 0xffff, 0xffff, 0x8007, 0xb7, 0xffff, 0xffff, 0xffff, 0xc6, 0xffff, 0xffff, 0xe7ff, 0x800f, 0xaf, 0xff0f, 0xffff, 0xffff, 0x83f8, 0xb1, 0xdfff, 0xffff, 0xffff, 0xbf83, 0xb3, 0xffff, 0xfff1, 0xffff, 0xfe01, 0x8001, 0xa4, 0xffff, 0xfc0f, 0xbfff, 0xf000, 0x8007, 0xa5, 0xffff, 0x803f, 0xa0f0, 0xc000, 0x8007, 0xa3, 0xfffd, 0x83ff, 0x10, 0x8003, 0xb6, 0xffff, 0x80ff, 0xc000, 0xc3, 0xffff, 0x81ff, 0xd3, 0xffff, 0x80ff, 0xd3, 0xffff, 0x807f, 0xd3, 0xffff, 0x807f, 0xd3, 0xffff, 0x803f, 0xd0, 0xfff3, 0x80ff, 0xd2, 0xffef, 0x81ff, 0xd3, 0xffcf, 0x80ff, 0xdb, 0xb1ff, 0xe4, 0x803f];
    // let compressed = unsafe { slice::from_raw_parts(compressed.as_ptr().cast(), compressed.len()) };
    // let full_blocks = decompress(compressed, TOTAL_BLOCK_COUNT as usize);
    let mut count = 0;
    for x in 0..X_RESOLUTION {
        let re = (x as f64 / X_RESOLUTION as f64) * 4.0 - 2.0;
        count += (0..Y_RESOLUTION).into_par_iter().map(|y| {
            let im = (y as f64 / Y_RESOLUTION as f64) * 4.0 - 2.0;
            let in_set = mandelbrot_explicit(re, im, exp);
            // if in_set {
            //     let full = pixel_is_interesting(x, y, &full_blocks);
            //     let interesting = pixel_is_interesting(x, y, &interesting_blocks);
            //     assert!(full | interesting, "{x} {y} is not full and not interesting");
            // }
            in_set
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
        let re = (x as f32 / X_RESOLUTION as f32) * 4.0 - 2.0;
        count += (0..Y_RESOLUTION).into_par_iter().map(|y| {
            let im = (y as f32 / Y_RESOLUTION as f32) * 4.0 - 2.0;
            mandelbrot_explicit_half(re, im, exp)
        }).filter(|in_set| *in_set).count();
    }
    count
}

fn load_data(exp: f64) -> (u32, Vec<bool>) {
    // let index = ((exp - 2.0) * 1000.0) as usize;
    let (full_count, compressed_full, compressed_interesting) = data::BLOCK_DATA[0];
    _ = compressed_full;
    let compressed = compressed_interesting;
    let compressed = unsafe { slice::from_raw_parts(compressed.as_ptr().cast(), compressed.len()) };
    let decompressed = decompress(compressed, TOTAL_BLOCK_COUNT as usize);
    (full_count, decompressed)
}

fn pixel_is_interesting(x: u32, y: u32, blocks: &[bool]) -> bool {
    let block_x = x / X_BLOCK_SIZE;
    let block_y = y / Y_BLOCK_SIZE;
    let block_index = block_y * X_BLOCK_COUNT + block_x;
    blocks[block_index as usize]
}

fn mandelbrot_count_with_data(exp: f64) -> usize {
    let (full_count, interesting) = load_data(exp);
    interesting.into_par_iter().enumerate().filter(|(_, b)| *b).map(|(block_index, _)| {
        let block_x = block_index as u32 % X_BLOCK_COUNT;
        let block_y = block_index as u32 / X_BLOCK_COUNT;
        let x_min = block_x * X_BLOCK_SIZE;
        let y_min = block_y * Y_BLOCK_SIZE;
        let x_max = x_min + X_BLOCK_SIZE;
        let y_max = y_min + Y_BLOCK_SIZE;
        let mut count = 0;
        for x in x_min..x_max {
            let re = (x as f64 / X_RESOLUTION as f64) * 4.0 - 2.0;
            for y in y_min..y_max {
                let im = (y as f64 / Y_RESOLUTION as f64) * 4.0 - 2.0;
                let in_set = mandelbrot_explicit(re, im, exp);
                if in_set {
                    count += 1;
                }
            }
        }
        count
    }).sum::<usize>() + (full_count * TOTAL_BLOCK_SIZE) as usize
}

fn mandelbrot_count_with_data_half(exp: f32) -> usize {
    let (full_count, interesting) = load_data(exp as f64);
    interesting.into_par_iter().enumerate().filter(|(_, b)| *b).map(|(block_index, _)| {
        let block_x = block_index as u32 % X_BLOCK_COUNT;
        let block_y = block_index as u32 / X_BLOCK_COUNT;
        let x_min = block_x * X_BLOCK_SIZE;
        let y_min = block_y * Y_BLOCK_SIZE;
        let x_max = x_min + X_BLOCK_SIZE;
        let y_max = y_min + Y_BLOCK_SIZE;
        let mut count = 0;
        for x in x_min..x_max {
            let re = (x as f32 / X_RESOLUTION as f32) * 4.0 - 2.0;
            for y in y_min..y_max {
                let im = (y as f32 / Y_RESOLUTION as f32) * 4.0 - 2.0;
                let in_set = mandelbrot_explicit_half(re, im, exp);
                if in_set {
                    count += 1;
                }
            }
        }
        count
    }).sum::<usize>() + dbg!(full_count * TOTAL_BLOCK_SIZE) as usize
}

fn main() {
    // let x = 108;
    // let y = 719;
    // let re = (x as f64 / X_RESOLUTION as f64) * 4.0 - 2.0;
    // let im = (y as f64 / Y_RESOLUTION as f64) * 4.0 - 2.0;
    // let steps = 1000_000;
    // for i in 0..steps {
    //     let exp = 2.0 + 0.001 * i as f64 / steps as f64;
    //     let in_set = mandelbrot_explicit(re, im, exp);
    //     if in_set {
    //         println!("{exp}");
    //     }
    // }

    gpu::main();

    // let values = [2.5, 2.5001, 2.5002, 2.5003, 2.5004, 2.5005, 2.50061435163];
    // let values = (0..100).map(|i| 2.5 + i as f64 / 100_000.0);
    // for exp in values {
    //     let diff = mandelbrot_count(exp).abs_diff(mandelbrot_count_with_data(exp));
    //     println!("for exp {exp}: diff {diff}");
    // }

    // let start = Instant::now();
    // let count = mandelbrot_count(2.500);
    // println!("took {:?}", start.elapsed());
    // println!("count: {count}");
    // assert_eq!(count, 330238);
}
