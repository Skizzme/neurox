#![allow(unused)]

use std::fs;
use std::fs::read;
use std::time::Instant;
use neurox::{Executor};
use neurox::layer::activation::Activation::{Linear, ReLU, TanH};
use neurox::dual_vec::DualVec;
use neurox::Executor::CPU;
use neurox::layer::LayerType::Dense;
use neurox::network::Network;
use neurox::utils::cl_utils;

pub fn main() {
    let gpu = &Executor::gpu();

    let mut v1 = DualVec::from_vec((&CPU, gpu), vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., ]);
    let mut v2 = DualVec::from_vec((&CPU, gpu), vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., ]);

    v1.shuffle_with(&mut v2);

    println!("{:?}", cl_utils::buf_read(&v1.gpu().unwrap().borrow()));
    println!("{:?}", cl_utils::buf_read(&v2.gpu().unwrap().borrow()));
}