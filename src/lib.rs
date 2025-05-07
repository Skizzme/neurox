#![allow(unused)]

use ocl::ProQue;

use crate::Executor::GPU;

pub mod dual_vec;
pub mod layer;
pub mod utils;
pub mod network;


#[derive(Debug)]
pub enum Executor {
    GPU(ProQue),
    CPU,
}

impl Executor {
    pub fn gpu() -> Self {
        let src = include_str!("kernels.c");
        let pro_que = ProQue::builder().src(src).build().unwrap();
        GPU(pro_que)
    }
}

pub enum Optimizer {
    GradientDecent(f32),
}