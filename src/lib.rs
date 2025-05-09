#![allow(unused)]

use ocl::ProQue;
use crate::Executor::GPU;

pub mod dual_vec;
pub mod layer;
pub mod utils;
pub mod network;
pub mod loss;
pub mod activation;
pub mod error;


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

impl Optimizer {
    pub fn learn_rate(&self) -> f32 {
        match self {
            Optimizer::GradientDecent(lr) => *lr,
        }
    }
}
