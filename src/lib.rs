use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use ocl::ProQue;
use crate::dual_vec::DualVec;
use crate::Executor::{CPU, GPU};
use crate::layer::{Layer, LayerType};
use crate::layer::dense::Dense;
use crate::utils::vec_utils::{CursorReader, VecWriter};

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