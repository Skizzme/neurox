use std::cell::RefCell;
use std::rc::Rc;
use ocl::ProQue;
use crate::dual_vec::DualVec;
use crate::Executor::{CPU, GPU};
use crate::layer::{Layer, LayerType};

pub mod activation;
pub mod dual_vec;
pub mod layer;
pub mod utils;

pub struct Network<'a> {
    layers: Vec<Rc<RefCell<dyn Layer + 'a>>>,
}

impl<'a> Network<'a> {
    pub fn from_values(mut input_size: usize, layers_types: Vec<(&'a Executor, LayerType)>, values: Vec<u8>) -> Self {
        let mut layers = vec![];
        for i in 0..layers_types.len() {
            let (current_exec, l_type) = &layers_types[i];
            let prev_exec =
                if i > 0 {
                    layers_types[i-1].0
                } else {
                    &CPU
                };
            let next_exec =
                if i < layers_types.len() - 1 {
                    layers_types[i+1].0
                } else {
                    &CPU
                };

            let (layer, output_size) = l_type.layer((prev_exec, *current_exec, next_exec), input_size);

            layers.push(layer);
            input_size = output_size;
        }

        Network {
            layers,
        }
    }
    pub fn new(mut input_size: usize, layers_types: Vec<(&'a Executor, LayerType)>) -> Self {
        let mut layers = vec![];
        for i in 0..layers_types.len() {
            let (current_exec, l_type) = &layers_types[i];
            let prev_exec =
                if i > 0 {
                    layers_types[i-1].0
                } else {
                    &CPU
                };
            let next_exec =
                if i < layers_types.len() - 1 {
                    layers_types[i+1].0
                } else {
                    &CPU
                };

            let (layer, output_size) = l_type.layer((prev_exec, *current_exec, next_exec), input_size);

            layers.push(layer);
            input_size = output_size;
        }

        Network {
            layers,
        }
    }

    pub fn predict(&mut self, inputs: &mut DualVec) -> DualVec {
        let mut batch_size = self.layers[0].borrow_mut().forward(inputs);
        for i in 1..self.layers.len() {
            let layer = self.layers[i].clone();
            layer.borrow_mut().forward(self.layers[i-1].borrow_mut().activated_output(batch_size));
        }
        self.layers.last().unwrap().borrow_mut().activated_output(batch_size).clone()
    }
}

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