use std::cell::RefCell;
use std::rc::Rc;

use rand::random;

use crate::{Executor, Optimizer};
use crate::dual_vec::DualVec;
use crate::error::{Error, MismatchError, NetworkError};
use crate::Executor::{CPU, GPU};
use crate::layer::{Layer, LayerType};
use crate::layer::attention::Attention;
use crate::layer::dense::Dense;
use crate::loss::Loss;
use crate::utils::vec_utils::{CursorReader, VecWriter};

pub struct Network<'a> {
    layers: Vec<Rc<RefCell<dyn Layer<'a> + 'a>>>,
}

impl<'a> Network<'a> {

    pub fn new(mut input_size: usize, layers_types: &'a Vec<(&'a Executor, LayerType)>) -> Result<Self, Error> {
        if layers_types.len() == 0 {
            return Err(Error::Network(NetworkError::ZeroLayers))
        }
        let mut layers = vec![];
        for i in 0..layers_types.len() {
            let l_type = &layers_types[i].1;
            let current_exec= layers_types[i].0;
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

            let (layer, output_size) = l_type.layer((prev_exec, current_exec, next_exec), input_size);

            layers.push(layer);
            input_size = output_size;
        }

        Ok(Network {
            layers,
        })
    }

    pub fn predict(&mut self, inputs: &mut DualVec) -> DualVec {
        let mut batch_size = self.layers[0].borrow_mut().forward(inputs);
        for i in 1..self.layers.len() {
            let layer = self.layers[i].clone();
            layer.borrow_mut().forward(self.layers[i-1].borrow_mut().activated_output());
        }
        self.layers.last().unwrap().borrow_mut().activated_output().clone()
    }

    pub fn train(&mut self, inputs: &mut DualVec, targets: &mut DualVec, optimizer: Optimizer, loss: Loss, epochs: u32, batch_size: usize) -> Result<f32, Error> {
        let input_size = self.layers.first().unwrap().borrow().input_size();
        let output_size = self.layers.last().unwrap().borrow().output_size();

        let samples = inputs.len() / input_size;
        if (targets.len() / output_size != samples) {
            return Err(Error::Mismatch(MismatchError::Sample(samples, targets.len() / output_size)))
        }

        let mut best_loss = f32::INFINITY;
        for epoch in 0..epochs {
            for i in 0..samples / batch_size {
                let mut input_indices = Vec::new();
                let mut output_indices = Vec::new();
                for batch in 0..batch_size {
                    // TODO This should probably not be just random, and shouldn't pick the same sample more than once within the same batch
                    let sample = (random::<f64>() * samples as f64) as usize;
                    println!("sample: {}", sample);
                    input_indices.push(sample * input_size);
                    output_indices.push(sample * output_size);
                }

                self.layers[0].borrow_mut().dynamic_forward(&input_indices, inputs);
                for i in 1..self.layers.len() {
                    let layer = self.layers[i].clone();
                    layer.borrow_mut().forward(self.layers[i - 1].borrow_mut().activated_output());
                }
                let mut batch_ouput = self.layers.last().unwrap().borrow_mut().activated_output().clone();

                println!("in {:?} out {:?} target {:?} actual {:?}", input_indices, output_indices, targets, batch_ouput);
                let res = loss.calculate(&CPU, &mut batch_ouput, targets, &output_indices, batch_size);
                println!("error {:?}", res);
            }
        }

        Ok(best_loss)
    }

    pub fn as_bytes(&mut self) -> Vec<u8> {
        let mut writer = VecWriter::new();

        writer.usize(self.layers.len());

        for l in &self.layers {
            let mut layer = l.borrow_mut();
            writer.usize(layer.id());

            // Store the executor type as well
            match layer.exec() {
                CPU => writer.usize(0),
                GPU(_) => writer.usize(1),
            }
        }

        for l in &self.layers {
            l.borrow_mut().as_bytes(&mut writer);
        }

        writer.vec()
    }

    pub fn from_bytes(gpu_executor: Option<&'a Executor>, bytes: Vec<u8>) -> Network<'a> {
        let mut reader = CursorReader::new(bytes.as_slice());

        let layer_count = reader.usize();

        let cpu_exec = &CPU;
        let gpu_exec = gpu_executor.unwrap_or_else(|| &CPU); // If no GPU executor is provided, then default to using the CPU

        let mut layer_types = Vec::new();
        for i in 0..layer_count {
            let layer_type = reader.usize();
            let executor = match reader.usize() {
                1 => gpu_exec,
                _ => cpu_exec, // default should be CPU
            };
            layer_types.push((layer_type, executor));
        }

        let mut layers = Vec::new();
        let mut last_exec = cpu_exec;
        for i in 0..layer_count {
            let current_exec = layer_types[i].1;
            let next_exec = if i+1 >= layer_count {
                &CPU
            } else {
                layer_types[i+1].1
            };
            match layer_types[i].0 {
                0 => layers.push(Dense::from_bytes((last_exec, current_exec, next_exec), &mut reader)),
                1 => layers.push(Attention::from_bytes((last_exec, current_exec, next_exec), &mut reader)),
                v => {
                    eprintln!("Unknown layer type encountered while decoding layer from bytes. Value: {}", v)
                }
            }
            last_exec = current_exec;
        }

        Self {
            layers,
        }
    }
}