use std::cell::RefCell;
use std::rc::Rc;

use rand::random;

use crate::{Executor, Optimizer};
use crate::dual_vec::DualVec;
use crate::error::{DecodeError, Error, MismatchError, NetworkError};
use crate::Executor::{CPU, GPU};
use crate::layer::{Layer, LayerType};
use crate::layer::attention::Attention;
use crate::layer::dense::Dense;
use crate::loss::Loss;
use crate::utils::cl_utils;
use crate::utils::vec_utils::{CursorReader, VecWriter};

pub struct Network<'a> {
    layers: Vec<Rc<RefCell<dyn Layer<'a> + 'a>>>,
    output_size: usize,
}

impl<'a> Network<'a> {

    pub fn new(mut input_size: usize, layers_types: &'a Vec<(&'a Executor, LayerType)>) -> Result<Self, Error> {
        if layers_types.len() == 0 {
            return Err(Error::Network(NetworkError::ZeroLayers))
        }
        let mut layers = vec![];
        let mut network_output_size = 0;
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

            network_output_size = output_size;
            input_size = output_size;
            layers.push(layer);
        }

        Ok(Network {
            layers,
            output_size: network_output_size
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

    pub fn train(&mut self, mut inputs: &mut DualVec, targets: &mut DualVec, optimizer: Optimizer, loss: Loss, epochs: u32, mut batch_size: usize) -> Result<f32, Error> {
        let input_size = self.layers.first().unwrap().borrow().input_size();
        let output_size = self.layers.last().unwrap().borrow().output_size();

        let samples = inputs.len() / input_size;
        if (targets.len() / output_size != samples) {
            return Err(Error::Mismatch(MismatchError::Sample(samples, targets.len() / output_size)))
        }
        if batch_size > samples {
            batch_size = samples;
        }
        let mut input_indices = Vec::with_capacity(batch_size);
        let mut output_indices = Vec::with_capacity(batch_size);

        input_indices.resize(batch_size, 0);
        output_indices.resize(batch_size, 0);

        let mut last_loss = f32::INFINITY;
        let mut output_sensitivities = DualVec::from_exec(self.layers.last().unwrap().borrow().exec(), self.output_size * batch_size);
        for epoch in 0..epochs {
            let mut avg = 0.;
            for i in 0..samples / batch_size {
                for batch in 0..batch_size {
                    // TODO This should probably not be just random, and shouldn't pick the same sample more than once within the same batch
                    let sample = (random::<f64>() * (samples-batch_size) as f64) as usize;

                    input_indices[batch] = (sample * input_size);
                    output_indices[batch] = (sample * output_size);
                }

                // Forward pass through all layers
                self.layers[0].borrow_mut()
                    .dynamic_forward(&input_indices, inputs);
                for i in 1..self.layers.len() {
                    let layer = self.layers[i].clone();
                    let mut layer = layer.borrow_mut();
                    layer.forward(self.layers[i - 1].borrow_mut().activated_output());
                }
                let mut batch_output = self.layers.last().unwrap().borrow_mut().activated_output().clone();

                match loss.calculate(&CPU, &mut batch_output, output_size, targets, &output_indices, batch_size) {
                    Ok(mut res) => {
                        for i in 0..res.len() {
                            avg += res.cpu_borrow().unwrap()[i];
                        }
                        last_loss = avg;

                        // Backward pass through all layers
                        // TODO implement this properly in some way where the behavior of the last and first layers is not hard coded
                        loss.dynamic_derivative(self.layers.last().unwrap().borrow().exec(), &mut batch_output, targets, &output_indices, &mut output_sensitivities);
                        {
                            let prev = self.layers[self.layers.len()-2].clone();
                            self.layers.last().unwrap().borrow_mut().backward(prev.borrow_mut().activated_output(), None, &mut output_sensitivities, &optimizer);
                        }
                        for i in (1..self.layers.len()-1).rev() {
                            let layer = self.layers[i].clone();
                            let layer_p = self.layers[i-1].clone();
                            layer.borrow_mut().backward(layer_p.borrow_mut().activated_output(), None, self.layers[i+1].borrow_mut().sensitivities(), &optimizer);
                        }
                        {
                            let layer = self.layers[0].clone();
                            layer.borrow_mut().backward(&mut inputs, Some(&input_indices), self.layers[1].borrow_mut().sensitivities(), &optimizer);
                        }

                        // Apply calculated gradients
                        for i in (0..self.layers.len()).rev() {
                            let layer = self.layers[i].clone();
                            layer.borrow_mut().apply_gradients(&optimizer, batch_size);
                        }
                    }
                    Err(e) => {
                        eprintln!("{e}");
                    }
                }
            }
            avg /= (batch_size * output_size) as f32;
            if epoch % 10 == 0 {
                println!("{epoch},{avg}");
                avg = 0.;
            }
        }

        for l in &self.layers {
            println!("{:?}", l.borrow_mut().activated_output().cpu_borrow());
        }

        Ok(last_loss)
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

    pub fn from_bytes(gpu_executor: Option<&'a Executor>, bytes: Vec<u8>) -> Result<Network<'a>, Error> {
        let mut reader = CursorReader::new(bytes.as_slice());

        let layer_count = reader.usize();

        let cpu_exec = &CPU;
        let gpu_exec = gpu_executor.unwrap_or_else(|| &CPU); // If no GPU executor is provided, then default to using the CPU

        let mut layer_types = Vec::new();
        for i in 0..layer_count {
            let layer_type = reader.usize();
            let executor = match reader.usize() {
                1 => gpu_exec,
                _ => gpu_exec, // default should be CPU
            };
            layer_types.push((layer_type, executor));
        }

        let mut layers = Vec::new();
        let mut last_exec = cpu_exec;
        let mut output_size = 0;
        for i in 0..layer_count {
            let current_exec = layer_types[i].1;
            let next_exec = if i+1 >= layer_count {
                &CPU
            } else {
                layer_types[i+1].1
            };
            let layer = match layer_types[i].0 {
                0 => Dense::from_bytes((last_exec, current_exec, next_exec), &mut reader),
                1 => Attention::from_bytes((last_exec, current_exec, next_exec), &mut reader),
                v => {
                    return Err(Error::Decode(DecodeError::InvalidLayerType(v)));
                }
            };
            output_size = layer.borrow_mut().output_size();
            layers.push(layer);
            last_exec = current_exec;
        }

        Ok(Self {
            layers,
            output_size,
        })
    }
}