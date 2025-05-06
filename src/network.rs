use std::cell::RefCell;
use std::fmt;
use std::fmt::{Debug, DebugStruct, Formatter};
use std::io::{Read, Seek};
use std::rc::Rc;
use log::error;
use crate::dual_vec::DualVec;
use crate::Executor;
use crate::Executor::{CPU, GPU};
use crate::layer::{Layer, LayerType};
use crate::layer::attention::Attention;
use crate::layer::dense::Dense;
use crate::utils::vec_utils::{CursorReader, VecWriter};

pub struct Network<'a> {
    layers: Vec<Rc<RefCell<dyn Layer<'a> + 'a>>>,
}

impl<'a> Network<'a> {

    pub fn new(mut input_size: usize, layers_types: &'a Vec<(&'a Executor, LayerType)>) -> Self {
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

    pub fn to_bytes(&mut self) -> Vec<u8> {
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
            l.borrow_mut().to_bytes(&mut writer);
        }

        writer.vec()
    }

    pub fn from_bytes(gpu_executor: Option<&'a Executor>, bytes: Vec<u8>) -> Network {
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
            let next_exec = if i+1 <= layer_count {
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
        }

        let mut c = reader.cursor();
        let mut v = Vec::new();
        c.read_to_end(&mut v);
        println!("{} {:?}", c.position(), v);

        Self {
            layers,
        }
    }
}