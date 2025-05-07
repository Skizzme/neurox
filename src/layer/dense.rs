use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ocl::Kernel;

use crate::{Executor, Optimizer};
use crate::dual_vec::DualVec;
use crate::layer::activation::Activation;
use crate::layer::Layer;
use crate::utils::vec_utils::{CursorReader, VecWriter};

#[derive(Debug)]
pub struct Dense<'a> {
    exec: &'a Executor,
    execs:  (&'a Executor, &'a Executor, &'a Executor),
    size: usize,
    input_len: usize,

    activation: Activation,

    weights: DualVec,
    biases: DualVec,

    weight_mods: DualVec,
    bias_mods: DualVec,

    outputs: HashMap<usize, DualVec>,
    activated_outputs: HashMap<usize, DualVec>,
    sensitivities: HashMap<usize, DualVec>,

    forward_kernels: HashMap<u32, Kernel>,
}

impl<'a> Dense<'a> {

    pub fn new(exec: (&'a Executor, &'a Executor, &'a Executor), inputs: usize, size: usize, activation: Activation) -> Self {
        let c = exec.1; // current

        let mut a = Dense {
            exec: exec.1,
            execs: exec,
            size,
            input_len: inputs,

            activation,
            outputs: HashMap::new(),
            activated_outputs: HashMap::new(),

            sensitivities: HashMap::new(),

            weights: DualVec::from_exec(c, inputs * size),
            biases: DualVec::from_exec(c, size),

            weight_mods: DualVec::from_exec(c, inputs * size),
            bias_mods: DualVec::from_exec(c, size),
            forward_kernels: HashMap::new(),
        };
        a.weights.randomize(c);
        a.biases.randomize(c);
        a
    }

    fn setup_batch(&mut self, batch_size: usize) {

        if !self.outputs.contains_key(&batch_size) {
            let p_to_c = (self.execs.0, self.execs.1); // previous to current
            let c_to_n = (self.execs.1, self.execs.2); // current to next

            self.outputs.insert(batch_size, DualVec::from_execs(c_to_n, self.size*batch_size));
            self.activated_outputs.insert(batch_size, DualVec::from_execs(c_to_n, self.size*batch_size));
            self.sensitivities.insert(batch_size, DualVec::from_execs(p_to_c, self.input_len*batch_size));
        }
    }
}

impl<'a> Layer<'a> for Dense<'a> {
    fn forward(&mut self, activated_inputs: &mut DualVec) -> usize {
        let batch_size = activated_inputs.len() / self.input_len;

        self.setup_batch(batch_size);

        match self.exec {
            Executor::GPU(_) => {

            }
            Executor::CPU => {
                let activated_inputs = activated_inputs.cpu().unwrap();

                for batch in 0..batch_size {
                    let output_offset = batch * self.size;
                    let input_offset = batch * self.input_len;

                    let mut x = 0;
                    while x < self.size {
                        let outputs = self.outputs.get_mut(&batch_size).unwrap();
                        let output_index = x + output_offset;

                        outputs.cpu().unwrap().borrow_mut()[output_index] = self.biases.cpu().unwrap().borrow_mut()[x];

                        let mut y = 0;
                        while y < self.input_len {
                            let weight_index = (self.input_len * x) + y;
                            let in_value = activated_inputs.borrow_mut()[y + input_offset];

                            outputs.cpu().unwrap().borrow_mut()[output_index] += self.weights.cpu().unwrap().borrow_mut()[weight_index] * in_value;
                            y += 1
                        }
                        self.activated_outputs.get_mut(&batch_size).unwrap()
                            .cpu().unwrap().borrow_mut()[output_index] = self.activation.activate(self.outputs.get_mut(&batch_size).unwrap().cpu().unwrap().borrow()[output_index]);
                        x += 1;
                    }
                }
            }
        }
        batch_size
    }

    fn backward(&mut self, next_sensitivities: &DualVec, optimizer: &Optimizer)  {
        todo!()
    }

    fn activated_output(&mut self, batch_size: usize) -> &mut DualVec {
        self.activated_outputs.get_mut(&batch_size).unwrap()
    }

    fn as_bytes(&mut self, bytes: &mut VecWriter) {
        bytes.usize(self.input_len);
        bytes.usize(self.size);
        bytes.index(&self.activation);

        let weights = self.weights.cpu().unwrap();
        for w in weights.borrow().iter() {
            bytes.f32(*w);
        }

        let biases = self.biases.cpu().unwrap();
        for b in biases.borrow().iter() {
            bytes.f32(*b);
        }
    }

    fn from_bytes(exec: (&'a Executor, &'a Executor, &'a Executor), bytes: &mut CursorReader) -> Rc<RefCell<dyn Layer<'a> + 'a>> {
        let mut l = Dense::new(exec, bytes.usize(), bytes.usize(), bytes.indexed());

        let w = l.weights.cpu().unwrap();
        let mut weights = w.borrow_mut();
        for i in 0..weights.len() {
            weights[i] = bytes.f32();
        }

        let b = l.biases.cpu().unwrap();
        let mut biases = b.borrow_mut();
        for i in 0..biases.len() {
            biases[i] = bytes.f32();
        }

        Rc::new(RefCell::new(l))
    }

    fn id(&self) -> usize {
        0
    }

    fn exec(&self) -> &Executor {
        self.exec
    }

    fn weights(&self) -> &DualVec {
        &self.weights
    }
}