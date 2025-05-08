use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ocl::{Kernel, ProQue};

use crate::{Executor, Optimizer};
use crate::dual_vec::DualVec;
use crate::layer::activation::Activation;
use crate::layer::Layer;
use crate::utils::cl_utils::execute_kernel;
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

    // TODO change this to use max size buffer instead of hashmap
    outputs: DualVec,
    activated_outputs: DualVec,
    sensitivities: DualVec,

    forward_kernel: Option<Kernel>,
}

impl<'a> Dense<'a> {

    pub fn new(exec: (&'a Executor, &'a Executor, &'a Executor), inputs: usize, size: usize, activation: Activation) -> Self {
        let c = exec.1; // current
        let p_to_c = (exec.0, exec.1); // previous to current
        let c_to_n = (exec.1, exec.2); // current to next

        let mut a = Dense {
            exec: exec.1,
            execs: exec,
            size,
            input_len: inputs,

            activation,
            outputs: DualVec::from_execs(c_to_n, size),
            activated_outputs: DualVec::from_execs(c_to_n, size),

            sensitivities: DualVec::from_execs(p_to_c,inputs),

            weights: DualVec::from_exec(c, inputs * size),
            biases: DualVec::from_exec(c, size),

            weight_mods: DualVec::from_exec(c, inputs * size),
            bias_mods: DualVec::from_exec(c, size),
            forward_kernel: None,
        };

        a.weights.randomize(c);
        a.biases.randomize(c);

        a
    }

    fn ensure_batch_size(&mut self, batch_size: usize) {
        if self.outputs.capacity() < batch_size * self.size {
            self.outputs.expand_to(self.size * batch_size);
            self.activated_outputs.expand_to(self.size * batch_size);
            self.sensitivities.expand_to(self.input_len * batch_size);
        }
    }

    fn gpu_forward(&mut self, activated_inputs: &mut DualVec, batch_size: usize, pq: &ProQue) {
        let outputs = self.outputs.gpu().unwrap();
        let mut outputs = outputs.borrow_mut();
        outputs.cmd().fill(0., None).enq();

        if self.forward_kernel.is_none() {
            self.forward_kernel = Some(
                pq.kernel_builder("forward")
                    .arg(self.input_len as u64)
                    .arg(self.size as u64)
                    .arg(&*self.weights.gpu().unwrap().borrow())
                    .arg(&*self.biases.gpu().unwrap().borrow())
                    .arg(&*activated_inputs.gpu().unwrap().borrow())
                    .arg(&*outputs)
                    .arg(&*self.activated_outputs.gpu().unwrap().borrow())
                    .arg_named("bo_i", 0u64)
                    .arg_named("bo_o", 0u64)
                    .build().unwrap()
            );
        }

        if let Some(kernel) = &self.forward_kernel {
            for batch in 0..batch_size {
                kernel.set_arg("bo_i", (batch * self.input_len) as u64);
                kernel.set_arg("bo_o", (batch * self.size) as u64);

                unsafe {
                    execute_kernel(pq, &kernel, (self.size, self.input_len));
                }

                self.activated_outputs.updated_gpu();
                self.outputs.updated_gpu();
            }
        }
    }

    fn cpu_forward(&mut self, activated_inputs: &mut DualVec, batch_size: usize) {
        let activated_inputs = activated_inputs.cpu().unwrap();

        for batch in 0..batch_size {
            let output_offset = batch * self.size;
            let input_offset = batch * self.input_len;

            let mut x = 0;
            while x < self.size {
                let outputs = self.outputs.cpu().unwrap();
                let mut outputs = outputs.borrow_mut();
                let output_index = x + output_offset;

                outputs[output_index] = self.biases.cpu().unwrap().borrow_mut()[x];

                let mut y = 0;
                while y < self.input_len {
                    let weight_index = (self.input_len * x) + y;
                    let in_value = activated_inputs.borrow_mut()[y + input_offset];

                    outputs[output_index] += self.weights.cpu().unwrap().borrow_mut()[weight_index] * in_value;
                    y += 1
                }
                self.activated_outputs.cpu().unwrap().borrow_mut()[output_index] = self.activation.activate(outputs[output_index]);

                self.activated_outputs.updated_cpu();
                self.outputs.updated_cpu();

                x += 1;
            }
        }
    }
}

impl<'a> Layer<'a> for Dense<'a> {
    fn forward(&mut self, activated_inputs: &mut DualVec) -> usize {
        let batch_size = activated_inputs.len() / self.input_len;

        self.ensure_batch_size(batch_size);

        match self.exec {
            Executor::GPU(pq) => self.gpu_forward(activated_inputs, batch_size, pq),
            Executor::CPU => self.cpu_forward(activated_inputs, batch_size),
        }

        batch_size
    }

    fn backward(&mut self, next_sensitivities: &DualVec, optimizer: &Optimizer)  {
        todo!()
    }

    fn activated_output(&mut self) -> &mut DualVec {
        &mut self.activated_outputs
    }

    fn as_bytes(&mut self, bytes: &mut VecWriter) {
        let weights = self.weights.cpu().unwrap();
        let biases = self.biases.cpu().unwrap();
        bytes.reserve((3 * 8) + (weights.borrow().len() + biases.borrow().len()) * 4);

        bytes.usize(self.input_len);
        bytes.usize(self.size);
        bytes.index(&self.activation);

        for w in weights.borrow().iter() {
            bytes.f32(*w);
        }

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

        l.weights.updated_cpu();
        l.biases.updated_cpu();

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