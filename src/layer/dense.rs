use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use ocl::{Kernel, ProQue};
use ocl::builders::KernelBuilder;

use crate::{Executor, Optimizer};
use crate::activation::Activation;
use crate::dual_vec::DualVec;
use crate::layer::Layer;
use crate::utils::cl_utils::execute_kernel;
use crate::utils::{cl_utils, gpu_math};
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

    outputs: DualVec,
    activated_outputs: DualVec,
    sensitivities: DualVec,

    forward_kernel: Option<Kernel>,
    backward_kernel: Option<Kernel>,
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
            backward_kernel: None,
        };

        a.weights.randomize(c, (a.weights.len() as f32).cbrt());
        a.biases.randomize(c, (a.bias_mods.len() as f32).sqrt());

        // a.weights.randomize(c, 10.);
        // a.biases.randomize(c, 10.);

        a
    }

    fn ensure_batch_size(&mut self, batch_size: usize) -> bool {
        let target = batch_size * self.size;
        if self.outputs.len() < target {
            self.outputs.expand_to(self.size * batch_size);
            self.activated_outputs.expand_to(self.size * batch_size);
            self.sensitivities.expand_to(self.input_len * batch_size);
            true
        } else if self.outputs.len() > target {
            self.outputs.truncate_to(self.size * batch_size);
            self.activated_outputs.truncate_to(self.size * batch_size);
            self.sensitivities.truncate_to(self.input_len * batch_size);
            true
        } else {
            false
        }
    }

    fn gpu_forward(&mut self, positions: &Vec<usize>, activated_inputs: &mut DualVec, pq: &ProQue) {
        {
            let outputs = self.outputs.gpu().unwrap();
            let mut outputs = outputs.borrow_mut();
            outputs.cmd().fill(0., None).enq();

            if self.forward_kernel.is_none() {
                let activation_id: usize = (&self.activation).into();
                self.forward_kernel = Some(
                    pq.kernel_builder("forward")
                        .arg(activation_id as u64)
                        .arg(self.input_len as u64)
                        .arg(self.size as u64)
                        .arg(&*self.weights.gpu_borrow().unwrap())
                        .arg(&*self.biases.gpu_borrow().unwrap())
                        .arg_named("inputs", &*activated_inputs.gpu_borrow().unwrap())
                        .arg(&*outputs)
                        .arg(&*self.activated_outputs.gpu().unwrap().borrow())
                        .arg_named("bo_i", 0u64)
                        .arg_named("bo_o", 0u64)
                        .build().unwrap()
                );
            }

            if let Some(kernel) = &self.forward_kernel {
                for batch in 0..positions.len() {
                    kernel.set_arg("bo_i", positions[batch] as u64);
                    kernel.set_arg("bo_o", (batch * self.size) as u64);
                    kernel.set_arg("inputs", &*activated_inputs.gpu_borrow().unwrap());

                    unsafe {
                        execute_kernel(pq, &kernel, (self.size, self.input_len));
                    }
                }

                self.activated_outputs.updated_gpu();
                self.outputs.updated_gpu();
            }
        }
    }

    fn cpu_forward(&mut self, positions: &Vec<usize>, activated_inputs: &mut DualVec) {
        {
            let activated_inputs = activated_inputs.cpu_borrow().unwrap();
            let biases = self.biases.cpu_borrow().unwrap();
            let weights = self.weights.cpu_borrow().unwrap();
            let mut outputs = self.outputs.cpu_borrow().unwrap();
            let mut activated_outputs = self.activated_outputs.cpu_borrow().unwrap();

            for batch in 0..positions.len() {
                let output_offset = batch * self.size;
                let input_offset = positions[batch];

                let mut x = 0;
                while x < self.size {
                    let output_index = x + output_offset;

                    outputs[output_index] = biases[x];

                    let mut y = 0;
                    while y < self.input_len {
                        let weight_index = (self.input_len * x) + y;
                        let in_value = activated_inputs[y + input_offset];

                        outputs[output_index] += weights[weight_index] * in_value;
                        y += 1
                    }
                    activated_outputs[output_index] = self.activation.activate(outputs[output_index]);

                    x += 1;
                }
            }
        }

        self.activated_outputs.updated_cpu();
        self.outputs.updated_cpu();
    }

    fn gpu_backward(&mut self, inputs: &mut DualVec, input_indices: Option<&Vec<usize>>, in_sensitivities: &mut DualVec, optimizer: &Optimizer, batch_size: usize, pq: &ProQue) {
        let lr = optimizer.learn_rate();

        self.sensitivities.gpu_borrow().unwrap().cmd().fill(0., None).enq();

        if self.backward_kernel.is_none() {
            self.backward_kernel = Some(pq.kernel_builder("backward")
                .arg(<&Activation as Into<usize>>::into((&self.activation)) as u64)
                .arg(self.input_len as u64)
                .arg(lr)
                .arg_named("inputs", &*inputs.gpu_borrow().unwrap())
                .arg(&*self.outputs.gpu_borrow().unwrap())
                .arg_named("in_sensitivities", &*in_sensitivities.gpu_borrow().unwrap())
                .arg(&*self.weights.gpu_borrow().unwrap())
                .arg(&*self.biases.gpu_borrow().unwrap())
                .arg(&*self.weight_mods.gpu_borrow().unwrap())
                .arg(&*self.bias_mods.gpu_borrow().unwrap())
                .arg(&*self.sensitivities.gpu_borrow().unwrap())
                .arg_named("bo_i", 0u64)
                .arg_named("bo_s", 0u64)
                .arg_named("bo_o", 0u64)
                .build().unwrap());
        }

        if let Some(k) = &self.backward_kernel {
            for batch in 0..batch_size {
                let out_offset = batch * self.size;
                let in_offset = match input_indices {
                    None => batch * self.input_len,
                    Some(indices) => {
                        indices[batch]
                    },
                };
                k.set_arg("bo_i", in_offset as u64);
                k.set_arg("bo_s", (batch * self.input_len) as u64);
                k.set_arg("bo_o", out_offset as u64);
                k.set_arg("inputs", &*inputs.gpu_borrow().unwrap());
                k.set_arg("in_sensitivities", &*in_sensitivities.gpu_borrow().unwrap());

                unsafe {
                    execute_kernel(pq, k, (self.size, self.input_len));
                }
            }

            self.sensitivities.updated_gpu();
            self.bias_mods.updated_gpu();
            self.weight_mods.updated_gpu();
        }
    }

    fn cpu_backward(&mut self, inputs: &mut DualVec, input_indices: Option<&Vec<usize>>, in_sensitivities: &mut DualVec, optimizer: &Optimizer, batch_size: usize) {
        let lr = optimizer.learn_rate();

        {
            let in_sensitivities = in_sensitivities.cpu_borrow().unwrap(); // 0

            let mut outputs = self.outputs.cpu_borrow().unwrap();

            let mut sensitivities = self.sensitivities.cpu_borrow().unwrap(); // -1
            sensitivities.fill(0.);

            let biases = self.biases.cpu_borrow().unwrap();
            let weights = self.weights.cpu_borrow().unwrap();

            let mut weight_mods = self.weight_mods.cpu_borrow().unwrap();
            let mut bias_mods = self.bias_mods.cpu_borrow().unwrap();
            let inputs = inputs.cpu_borrow().unwrap();

            for batch in 0..batch_size {
                let out_offset = batch * self.size;
                let in_offset = match &input_indices {
                    None => batch * self.input_len,
                    Some(indices) => indices[batch],
                };
                for x in 0..self.size {
                    let gradient = self.activation.derivative(outputs[x + out_offset]) * in_sensitivities[x + out_offset];

                    bias_mods[x] -= (lr * gradient);
                    for y in 0..self.input_len {
                        let weight_index = (self.input_len * x) + y;

                        weight_mods[weight_index] -= (lr * inputs[y + in_offset] * gradient);

                        sensitivities[y + (batch * self.input_len)] += gradient * weights[weight_index];
                    }
                }
            }
        }

        self.sensitivities.updated_cpu();
        self.weight_mods.updated_cpu();
        self.bias_mods.updated_cpu();
    }

    fn gpu_apply(&mut self, optimizer: &Optimizer, batch_size: usize, pq: &ProQue) {
        match optimizer {
            Optimizer::GradientDecent(lr) => {
                let lr = *lr;

                gpu_math::mult_second_and_add(
                    pq,
                    &*self.weights.gpu_borrow().unwrap(),
                    &*self.weight_mods.gpu_borrow().unwrap(),
                    1. / batch_size as f32
                );
                gpu_math::mult_second_and_add(
                    pq,
                    &*self.biases.gpu_borrow().unwrap(),
                    &*self.bias_mods.gpu_borrow().unwrap(),
                    1. / batch_size as f32
                );

                self.weight_mods.gpu_borrow().unwrap().cmd().fill(0., None).enq();
                self.bias_mods.gpu_borrow().unwrap().cmd().fill(0., None).enq();

                self.weight_mods.updated_gpu();
                self.bias_mods.updated_gpu();
                self.weights.updated_gpu();
                self.biases.updated_gpu();
            }
        }
    }

    fn cpu_apply(&mut self, optimizer: &Optimizer, batch_size: usize) {
        match optimizer {
            Optimizer::GradientDecent(lr) => {
                let lr = *lr;
                let i_batch_size = 1. / batch_size as f32;

                if let (
                    Some(mut weights), Some(mut biases),
                    Some(mut w_mods), Some(mut b_mods)
                ) = (
                    self.weights.cpu_borrow(), self.biases.cpu_borrow(),
                    self.weight_mods.cpu_borrow(), self.bias_mods.cpu_borrow()
                ) {
                    for i in 0..weights.len() {
                        weights[i] += w_mods[i] * i_batch_size;
                    }
                    for i in 0..biases.len() {
                        biases[i] += b_mods[i] * i_batch_size;
                    }

                    w_mods.fill(0.);
                    b_mods.fill(0.);
                }

                self.weight_mods.updated_cpu();
                self.bias_mods.updated_cpu();
                self.weights.updated_cpu();
                self.biases.updated_cpu();
            }
        }
    }
}

impl<'a> Layer<'a> for Dense<'a> {
    fn dynamic_forward(&mut self, positions: &Vec<usize>, inputs: &mut DualVec) {
        if self.ensure_batch_size(positions.len()) {
            // If the batch size changes, the GPU buffers will change, and so the kernels must be
            // re-created in order to not maintain old buffer references
            self.forward_kernel = None;
            self.backward_kernel = None;
        }

        match self.exec {
            Executor::GPU(pq) => self.gpu_forward(positions, inputs, pq),
            Executor::CPU => self.cpu_forward(positions, inputs),
        }
    }

    fn forward(&mut self, activated_inputs: &mut DualVec) -> usize {
        let batch_size = activated_inputs.len() / self.input_len;

        let mut positions = vec![];
        for i in 0..batch_size {
            positions.push(i * self.input_len);
        }

        self.dynamic_forward(&positions, activated_inputs);

        batch_size
    }

    fn backward(&mut self, inputs: &mut DualVec, input_indices: Option<&Vec<usize>>, in_sensitivities: &mut DualVec, optimizer: &Optimizer)  {
        let batch_size = in_sensitivities.len() / self.size;

        match &self.exec {
            Executor::GPU(pq) => self.gpu_backward(inputs, input_indices, in_sensitivities, optimizer, batch_size, pq),
            Executor::CPU => self.cpu_backward(inputs, input_indices, in_sensitivities, optimizer, batch_size),
        }
    }

    fn apply_gradients(&mut self, optimizer: &Optimizer, batch_size: usize) {
        match self.exec {
            Executor::GPU(pq) => self.gpu_apply(optimizer, batch_size, pq),
            Executor::CPU => self.cpu_apply(optimizer, batch_size),
        }
    }

    fn as_bytes(&mut self, bytes: &mut VecWriter) {
        let weights = self.weights.cpu_borrow().unwrap();
        let biases = self.biases.cpu_borrow().unwrap();
        bytes.reserve((3 * 8) + (weights.len() + biases.len()) * 4);

        bytes.usize(self.input_len);
        bytes.usize(self.size);
        bytes.index(&self.activation);

        for w in weights.iter() {
            bytes.f32(*w);
        }

        for b in biases.iter() {
            bytes.f32(*b);
        }
    }

    fn from_bytes(exec: (&'a Executor, &'a Executor, &'a Executor), bytes: &mut CursorReader) -> Rc<RefCell<dyn Layer<'a> + 'a>> {
        let mut l = Dense::new(exec, bytes.usize(), bytes.usize(), bytes.indexed());

        if let Some(mut weights) = l.weights.cpu_borrow() {
            for i in 0..weights.len() {
                weights[i] = bytes.f32();
            }
        }

        if let Some(mut biases) = l.biases.cpu_borrow() {
            for i in 0..biases.len() {
                biases[i] = bytes.f32();
            }
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

    fn values(&self) -> Vec<&DualVec> {
        vec![&self.weights, &self.biases]
    }

    fn input_size(&self) -> usize {
        self.input_len
    }

    fn output_size(&self) -> usize {
        self.size
    }

    fn activated_output(&mut self) -> &mut DualVec {
        &mut self.activated_outputs
    }

    fn sensitivities(&mut self) -> &mut DualVec {
        &mut self.sensitivities
    }
}