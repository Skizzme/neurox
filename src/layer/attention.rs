use crate::dual_vec::DualVec;
use crate::{Executor, Optimizer};
use crate::layer::Layer;

#[derive(Debug)]
pub struct Attention<'a> {
    exec: &'a Executor,
}

impl<'a> Attention<'a> {
    pub fn new(exec: (&'a Executor, &'a Executor, &'a Executor), inputs: usize, head_count: usize, internal: usize, characteristics: usize, output: usize) -> Self {
        let p_to_c = (exec.0, exec.1); // previous to current
        let c_to_n = (exec.1, exec.2); // current to next
        let c = exec.1; // current
        Attention {
            exec: c,
        }
    }
}

impl<'a> Layer for Attention<'a> {
    fn forward(&mut self, inputs: &mut DualVec) -> usize {
        todo!()
    }

    fn backward(&mut self, next_gradients: &DualVec, optimizer: &Optimizer) {
        todo!()
    }

    fn activated_output(&mut self, batch_size: usize) -> &mut DualVec {
        todo!()
    }
}