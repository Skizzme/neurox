use std::cell::RefCell;
use std::rc::Rc;

use crate::{Executor, Optimizer};
use crate::dual_vec::DualVec;
use crate::layer::Layer;
use crate::utils::vec_utils::{CursorReader, VecWriter};

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

impl<'a> Layer<'a> for Attention<'_> {
    fn forward(&mut self, inputs: &mut DualVec) -> usize {
        todo!()
    }

    fn backward(&mut self, next_gradients: &mut DualVec, optimizer: &Optimizer) {
        todo!()
    }

    fn activated_output(&mut self) -> &mut DualVec {
        todo!()
    }

    fn as_bytes(&mut self, writer: &mut VecWriter) {
        todo!()
    }

    fn from_bytes(exec: (&'a Executor, &'a Executor, &'a Executor), bytes: &mut CursorReader) -> Rc<RefCell<dyn Layer<'a> + 'a>> {
        todo!()
    }

    fn id(&self) -> usize {
        1
    }

    fn exec(&self) -> &Executor {
        self.exec
    }

    fn weights(&self) -> &DualVec {
        todo!()
    }
}