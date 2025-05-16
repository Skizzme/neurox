use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::{Executor, Optimizer};
use crate::activation::Activation;
use crate::dual_vec::DualVec;
use crate::layer::attention::Attention;
use crate::layer::dense::Dense;
use crate::utils::vec_utils::{CursorReader, VecWriter};

pub mod dense;
pub mod attention;

pub trait Layer<'a> {
    fn dynamic_forward(&mut self, positions: &Vec<usize>, inputs: &mut DualVec);
    fn forward(&mut self, activated_inputs: &mut DualVec) -> usize;

    fn backward(&mut self, inputs: &mut DualVec, input_indices: Option<&Vec<usize>>, gradients: &mut DualVec, optimizer: &Optimizer);
    fn apply_gradients(&mut self, optimizer: &Optimizer, batch_size: usize);

    fn as_bytes(&mut self, writer: &mut VecWriter);
    fn from_bytes(exec: (&'a Executor, &'a Executor, &'a Executor), bytes: &mut CursorReader)
        -> Rc<RefCell<dyn Layer<'a> + 'a>> where Self: Sized;

    fn id(&self) -> usize;
    fn exec(&self) -> &Executor;

    fn values(&self) -> Vec<&DualVec>;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    fn activated_output(&mut self) -> &mut DualVec;
    fn sensitivities(&mut self) -> &mut DualVec;
}

#[derive(Clone, Debug)]
pub enum LayerType {
    /// size, activation
    Dense(usize, Activation),
    /// head count, internal size (d_k), characteristics (d_model), output size (d_v)
    Attention(usize, usize, usize, usize),
}

impl LayerType {
    pub fn layer<'a>(&'a self, exec: (&'a Executor, &'a Executor, &'a Executor), inputs: usize) -> (Rc<RefCell<dyn Layer<'a> + 'a>>, usize) {
        match self {
            LayerType::Dense(s, a) => (Rc::new(RefCell::new(Dense::new(exec, inputs, *s, a.clone()))), *s),
            LayerType::Attention(h, i, m, o) => (Rc::new(RefCell::new(Attention::new(exec, inputs, *h, *i, *m, *o))), *o),
        }
    }
}