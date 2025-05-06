use std::cell::RefCell;
use std::rc::Rc;
use crate::activation::Activation;
use crate::dual_vec::DualVec;
use crate::{Executor, Optimizer};
use crate::layer::attention::Attention;
use crate::layer::dense::Dense;

pub mod dense;
pub mod attention;

impl Default for Box<dyn Layer> {
    fn default() -> Self {
        Box::new(DummyLayer {})
    }
}

pub trait LayerCodec {
    fn to_values(&mut self) -> Vec<u8>;
    fn set_values(&mut self, values: Vec<u8>);
}

pub trait Layer {
    fn forward(&mut self, activated_inputs: &mut DualVec) -> usize;
    fn backward(&mut self, next_sensitivities: &DualVec, optimizer: &Optimizer);

    fn activated_output(&mut self, batch_size: usize) -> &mut DualVec;
}

#[derive(Clone, Debug)]
pub enum LayerType {
    /// size, activation
    Dense(usize, Activation),
    /// head count, internal size (d_k), characteristics (d_model), output size (d_v)
    Attention(usize, usize, usize, usize),
}

impl LayerType {
    pub fn layer<'a>(&self, exec: (&'a Executor, &'a Executor, &'a Executor), inputs: usize) -> (Rc<RefCell<dyn Layer + 'a>>, usize) {
        match self {
            LayerType::Dense(s, a) => (Rc::new(RefCell::new(Dense::new(exec, inputs, *s, a.clone()))), *s),
            LayerType::Attention(h, i, m, o) => (Rc::new(RefCell::new(Attention::new(exec, inputs, *h, *i, *m, *o))), *o),
        }
    }

    pub fn from_values<'a>(&self, exec: (&'a Executor, &'a Executor, &'a Executor), inputs: usize) -> (Rc<RefCell<dyn Layer + 'a>>, usize) {
        match self {
            LayerType::Dense(s, a) => (Rc::new(RefCell::new(Dense::new(exec, inputs, *s, a.clone()))), *s),
            LayerType::Attention(h, i, m, o) => (Rc::new(RefCell::new(Attention::new(exec, inputs, *h, *i, *m, *o))), *o),
        }
    }
}


struct DummyLayer {}
impl Layer for DummyLayer {
    fn forward(&mut self, activated_inputs: &mut DualVec) -> usize {
        0
    }

    fn backward(&mut self, next_sensitivities: &DualVec, optimizer: &Optimizer) {

    }

    fn activated_output(&mut self, batch_size: usize) -> &mut DualVec {
        todo!()
    }
}