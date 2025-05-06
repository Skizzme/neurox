use std::cell::RefCell;
use std::rc::Rc;
use activation::Activation;
use crate::dual_vec::DualVec;
use crate::{Executor, Optimizer};
use crate::layer::attention::Attention;
use crate::layer::dense::Dense;
use crate::utils::vec_utils::{CursorReader, VecWriter};

pub mod dense;
pub mod attention;
pub mod activation;

impl Default for Box<dyn Layer> {
    fn default() -> Self {
        Box::new(DummyLayer {})
    }
}

pub trait Layer<'a> {
    fn forward(&mut self, activated_inputs: &mut DualVec) -> usize;
    fn backward(&mut self, next_sensitivities: &DualVec, optimizer: &Optimizer);
    fn activated_output(&mut self, batch_size: usize) -> &mut DualVec;

    fn to_bytes(&mut self, writer: &mut VecWriter);
    fn from_bytes(exec: (&'a Executor, &'a Executor, &'a Executor), bytes: &mut CursorReader) -> Rc<RefCell<dyn Layer + 'a>> where Self: Sized;

    fn id(&self) -> usize;
    fn exec(&self) -> &Executor;

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

    fn to_bytes(&mut self, writer: &mut VecWriter) {
    }

    fn from_bytes<'a>(exec: (&'a Executor, &'a Executor, &'a Executor), bytes: &mut CursorReader) -> Rc<RefCell<dyn Layer>> {
        Rc::new(RefCell::new(DummyLayer {}))
    }

    fn id(&self) -> usize {
        todo!()
    }

    fn exec(&self) -> &Executor {
        &Executor::CPU
    }
}
