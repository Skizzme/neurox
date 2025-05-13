use crate::dual_vec::DualVec;
use crate::error::Error;
use crate::error::Error::UnavailableBuffer;
use crate::Executor;
use crate::Executor::{CPU, GPU};

pub enum Loss {
    Categorical,
    MeanSquared,
}

impl Loss {
    pub fn calculate(&self, exec: &Executor, actual: &mut DualVec, output_size: usize, target: &mut DualVec, target_indices: &Vec<usize>, batch_size: usize) -> Result<DualVec, Error> {
        match self {
            Loss::Categorical => match exec {
                GPU(_) => todo!(),
                Executor::CPU => {
                    todo!()
                }
            },
            Loss::MeanSquared => match exec {
                GPU(_) => todo!(),
                Executor::CPU => {
                    if let (Some(actual), Some(target)) = (actual.cpu_borrow(), target.cpu_borrow()) {
                        let mut losses = Vec::with_capacity(actual.len());
                        for batch in 0..batch_size {
                            let mut error = 0.;
                            for i in 0..output_size {
                                error += (actual[i + batch * output_size] - target[i + target_indices[batch]]).powf(2.0);
                            }
                            losses.push(error / actual.len() as f32);
                        }
                        Ok(DualVec::from_vec((&CPU, exec), losses))
                    } else {
                        Err(UnavailableBuffer("CPU buffer was not available when calculating error".to_string()))
                    }
                }
            }
        }
    }

    pub fn dynamic_derivative(&self, exec: &Executor, actual: &mut DualVec, target: &mut DualVec, target_indices: &Vec<usize>) -> DualVec {

        match self {
            Loss::Categorical => match exec {
                GPU(_) => todo!(),
                Executor::CPU => todo!(),
            }
            Loss::MeanSquared => match exec {
                GPU(_) => todo!(),
                Executor::CPU => {
                    let mut out = DualVec::from_exec(exec, actual.len());
                    if let (Some(mut gradients), Some(actual), Some(target)) =
                        (out.cpu_borrow(), actual.cpu_borrow(), target.cpu_borrow()) {
                        for i in 0..actual.len() {
                            gradients[i] = (2. * (actual[i] - target[target_indices[i]+i]) / actual.len() as f32);
                        }
                    }

                    out
                }
            }
        }
    }
}