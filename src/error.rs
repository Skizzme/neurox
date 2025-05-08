use crate::dual_vec::DualVec;
use crate::Executor;
use crate::Executor::GPU;

pub enum Error {
    Categorical,
    MeanSquared,
}

impl Error {
    pub fn calculate(&self, exec: &Executor, actual: &mut DualVec, target: &mut DualVec) -> f32 {
        match self {
            Error::Categorical => match exec {
                GPU(_) => todo!(),
                Executor::CPU => {
                    todo!()
                }
            },
            Error::MeanSquared => match exec {
                GPU(_) => todo!(),
                Executor::CPU => {
                    let actual = actual.cpu_borrow().unwrap();
                    let target = target.cpu_borrow().unwrap();
                    let mut error = 0.;
                    for i in 0..actual.len() {
                        error += (actual[i]-target[i]).powf(2.0);
                    }
                    error / actual.len() as f32
                }
            }
        }
    }

    pub fn derivative(&self, exec: &Executor, actual: &mut DualVec, target: &mut DualVec) -> DualVec {
        todo!()
    }

    pub fn dynamic_derivative(&self, exec: &Executor, actual: &mut DualVec, target: &mut DualVec, target_positions: &Vec<usize>) -> DualVec {

        match self {
            Error::Categorical => match exec {
                GPU(_) => todo!(),
                Executor::CPU => todo!(),
            }
            Error::MeanSquared => match exec {
                GPU(_) => todo!(),
                Executor::CPU => {
                    let mut out = DualVec::from_exec(exec, actual.len());
                    {
                        let mut gradients = out.cpu_borrow().unwrap();

                        let actual = actual.cpu_borrow().unwrap();
                        let target = target.cpu_borrow().unwrap();
                        for i in 0..actual.len() {
                            gradients[i] = (2. * (actual[i] - target[i]) / actual.len() as f32);
                        }
                    }

                    out
                }
            }
        }
    }
}