use std::cell::RefCell;
use std::rc::Rc;
use ocl::Buffer;
use rand::random;
use crate::Executor;
use crate::Executor::GPU;
use crate::utils::cl_utils;

#[derive(Debug)]
pub struct DualVec {
    len: usize,
    cpu: (Option<Rc<RefCell<Vec<f32>>>>, bool),
    gpu: (Option<Rc<RefCell<Buffer<f32>>>>, bool),
}


impl DualVec {
    pub fn from_execs(exec: (&Executor, &Executor), len: usize) -> Self {
        let gpu = match exec {
            (Executor::GPU(q), _) | (_, Executor::GPU(q)) => Some(Rc::new(RefCell::new(cl_utils::new_buffer(q, len)))),
            _ => None,
        };
        let cpu = match exec {
            (Executor::CPU, _) | (_, Executor::CPU) => Some(Rc::new(RefCell::new(vec![0.; len]))),
            _ => None,
        };
        DualVec {
            len,
            cpu: (cpu, false),
            gpu: (gpu, false),
        }
    }

    pub fn randomize(&mut self, exec: &Executor) {
        match exec {
            GPU(_) => {}
            Executor::CPU => {
                match self.cpu() {
                    None => {}
                    Some(vec) => {
                        for i in 0..self.len {
                            vec.borrow_mut()[i] = (random::<f32>() * 2.0 - 1.0) / self.len as f32 * 4.;
                        }
                    }
                }
            }
        }
    }

    pub fn from_exec(exec: &Executor, len: usize) -> Self {
        DualVec {
            len,
            cpu: (match exec {
                Executor::GPU(_) => None,
                Executor::CPU => Some(Rc::new(RefCell::new(vec![0.; len]))),
            }, false),
            gpu: (match exec {
                Executor::GPU(q) => Some(Rc::new(RefCell::new(cl_utils::new_buffer(q, len)))),
                Executor::CPU => None,
            }, false)
        }
    }

    pub fn clear(&mut self) {
        if self.cpu.1 {
            self.cpu.0.as_mut().unwrap().borrow_mut().fill(0.);
            self.cpu.1 = false;
        }
        if self.gpu.1 {
            self.gpu.0.as_ref().unwrap().borrow_mut().cmd().fill(0., None).enq().unwrap();
            self.gpu.1 = false;
        }
    }

    pub fn gpu(&mut self) -> Option<Rc<RefCell<Buffer<f32>>>> {
        if self.gpu.0.is_none() {
            return None;
        }

        if !self.gpu.1 && self.cpu.1 {
            cl_utils::buf_write(&*self.gpu.0.clone().unwrap().borrow_mut(), &*self.cpu.0.clone().unwrap().borrow_mut());
        }
        self.gpu.0.clone()
    }

    pub fn cpu(&mut self) -> Option<Rc<RefCell<Vec<f32>>>> {
        if self.cpu.0.is_none() {
            return None;
        }

        if !self.cpu.1 && self.gpu.1 {
            cl_utils::read_to(&*self.gpu.0.clone().unwrap().borrow_mut(), &mut *self.cpu.0.clone().unwrap().borrow_mut());
        }
        self.cpu.0.clone()
    }

    pub fn update_gpu(&mut self) {
        if self.gpu.0.is_some() {
            self.gpu.1 = true;
        }
    }

    pub fn update_cpu(&mut self) {
        if self.cpu.0.is_some() {
            self.cpu.1 = true;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Clone for DualVec {
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            cpu: self.cpu.clone(),
            gpu: self.gpu.clone(),
        }
    }
}