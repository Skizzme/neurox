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
    capacity: usize,
    cpu: (Option<Rc<RefCell<Vec<f32>>>>, bool),
    gpu: (Option<Rc<RefCell<Buffer<f32>>>>, bool),
}


impl DualVec {
    pub fn from_execs(exec: (&Executor, &Executor), len: usize) -> Self {
        let gpu = match exec {
            (Executor::GPU(q), _) | (_, Executor::GPU(q)) => Some(Rc::new(RefCell::new(cl_utils::new_buffer(q, len)))),
            _ => None,
        };
        // Ensures that only a copy on both the GPU and CPU are made only if it will
        // be used on both, otherwise it would waste memory creating unused copies.
        let cpu = match exec {
            (Executor::CPU, _) | (_, Executor::CPU) => Some(Rc::new(RefCell::new(vec![0.; len]))),
            _ => None,
        };

        DualVec {
            len,
            capacity: len,
            cpu: (cpu, false),
            gpu: (gpu, false),
        }
    }

    pub fn randomize(&mut self, exec: &Executor) {
        match exec {
            GPU(p) => {
                if let Some(buf) = &self.gpu.0 {
                    cl_utils::randomize_buffer(&*buf.borrow(), 256, self.len as f32/ 10., p);
                    self.updated_gpu();
                }
            }
            Executor::CPU => {
                if let Some(vec) = self.cpu() {
                    for i in 0..self.len {
                        vec.borrow_mut()[i] = (random::<f32>() * 2.0 - 1.0) / self.len as f32 * 4.;
                    }
                    self.updated_cpu();
                }
            }
        }
    }

    pub fn from_exec(exec: &Executor, len: usize) -> Self {
        DualVec {
            len,
            capacity: len,
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
            // Can't create a GPU buffer, since a Queue is required and should have been when creating the buffer
            return None;
        }

        if self.cpu.1 {
            cl_utils::buf_write(&*self.gpu.0.clone().unwrap().borrow_mut(), &*self.cpu.0.clone().unwrap().borrow_mut());
            self.cpu.1 = false;
        }
        self.gpu.0.clone()
    }

    pub fn cpu(&mut self) -> Option<Rc<RefCell<Vec<f32>>>> {
        if self.cpu.0.is_none() {
            let mut vec = vec![0.; self.len];
            vec.reserve_exact(self.capacity - self.len);
            self.cpu.0.replace(Rc::new(RefCell::new(vec)));
        }

        if self.gpu.1 {
            cl_utils::read_to(&*self.gpu.0.clone().unwrap().borrow_mut(), &mut *self.cpu.0.clone().unwrap().borrow_mut());
            self.gpu.1 = false;
        }
        self.cpu.0.clone()
    }

    pub fn updated_gpu(&mut self) {
        if self.gpu.0.is_some() {
            self.gpu.1 = true;
        }
    }

    pub fn updated_cpu(&mut self) {
        if self.cpu.0.is_some() {
            self.cpu.1 = true;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn truncate_to(&mut self, size: usize) {
        // No modifications to the GPU buffer are necessary with OpenCL, since all
        // sizes are passed through the kernel parameters, and so no actual change is needed.
        if let Some(cpu) = &self.cpu.0 {
            let mut vec = cpu.borrow_mut();
            if vec.len() > size {
                vec.truncate(size);
            }
        }
        self.len = self.len.min(size);
    }

    /// Expands the necessary GPU and/or CPU storage to have a capacity of size
    pub fn expand_to(&mut self, size: usize) {
        if let Some(gpu) = &self.gpu.0 {
            let mut new_buf = None;
            {
                let buf = gpu.borrow_mut();
                if buf.len() < size {
                    let tmp_buffer = Buffer::builder()
                        .queue(buf.default_queue().unwrap().clone())
                        .len(size)
                        .build()
                        .expect("Failed to create new buffer");

                    buf.copy(&tmp_buffer, None, None).enq()
                        .expect("Failed to copy buffer");

                    new_buf = Some(tmp_buffer);
                }
            }
            if let Some(buf) = new_buf {
                gpu.replace(buf);
            }
        }

        if let Some(cpu) = &self.cpu.0 {
            let mut vec = cpu.borrow_mut();

            let capacity = vec.capacity();
            if capacity < size {
                vec.reserve_exact(size - capacity);
                vec.resize(size, 0.);
            }
        }
        self.capacity = self.capacity.max(size);
        self.len = self.capacity;
    }
}

impl Clone for DualVec {
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            capacity: self.capacity,
            cpu: self.cpu.clone(),
            gpu: self.gpu.clone(),
        }
    }
}