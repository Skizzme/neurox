use std::cell::RefCell;
use std::mem;
use std::rc::Rc;
use num_format::Locale::ne;
use ocl::{Buffer, ProQue};
use rand::random;
use crate::cl_utils;
use crate::network::Executor::{CPU, GPU};
use crate::network::layer::{Layer, LayerType};
use crate::network_old::cpu::activate;

pub mod layer;



