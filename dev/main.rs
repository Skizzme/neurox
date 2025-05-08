#![allow(unused)]

use std::fs;
use std::fs::read;
use std::time::Instant;
use neurox::{Executor};
use neurox::layer::activation::Activation::{Linear, ReLU, TanH};
use neurox::dual_vec::DualVec;
use neurox::Executor::CPU;
use neurox::layer::LayerType::Dense;
use neurox::network::Network;

pub fn main() {
    let gpu = &Executor::gpu();
    let layers = &vec![
        // (gpu, Attention(6, 12, 20, 44)),
        // (gpu, Dense(46, ReLU)),
        // (&CPU, Dense(2056, TanH)),
        // (&CPU, Dense(1024, TanH)),
        (gpu, Dense(1024, ReLU)),
        (gpu, Dense(2056, Linear)),
        (gpu, Dense(2056, TanH)),
        (gpu, Dense(2056, Linear)),
        (gpu, Dense(2, ReLU)),
        (gpu, Dense(2, TanH)),
        // (gpu, Dense(4, TanH)),
        // (gpu, Dense(2, TanH)),
    ];
    let mut network = Network::new(2, layers);
    // let mut network = Network::from_bytes(Some(gpu), read("test.neurox").unwrap());
    // let mut network = Network::from_bytes(None, read("test.neurox").unwrap());

    // [-0.86633563, -0.56801283, -0.86633563, -0.56801283, -0.86633563, -0.56801283]
    //

    // the DualVec is a single object representing a CPU and GPU vec / array that makes using it easier,
    // and allows for different layers to be moved from the GPU to CPU or visa-versa for whatever reason
    // while still allowing a very general implementation of each.
    let mut input = DualVec::from_execs((&CPU, gpu), 6); // if this list size was a multiple of the input size, it would automatically do everything required to batch a single call
    // i will make it so that lists like this dont have to be cleared and then appended
    input.cpu().unwrap().borrow_mut().clear();
    input.cpu().unwrap().borrow_mut().append(&mut vec![-0.7, 0.3, 0.1, 0.5, -0.7, 0.3]);
    input.updated_cpu();

    let st = Instant::now();
    let mut output = network.predict(&mut input);
    let d = st.elapsed();
    let output_vec = output.cpu().unwrap().borrow().clone(); // returns Vec<f32>

    println!("{:?} {:?}", output_vec, d);

    let st = Instant::now();
    let mut output = network.predict(&mut input);
    let d = st.elapsed();
    let output_vec = output.cpu().unwrap().borrow().clone(); // returns Vec<f32>

    println!("{:?} {:?}", output_vec, d);

    // i haven't yet implemented a training method, but it will work similarly easily.

    let st = Instant::now();
    let bytes = network.as_bytes();
    let d = st.elapsed();

    // fs::write("test.neurox", &bytes);

    println!("to bytes in {:?}", d);
    // could then write bytes to a file and load them like so
    let st = Instant::now();
    let loaded_network = Network::from_bytes(Some(gpu), bytes);
    let d = st.elapsed();
    println!("loaded in {:?}", d);
}