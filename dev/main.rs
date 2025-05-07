#![allow(unused)]

use std::time::Instant;
use neurox::{Executor};
use neurox::layer::activation::Activation::{Linear, TanH};
use neurox::dual_vec::DualVec;
use neurox::Executor::CPU;
use neurox::layer::LayerType::Dense;
use neurox::network::Network;

pub fn main() {
    let gpu = &Executor::gpu();
    let layers = &vec![
        // (gpu, Attention(6, 12, 20, 44)),
        // (gpu, Dense(46, ReLU)),
        (&CPU, Dense(2056, Linear)),
        (&CPU, Dense(1024, Linear)),
        (&CPU, Dense(4, Linear)),
        (&CPU, Dense(2, Linear))
    ];
    let mut network = Network::new(2, layers);

    // the DualVec is a single object representing a CPU and GPU vec / array that makes using it easier,
    // and allows for different layers to be moved from the GPU to CPU or visa-versa for whatever reason
    // while still allowing a very general implementation of each.
    let mut input = DualVec::from_exec(&CPU, 6); // if this list size was a multiple of the input size, it would automatically do everything required to batch a single call
    // i will make it so that lists like this dont have to be cleared and then appended
    input.cpu().unwrap().borrow_mut().clear();
    input.cpu().unwrap().borrow_mut().append(&mut vec![-0.7, 0.3, 0.1, 0.5, -0.7, 0.3]);

    let mut output = network.predict(&mut input);
    let output_vec = output.cpu().unwrap().borrow().clone(); // returns Vec<f32>

    println!("{:?}", output_vec);
    let st = Instant::now();
    let mut output = network.predict(&mut input);
    let d = st.elapsed();
    let output_vec = output.cpu().unwrap().borrow().clone(); // returns Vec<f32>

    println!("{:?} {:?}", output_vec, d);

    // i haven't yet implemented a training method, but it will work similarly easily.

    let st = Instant::now();
    let bytes = network.as_bytes();
    let d = st.elapsed();
    println!("to bytes in {:?}", d);
    // could then write bytes to a file and load them like so
    let st = Instant::now();
    let loaded_network = Network::from_bytes(Some(gpu), bytes);
    let d = st.elapsed();
    println!("loaded in {:?}", d);
}