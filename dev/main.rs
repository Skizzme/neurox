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
        (&CPU, Dense(1024, Linear)),
        // (&CPU, Dense(2056, Linear)),
        (&CPU, Dense(4, Linear)),
        (&CPU, Dense(2, Linear))
    ];
    let mut network = Network::new(2, layers);
    let mut input = DualVec::from_exec(&CPU, 2);
    input.cpu().unwrap().borrow_mut().clear();
    input.cpu().unwrap().borrow_mut().append(&mut vec![-0.7, 0.3]);

    let st = Instant::now();
    let mut output = network.predict(&mut input);
    let d = st.elapsed();
    println!("before {:?} in {:?} {:?}", input, output, d);
    let st = Instant::now();
    let bytes = network.to_bytes();
    let d = st.elapsed();
    println!("to bytes in {:?}", d);

    let st = Instant::now();
    let mut after = Network::from_bytes(Some(gpu), bytes.clone());
    let d = st.elapsed();
    println!("from bytes in {:?}", d);

    let a_output = after.predict(&mut input);
    println!("after in {:?} out {:?}", input, a_output);
}